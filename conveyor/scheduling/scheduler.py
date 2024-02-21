from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Tuple
from attr import dataclass
from conveyor.scheduling.cache_manager import CacheManager
from conveyor.scheduling.context import InferenceContext, InferenceState, RequestInfo
from conveyor.scheduling.request_pool import RequestPool
from transformers import PretrainedConfig
import torch
from torch import nn
import conveyor
import importlib


@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    for module_path in (Path(conveyor.__file__).parent / "models").glob("*.py"):
        module = importlib.import_module(f"conveyor.models.{module_path.stem}")
        if hasattr(module, "EntryClass"):
            model_arch_name_to_cls[module.EntryClass.__name__] = module.EntryClass
    return model_arch_name_to_cls


@dataclass
class SchedulerContext:
    requests: list[RequestInfo]
    pending_requests: list[RequestInfo]
    cache_manager: CacheManager

    # Batch info
    seq_lens: torch.Tensor
    completed_lens: torch.Tensor

    @classmethod
    def new() -> SchedulerContext:
        raise NotImplementedError

    def add_active_request(self, req: RequestInfo) -> None:
        self.requests.append(req)


def compute_page_needed(
    seq_lens: torch.Tensor, completed_lens: torch.Tensor, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    return page_needed, page_idx_start
    """
    # token index: [completed_lens, seq_lens-1]
    page_end = (seq_lens - 1) // page_size
    page_start = completed_lens // page_size
    page_start_not_allocated = completed_lens % page_size == 0
    page_needed = page_end - page_start + page_start_not_allocated
    return page_needed, page_start


class ScheduleEngine:
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.model = ScheduleEngine.load_model(config)
        self.cache_manager = CacheManager(256)  # TODO: FIXME
        self.request_pool = RequestPool()
        self.max_concurrent_requests = 16
        self.context = SchedulerContext.new()

    @torch.inference_mode()
    def iteration_step(self):
        if self.new_request_available():
            new_request = self.request_pool.pop_request()
            self.context.add_active_request(new_request)
            self.forward_prefill(self.context)
        else:
            self.forward_decode(self.context)

    def new_request_available(self) -> bool:
        # TODO: better policy
        return (
            len(self.request_pool.queued_requests) > 0
            and len(self.context.requests) < self.max_concurrent_requests
        )

    @staticmethod
    def load_model(config: PretrainedConfig) -> nn.Module:
        def get_model_cls_by_arch_name(model_arch_names):
            model_arch_name_to_cls = import_model_classes()
            model_class = None
            for arch in model_arch_names:
                if arch in model_arch_name_to_cls:
                    model_class = model_arch_name_to_cls[arch]
                    break
            else:
                raise ValueError(
                    f"Unsupported architectures: {arch}. "
                    f"Supported list: {list(model_arch_name_to_cls.keys())}"
                )
            return model_class

        architectures = getattr(config.hf_config, "architectures", [])
        model_class = get_model_cls_by_arch_name(architectures)

        # Load weights
        linear_method = None
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        with torch.device("cuda"):
            model = model_class(config=config.hf_config, linear_method=linear_method)
        model.load_weights(
            config.path,
            cache_dir=None,
            load_format="auto",
            revision=None,
        )
        torch.set_default_dtype(old_dtype)
        return model.eval()

    def manage_memory(self) -> None:
        pass

    def add_new_request(self, req: RequestInfo) -> None:
        self.request_pool.add_request(req)

    def forward_prefill(self, sched_ctx: SchedulerContext) -> None:
        self.manage_memory()
        req_ids = torch.tensor([req.id for req in sched_ctx.requests])

        # calculate how many pages to allocate
        page_needed, page_idx_start = compute_page_needed(
            sched_ctx.seq_lens, sched_ctx.completed_lens, self.cache_manager.page_size
        )

        new_page_idx = self.cache_manager.alloc_pages(page_needed.sum().item())
        if new_page_idx is None:
            raise RuntimeError("No free pages")
        range_idx = torch.zeros((page_needed.size(0) + 1,), dtype=torch.int64)
        range_idx[1:] = page_needed.cumsum(dim=0)
        for i in range(page_needed.size(0)):
            self.cache_manager.req_page_mapping[
                req_ids[i],
                page_idx_start[i] : (
                    page_idx_start[i] + range_idx[i + 1] - range_idx[i]
                ),
            ] = new_page_idx[range_idx[i] : range_idx[i + 1]]

        inference_ctx = InferenceContext.new(
            InferenceState.PREFILL,
            self.config,
            sched_ctx.cache_manager,
            req_ids,
            sched_ctx.seq_lens,
            sched_ctx.completed_lens,
        )
        self.model.forward(req_ids, sched_ctx.seq_lens, inference_ctx)

    def forward_decode(self, sched_ctx: SchedulerContext) -> None:
        self.manage_memory()
        fill_pos = sched_ctx.seq_lens.clone()
        req_ids = torch.tensor([req.id for req in sched_ctx.requests])

        sched_ctx.seq_lens.add_(1)
        new_page_idx = self.cache_manager.alloc_pages(
            fill_pos[fill_pos % self.cache_manager.page_size == 0].count_nonzero()
        )
        if new_page_idx is None:
            raise RuntimeError("No free pages")
        self.cache_manager.req_page_mapping[
            req_ids, fill_pos // self.cache_manager.page_size
        ] = new_page_idx

        inference_ctx = InferenceContext.new(
            InferenceState.DECODE,
            self.config,
            sched_ctx.cache_manager,
            req_ids,
            sched_ctx.seq_lens,
            sched_ctx.completed_lens,
        )
        self.model.forward(req_ids, fill_pos, inference_ctx)
