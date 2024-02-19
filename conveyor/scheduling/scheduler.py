from __future__ import annotations
from typing import Tuple
from attr import dataclass
from conveyor.scheduling.cache_manager import CacheManager
from conveyor.scheduling.context import InferenceContext, InferenceState, RequestInfo
from conveyor.scheduling.request_pool import RequestPool
from transformers import PretrainedConfig
import torch
from torch import nn


@dataclass
class SchedulerContext:
    requests: list[RequestInfo]
    cache_manager: CacheManager

    # Batch info
    seq_lens: torch.Tensor
    completed_lens: torch.Tensor

    @classmethod
    def new() -> SchedulerContext:
        raise NotImplementedError


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
        self.cache_manager = CacheManager(256)
        self.request_pool = RequestPool()
        self.max_concurrent_requests = 16

    @torch.inference_mode()
    def iteration_step(self):
        pass

    @staticmethod
    def load_model(config: PretrainedConfig) -> nn.Module:
        raise NotImplementedError

    def manage_memory(self) -> None:
        pass

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
