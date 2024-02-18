from __future__ import annotations
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
    prefix_lens: torch.Tensor

    @classmethod
    def new() -> SchedulerContext:
        raise NotImplementedError


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

    def forward_decode(self, sched_ctx: SchedulerContext) -> None:
        self.manage_memory()
        fill_pos = sched_ctx.seq_lens.clone()
        req_ids = torch.tensor([req.id for req in sched_ctx.requests])

        sched_ctx.seq_lens.add_(1)
        self.cache_manager.alloc_new_reqs(
            fill_pos[fill_pos % self.cache_manager.page_size == 0].count_nonzero()
        )

        inference_ctx = InferenceContext.new(
            InferenceState.DECODE,
            self.config,
            sched_ctx.cache_manager,
            req_ids,
            sched_ctx.seq_lens,
            sched_ctx.prefix_lens,
        )
        self.model.forward(req_ids, fill_pos, inference_ctx)
