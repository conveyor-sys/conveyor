from __future__ import annotations
from enum import Enum
from typing import Optional

from attr import dataclass
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from transformers import PretrainedConfig

from conveyor.scheduling.cache_manager import CacheManager


class InferenceState(Enum):
    PREFILL = 0
    DECODE = 1
    APPEND = 2
    AWAIT = 3


@dataclass
class InferenceContext:
    state: InferenceState

    # Inference state
    req_ids: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor

    # KV cache
    cache_manager: CacheManager
    kv_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_last_page_lens: torch.Tensor
    qo_indptr: torch.Tensor
    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper]
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper]

    @classmethod
    def new(
        self,
        state: InferenceState,
        config: PretrainedConfig,
        cache_manager: CacheManager,
        req_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> InferenceContext:
        batch_size = req_ids.size(0)
        kv_indptr = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=config.device
        )
        kv_indptr[1:] = seq_lens.cumsum(dim=0)
        kv_page_index = torch.cat(
            [
                cache_manager.req_page_mapping[
                    req_ids[i], torch.ceil_div(seq_lens[i], cache_manager.page_size)
                ]
                for i in range(batch_size)
            ],
            dim=0,
        ).contiguous()
        # TODO: last page lens range?
        kv_last_page_lens = torch.remainder(seq_lens, cache_manager.page_size) + 1

        raise NotImplementedError


@dataclass
class BatchContext:
    @classmethod
    def new(self) -> BatchContext:
        raise NotImplementedError
