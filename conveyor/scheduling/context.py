from __future__ import annotations
from enum import Enum
from typing import Optional

import datetime

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
    qo_indptr: Optional[torch.Tensor]  # only used in PREFILL/APPEND state
    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper]
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper]

    @classmethod
    def new(
        cls,
        state: InferenceState,
        config: PretrainedConfig,
        cache_manager: CacheManager,
        req_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> InferenceContext:
        batch_size = req_ids.size(0)
        # KV cache
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
        kv_last_page_lens = torch.remainder(seq_lens, cache_manager.page_size) + 1

        # TODO: end_forward
        match state:
            case InferenceState.PREFILL | InferenceState.APPEND:
                qo_indptr = torch.zeros(
                    (batch_size + 1,), dtype=torch.int32, device=config.device
                )
                qo_indptr[1:] = (seq_lens - prefix_lens).cumsum(dim=0)
                prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper()
                prefill_wrapper.begin_forward(
                    qo_indptr,
                    batch_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                )
                decode_wrapper = None
            case _:
                qo_indptr = None
                prefill_wrapper = None
                decode_wrapper = BatchDecodeWithPagedKVCacheWrapper()
                decode_wrapper.begin_forward(
                    kv_indptr,
                    kv_last_page_lens,
                    batch_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    1,
                )

        return cls(
            state=state,
            req_ids=req_ids,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            cache_manager=cache_manager,
            kv_indptr=kv_indptr,
            kv_page_indices=kv_page_index,
            kv_last_page_lens=kv_last_page_lens,
            qo_indptr=qo_indptr,
            prefill_wrapper=prefill_wrapper,
            decode_wrapper=decode_wrapper,
        )


class RequestState(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2


class RequestInfo:
    def __init__(self, req_id: int, input_text: str, tokenizer, state: RequestState):
        self.req_id = req_id
        self.input_text = input_text
        self.tokenizer = tokenizer
        self.state = state
        self.estimated_pending_ddl: Optional[datetime.datetime] = None
