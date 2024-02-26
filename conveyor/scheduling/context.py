from __future__ import annotations
from enum import Enum
from typing import List, Optional

import datetime

from attr import dataclass
import math
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from conveyor.models.config import ModelConfig

from conveyor.scheduling.cache_manager import CacheManager
import logging

logging = logging.getLogger(__name__)


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
    # the start offset of token whose KV-cache have not been computed
    filling_start_offset: torch.Tensor

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
        config: ModelConfig,
        cache_manager: CacheManager,
        req_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        filling_start_offset: torch.Tensor,
    ) -> InferenceContext:
        batch_size = req_ids.size(0)
        # KV cache
        kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = seq_lens.cumsum(dim=0)
        kv_page_index = torch.cat(
            [
                cache_manager.req_page_mapping[
                    req_ids[i],
                    : int(math.ceil((int(seq_lens[i]) + 1) / cache_manager.page_size)),
                ]
                for i in range(batch_size)
            ],
            dim=0,
        ).contiguous()
        kv_last_page_lens = (
            torch.remainder(seq_lens - 1, cache_manager.page_size) + 1
        ).int()
        workspace_buffer = torch.empty(
            32 * 1024 * 1024, dtype=torch.int8, device="cuda"
        )
        qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1:] = (seq_lens - filling_start_offset).cumsum(dim=0)

        match state:
            case InferenceState.PREFILL | InferenceState.APPEND:
                prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
                prefill_wrapper.begin_forward(
                    qo_indptr,
                    kv_indptr,
                    kv_page_index,
                    kv_last_page_lens,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                )
                decode_wrapper = None
            case _:
                prefill_wrapper = None
                decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer)
                decode_wrapper.begin_forward(
                    kv_indptr,
                    kv_page_index,
                    kv_last_page_lens,
                    batch_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    1,
                )
        logging.debug(
            f"InferenceContext::new(): state={state}, req_ids={req_ids}, seq_lens={seq_lens}, filling_start_offset={filling_start_offset}, kv_indptr={kv_indptr}, kv_page_index={kv_page_index}, kv_last_page_lens={kv_last_page_lens}, qo_indptr={qo_indptr}"
        )

        return cls(
            state=state,
            req_ids=req_ids,
            seq_lens=seq_lens,
            filling_start_offset=filling_start_offset,
            cache_manager=cache_manager,
            kv_indptr=kv_indptr,
            kv_page_indices=kv_page_index,
            kv_last_page_lens=kv_last_page_lens,
            qo_indptr=qo_indptr,
            prefill_wrapper=prefill_wrapper,
            decode_wrapper=decode_wrapper,
        )

    def drop(self) -> None:
        if self.state in {InferenceState.PREFILL, InferenceState.APPEND}:
            self.prefill_wrapper.end_forward()
        else:
            self.decode_wrapper.end_forward()
        self.prefill_wrapper = None
        self.decode_wrapper = None


class RequestState(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2


class RequestInfo:
    def __init__(self, req_id: int, input_text: str, tokenizer, state: RequestState):
        self.req_id = req_id
        self.input_text = input_text
        self.tokens: List = tokenizer.encode(input_text)
        self.state = state
        self.estimated_pending_ddl: Optional[datetime.datetime] = None
