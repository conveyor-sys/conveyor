from __future__ import annotations
from enum import Enum

from attr import dataclass
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from transformers import PretrainedConfig


class InferenceState(Enum):
    PREFILL = 0
    DECODE = 1
    APPEND = 2
    AWAIT = 3


@dataclass
class InferenceContext:
    state: InferenceState
    kv_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_last_page_lens: torch.Tensor
    qo_indptr: torch.Tensor
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper

    @classmethod
    def new(self, state: InferenceState, config: PretrainedConfig) -> InferenceContext:

        pass
