from __future__ import annotations
from enum import Enum

from attr import dataclass
import torch


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

    @classmethod
    def new(self, state: InferenceState) -> InferenceContext:
        pass
