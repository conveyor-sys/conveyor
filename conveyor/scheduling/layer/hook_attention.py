import torch
import logging
from torch import nn

from conveyor.scheduling.context import InferenceContext, InferenceState

logging = logging.getLogger(__name__)


class HookAttention(nn.Module):
    def __init__(
        self, num_heads: int, head_dim: int, scaling, num_kv_heads: int, layer_id: int
    ):
        super().__init__()
        self.num_q_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: InferenceContext,
    ) -> torch.Tensor:
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        match context.state:
            case InferenceState.PREFILL:
                return self.prefill_forward(q, k, v, context)
            case InferenceState.DECODE:
                return self.decode_forward(q, k, v, context)
            case InferenceState.APPEND:
                return self.prefill_forward(q, k, v, context)
            case InferenceState.AWAIT:
                logging.error("HookAttention: no-op in AWAIT state")
                exit(1)

    def prefill_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: InferenceContext,
    ) -> torch.Tensor:
        self.store_kv_cache(k, v, context)
        output: torch.Tensor = context.prefill_wrapper.forward(
            q.contiguous().view(-1, self.num_q_heads, self.head_dim),
            # TODO
        )
        return output.view(-1, self.num_q_heads * self.head_dim)

    def decode_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: InferenceContext,
    ) -> torch.Tensor:
        self.store_kv_cache(k, v, context)
        output: torch.Tensor = context.decode_wrapper.forward(
            q.contiguous().view(-1, self.num_q_heads, self.head_dim),
            # TODO
        )
        return output.view(-1, self.num_q_heads * self.head_dim)

    def store_kv_cache(
        self, k_cache: torch.Tensor, v_cache: torch.Tensor, context: InferenceContext
    ):
        key_buf = context.cache_manager.get_key_storage(self.layer_id)
        value_buf = context.cache_manager.get_value_storage(self.layer_id)
        kv_len = k_cache.size(0)
        raise NotImplementedError
