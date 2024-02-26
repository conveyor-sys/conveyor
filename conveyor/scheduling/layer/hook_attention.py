import torch
import logging
from torch import nn

from conveyor.scheduling.context import InferenceContext, InferenceState
import flashinfer

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
        k = k.contiguous().view(-1, self.num_kv_heads, self.head_dim)
        v = v.contiguous().view(-1, self.num_kv_heads, self.head_dim)
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
            context.cache_manager.kv_storage[self.layer_id],
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
            context.cache_manager.kv_storage[self.layer_id],
        )
        return output.view(-1, self.num_q_heads * self.head_dim)

    def store_kv_cache(
        self, k_cache: torch.Tensor, v_cache: torch.Tensor, context: InferenceContext
    ):
        kv_data = context.cache_manager.kv_storage[self.layer_id]
        logging.debug(
            f"store_kv_cache()[{self.layer_id}]: kv_data={kv_data.shape}, k_cache={k_cache.shape}, v_cache={v_cache.shape}"
        )
        flashinfer.append_paged_kv_cache(
            k_cache,
            v_cache,
            context.qo_indptr,
            kv_data,
            context.kv_page_indices,
            context.kv_indptr,
            context.kv_last_page_lens,
        )
