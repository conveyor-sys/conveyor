import torch
from torch import nn
from vllm.model_executor.parallel_utils.communication_op import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)


from conveyor.scheduling.context import InferenceContext, InferenceState
import logging

logging = logging.getLogger(__name__)


class LogitsAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tensor_parallel_size = get_tensor_model_parallel_world_size()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        context: InferenceContext,
    ):
        # TODO: log probability?
        logging.debug(
            f"LogitsAgent: input_ids.shape={input_ids.shape}, hidden_states.shape={hidden_states.shape}, weight.shape={weight.shape}"
        )
        if context.state == InferenceState.DECODE:
            last_hidden = hidden_states
        else:
            last_index = (
                torch.cumsum(
                    context.seq_lens - context.filling_start_offset,
                    dim=0,
                    dtype=torch.int64,
                )
                - 1
            )
            logging.debug(f"LogitsAgent: last_index={last_index}")
            last_hidden = hidden_states[last_index]
            hidden_states = None

        logging.debug(
            f"LogitsAgent: last_hidden.shape={last_hidden.shape}, weight.T.shape={weight.T.shape}"
        )
        logits = torch.matmul(last_hidden, weight.T)
        if self.tensor_parallel_size > 1:
            logits = tensor_model_parallel_all_gather(logits)
        return logits[:, : self.config.vocab_size], (None, None, None)
