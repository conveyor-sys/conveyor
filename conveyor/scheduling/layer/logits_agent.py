import torch
from torch import nn
from vllm.model_executor.parallel_utils.communication_op import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)


from conveyor.scheduling.context import InferenceContext, InferenceState


class LogitsAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tensor_parallel_size = get_tensor_model_parallel_world_size()

    def forward(
        self,
        input_ids,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        context: InferenceContext,
    ):
        # TODO: log probability?
        if context.state == InferenceState.DECODE:
            logits = weight.matmul(hidden_states)
            if self.tensor_parallel_size > 1:
                logits = tensor_model_parallel_all_gather(logits)
            return logits[:, :, self.config.vocab_size], (None, None, None)
        else:
            raise NotImplementedError
