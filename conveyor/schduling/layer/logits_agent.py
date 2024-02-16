import torch
from torch import nn

from conveyor.schduling.context import InferenceContext


class LogitsAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_ids, hidden_states, weight, context: InferenceContext):
        pass
