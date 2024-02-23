from typing import Any, List
from typing import Optional
from transformers import AutoConfig

CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]


def get_context_length(config):
    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return val
    return 2048


class ModelConfig:
    def __init__(
        self,
        path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        model_arch_override: Optional[List[str] | str] = None,
    ) -> None:
        self.path = path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.hf_config = AutoConfig.from_pretrained(
            self.path, trust_remote_code=trust_remote_code, revision=revision
        )
        if model_arch_override is not None:
            if isinstance(model_arch_override, str):
                self.hf_config.architectures = [model_arch_override]
            elif isinstance(model_arch_override, list):
                self.hf_config.architectures = model_arch_override

        # Unify the config keys for hf_config
        self.context_len: int = get_context_length(self.hf_config)
        self.head_dim: int = (
            self.hf_config.hidden_size // self.hf_config.num_attention_heads
        )
        self.num_attention_heads: int = self.hf_config.num_attention_heads
        self.num_key_value_heads: int = getattr(
            self.hf_config, "num_key_value_heads", None
        )
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size: int = self.hf_config.hidden_size
        self.num_hidden_layers: int = self.hf_config.num_hidden_layers
        self.vocab_size: int = self.hf_config.vocab_size
