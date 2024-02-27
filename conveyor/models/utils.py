from functools import lru_cache
from pathlib import Path
import importlib
import logging

import torch
from torch import nn

import conveyor
from conveyor.models.config import ModelConfig
from transformers import AutoTokenizer


@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    for module_path in (Path(conveyor.__file__).parent / "models").glob("*.py"):
        module = importlib.import_module(f"conveyor.models.{module_path.stem}")
        if hasattr(module, "EntryClass"):
            model_arch_name_to_cls[module.EntryClass.__name__] = module.EntryClass
    logging.debug(f"Loaded model classes: {model_arch_name_to_cls.keys()}")
    return model_arch_name_to_cls


def load_model(config: ModelConfig) -> nn.Module:
    def get_model_cls_by_arch_name(model_arch_names):
        model_arch_name_to_cls = import_model_classes()
        model_class = None
        for arch in model_arch_names:
            if arch in model_arch_name_to_cls:
                model_class = model_arch_name_to_cls[arch]
                break
        else:
            raise ValueError(
                f"Unsupported architectures: {arch}. "
                f"Supported list: {list(model_arch_name_to_cls.keys())}"
            )
        return model_class

    architectures = getattr(config.hf_config, "architectures", [])
    logging.debug(
        f"Loading model with architectures: {architectures}, config: {config.hf_config}"
    )
    model_class = get_model_cls_by_arch_name(architectures)

    # Load weights
    linear_method = None
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with torch.device("cuda"):
        model = model_class(config=config.hf_config, linear_method=linear_method)
    model.load_weights(
        config.path,
        cache_dir=None,
        load_format="auto",
        revision=None,
    )
    torch.set_default_dtype(old_dtype)
    return model.eval()


def load_tokenizer(path: str, trust_remote_code: bool = True, revision: str = None):
    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=trust_remote_code, revision=revision
    )
