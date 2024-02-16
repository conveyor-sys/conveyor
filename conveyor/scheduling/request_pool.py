from __future__ import annotations
from enum import Enum

from attr import dataclass
import torch


class Request:
    def __init__(self, id, prefix: str):
        self.id = id
        self.content = prefix
        # todo: tokenize content
        raise NotImplementedError


class RequestPool:
    def __init__(self):
        raise NotImplementedError
