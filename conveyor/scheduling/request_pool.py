from __future__ import annotations
from enum import Enum

from attr import dataclass
import torch

from conveyor.scheduling.context import RequestInfo


class RequestPool:
    def __init__(self):
        self.queued_requests = []

    def add_request(self, req: RequestInfo):
        self.queued_requests.append(req)

    def pop_request(self) -> RequestInfo:
        return self.queued_requests.pop(0)
