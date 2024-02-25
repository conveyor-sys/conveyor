from __future__ import annotations
from enum import Enum
from typing import List

from attr import dataclass
import torch

from conveyor.scheduling.context import RequestInfo, RequestState


class RequestPool:
    def __init__(self, tokenizer):
        self.queued_requests: List[RequestInfo] = []
        self.req_id_cnt = 0
        self.tokenizer = tokenizer

    def _add_request(self, req: RequestInfo):
        self.queued_requests.append(req)

    def add_request(self, text: str):
        self._add_request(
            RequestInfo(
                req_id=self.req_id_cnt,
                input_text=text,
                tokenizer=self.tokenizer,
                state=RequestState.PENDING,
            )
        )
        self.req_id_cnt += 1

    def pop_request(self) -> RequestInfo:
        return self.queued_requests.pop(0)
