from __future__ import annotations
from typing import List

from conveyor.scheduling.context import RequestInfo


class RequestPool:
    def __init__(self, tokenizer, callback):
        self.queued_requests: List[RequestInfo] = []
        self.req_id_cnt = 0
        self.tokenizer = tokenizer
        self.callback = callback

    def _add_request(self, req: RequestInfo):
        self.queued_requests.append(req)

    def add_request(self, text: str) -> int:
        assert len(text) > 0
        req_id = self.req_id_cnt
        self._add_request(
            RequestInfo(
                req_id=req_id,
                input_text=text,
                tokenizer=self.tokenizer,
                callback=self.callback,
            )
        )
        self.req_id_cnt += 1
        return req_id

    def pop_request(self) -> RequestInfo:
        return self.queued_requests.pop(0)
