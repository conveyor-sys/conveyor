from __future__ import annotations
from typing import Optional
import torch
import logging

logging = logging.getLogger("CacheManager")


class CacheManager:
    def __init__(
        self,
        max_request: int,
        page_num: int,
        page_size: int,
        max_page_per_req: int,
        dtype,
        kv_head_num: int,
        head_dim: int,
        layer_num: int,
        device="cuda",
    ) -> None:
        self.page_num = page_num
        self.page_size = page_size

        # requests to pages
        self.free_req_slots = max_request
        self.free_pages_cnt = page_num
        self.req_state = torch.zeros((max_request,), dtype=torch.bool, device=device)
        self.req_page_mapping = torch.empty(
            (max_request, max_page_per_req),
            dtype=torch.int32,
            device=device,
        )

        # tokens to kv cache
        self.page_state = torch.zeros((page_num,), dtype=torch.bool, device=device)
        self.kv_storage = [
            torch.empty(
                (page_num, 2, page_size, kv_head_num, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]

    def alloc_new_reqs(self, num_reqs) -> Optional[torch.Tensor]:
        if self.free_req_slots < num_reqs:
            logging.warning("No free request slots")
            return None
        new_req_idx = torch.nonzero(self.req_state == 0).squeeze(1)[:num_reqs]
        self.req_state[new_req_idx] = True
        self.free_req_slots -= num_reqs
        return new_req_idx

    def free_reqs(self, req_idx: torch.Tensor | int) -> None:
        self.req_state[req_idx] = False
        self.free_req_slots += (
            req_idx.size(0) if isinstance(req_idx, torch.Tensor) else 1
        )

    def alloc_pages(self, page_num: int) -> Optional[torch.Tensor]:
        if self.free_pages_cnt < page_num:
            logging.warning("No free page slots")
            return None
        new_page_idx = torch.nonzero(self.page_state == 0).squeeze(1)[:page_num]
        self.page_state[new_page_idx] = True
        self.free_pages_cnt -= page_num
        return new_page_idx

    def free_pages(self, page_idx: torch.Tensor | int) -> None:
        self.page_state[page_idx] = False
        self.free_pages_cnt += (
            page_idx.size(0) if isinstance(page_idx, torch.Tensor) else 1
        )
