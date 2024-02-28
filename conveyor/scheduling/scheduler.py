from __future__ import annotations
from typing import List, Tuple
from attr import dataclass
from conveyor.models.config import ModelConfig
from conveyor.models.utils import load_model, load_tokenizer
from conveyor.scheduling.cache_manager import CacheManager
from conveyor.scheduling.context import InferenceContext, InferenceState, RequestInfo
from conveyor.scheduling.request_pool import RequestPool
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel
import torch
import logging

logging = logging.getLogger(__name__)


@dataclass
class ReqRuntimeStat:
    req_id: int
    seq_len: int
    completed_len: int


@dataclass
class SchedulerContext:
    requests: list[RequestInfo]
    pending_requests: list[RequestInfo]
    cache_manager: CacheManager
    req_runtime_stats: dict[int, ReqRuntimeStat]

    # Batch info
    seq_lens: torch.Tensor
    completed_lens: torch.Tensor

    # Volatile state

    @classmethod
    def new(
        cls,
        reqs: list[RequestInfo],
        cache_manager: CacheManager,
        completed_lens: torch.Tensor = None,
    ) -> SchedulerContext:
        if completed_lens is None:
            completed_lens = torch.zeros(len(reqs), dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor(
            [len(req.tokens) for req in reqs], dtype=torch.int64, device="cuda"
        )
        req_runtime_stats = {
            req.req_id: ReqRuntimeStat(
                req.req_id,
                len(req.tokens),
                completed_lens[idx],
            )
            for idx, req in enumerate(reqs)
        }
        return cls(
            requests=reqs,
            pending_requests=[],
            cache_manager=cache_manager,
            req_runtime_stats=req_runtime_stats,
            seq_lens=seq_lens,
            completed_lens=completed_lens,
        )

    def add_active_request(self, req: RequestInfo) -> None:
        self.requests.append(req)
        self.req_runtime_stats[req.req_id] = ReqRuntimeStat(
            req.req_id, len(req.tokens), 0
        )
        # self.seq_lens = torch.cat(
        #     [self.seq_lens, torch.tensor([len(req.tokens)], device="cuda")]
        # )
        # self.completed_lens = torch.cat(
        #     [self.completed_lens, torch.tensor([0], device="cuda")]
        # )

    # Update the context state after a forward pass
    def update_req(self, logits: torch.Tensor) -> None:
        assert len(logits) == len(self.requests)
        for i, req in enumerate(self.requests):
            self.req_runtime_stats[req.req_id].completed_len = self.req_runtime_stats[
                req.req_id
            ].seq_len
            self.req_runtime_stats[req.req_id].seq_len += 1
            req.tokens.append(logits[i].item())
        self.completed_lens = self.seq_lens.clone()

    # Extend the request with string appended to the end
    def extend_req_with_str(self, req_id: int, new_content: str) -> None:
        for idx, req in enumerate(self.requests):
            if req.req_id == req_id:
                req.extend_str_no_re_encoding(new_content)
                self.seq_lens[idx] = len(req.tokens)
                self.req_runtime_stats[req.req_id].seq_len = len(req.tokens)
                return
        logging.warning(f"Request {req_id} not found")

    def _recompute_batch_state(self) -> None:
        self.seq_lens = torch.tensor(
            [len(req.tokens) for req in self.requests],
            dtype=torch.int64,
            device="cuda",
        )
        self.completed_lens = torch.tensor(
            [self.req_runtime_stats[req.req_id].completed_len for req in self.requests],
            dtype=torch.int64,
            device="cuda",
        )


def compute_page_needed(
    seq_lens: torch.Tensor, completed_lens: torch.Tensor, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    return page_needed, page_idx_start
    """
    # token index: [completed_lens, seq_lens-1]
    page_end = (seq_lens - 1) // page_size
    page_start = completed_lens // page_size
    page_start_not_allocated = completed_lens % page_size == 0
    page_needed = page_end - page_start + page_start_not_allocated
    return page_needed, page_start


class ScheduleEngine:
    def __init__(self, config: ModelConfig):
        nccl_port = 5000
        tp_size = 1
        tp_rank = 0
        # init global context
        torch.cuda.set_device(tp_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=tp_size,
            rank=tp_rank,
            init_method=f"tcp://127.0.0.1:{nccl_port}",
        )
        initialize_model_parallel(tensor_model_parallel_size=tp_size)

        page_size = 8
        max_request = 32
        page_num = 2048
        total_memory_usage_gb = (
            config.num_hidden_layers
            * page_num
            * page_size
            * config.num_attention_heads
            * config.head_dim
            * 2
            / 1024
            / 1024
            / 1024
        )
        logging.info(
            f"Initializing cache manager with page_num={page_num}, page_size={page_size}, GPU Memory Usage = {total_memory_usage_gb} GB"
        )

        self.config = config
        self.model = load_model(config)
        self.tokenizer = load_tokenizer(config.path)
        self.cache_manager = CacheManager(
            max_request=max_request,
            page_num=page_num,
            page_size=page_size,
            max_page_per_req=config.context_len // page_size,
            dtype=torch.float16,
            kv_head_num=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_num=config.num_hidden_layers,
            device="cuda",
        )
        self.request_pool = RequestPool(self.tokenizer)
        self.max_concurrent_requests = 16
        self.context = SchedulerContext.new([], self.cache_manager)

    @torch.inference_mode()
    def iteration_step(self):
        if self.new_request_available():
            new_request = self.request_pool.pop_request()
            self.context.add_active_request(new_request)
        next_operation = self.schedule_next_operation()
        logging.debug(f"Scheduler: next operation={next_operation}")
        match next_operation:
            case InferenceState.APPEND:
                logits, _ = self.forward_append(self.context)
            case InferenceState.DECODE:
                logits, _ = self.forward_decode(self.context)
            case InferenceState.AWAIT:
                logging.error("Scheduler: no-op in AWAIT state")
                return
        result = self.sample_logits(logits)
        self.context.update_req(result)
        return result

    def sample_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

    def new_request_available(self) -> bool:
        # TODO: better policy
        return (
            len(self.request_pool.queued_requests) > 0
            and len(self.context.requests) < self.max_concurrent_requests
        )

    def manage_memory(self) -> None:
        pass

    def add_new_request(self, req: RequestInfo) -> None:
        self.request_pool.add_request(req)

    def extend_req_with_str(self, req_id: int, new_content: str) -> None:
        self.context.extend_req_with_str(req_id, new_content)

    def forward_append(self, sched_ctx: SchedulerContext) -> None:
        self.manage_memory()
        req_ids = torch.tensor(
            [req.req_id for req in sched_ctx.requests], device="cuda"
        )

        # calculate how many pages to allocate
        page_needed, page_idx_start = compute_page_needed(
            sched_ctx.seq_lens, sched_ctx.completed_lens, self.cache_manager.page_size
        )

        page_num_required = int(page_needed.sum().item())
        if page_num_required > 0:
            new_page_idx = self.cache_manager.alloc_pages(page_num_required).int()
            if new_page_idx is None:
                raise RuntimeError("No free pages")
            range_idx = torch.zeros((page_needed.size(0) + 1,), dtype=torch.int64)
            range_idx[1:] = page_needed.cumsum(dim=0)
            for i in range(page_needed.size(0)):
                self.cache_manager.req_page_mapping[
                    req_ids[i],
                    page_idx_start[i] : (
                        page_idx_start[i] + range_idx[i + 1] - range_idx[i]
                    ),
                ] = new_page_idx[range_idx[i] : range_idx[i + 1]]

        inference_ctx = InferenceContext.new(
            InferenceState.APPEND,
            self.config,
            sched_ctx.cache_manager,
            req_ids,
            sched_ctx.seq_lens,
            sched_ctx.completed_lens,
        )

        flatten_tokens = []
        for req in sched_ctx.requests:
            flatten_tokens.extend(req.tokens)
        tokens = torch.tensor(flatten_tokens, dtype=torch.int32, device="cuda")

        logging.debug(
            f"Forward append(): req_ids={req_ids}, seq_lens={sched_ctx.seq_lens}, completed_lens={sched_ctx.completed_lens}, tokens={tokens}"
        )

        return self.model.forward(tokens, sched_ctx.seq_lens, inference_ctx)

    def forward_decode(self, sched_ctx: SchedulerContext) -> None:
        self.manage_memory()
        fill_pos = sched_ctx.seq_lens.clone()
        req_ids = torch.tensor(
            [req.req_id for req in sched_ctx.requests], dtype=torch.int64, device="cuda"
        )

        sched_ctx.seq_lens.add_(1)
        page_num_required = int(
            fill_pos[fill_pos % self.cache_manager.page_size == 0]
            .count_nonzero()
            .item()
        )
        if page_num_required > 0:
            new_page_idx = self.cache_manager.alloc_pages(page_num_required).int()
            if new_page_idx is None:
                raise RuntimeError("No free pages")
            # logging.debug(
            #     f"Allocated pages: req_ids={req_ids}, fill_pos={fill_pos}, new_page_idx={new_page_idx}"
            # )
            self.cache_manager.req_page_mapping[
                req_ids, fill_pos // self.cache_manager.page_size
            ] = new_page_idx

        inference_ctx = InferenceContext.new(
            InferenceState.DECODE,
            self.config,
            sched_ctx.cache_manager,
            req_ids,
            sched_ctx.seq_lens,
            sched_ctx.completed_lens,
        )

        tokens = torch.tensor(
            [req.tokens[-1] for req in sched_ctx.requests],
            dtype=torch.int32,
            device="cuda",
        )

        return self.model.forward(tokens, fill_pos, inference_ctx)

    def schedule_next_operation(self) -> InferenceState:
        prefill_list = []
        for req in self.context.requests:
            if self.context.req_runtime_stats[req.req_id].completed_len < len(
                req.tokens
            ):
                prefill_list.append(req)
        # if any prefillable, do prefill
        if len(prefill_list) > 0:
            decode_list = []
            for req in prefill_list:
                if self.context.req_runtime_stats[req.req_id].completed_len == len(
                    req.tokens
                ):
                    decode_list.append(req)
            self.context.pending_requests = decode_list + self.context.pending_requests
            self.context.requests = prefill_list
            self.context._recompute_batch_state()
            return InferenceState.APPEND
        else:
            changed = False
            while (
                len(self.context.pending_requests) > 0
                and len(self.context.requests) < self.max_concurrent_requests
            ):
                self.context.requests.append(self.context.pending_requests.pop(0))
                changed = True
            if changed:
                self.context._recompute_batch_state()
            return InferenceState.DECODE
