from __future__ import annotations
from datetime import datetime
import time
from typing import Optional, Tuple
from attr import dataclass
from conveyor.models.config import ModelConfig
from conveyor.models.utils import load_model, load_tokenizer
from conveyor.plugin.scheduler import PluginScheduler
from conveyor.scheduling.cache_manager import CacheManager
from conveyor.scheduling.context import InferenceContext, InferenceState, RequestInfo
from conveyor.scheduling.request_pool import RequestPool
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel
import torch
import numpy as np
from conveyor.utils import getLogger

logging = getLogger(__name__)


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
        logging.debug(f"Current requests: {[req.req_id for req in self.requests]}")

    def add_inactive_request(self, req: RequestInfo) -> None:
        self.pending_requests.append(req)
        self.req_runtime_stats[req.req_id] = ReqRuntimeStat(
            req.req_id, len(req.tokens), 0
        )
        logging.debug(
            f"Current pending requests: {[req.req_id for req in self.pending_requests]}"
        )

    # Update the context state after a forward pass
    def update_req(self, logits: torch.Tensor) -> None:
        assert len(logits) == len(self.requests)
        for i, req in enumerate(self.requests):
            req.tokens.append(logits[i].item())
            req.evaluate_parser(logits[i].item())
            self.req_runtime_stats[req.req_id].completed_len = self.req_runtime_stats[
                req.req_id
            ].seq_len
            self.req_runtime_stats[req.req_id].seq_len = len(req.tokens)

        self.completed_lens = self.seq_lens.clone()
        self.seq_lens.add_(1)

    # Extend the request with string appended to the end
    def extend_req_with_str(self, req_id: int, new_content: str) -> int:
        for idx, req in enumerate(self.requests):
            if req.req_id == req_id:
                length = req.extend_str_no_re_encoding(new_content)
                self.seq_lens[idx] = len(req.tokens)
                self.req_runtime_stats[req.req_id].seq_len = len(req.tokens)
                return length
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

    def _req_in_pending(self, req_id: int) -> bool:
        for req in self.pending_requests:
            if req.req_id == req_id:
                return True
        return False

    def page_used_by(self, req_id: int) -> int:
        page_cnt = (
            self.req_runtime_stats[req_id].completed_len // self.cache_manager.page_size
        )
        if (
            self.req_runtime_stats[req_id].completed_len % self.cache_manager.page_size
            != 0
        ):
            page_cnt += 1
        return page_cnt

    def drop_kv_cache(self, req_id: int, drop_page_num: int) -> None:
        if not self._req_in_pending(req_id):
            raise RuntimeError(f"Request {req_id} is not in pending state")

        for req in self.requests:
            if req.req_id == req_id:
                break
        page_cnt = self.page_used_by(req_id)
        drop_page_idx = self.cache_manager.req_page_mapping[
            req_id, drop_page_num:page_cnt
        ]
        self.cache_manager.free_pages(drop_page_idx)
        # update stats
        self.req_runtime_stats[req_id].completed_len = (
            page_cnt - drop_page_num
        ) * self.cache_manager.page_size


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
    return page_needed, page_start + (~page_start_not_allocated)


class ScheduleEngine:
    def __init__(
        self,
        config: ModelConfig,
        parser_cls,
        plugin_scheduler: PluginScheduler,
        max_concurrent_requests: int = 16,
        sequential_call: bool = False,
    ):
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
        self.plugin_scheduler = plugin_scheduler
        self.request_pool = RequestPool(
            self.tokenizer,
            lambda tokenizer, id: parser_cls(
                tokenizer,
                id,
                lambda id, plugin_name: plugin_scheduler.start_plugin(id, plugin_name),
                lambda id, dat: plugin_scheduler.process_new_data(id, dat),
                lambda id: plugin_scheduler.finish_plugin(id, sequential_call),
            ),
        )
        self.max_concurrent_requests = max_concurrent_requests
        self.context = SchedulerContext.new([], self.cache_manager)

    def sample_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

    def new_request_available(self) -> bool:
        # TODO: better policy
        return (
            len(self.request_pool.queued_requests) > 0
            and len(self.context.requests) < self.max_concurrent_requests
        )

    def manage_memory(self) -> None:
        used, total = self.context.cache_manager.page_usage()
        if used / total > 0.9:
            logging.warning(f"Memory usage: {used}/{total}")
            evicted_page = 0
            for req in self.context.pending_requests:
                if req.invocation_timestamp is not None:
                    # leave 10 pages at max
                    req_page = self.context.page_used_by(req.req_id)
                    if req_page > 10:
                        p = req_page - 10
                        self.context.drop_kv_cache(req.req_id, p)
                        evicted_page += p
            logging.warning(
                f"Evicted {evicted_page} pages. Current usage: {used}/{total}"
            )

    @torch.inference_mode()
    def add_new_request(self, req: RequestInfo) -> None:
        self.request_pool.add_request(req)

    @torch.inference_mode()
    def extend_req_with_str(self, req_id: int, new_content: str) -> int:
        return self.context.extend_req_with_str(req_id, new_content)

    @torch.inference_mode()
    def iteration_step(
        self,
        remove_finished: bool = True,
        unload_stop: bool = False,
        manually_poll: bool = False,
    ) -> list[RequestInfo]:
        self.roundrobin_policy()
        # self.fcfs_policy()
        next_operation = self.schedule_next_operation()
        match next_operation:
            case InferenceState.APPEND:
                logits, _ = self.forward_append(self.context)
            case InferenceState.DECODE:
                logits, _ = self.forward_decode(self.context)
            case InferenceState.AWAIT:
                if not manually_poll:
                    self.poll_plugin()
                time.sleep(0.02)
                return
        result = self.sample_logits(logits).to("cpu")
        self.context.update_req(result)
        if not manually_poll:
            self.poll_plugin()
        if remove_finished:
            finished = self.remove_finished(result, skip_stop=unload_stop)
            return finished
        else:
            if unload_stop:
                return self.unload_to_pending(result)
            else:
                return []

    def poll_plugin(self) -> list[Tuple[str, Tuple[str]]]:
        keys = list(self.plugin_scheduler.waiting_queue.keys())
        result = []
        for k in keys:
            res = self.plugin_scheduler.poll_finished(k)
            if res is not None:
                logging.debug(f"Plugin {k} finished: {res[0]}")
                result.append((k, res))
        return result

    def prepare_kv_page(
        req_ids: torch.Tensor, sched_ctx: SchedulerContext, cache_manager: CacheManager
    ) -> None:
        # calculate how many pages to allocate
        page_needed, page_idx_start = compute_page_needed(
            sched_ctx.seq_lens, sched_ctx.completed_lens, cache_manager.page_size
        )

        page_num_required = int(page_needed.sum().item())
        if page_num_required > 0:
            new_page_idx = cache_manager.alloc_pages(page_num_required).int()
            if new_page_idx is None:
                raise RuntimeError("No free pages")
            range_idx = torch.zeros(
                (page_needed.size(0) + 1,), dtype=torch.int64, device="cpu"
            )
            range_idx[1:] = page_needed.cumsum(dim=0)
            for i in range(page_needed.size(0)):
                cache_manager.req_page_mapping[
                    req_ids[i],
                    page_idx_start[i] : (
                        page_idx_start[i] + range_idx[i + 1] - range_idx[i]
                    ),
                ] = new_page_idx[range_idx[i] : range_idx[i + 1]]

    def forward_append(self, sched_ctx: SchedulerContext) -> None:
        self.manage_memory()
        req_ids = torch.tensor(
            [req.req_id for req in sched_ctx.requests], device="cuda"
        )

        ScheduleEngine.prepare_kv_page(req_ids, sched_ctx, self.cache_manager)

        inference_ctx = InferenceContext.new(
            InferenceState.APPEND,
            self.config,
            sched_ctx.cache_manager,
            req_ids,
            sched_ctx.seq_lens,
            sched_ctx.completed_lens,
        )

        seq_lens_np = sched_ctx.seq_lens.cpu().numpy()
        completed_lens_np = sched_ctx.completed_lens.cpu().numpy()

        flatten_tokens = []
        for idx, req in enumerate(sched_ctx.requests):
            flatten_tokens.extend(req.tokens[completed_lens_np[idx] :])
        tokens = torch.tensor(flatten_tokens, dtype=torch.int32, device="cuda")

        positions = torch.tensor(
            np.concatenate(
                [
                    np.arange(completed_lens_np[i], seq_lens_np[i])
                    for i in range(len(seq_lens_np))
                ],
                axis=0,
            ),
            device="cuda",
        )

        logging.debug(
            f"Forward append(): req_ids={req_ids}, seq_lens={sched_ctx.seq_lens}, completed_lens={sched_ctx.completed_lens}, positions={positions}, tokens={tokens}"
        )

        return self.model.forward(tokens, positions, inference_ctx)

    def forward_decode(self, sched_ctx: SchedulerContext) -> None:
        self.manage_memory()
        fill_pos = sched_ctx.seq_lens.clone()
        req_ids = torch.tensor(
            [req.req_id for req in sched_ctx.requests], dtype=torch.int64, device="cuda"
        )

        ScheduleEngine.prepare_kv_page(req_ids, sched_ctx, self.cache_manager)

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

    #   -----------------------------------------------------
    #                   Scheduling Logic
    #   -----------------------------------------------------

    def schedule_next_operation(self) -> InferenceState:
        prefill_list = []
        for req in self.context.requests:
            if (
                self.context.req_runtime_stats[req.req_id].completed_len
                < len(req.tokens) - 1
            ):
                prefill_list.append(req)
        # if any prefillable, do prefill
        if len(prefill_list) > 0:
            decode_list = []
            for req in self.context.requests:
                if (
                    self.context.req_runtime_stats[req.req_id].completed_len
                    >= len(req.tokens) - 1
                ):
                    decode_list.append(req)
            self.context.pending_requests = decode_list + self.context.pending_requests
            self.context.requests = prefill_list
            self.context._recompute_batch_state()
            logging.debug(
                f"Next Prefill: requests={[r.req_id for r in self.context.requests]}, pending={[r.req_id for r in self.context.pending_requests]}"
            )
            return InferenceState.APPEND
        else:
            changed = False
            pending_removal = []
            for pending_req in self.context.pending_requests:
                if len(self.context.requests) >= self.max_concurrent_requests:
                    break
                elif pending_req.invocation_timestamp is None:
                    self.context.requests.append(pending_req)
                    pending_removal.append(pending_req.req_id)
                    changed = True

            if changed:
                self.context.pending_requests = [
                    x
                    for x in self.context.pending_requests
                    if x.req_id not in pending_removal
                ]
                logging.debug(
                    f"Next Decode: requests={[r.req_id for r in self.context.requests]}, pending={[r.req_id for r in self.context.pending_requests]}"
                )
                self.context._recompute_batch_state()
            if len(self.context.requests) > 0:
                return InferenceState.DECODE
            else:
                return InferenceState.AWAIT

    def fcfs_policy(self) -> None:
        while self.new_request_available():
            new_request = self.request_pool.pop_request()
            logging.info(f"Scheduler: new request={new_request.req_id}")
            self.context.add_active_request(new_request)

    def roundrobin_policy(self) -> None:
        changed = False
        while len(self.request_pool.queued_requests) > 0:
            new_request = self.request_pool.pop_request()
            logging.info(f"Scheduler: new request={new_request.req_id}")
            if len(self.context.requests) < self.max_concurrent_requests:
                self.context.add_active_request(new_request)
                changed = True
            else:
                self.context.add_inactive_request(new_request)
            self.context.req_runtime_stats[new_request.req_id].roundrobin_counter = 0

        new_requests = []
        new_pending = []
        for req in self.context.requests:
            kept = True
            if self.context.req_runtime_stats[req.req_id].roundrobin_counter >= 10:
                if len(self.context.pending_requests) > 0:
                    self.context.req_runtime_stats[req.req_id].roundrobin_counter = 0
                    new_pending.append(req)
                    new_requests.append(self.context.pending_requests.pop(0))
                    kept = False
                    changed = True
            if kept:
                new_requests.append(req)
        self.context.requests = new_requests
        self.context.pending_requests.extend(new_pending)

        for req in self.context.requests:
            self.context.req_runtime_stats[req.req_id].roundrobin_counter += 1
        if changed:
            self.context._recompute_batch_state()

    def cfs_policy(self) -> None:
        while len(self.request_pool.queued_requests) > 0:
            new_request = self.request_pool.pop_request()
            logging.info(f"Scheduler: new request={new_request.req_id}")
            if len(self.context.requests) < self.max_concurrent_requests:
                self.context.add_active_request(new_request)
            else:
                self.context.add_inactive_request(new_request)

    def unload_to_pending(self, logits: torch.Tensor) -> Optional[list[RequestInfo]]:
        unloaded = []
        for i, req in enumerate(self.context.requests):
            # <|stop|>
            if logits[i] == 32003:
                if req.req_id in self.plugin_scheduler.waiting_queue:
                    req.invocation_timestamp = datetime.now()
                    unloaded.append(req)
        if len(unloaded) > 0:
            self.context.pending_requests.extend(unloaded)
            self.context.requests = [
                req for req in self.context.requests if req not in unloaded
            ]
            self.context._recompute_batch_state()
            logging.debug(
                f"Unloaded: requests={[r.req_id for r in self.context.requests]}, pending={[r.req_id for r in self.context.pending_requests]}"
            )
        return unloaded

    # Require _recompute_batch_state() to be called after this method
    def evict_worst_roundrobin(self) -> bool:
        if len(self.context.requests) < self.max_concurrent_requests:
            return True
        worst_req_idx = 0
        for idx, req in enumerate(self.context.requests):
            if idx == 0:
                continue
            if (
                self.context.req_runtime_stats[req.req_id].roundrobin_counter
                > self.context.req_runtime_stats[
                    self.context.requests[worst_req_idx].req_id
                ].roundrobin_counter
            ):
                worst_req_idx = idx
        evicted_req = self.context.requests.pop(worst_req_idx)
        evicted_req.roundrobin_counter = 0
        self.context.pending_requests.append(evicted_req)
        return True

    def reload_from_pending(self, req_id: str, self_evict) -> None:
        add_now = self_evict()
        for req in self.context.pending_requests:
            if req.req_id == req_id:
                req.invocation_timestamp = None
                if add_now:
                    self.context.requests.append(req)
                    self.context.pending_requests = [
                        r for r in self.context.pending_requests if r.req_id != req_id
                    ]
                    if req.roundrobin_counter is not None:
                        req.roundrobin_counter = -10
                    self.context._recompute_batch_state()
                    logging.debug(
                        f"Reloaded: requests={[r.req_id for r in self.context.requests]}, pending={[r.req_id for r in self.context.pending_requests]}"
                    )
                return
        logging.error(f"Request {req_id} not found in pending requests")

    def remove_finished(
        self, logits: torch.Tensor, skip_stop=False
    ) -> list[RequestInfo]:
        finished = []
        # check eos in logits and remove finished requests
        for i, req in enumerate(self.context.requests):
            # </s> or <|stop|>
            if logits[i] == self.tokenizer.eos_token_id or (
                not skip_stop and logits[i] == 32003
            ):
                finished.append(req)
                self.context.req_runtime_stats.pop(req.req_id)

        if len(finished) > 0:
            # remove finished requests
            new_requests = []
            removed_ids = [req.req_id for req in finished]
            for req in self.context.requests:
                if req.req_id not in removed_ids:
                    new_requests.append(req)
            self.context.requests = new_requests
            self.context._recompute_batch_state()
            logging.debug(
                f"Finished: requests={[r.req_id for r in self.context.requests]}, pending={[r.req_id for r in self.context.pending_requests]}"
            )
        return finished
