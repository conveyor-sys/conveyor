from conveyor.models.config import ModelConfig
from conveyor.plugin.scheduler import PluginScheduler
from conveyor.scheduling.parsing import (
    FunctionaryParser,
    PlanningParser,
    PythonParser,
)
from conveyor.scheduling.scheduler import ScheduleEngine
import time
import sys
from misc.functionary import generate_functionary_input
from _private.access_token import set_hf_token


from conveyor.utils import clear_other_logger, getLogger

clear_other_logger()
logging = getLogger("conveyor.main")


def long_text(num: int = 50):
    return " ".join(["me"] * num)


tools = [  # For functionary-7b-v2 we use "tools"; for functionary-7b-v1.4 we use "functions" = [{"name": "get_current_weather", "description":..., "parameters": ....}]
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search on Google",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "site": {
                        "type": "string",
                        "description": "The website filter for the search, in a host form. e.g. www.google.com or en.wikipedia.org",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def eval_search(lazy: bool) -> float:
    model_name = "meetkai/functionary-small-v2.2"
    plugin_scheduler = PluginScheduler(lazy=lazy)
    engine = ScheduleEngine(
        ModelConfig(model_name),
        FunctionaryParser,
        plugin_scheduler,
        sequential_call=True,
    )
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request(
        generate_functionary_input(
            messages=[
                {
                    "role": "user",
                    "content": "Show me how to write hello world in python and c++ and java respectively with google tool, and only use result from stackoverflow.com",
                }
            ],
            tools=tools,
        )
        + "\n<|from|> assistant\n<|recipient|>"
    )
    # let parser aware of <|recipient|> token
    engine.request_pool.queued_requests[0].parser.buffer.append(32001)
    init_tokens_len = len(engine.request_pool.queued_requests[0].tokens)
    i = 0
    finished = None
    time_start = time.perf_counter()
    while i < 500:
        finished = engine.iteration_step()
        if finished:
            break
        i += 1
    print(f"<DECODE_INFO> {time.perf_counter() - time_start}", file=sys.stderr)

    if finished:
        res = None
        if not plugin_scheduler.lazy:
            while len(plugin_scheduler.waiting_queue) > 0:
                res = plugin_scheduler.poll_finished(
                    list(plugin_scheduler.waiting_queue.keys())[0]
                )
        if len(plugin_scheduler.lazy_queue) > 0:
            cur_id = list(plugin_scheduler.lazy_queue.keys())[0]
            while (
                cur_id in plugin_scheduler.lazy_queue
                and len(plugin_scheduler.lazy_queue[cur_id]) > 0
            ):
                plugin_scheduler.flush_lazy_sequentially(cur_id)
                while len(plugin_scheduler.waiting_queue) > 0:
                    res = plugin_scheduler.poll_finished(cur_id)
        time_end = time.perf_counter()
        logging.info(f"Plugin result: {res}")
        logging.info(f"Finished: {finished[0].decode()}")
        logging.info(
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end-time_start)} tokens/s with {i} iterations"
        )
        logging.info(f"Time: {time_end-time_start} s")
        ret_val = time_end - time_start
    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
        ret_val = -1
    plugin_scheduler.join_all()
    return ret_val


def eval_python(req: str, lazy: bool) -> float:
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    logging.info(f"Loading model {model_name}")
    plugin_scheduler = PluginScheduler(lazy)
    engine = ScheduleEngine(ModelConfig(model_name), PythonParser, plugin_scheduler)
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request(req)
    init_tokens_len = len(engine.request_pool.queued_requests[0].tokens)
    i = 0
    time_start = time.perf_counter()
    while i < 500:
        finished = engine.iteration_step()
        if finished:
            break
        i += 1
    print(f"<DECODE_INFO> {time.perf_counter() - time_start}", file=sys.stderr)

    if plugin_scheduler.lazy:
        plugin_scheduler.flush_lazy(list(plugin_scheduler.lazy_queue.keys())[0])

    if finished:
        while len(plugin_scheduler.waiting_queue) > 0:
            res = plugin_scheduler.poll_finished(
                list(plugin_scheduler.waiting_queue.keys())[0]
            )
        time_end = time.perf_counter()
        logging.info(f"Finished: {finished[0].decode()}")
        logging.info(
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end - time_start)} tokens/s with {i} iterations"
        )
        logging.info(f"Time: {time_end - time_start} s")
        ret_val = time_end - time_start
    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
        ret_val = -1
    plugin_scheduler.join_all()
    return ret_val


def eval_planning(lazy: bool) -> float:
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    logging.info(f"Loading model {model_name}")
    plugin_scheduler = PluginScheduler(lazy)
    engine = ScheduleEngine(ModelConfig(model_name), PlanningParser, plugin_scheduler)
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request("""Please repeat the following text wrapped by >>> and <<< only once and stop immediately:
>>>
$1 = search(Microsoft Market Cap)
$2 = search(Apple Market Cap)
$3 = compute_ratio($1, $2)
$4 = format_ratio($3)
<<<
""")
    init_tokens_len = len(engine.request_pool.queued_requests[0].tokens)
    i = 0
    time_start = time.perf_counter()
    while i < 500:
        finished = engine.iteration_step()
        if finished:
            break
        i += 1
    print(f"<DECODE_INFO> {time.perf_counter() - time_start}", file=sys.stderr)

    if plugin_scheduler.lazy:
        plugin_scheduler.flush_lazy(list(plugin_scheduler.lazy_queue.keys())[0])

    if finished:
        while len(plugin_scheduler.waiting_queue) > 0:
            res = plugin_scheduler.poll_finished(
                list(plugin_scheduler.waiting_queue.keys())[0]
            )
        time_end = time.perf_counter()
        logging.info(f"Finished: {finished[0].decode()}")
        logging.info(
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end - time_start)} tokens/s with {i} iterations"
        )
        logging.info(f"Time: {time_end - time_start} s")
        ret_val = time_end - time_start
    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
        ret_val = -1
    plugin_scheduler.join_all()
    return ret_val


def eval_validation(lazy: bool) -> float:
    model_name = "meetkai/functionary-small-v2.2"
    plugin_scheduler = PluginScheduler(lazy=lazy)
    engine = ScheduleEngine(
        ModelConfig(model_name),
        FunctionaryParser,
        plugin_scheduler,
        sequential_call=True,
    )
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request(
        generate_functionary_input(
            messages=[
                {
                    "role": "user",
                    "content": "Show me the local news of New York on April 1st, 2024 full day.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "query_local_news",
                        "description": "Get information of the local news in the designated city and time",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "time": {
                                    "type": "string",
                                    "description": "Time of the news, e.g. 2021-10-01",
                                },
                                "duration": {
                                    "type": "string",
                                    "description": "Duration of the news, e.g. 1 hour",
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        )
        + "\n<|from|> assistant\n<|recipient|>"
    )
    # let parser aware of <|recipient|> token
    engine.request_pool.queued_requests[0].parser.buffer.append(32001)
    init_tokens_len = len(engine.request_pool.queued_requests[0].tokens)
    i = 0
    finished = None
    res = None
    time_start = time.perf_counter()
    while i < 500:
        finished = engine.iteration_step(manually_poll=True)
        res = engine.poll_plugin()
        if len(res) > 0:
            res = res[0][1]
            break
        if finished:
            break
        i += 1
    print(f"<DECODE_INFO> {time.perf_counter() - time_start}", file=sys.stderr)

    if finished:
        if not plugin_scheduler.lazy:
            while len(plugin_scheduler.waiting_queue) > 0:
                res = plugin_scheduler.poll_finished(
                    list(plugin_scheduler.waiting_queue.keys())[0]
                )
        if len(plugin_scheduler.lazy_queue) > 0:
            cur_id = list(plugin_scheduler.lazy_queue.keys())[0]
            while (
                cur_id in plugin_scheduler.lazy_queue
                and len(plugin_scheduler.lazy_queue[cur_id]) > 0
            ):
                plugin_scheduler.flush_lazy_sequentially(cur_id)
                while len(plugin_scheduler.waiting_queue) > 0:
                    res = plugin_scheduler.poll_finished(cur_id)
        time_end = time.perf_counter()
        logging.info(f"Plugin result: {res}")
        logging.info(f"Finished: {finished[0].decode()}")
        logging.info(
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end-time_start)} tokens/s with {i} iterations"
        )
        logging.info(f"Time: {time_end-time_start} s")
        ret_val = time_end - time_start
    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
        ret_val = -1
    plugin_scheduler.join_all()
    return res[0]


def eval_sqlite(lazy: bool) -> float:
    model_name = "meetkai/functionary-small-v2.2"
    plugin_scheduler = PluginScheduler(lazy=lazy)
    engine = ScheduleEngine(
        ModelConfig(model_name),
        FunctionaryParser,
        plugin_scheduler,
        sequential_call=True,
    )
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request(
        generate_functionary_input(
            messages=[
                {
                    "role": "user",
                    "content": "Show me all the names of employees.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "query_database",
                        "description": "Query the database to get info for employees",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Name of queried employees, all to query all employees",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )
        + "\n<|from|> assistant\n<|recipient|>"
    )
    # let parser aware of <|recipient|> token
    engine.request_pool.queued_requests[0].parser.buffer.append(32001)
    init_tokens_len = len(engine.request_pool.queued_requests[0].tokens)
    i = 0
    finished = None
    time_start = time.perf_counter()
    while i < 500:
        finished = engine.iteration_step()
        if finished:
            break
        i += 1
    print(f"<DECODE_INFO> {time.perf_counter() - time_start}", file=sys.stderr)

    if finished:
        res = None
        if not plugin_scheduler.lazy:
            while len(plugin_scheduler.waiting_queue) > 0:
                res = plugin_scheduler.poll_finished(
                    list(plugin_scheduler.waiting_queue.keys())[0]
                )
        if len(plugin_scheduler.lazy_queue) > 0:
            cur_id = list(plugin_scheduler.lazy_queue.keys())[0]
            while (
                cur_id in plugin_scheduler.lazy_queue
                and len(plugin_scheduler.lazy_queue[cur_id]) > 0
            ):
                plugin_scheduler.flush_lazy_sequentially(cur_id)
                while len(plugin_scheduler.waiting_queue) > 0:
                    res = plugin_scheduler.poll_finished(cur_id)
        time_end = time.perf_counter()
        logging.info(f"Plugin result: {res}")
        logging.info(f"Finished: {finished[0].decode()}")
        logging.info(
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end-time_start)} tokens/s with {i} iterations"
        )
        logging.info(f"Time: {time_end-time_start} s")
        ret_val = time_end - time_start
    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
        ret_val = -1
    plugin_scheduler.join_all()
    return ret_val


def eval_calculator(lazy: bool) -> float:
    model_name = "meetkai/functionary-small-v2.2"
    plugin_scheduler = PluginScheduler(lazy=lazy)
    engine = ScheduleEngine(
        ModelConfig(model_name),
        FunctionaryParser,
        plugin_scheduler,
        sequential_call=True,
    )
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request(
        generate_functionary_input(
            messages=[
                {
                    "role": "user",
                    "content": "What is the result of 200 * 701",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Evaluate expression",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "expression need evaluating, in the form of python expression",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )
        + "\n<|from|> assistant\n<|recipient|>"
    )
    # let parser aware of <|recipient|> token
    engine.request_pool.queued_requests[0].parser.buffer.append(32001)
    init_tokens_len = len(engine.request_pool.queued_requests[0].tokens)
    i = 0
    finished = None
    time_start = time.perf_counter()
    while i < 500:
        finished = engine.iteration_step()
        if finished:
            break
        i += 1
    print(f"<DECODE_INFO> {time.perf_counter() - time_start}", file=sys.stderr)

    if finished:
        res = None
        if not plugin_scheduler.lazy:
            while len(plugin_scheduler.waiting_queue) > 0:
                res = plugin_scheduler.poll_finished(
                    list(plugin_scheduler.waiting_queue.keys())[0]
                )
        if len(plugin_scheduler.lazy_queue) > 0:
            cur_id = list(plugin_scheduler.lazy_queue.keys())[0]
            while (
                cur_id in plugin_scheduler.lazy_queue
                and len(plugin_scheduler.lazy_queue[cur_id]) > 0
            ):
                plugin_scheduler.flush_lazy_sequentially(cur_id)
                while len(plugin_scheduler.waiting_queue) > 0:
                    res = plugin_scheduler.poll_finished(cur_id)
        time_end = time.perf_counter()
        logging.info(f"Plugin result: {res}")
        logging.info(f"Finished: {finished[0].decode()}")
        logging.info(
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end-time_start)} tokens/s with {i} iterations"
        )
        logging.info(f"Time: {time_end-time_start} s")
        ret_val = time_end - time_start
    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
        ret_val = -1
    plugin_scheduler.join_all()
    return ret_val


def eval_scheduling():
    model_name = "meetkai/functionary-small-v2.2"
    logging.info(f"Loading model {model_name}")
    plugin_scheduler = PluginScheduler()
    engine = ScheduleEngine(
        ModelConfig(model_name),
        FunctionaryParser,
        plugin_scheduler,
        max_concurrent_requests=1,
    )
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request(
        # "List 10 famous mathematicians and their contributions."
        generate_functionary_input(
            messages=[
                {
                    "role": "user",
                    "content": "Show me how to write hello world in python with google search tool ",
                }
            ],
            tools=tools,
        )
        + "\n<|from|> assistant\n<|recipient|>"
    )
    req_id2 = engine.request_pool.add_request(
        # "Write an email to manager about this quarter's performance in a financial company."
        generate_functionary_input(
            messages=[
                {
                    "role": "user",
                    "content": "Write an email to manager about this quarter's performance in a financial company without using tool.",
                }
            ],
            tools=tools,
        )
        + "\n<|from|> assistant\n<|recipient|>all\n"
    )
    engine.request_pool.queued_requests[0].parser.buffer.append(32001)
    # engine.request_pool.queued_requests[1].parser.buffer.append(32001)
    init_tokens_len = len(engine.request_pool.queued_requests[0].tokens)
    i = 0
    time_start = time.perf_counter()
    first_stop = False
    while i < 1000:
        unloaded = engine.iteration_step(
            remove_finished=first_stop, unload_stop=not first_stop, manually_poll=True
        )
        if unloaded:
            if not first_stop:
                logging.debug(f"Unloaded: {unloaded[0].req_id}")
                first_stop = True
            else:
                break
        if len(plugin_scheduler.waiting_queue) > 0:
            # process I/O reqs manually
            got = engine.poll_plugin()
            for req_id, res in got:
                engine.reload_from_pending(req_id, engine.evict_worst_roundrobin)
                logging.info(f"Reload {req_id}")
                # manually hack [[{},{},...]]
                res = res[0][:3]
                engine.context.extend_req_with_str(
                    req_id,
                    f"\n<|from|>search\n<|recipient|>all\n<|content|>{str(res)}\n<|from|>assistant\n<|recipient|>all\n<|content|>",
                )
        i += 1

    if plugin_scheduler.lazy:
        plugin_scheduler.flush_lazy(list(plugin_scheduler.lazy_queue.keys())[0])

    if unloaded:
        while len(plugin_scheduler.waiting_queue) > 0:
            res = plugin_scheduler.poll_finished(
                list(plugin_scheduler.waiting_queue.keys())[0]
            )
        time_end = time.perf_counter()
        logging.info(f"Finished count={len(unloaded)} iter={i}: {unloaded[0].decode()}")
        final_tokens_len = len(unloaded[0].tokens)
    else:
        time_end = time.perf_counter()
        logging.info("Ongoing: " + engine.context.requests[0].decode())
        final_tokens_len = len(engine.context.requests[0].tokens)

    logging.info(f"Req1: {engine.context.pending_requests[0].decode()}")
    logging.info(
        f"Speed: {(final_tokens_len-init_tokens_len)/(time_end - time_start)} tokens/s"
    )
    logging.info(f"Time: {time_end - time_start} s")
    plugin_scheduler.join_all()
    return time_end - time_start


def eval_python_wrapper(lazy: bool) -> float:
    # eval_python("Plotting a sine wave in python. ONLY output code", lazy=False)
    # eval_python("Plotting a sine wave in python with torch. ONLY output code without trailing explanation.", lazy=False)
    # eval_python("Plotting a cosine wave in python with torch and matplotlib. ONLY output code without trailing explanation.", lazy=True)
    return eval_python(
        "Plotting a sine wave in python with torch and matplotlib. ONLY output code without trailing explanation.",
        lazy=lazy,
    )


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(
            "Usage: python main.py [python|scheduling|search|planning|validation|sqlite] [lazy?]"
        )
        sys.exit(1)
    if len(sys.argv) == 3 and sys.argv[2] == "lazy":
        lazy = True
    else:
        lazy = False
    set_hf_token()
    match sys.argv[1]:
        case "python":
            result = eval_python_wrapper(lazy)
        case "search":
            result = eval_search(lazy)
        case "scheduling":
            result = eval_scheduling()
        case "planning":
            result = eval_planning(lazy)
        case "validation":
            result = eval_validation(lazy)
        case "sqlite":
            result = eval_sqlite(lazy)
        case "calculator":
            result = eval_calculator(lazy)
        case _:
            print(
                "Usage: python main.py [python|scheduling|search|planning|validation|sqlite] [lazy?]"
            )
            sys.exit(1)
    print(f"Result: {result}", file=sys.stderr)
