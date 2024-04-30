from conveyor.models.config import ModelConfig
from conveyor.plugin.scheduler import PluginScheduler
from conveyor.scheduling.parsing import BaseParser, FunctionaryParser, PythonParser
from conveyor.scheduling.scheduler import ScheduleEngine
import time
from misc.functionary import generate_functionary_input
from _private.access_token import set_hf_token


from conveyor.utils import getLogger

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


def eval_weather():
    model_name = "meetkai/functionary-small-v2.2"
    plugin_scheduler = PluginScheduler(lazy=False)
    engine = ScheduleEngine(
        ModelConfig(model_name), FunctionaryParser, plugin_scheduler
    )
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request(
        generate_functionary_input(
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
                }
            ],
            tools=[
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
                }
            ],
        )
        + "\n<|from|> assistant\n<|recipient|>"
    )
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

    if plugin_scheduler.lazy:
        plugin_scheduler.flush_lazy(list(plugin_scheduler.lazy_queue.keys())[0])

    if finished:
        res = None
        while len(plugin_scheduler.waiting_queue) > 0:
            res = plugin_scheduler.poll_finished(
                list(plugin_scheduler.waiting_queue.keys())[0]
            )
        time_end = time.perf_counter()
        logging.info(f"Plugin result: {res}")
        logging.info(f"Finished: {finished[0].decode()}")
        logging.info(
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end-time_start)} tokens/s"
        )
        logging.info(f"Time: {time_end-time_start} s")
    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
    plugin_scheduler.join_all()


def eval_search():
    model_name = "meetkai/functionary-small-v2.2"
    plugin_scheduler = PluginScheduler(lazy=True)
    engine = ScheduleEngine(
        ModelConfig(model_name), FunctionaryParser, plugin_scheduler
    )
    logging.info(f"Model {model_name} loaded")
    req_id = engine.request_pool.add_request(
        # "Describe the basic components of a neural network and how it can be trained"
        # "[INST]Describe the basic components of a neural network and how it can be trained. [/INST]"
        # "\nAnd tell me how to write the Greatest common divisor algorithm in Python? Show me the code."
        generate_functionary_input(
            messages=[
                {
                    "role": "user",
                    "content": "Show me the latitude and altitude of the Eiffel Tower, White House, Tokyo Tree, and the Great Wall of China respectively",
                }
            ],
            tools=tools,
        )
        + "\n<|from|> assistant\n<|recipient|>"
    )
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

    if plugin_scheduler.lazy:
        plugin_scheduler.flush_lazy(list(plugin_scheduler.lazy_queue.keys())[0])

    if finished:
        while len(plugin_scheduler.waiting_queue) > 0:
            res = plugin_scheduler.poll_finished(
                list(plugin_scheduler.waiting_queue.keys())[0]
            )
        time_end = time.perf_counter()
        logging.info(f"Plugin result: {res}")
        logging.info(f"Finished: {finished[0].decode()}")
        logging.info(
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end-time_start)} tokens/s"
        )
        logging.info(f"Time: {time_end-time_start} s")
    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
    plugin_scheduler.join_all()


def eval_python(req: str, lazy: bool):
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
            f"Speed: {(len(finished[0].tokens)-init_tokens_len)/(time_end - time_start)} tokens/s"
        )
        logging.info(f"Time: {time_end - time_start} s")

    else:
        logging.info("Ongoing: " + engine.context.requests[0].decode())
    plugin_scheduler.join_all()


def eval_scheduling():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    logging.info(f"Loading model {model_name}")
    plugin_scheduler = PluginScheduler()
    engine = ScheduleEngine(
        ModelConfig(model_name),
        PythonParser,
        plugin_scheduler,
        max_concurrent_requests=1,
    )
    logging.info(f"Model {model_name} loaded")
    # req_id = engine.request_pool.add_request("Plotting a sine wave in python.")
    req_id = engine.request_pool.add_request(
        "List 10 famous mathematicians and their contributions."
    )
    req_id2 = engine.request_pool.add_request(
        "Write an email to manager about this quarter's performance in a financial company."
    )
    init_tokens_len = len(engine.request_pool.queued_requests[0].tokens)
    i = 0
    time_start = time.perf_counter()
    while i < 500:
        finished = engine.iteration_step()
        if finished:
            break
        i += 1

    if plugin_scheduler.lazy:
        plugin_scheduler.flush_lazy(list(plugin_scheduler.lazy_queue.keys())[0])

    if finished:
        while len(plugin_scheduler.waiting_queue) > 0:
            res = plugin_scheduler.poll_finished(
                list(plugin_scheduler.waiting_queue.keys())[0]
            )
        time_end = time.perf_counter()
        logging.info(f"Finished: {finished[0].decode()}")
        final_tokens_len = len(finished[0].tokens)
    else:
        time_end = time.perf_counter()
        logging.info("Ongoing: " + engine.context.requests[0].decode())
        final_tokens_len = len(engine.context.req_runtime_stats[req_id].tokens)

    logging.info(
        f"Speed: {(final_tokens_len-init_tokens_len)/(time_end - time_start)} tokens/s"
    )
    logging.info(f"Time: {time_end - time_start} s")
    plugin_scheduler.join_all()


def main100():
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "meetkai/functionary-small-v2.2"
    # config_path = "_private/mistral.json"
    logging.info(f"Loading model {model_name}")
    engine = ScheduleEngine(ModelConfig(model_name))
    logging.info(f"Model {model_name} loaded")
    # engine.request_pool.add_request("Hello, how are you?")
    # engine.request_pool.add_request("How are you?")
    req_id = engine.request_pool.add_request(
        # "Describe the basic components of a neural network and how it can be trained"
        # "[INST]Describe the basic components of a neural network and how it can be trained. [/INST]"
        # "\nAnd tell me how to write the Greatest common divisor algorithm in Python? Show me the code."
        generate_functionary_input() + "\n<|from|> assistant\n<|recipient|>"
    )
    # req2_id = engine.request_pool.add_request(json.dumps({"tools": tools, "messages": messages}))
    logging.info("Request added")
    print(engine.request_pool.queued_requests[0].tokens)

    logging.info("Prefill")
    prefill_len = len(engine.request_pool.queued_requests[0].tokens)
    prefill_start = time.perf_counter()
    r = engine.iteration_step(remove_finished=False)
    prefill_end = time.perf_counter()
    print(f"Result: {r}")
    print(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )

    req2_id = engine.request_pool.add_request("[INST]Generate a JSON. [/INST]")
    for _ in range(50):
        engine.iteration_step(remove_finished=False)

    logging.info("Decode")
    generation_num = 120
    decode_start = time.perf_counter()
    for _ in range(generation_num):
        engine.iteration_step(remove_finished=False)
    decode_end = time.perf_counter()

    logging.info("Add another request")
    extend_len = engine.extend_req_with_str(
        req2_id,
        "</s>\n<s>[INST]Let's stop here and continue later. Now tell me how to write a GCD in Python?[/INST]",
    )
    # extend_len = engine.extend_req_with_str(
    #     req_id,
    #     long_text(1024)
    # )
    extend_start = time.perf_counter()
    engine.iteration_step(remove_finished=False)
    extend_end = time.perf_counter()

    for _ in range(20):
        engine.iteration_step(remove_finished=False)

    req3_id = engine.request_pool.add_request(
        "[INST]How does operating system do scheduling? [/INST]"
    )

    extra_rounds = 100
    for i in range(extra_rounds):
        if i % 100 == 0:
            logging.info(f"Extra: {i/extra_rounds*100:.2f}%")
        engine.iteration_step(remove_finished=False)

    logging.info(
        f"Engine CTX: requests={[r.req_id for r in engine.context.requests]}, pending={[r.req_id for r in engine.context.pending_requests]}, stats={engine.context.req_runtime_stats}"
    )

    logging.info(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )
    for i in range(3):
        logging.info(
            f"Req{engine.context.requests[i].req_id}: {engine.context.requests[i].decode()}"
        )
    logging.info(
        f"Prefill speed: {prefill_len/(prefill_end - prefill_start)} tokens/s, time: {prefill_end - prefill_start} s"
    )
    logging.info(
        f"Extend speed: {extend_len/(extend_end - extend_start)} tokens/s, time: {extend_end - extend_start} s (len={extend_len})"
    )
    logging.info(
        f"Decode speed: {generation_num/(decode_end - decode_start)} tokens/s, time: {decode_end - decode_start} s"
    )
    page_used, page_all = engine.context.cache_manager.page_usage()
    logging.info(f"KV Cache usage: {page_used/page_all*100:.2f}%")


def good_python_example():
    set_hf_token()
    # eval_python("Plotting a sine wave in python. ONLY output code", lazy=False)
    # eval_python("Plotting a sine wave in python with torch. ONLY output code without trailing explanation.", lazy=False)
    # eval_python("Plotting a cosine wave in python with torch and matplotlib. ONLY output code without trailing explanation.", lazy=True)
    eval_python(
        "Plotting a sine wave in python with torch and matplotlib. ONLY output code without trailing explanation.",
        lazy=False,
    )


if __name__ == "__main__":
    set_hf_token()
    eval_scheduling()
