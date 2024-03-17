from conveyor.models.config import ModelConfig
from conveyor.scheduling.scheduler import ScheduleEngine
import time
import json
from misc.functionary import generate_functionary_input

from conveyor.utils import getLogger

logging = getLogger("conveyor.main")


def long_text(num: int = 50):
    return " ".join(["me"] * num)


def main():
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
    r = engine.iteration_step()
    prefill_end = time.perf_counter()
    print(f"Result: {r}")
    print(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )

    req2_id = engine.request_pool.add_request("[INST]Generate a JSON. [/INST]")
    for _ in range(50):
        engine.iteration_step()

    logging.info("Decode")
    generation_num = 120
    decode_start = time.perf_counter()
    for _ in range(generation_num):
        engine.iteration_step()
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
    engine.iteration_step()
    extend_end = time.perf_counter()

    for _ in range(20):
        engine.iteration_step()

    req3_id = engine.request_pool.add_request(
        "[INST]How does operating system do scheduling? [/INST]"
    )

    extra_rounds = 100
    for i in range(extra_rounds):
        if i % 100 == 0:
            logging.info(f"Extra: {i/extra_rounds*100:.2f}%")
        engine.iteration_step()

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

    ids = [
        1,
        32002,
        1587,
        13,
        32001,
        544,
        13,
        32000,
        589,
        10731,
        286,
        908,
        20343,
        369,
        1023,
        347,
        1987,
        739,
        4892,
        28723,
        13,
        14147,
        5572,
        371,
        13,
        13,
        421,
        2483,
        272,
        1868,
        8086,
        13,
        1123,
        625,
        28730,
        3022,
        28730,
        769,
        1223,
        327,
        9453,
        28747,
        371,
        13,
        421,
        415,
        2990,
        304,
        1665,
        28725,
        317,
        28723,
        28721,
        28723,
        3652,
        9686,
        28725,
        9461,
        28723,
        13,
        2733,
        28747,
        1423,
        28725,
        13,
        1542,
        953,
        707,
        28745,
        13,
        13,
        28752,
        589,
        4772,
        5572,
        13,
        32002,
        1587,
        13,
        32001,
        544,
        13,
        32000,
        330,
        10706,
        1444,
        264,
        13903,
        2188,
        304,
        396,
        18278,
        10895,
        13892,
        28723,
        415,
        13892,
        5212,
        10865,
        28725,
        10537,
        28725,
        304,
        27057,
        11194,
        298,
        272,
        2188,
        28742,
        28713,
        4224,
        28723,
        415,
        13892,
        6470,
        5572,
        395,
        7658,
        2787,
        739,
        4892,
        13,
        32002,
        2188,
        13,
        32001,
        544,
        13,
        32000,
        1824,
        349,
        272,
        8086,
        354,
        315,
        13511,
        9258,
        28804,
        13,
        32002,
        13892,
        13,
        32001,
    ]
    print(engine.context.requests[0].tokenizer.decode(ids))


if __name__ == "__main__":
    main()
