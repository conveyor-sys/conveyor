from conveyor.models.config import ModelConfig
from conveyor.scheduling.scheduler import ScheduleEngine
import logging
import time
import os

logging.basicConfig(level=logging.NOTSET)

def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # config_path = "_private/mistral.json"
    logging.info(f"Loading model {model_name}")
    engine = ScheduleEngine(ModelConfig(model_name))
    logging.info(f"Model {model_name} loaded")
    # engine.request_pool.add_request("Hello, how are you?")
    # engine.request_pool.add_request("How are you?")
    req_id = engine.request_pool.add_request(
        # "Describe the basic components of a neural network and how it can be trained"
        "[INST]Describe the basic components of a neural network and how it can be trained. [/INST]"
        # "\nAnd tell me how to write the Greatest common divisor algorithm in Python? Show me the code."
    )
    req2_id = engine.request_pool.add_request("[INST]Generate a JSON. [/INST]")
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

    logging.info("Decode")
    generation_num = 60
    decode_start = time.perf_counter()
    for _ in range(generation_num):
        engine.iteration_step()
    decode_end = time.perf_counter()

    logging.info("Add another request")
    extend_len = engine.extend_req_with_str(
        req_id,
        "</s>\n<s>[INST]Let's stop here and continue later. Now tell me how to write a GCD in Python?[/INST]",
    )
    extend_start = time.perf_counter()
    engine.iteration_step()
    extend_end = time.perf_counter()

    for _ in range(100):
        engine.iteration_step()

    print(
        f"Engine CTX: requests={[r.req_id for r in engine.context.requests]}, pending={[r.req_id for r in engine.context.pending_requests]}, stats={engine.context.req_runtime_stats}"
    )

    print(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )
    print(f"Req: {engine.context.requests[0].decode()}")
    print(f"Req2: {engine.context.requests[1].decode()}")
    print(f"Prefill speed: {prefill_len/(prefill_end - prefill_start)} tokens/s, time: {prefill_end - prefill_start} s")
    print(f"Extend speed: {extend_len/(extend_end - extend_start)} tokens/s, time: {extend_end - extend_start} s")
    print(f"Decode speed: {generation_num/(decode_end - decode_start)} tokens/s, time: {decode_end - decode_start} s")


if __name__ == "__main__":
    main()
