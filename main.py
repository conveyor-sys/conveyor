from conveyor.models.config import ModelConfig
from conveyor.scheduling.scheduler import ScheduleEngine
import logging
import time

logging.basicConfig(level=logging.NOTSET)


def main():
    # model_name = "meta/Llama-2-7b-chat-hf"
    # model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    logging.info(f"Loading model {model_name}")
    engine = ScheduleEngine(ModelConfig(model_name))
    logging.info(f"Model {model_name} loaded")
    # engine.request_pool.add_request("Hello, how are you?")
    # engine.request_pool.add_request("How are you?")
    engine.request_pool.add_request(
        "Describe the basic components of a neural network and how it can be trained."
    )
    logging.info("Request added")
    print(engine.request_pool.queued_requests[0].tokens)

    logging.info("First step")
    prefill_len = len(engine.request_pool.queued_requests[0].tokens)
    prefill_start = time.perf_counter()
    r = engine.iteration_step()
    prefill_end = time.perf_counter()
    print(f"Result: {r}")
    print(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )

    logging.info("More steps")
    generation_num = 500
    decode_start = time.perf_counter()
    for _ in range(generation_num):
        engine.iteration_step()
    decode_end = time.perf_counter()
    print(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )
    print(f"Req: {engine.context.requests[0].decode()}")
    print(f"Prefill speed: {prefill_len/(prefill_end - prefill_start)} tokens/s")
    print(f"Decode speed: {generation_num/(decode_end - decode_start)} tokens/s")


if __name__ == "__main__":
    main()
