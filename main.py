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
    req_id = engine.request_pool.add_request(
        "[INST]Describe the basic components of a neural network and how it can be trained.[/INST]"
        # "\nAnd tell me how to write the Greatest common divisor algorithm in Python? Show me the code."
    )
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
    generation_num = 15
    decode_start = time.perf_counter()
    for _ in range(generation_num):
        engine.iteration_step()
    decode_end = time.perf_counter()

    logging.info("Add another request")
    engine.extend_req_with_str(req_id, "</s>\n<s>[INST]Let's stop here and continue later. Now tell me how to write a GCD in Python?[/INST]")
    engine.iteration_step()

    for _ in range(350):
        engine.iteration_step()

    print(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )
    print(f"Req: {engine.context.requests[0].decode()}")
    print(f"Prefill speed: {prefill_len/(prefill_end - prefill_start)} tokens/s")
    print(f"Decode speed: {generation_num/(decode_end - decode_start)} tokens/s")


if __name__ == "__main__":
    main()
