from conveyor.models.config import ModelConfig
from conveyor.scheduling.scheduler import ScheduleEngine
import logging

logging.basicConfig(level=logging.NOTSET)


def main():
    # model_name = "meta/Llama-2-7b-chat-hf"
    # model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    logging.info(f"Loading model {model_name}")
    engine = ScheduleEngine(ModelConfig(model_name))
    logging.info(f"Model {model_name} loaded")
    engine.request_pool.add_request("Hello, how are you?")
    logging.info("Request added")
    print(engine.request_pool.queued_requests[0].tokens)

    logging.info("First step")
    r = engine.iteration_step()
    print(f"Result: {r}")
    print(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )

    logging.info("Second step")
    r = engine.iteration_step()
    print(f"Result: {r}")
    print(
        f"SchedulerContext: requests={[r.tokens for r in engine.context.requests]}, stats={engine.context.req_runtime_stats}, seq_lens={engine.context.seq_lens}, completed_lens={engine.context.completed_lens}"
    )


if __name__ == "__main__":
    main()
