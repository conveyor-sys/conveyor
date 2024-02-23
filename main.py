from conveyor.models.config import ModelConfig
from conveyor.scheduling.scheduler import ScheduleEngine
import logging

logging.basicConfig(level=logging.NOTSET)


def main():
    # model_name = "meta/Llama-2-7b-chat-hf"
    model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
    logging.info(f"Loading model {model_name}")
    engine = ScheduleEngine(
        ModelConfig(model_name, model_arch_override="LlamaForCausalLM")
    )
    logging.info(f"Model {model_name} loaded")


if __name__ == "__main__":
    main()
