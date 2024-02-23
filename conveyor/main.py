from conveyor.models.config import ModelConfig
from conveyor.scheduling.scheduler import ScheduleEngine


def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    engine = ScheduleEngine(ModelConfig(model_name))


if __name__ == "__main__":
    main()
