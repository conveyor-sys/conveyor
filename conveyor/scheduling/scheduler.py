from conveyor.scheduling.request_pool import RequestPool
from transformers import PretrainedConfig


class ScheduleEngine:
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.request_pool = RequestPool()
