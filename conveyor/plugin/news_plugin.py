from conveyor.plugin.base_plugin import BasePlugin
from conveyor.utils import getLogger
import datetime

logging = getLogger(__name__)


class LocalNewsPlugin(BasePlugin):
    def __init__(self, lazy: bool = False):
        super().__init__()
        self.lazy = lazy
        self.buf = []
        self.answer = None
        self.abort = False
        self.start_time = datetime.datetime.now()

    def process_new_dat(self, data: dict):
        if self.lazy:
            self.buf.append(data)
        else:
            self.compute(data)

    def finish(self):
        if self.lazy:
            for data in self.buf:
                self.compute(data)
        return self.answer

    def compute(self, data: dict):
        if data.get("location") is not None:
            if "," not in data["location"]:
                self.abort = True
                end_time = datetime.datetime.now()
                dur = (end_time - self.start_time).total_seconds()
                print(f"Plugin syntax error detected: {dur}s")
                self.answer = dur
