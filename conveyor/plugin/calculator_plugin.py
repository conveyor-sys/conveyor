import sys
import time
from conveyor.plugin.base_plugin import BasePlugin
from conveyor.utils import getLogger

logging = getLogger(__name__)


class CalculatorPlugin(BasePlugin):
    def __init__(self, lazy: bool = False):
        super().__init__()
        self.lazy = lazy
        self.buf = None
        self.answer = None
        self.time = 0

    def process_new_dat(self, data: dict):
        if self.lazy:
            self.buf = data
        else:
            self.compute(data)

    def finish(self):
        if self.lazy:
            self.compute(self.buf)
        print(f"<PLUGIN_INFO> {self.time}", file=sys.stderr)
        return self.answer

    def compute(self, data: dict):
        start = time.perf_counter()
        query = data.get("query")
        self.answer = eval(query)
        end = time.perf_counter()
        self.time += end - start

