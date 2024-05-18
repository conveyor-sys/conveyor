from conveyor.plugin.base_plugin import BasePlugin
from conveyor.utils import getLogger

logging = getLogger(__name__)


class CalculatorPlugin(BasePlugin):
    def __init__(self, lazy: bool = False):
        super().__init__()
        self.lazy = lazy
        self.buf = None
        self.answer = None

    def process_new_dat(self, data: dict):
        if self.lazy:
            self.buf = data
        else:
            self.compute(data)

    def finish(self):
        if self.lazy:
            self.compute(self.buf)
        return self.answer

    def compute(self, data: dict):
        query = data.get("query")
        self.answer = eval(query)
