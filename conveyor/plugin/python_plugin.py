from conveyor.plugin.base_plugin import BasePlugin
from conveyor.utils import getLogger

from io import StringIO
import sys
import time

logging = getLogger(__name__)


class PythonPlugin(BasePlugin):
    def __init__(self, lazy: bool = False):
        super().__init__()
        self.old_stdout = sys.stdout
        self.new_stdout = StringIO()
        self.global_vars = {}
        self.text_buffer = []
        self.start_time = None
        self.lazy = lazy
        sys.stdout = self.new_stdout

    def process_new_dat(self, data: str):
        if data.startswith("```python"):
            print("PythonPlugin: Executing python code", file=sys.stderr)
            return None
        elif data.strip() == "```":
            print("PythonPlugin: Finished python code", file=sys.stderr)
            return None
        try:
            if self.lazy:
                self.text_buffer.append(data)
                return None
            else:
                exec(data, self.global_vars)
                return None
        except Exception as e:
            return e

    def finish(self):
        if self.lazy:
            start_time = time.perf_counter()
            try:
                exec("\n".join(self.text_buffer), self.global_vars)
            except Exception as e:
                return e
            end_time = time.perf_counter()
            logging.info(f"PythonPlugin: Lazy Execution time: {end_time-start_time}")
        val = self.new_stdout.getvalue()
        sys.stdout = self.old_stdout
        return val
