from conveyor.plugin.base_plugin import BasePlugin
from conveyor.utils import getLogger

from io import StringIO
import sys

logging = getLogger(__name__)


class PythonPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.old_stdout = sys.stdout
        self.new_stdout = StringIO()
        sys.stdout = self.new_stdout

    def process_new_dat(self, data: str):
        if data.startswith("```python"):
            logging.debug("PythonPlugin: Executing python code")
            return None
        elif data.strip() == "```":
            logging.debug("PythonPlugin: Finished executing python code")
            return None
        try:
            exec(data)
            return None
        except Exception as e:
            return e

    def finish(self):
        val = self.new_stdout.getvalue()
        sys.stdout = self.old_stdout
        return val
