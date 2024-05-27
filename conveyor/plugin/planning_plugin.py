import sys
import time
from turtle import st
from conveyor.plugin.base_plugin import BasePlugin
from conveyor.plugin.search_plugin import search
from conveyor.utils import getLogger
import requests

logging = getLogger(__name__)


class PlanningPlugin(BasePlugin):
    def __init__(self, lazy: bool = False):
        super().__init__()
        self.query = None
        self.session = None
        self.lazy = lazy
        self.data = {}
        self.buffer = []
        self.time = 0
        if not self.lazy and self.session is None:
            self.session = requests.Session()
            self.session.get("https://www.google.com/generate_204")

    def process_new_dat(self, data: str):
        try:
            if not self.lazy:
                start = time.perf_counter()
                self.process_line(data)
                end = time.perf_counter()
                self.time += end - start
            else:
                self.buffer.append(data)
            return None
        except Exception as e:
            return e

    def finish(self):
        if not self.lazy:
            print(f"<PLUGIN_INFO> {self.time}", file=sys.stderr)
            return self.data["4"]
        else:
            start = time.perf_counter()
            for line in self.buffer:
                self.process_line(line)
            end = time.perf_counter()
            self.time += end - start
            print(f"<PLUGIN_INFO> {self.time}", file=sys.stderr)
            return self.data

    def process_line(self, line: str):
        if line.startswith("$1"):
            search(self.session, "hi", "en", "us")
            self.data["1"] = 200
        elif line.startswith("$2"):
            search(self.session, "hello", "en", "us")
            self.data["2"] = 300
        elif line.startswith("$3"):
            self.data["3"] = self.data["1"] / self.data["2"]
        elif line.startswith("$4"):
            self.data["4"] = f"Value of $3 is {self.data['3']}"
        elif line.startswith(">>>") or line.startswith("<<<"):
            pass
        else:
            logging.warn(f"Unknown line: {line}")
