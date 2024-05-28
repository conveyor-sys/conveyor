import sys
import time
from conveyor.plugin.base_plugin import BasePlugin
from conveyor.utils import getLogger
import sqlite3

logging = getLogger(__name__)


class SqlitePlugin(BasePlugin):
    def __init__(self, lazy: bool = False):
        super().__init__()
        self.lazy = lazy
        self.connection = None
        self.cursor = None
        self.buf = None
        self.answer = None
        self.time = 0

    def post_init(self):
        start = time.perf_counter()
        if not self.lazy:
            self.connection = sqlite3.connect("test/test.sqlite3")
            self.cursor = self.connection.cursor()
        end = time.perf_counter()
        self.time += end - start
        return super().post_init()

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
        if self.connection is None:
            self.connection = sqlite3.connect("test/test.sqlite3")
            self.cursor = self.connection.cursor()
        if data.get("query") is not None:
            self.cursor.execute("SELECT * FROM employees ORDER BY name")
            self.answer = self.cursor.fetchall()
        end = time.perf_counter()
        self.time += end - start
