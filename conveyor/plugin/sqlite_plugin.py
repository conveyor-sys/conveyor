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

    def post_init(self):
        if not self.lazy:
            self.connection = sqlite3.connect("_private/test.sqlite3")
            self.cursor = self.connection.cursor()
        return super().post_init()

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
        if self.connection is None:
            self.connection = sqlite3.connect("_private/test.sqlite3")
            self.cursor = self.connection.cursor()
        if data.get("query") is not None:
            self.cursor.execute("SELECT * FROM employees ORDER BY name")
            self.answer = self.cursor.fetchall()
