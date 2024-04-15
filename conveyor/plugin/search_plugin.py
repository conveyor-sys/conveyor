from conveyor.plugin.base_plugin import BasePlugin
from conveyor.utils import getLogger

logging = getLogger(__name__)


class SearchPlugin(BasePlugin):
    def __init__(self):
        super().__init__()

    def process_new_dat(self, data: str):
        try:
            # todo
            print(f"SearchPlugin WIP: {data}")
            return None
        except Exception as e:
            return e

    def finish(self):
        pass
