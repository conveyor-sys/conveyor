from typing import Dict, List, Optional

import multiprocessing as mp
import sys
import os
import time
from multiprocessing.connection import Connection

from conveyor.plugin.base_plugin import BasePlugin
from conveyor.plugin.python_plugin import PythonPlugin
from conveyor.plugin.search_plugin import SearchPlugin

from conveyor.utils import getLogger

logging = getLogger(__name__)


finish_str = "@[finish]"


def plugin_loop(plugin: BasePlugin, ep: Connection):
    print("Starting plugin loop", file=sys.stderr)
    while True:
        data = ep.recv()
        if data == finish_str:
            ep.send(plugin.finish())
            return
        err = plugin.process_new_dat(data)
        if err is not None:
            print(f"Error in plugin: {err}", file=sys.stderr)
            raise err


class PluginInstance:
    def __init__(self, plugin: BasePlugin) -> None:
        local_ep, plugin_ep = mp.Pipe()
        self.local_pipe = local_ep
        self.process = mp.Process(target=plugin_loop, args=(plugin, plugin_ep))
        self.process.start()
        logging.debug(f"[PluginInstance] Process started: {self.process.pid}")


class PluginScheduler:
    def __init__(self) -> None:
        self.plugin_map: Dict[str, PluginInstance] = {}
        self.waiting_queue: Dict[str, PluginInstance] = {}
        pass

    def start_plugin(self, client_id: str, plugin_name: str):
        assert self.plugin_map.get(client_id) is None
        match plugin_name:
            case "python":
                plugin = PythonPlugin()
                logging.debug(f"[PluginScheduler:{client_id}] Starting python plugin")
            case "search":
                plugin = SearchPlugin()
                logging.debug(f"[PluginScheduler:{client_id}] Starting search plugin")
            case _:
                raise ValueError(f"Invalid plugin name: >>{plugin_name}<<")
        self.plugin_map[client_id] = PluginInstance(plugin)
        pass

    def process_new_data(self, client_id: str, data: str):
        plugin = self.plugin_map.get(client_id)
        assert plugin is not None
        plugin.local_pipe.send(data)

    def finish_plugin(self, client_id: str):
        plugin = self.plugin_map.get(client_id)
        logging.debug(f"[PluginScheduler:{client_id}] Finishing plugin")
        assert plugin is not None
        plugin.local_pipe.send(finish_str)
        self.waiting_queue[client_id] = plugin
        del self.plugin_map[client_id]

    def poll_finished(self, client_id: str) -> Optional[List]:
        plugin = self.waiting_queue.get(client_id)
        assert plugin is not None
        if plugin.local_pipe.poll():
            res = plugin.local_pipe.recv()
            logging.debug(f"[PluginScheduler:{client_id}] Finished: {res}")
            plugin.process.join()
            logging.debug(f"[PluginScheduler:{client_id}] Process joined")
            del self.waiting_queue[client_id]
            return [res]
        return None
