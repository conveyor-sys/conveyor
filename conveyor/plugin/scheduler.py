from typing import Dict

import multiprocessing as mp
import sys
from multiprocessing.connection import Connection

from conveyor.plugin.base_plugin import BasePlugin
from conveyor.plugin.python_plugin import PythonPlugin
from conveyor.plugin.search_plugin import SearchPlugin


finish_str = "@[finish]"


def plugin_loop(plugin: BasePlugin, pipe_in: Connection, pipe_out: Connection):
    print("Starting plugin loop", file=sys.stderr)
    while True:
        data = pipe_in.recv()
        if data == finish_str:
            pipe_out.send(plugin.finish())
            return
        err = plugin.process_new_dat(data)
        if err is not None:
            print(f"Error in plugin: {err}", file=sys.stderr)
            raise err


class PluginInstance:
    def __init__(self, plugin: BasePlugin) -> None:
        sending_pipe, receiving_pipe = mp.Pipe()
        self.process = mp.Process(
            target=plugin_loop, args=(plugin, sending_pipe, receiving_pipe)
        )
        self.sending_pipe = sending_pipe
        self.receiving_pipe = receiving_pipe
        self.process.start()


class PluginScheduler:
    def __init__(self) -> None:
        self.plugin_map: Dict[str, PluginInstance] = {}
        self.waiting_queue = {}
        pass

    def start_plugin(self, client_id: str, plugin_name: str):
        assert self.plugin_map.get(client_id) is None
        match plugin_name:
            case "python":
                plugin = PythonPlugin()
            case "search":
                plugin = SearchPlugin()
            case _:
                raise ValueError("Invalid plugin name")
        self.plugin_map[client_id] = PluginInstance(plugin)
        pass

    def process_new_data(self, client_id: str, data: str):
        plugin = self.plugin_map.get(client_id)
        assert plugin is not None
        plugin.sending_pipe.send(data)

    def finish_plugin(self, client_id: str):
        plugin = self.plugin_map.get(client_id)
        assert plugin is not None
        plugin.sending_pipe.send(finish_str)

    def poll_finished(self, client_id: str):
        plugin = self.plugin_map.get(client_id)
        assert plugin is not None
        if plugin.receiving_pipe.poll():
            return plugin.receiving_pipe.recv()
        return None
