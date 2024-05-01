from tkinter import S
from typing import Dict, List, Optional

import multiprocessing as mp
import sys
import os
import time
from multiprocessing.connection import Connection

from conveyor.plugin.base_plugin import BasePlugin, PlaceholderPlugin
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
    def __init__(self, lazy: bool = False) -> None:
        self.plugin_map: Dict[str, PluginInstance] = {}
        self.waiting_queue: Dict[str, List[PluginInstance]] = {}
        self.join_queue = []
        self.lazy = lazy
        self.lazy_queue: Dict[str, List[PluginInstance]] = {}
        pass

    def start_plugin(self, client_id: str, plugin_name: str):
        assert self.plugin_map.get(client_id) is None
        match plugin_name.strip():
            case "python":
                plugin = PythonPlugin(self.lazy)
                logging.debug(f"[PluginScheduler:{client_id}] Starting python plugin")
            case "search":
                plugin = SearchPlugin(self.lazy)
                logging.debug(f"[PluginScheduler:{client_id}] Starting search plugin")
            case _:
                plugin = PlaceholderPlugin()
                logging.warn(
                    f"[PluginScheduler:{client_id}] Starting placeholder plugin for {plugin_name}"
                )
        self.plugin_map[client_id] = PluginInstance(plugin)
        pass

    def process_new_data(self, client_id: str, data: str):
        plugin = self.plugin_map.get(client_id)
        if plugin is None:
            print(f"Plugin not found: {client_id}, data={data}", file=sys.stderr)
        assert plugin is not None
        plugin.local_pipe.send(data)

    def finish_plugin(self, client_id: str, sequential: bool = False):
        if self.lazy or (sequential and self.waiting_queue.get(client_id) is not None):
            # logging.warn(f"[PluginScheduler] sequential={sequential}")
            plugin = self.plugin_map.get(client_id)
            assert plugin is not None
            if self.lazy_queue.get(client_id) is None:
                self.lazy_queue[client_id] = []
            self.lazy_queue[client_id].append(plugin)
            del self.plugin_map[client_id]
            return
        # move from working slot to waiting queue
        plugin = self.plugin_map.get(client_id)
        logging.debug(f"[PluginScheduler:{client_id}] Finishing plugin")
        assert plugin is not None
        plugin.local_pipe.send(finish_str)
        if self.waiting_queue.get(client_id) is None:
            self.waiting_queue[client_id] = []
        self.waiting_queue[client_id].append(plugin)
        del self.plugin_map[client_id]

    def flush_lazy(self, client_id: str):
        if self.lazy_queue.get(client_id) is None:
            return
        for plugin in self.lazy_queue[client_id]:
            plugin.local_pipe.send(finish_str)
            if self.waiting_queue.get(client_id) is None:
                self.waiting_queue[client_id] = []
            self.waiting_queue[client_id].append(plugin)
        del self.lazy_queue[client_id]

    def poll_finished(self, client_id: str) -> Optional[List]:
        plugin_list = self.waiting_queue.get(client_id)
        assert plugin_list is not None
        for plugin in plugin_list:
            if plugin.local_pipe.poll():
                res = plugin.local_pipe.recv()
                # logging.debug(f"[PluginScheduler:{client_id}] Finished: {res}")
                self.join_queue.append(plugin.process)
                logging.debug(f"[PluginScheduler:{client_id}] Process joined")
                plugin_list.remove(plugin)
                if plugin_list == []:
                    del self.waiting_queue[client_id]
                return [res]
        return None

    def flush_lazy_sequentially(self, client_id: str):
        if self.lazy_queue.get(client_id) is None:
            return
        queue = self.lazy_queue[client_id]
        queue[0].local_pipe.send(finish_str)
        if self.waiting_queue.get(client_id) is None:
            self.waiting_queue[client_id] = []
        self.waiting_queue[client_id].append(queue[0])
        queue.pop(0)
        if len(queue) == 0:
            del self.lazy_queue[client_id]

    def join_all(self):
        proc_cnt = len(self.join_queue)
        for p in self.join_queue:
            p.join()
        self.join_queue.clear()
        logging.info(f"[PluginScheduler] All {proc_cnt} processes joined")
