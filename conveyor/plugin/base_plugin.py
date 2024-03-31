class BasePlugin:
    def __init__(self):
        pass

    async def process_new_arg(self, key: str, value: str):
        raise NotImplementedError

    async def finish(self):
        raise NotImplementedError
