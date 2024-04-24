class BasePlugin:
    def __init__(self):
        pass

    def process_new_dat(self, data):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError


class PlaceholderPlugin(BasePlugin):
    def __init__(self):
        super().__init__()

    def process_new_dat(self, data):
        return None

    def finish(self):
        return None
