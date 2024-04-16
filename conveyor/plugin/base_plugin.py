class BasePlugin:
    def __init__(self):
        pass

    def process_new_dat(self, data):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError
