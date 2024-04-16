from typing import Optional


class StreamingJsonObjParser:
    def __init__(self):
        self.current_key: Optional[str | StreamingJsonObjParser] = None
        self.current_obj: Optional[StreamingJsonObjParser | StreamingJsonObjParser] = (
            None
        )
        self.done = False

    def feed_char(self, char):
        if self.current_obj is None:
            if self.current_key is None:
                if char == '"':
                    self.current_key = StreamingJsonStringParser()
                elif char == "}":
                    self.done = True
                    return None
                else:
                    pass
            elif isinstance(self.current_key, StreamingJsonStringParser):
                key = self.current_key.feed_char(char)
                if key is not None:
                    self.current_key = key
            else:
                # expect value
                if char == '"':
                    self.current_obj = StreamingJsonStringParser()
                elif char == "{":
                    self.current_obj = StreamingJsonObjParser()
                else:
                    pass
        else:
            val = self.current_obj.feed_char(char)
            if val is not None:
                result = {self.current_key: val}
                self.current_key = None
                self.current_obj = None
                return result
        return None


class StreamingJsonStringParser:
    def __init__(self) -> None:
        self.buf = ""

    def feed_char(self, char) -> Optional[str]:
        if char == '"':
            return self.buf
        else:
            self.buf += char
        return None
