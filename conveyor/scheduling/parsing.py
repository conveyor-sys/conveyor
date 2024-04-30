from typing import Dict, List, Optional
from conveyor.plugin.json_parser import StreamingJsonObjParser

from conveyor.utils import getLogger

logging = getLogger(__name__)


class BaseParser:
    def enqueue(self, token) -> Optional[Dict | List]:
        raise NotImplementedError


class PlaceHolderParser(BaseParser):
    def __init__(self, tokenizer, client_id, start_cb, update_cb, finish_cb):
        self.buffer = []
        self.in_progress = False
        self.string = ""
        self.tokenizer = tokenizer
        self.start_cb = start_cb
        self.update_cb = update_cb
        self.finish_cb = finish_cb
        self.client_id = client_id

    def enqueue(self, token) -> Optional[Dict | List]:
        pass


# For mistral
class PythonParser(BaseParser):
    CRLF = 13

    def __init__(self, tokenizer, client_id, start_cb, update_cb, finish_cb):
        self.buffer = []
        self.in_progress = False
        self.string = ""
        self.tokenizer = tokenizer
        self.start_cb = start_cb
        self.update_cb = update_cb
        self.finish_cb = finish_cb
        self.client_id = client_id

    def enqueue(self, token) -> Optional[Dict | List]:
        self.buffer.append(token)
        match token:
            # CRLF, </s> or <|STOP|>
            case self.CRLF | self.tokenizer.eos_token_id | 32003:
                buf = None
                if self.string == "```python":
                    self.start_cb(self.client_id, "python")
                    self.in_progress = True
                if self.in_progress:
                    self.update_cb(self.client_id, self.string)
                    buf = self.buffer[:-1]
                if self.string == "```" or self.string.startswith("```</s>"):
                    self.finish_cb(self.client_id)
                    self.in_progress = False
                self.buffer = []
                self.string = ""
                # print(f"The string is: ::>{self.string}<::")
                return buf
            case _:
                new_str: str = self.tokenizer.convert_ids_to_tokens(token)
                if not isinstance(new_str, str):
                    new_str = new_str.decode("utf-8", errors="ignore")
                if new_str.startswith("▁"):
                    new_str = " " + new_str[1:]
                self.string += new_str
                return None


class FunctionaryParser(BaseParser):
    CONTENT = 32000
    RECIPIENT = 32001
    FROM = 32002
    STOP = 32003
    CRLF = 13

    def __init__(self, tokenizer, client_id, start_cb, update_cb, finish_cb):
        self.buffer = []
        self.string = ""
        # self.left_bracket_pos = []
        self.tokenizer = tokenizer
        self.start_cb = start_cb
        self.update_cb = update_cb
        self.finish_cb = finish_cb
        self.obj_parser = None
        self.client_id = client_id

    def enqueue(self, token) -> Optional[Dict | List]:
        self.buffer.append(token)
        match token:
            case self.STOP:
                buf = self.buffer
                self.buffer = []
                return buf
            case self.CRLF:
                if (
                    self.buffer[0] == self.CONTENT
                    or self.buffer[0] == self.RECIPIENT
                    or self.buffer[0] == self.FROM
                ):
                    if self.buffer[0] == self.RECIPIENT:
                        func_name = self.tokenizer.decode(self.buffer[1:]).strip()
                        logging.debug(f"Recipient: {func_name}")
                        self.start_cb(self.client_id, func_name)
                    buf = self.buffer
                    self.buffer = []
                    return buf
                else:
                    return None
            case _:
                new_str: str = self.tokenizer.convert_ids_to_tokens(token)
                if not isinstance(new_str, str):
                    new_str = new_str.decode("utf-8", errors="ignore")
                if new_str.startswith("▁"):
                    new_str = " " + new_str[1:]
                self.string += new_str
                for c in new_str:
                    if self.obj_parser is None and c == "{":
                        # TODO
                        self.obj_parser = StreamingJsonObjParser()
                        self.obj_parser.feed_char(c)
                    elif self.obj_parser is not None:
                        res = self.obj_parser.feed_char(c)
                        if res is not None:
                            self.update_cb(self.client_id, res)
                        if self.obj_parser.done:
                            self.finish_cb(self.client_id)
                            self.obj_parser = None
                return None
