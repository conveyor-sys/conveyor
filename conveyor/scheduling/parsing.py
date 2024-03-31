from typing import Dict, List, Optional

from conveyor.utils import getLogger

logging = getLogger(__name__)


class FunctionaryParser:
    CONTENT = 32000
    RECIPIENT = 32001
    FROM = 32002
    STOP = 32003
    CRLF = 13

    def __init__(self, tokenizer):
        self.buffer = []
        self.string = ""
        self.left_bracket_pos = []
        self.tokenizer = tokenizer

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
                    buf = self.buffer
                    self.buffer = []
                    return buf
                else:
                    return None
            case _:
                new_str: str = self.tokenizer.decode(token)
                self.string += new_str
                if "{" in new_str:
                    bracket_index = len(self.string) - len(new_str) + new_str.index("{")
                    self.left_bracket_pos.append(bracket_index)
                elif "}" in new_str:
                    bracket_index = len(self.string) - len(new_str) + new_str.index("}")
                    sub_str = self.string[self.left_bracket_pos[-1] : bracket_index + 1]
                    self.left_bracket_pos.pop()
                    logging.debug(f"Parser got: {sub_str}")
                return None
