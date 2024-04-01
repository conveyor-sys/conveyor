from typing import Dict, List, Optional

from conveyor.utils import getLogger

logging = getLogger(__name__)


class FunctionaryParser:
    CONTENT = 32000
    RECIPIENT = 32001
    FROM = 32002
    STOP = 32003
    CRLF = 13

    def __init__(self, tokenizer, callback):
        self.buffer = []
        self.string = ""
        self.left_bracket_pos = []
        self.tokenizer = tokenizer
        self.callback = callback

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
                        logging.debug(
                            f"Recipient: {self.tokenizer.decode(self.buffer[1:])}"
                        )
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
                if "{" in new_str:
                    bracket_index = len(self.string) - len(new_str) + new_str.index("{")
                    self.left_bracket_pos.append(bracket_index)
                elif "}" in new_str:
                    bracket_index = len(self.string) - len(new_str) + new_str.index("}")
                    sub_str = self.string[self.left_bracket_pos[-1] : bracket_index + 1]
                    self.left_bracket_pos.pop()
                    self.callback(sub_str)
                return None
