from copy import deepcopy
from typing import Dict, List, Optional
import jsonref

SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""
PYTHON_RUN_SYS_MSG = "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files."

from_token = "<|from|>"
recipient_token = "<|recipient|>"
content_token = "<|content|>"
stop_token = "<|stop|>"
# This token splits between function name and parameters
fn_param_sep_token = "\n<|content|> {"


def get_chat_template_jinja() -> str:
    chat_template = """{% for message in messages %}
    {% if message['role'] == 'user' or message['role'] == 'system' %}
        {{ '<|from|>' + message['role'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}<br>
    {% elif message['role'] == 'tool' %}
        {{ '<|from|>' + message['name'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}<br>
    {% else %}
        {% set contain_content='no'%}
        {% if message['content'] is not none %}
            {{ '<|from|>assistant\n<|recipient|>all\n<|content|>' + message['content'] }}<br>
            {% set contain_content='yes'%}
        {% endif %}
        {% if 'tool_calls' in message and message['tool_calls'] is not none %}
            {% for tool_call in message['tool_calls'] %}
                {% set prompt='<|from|>assistant\n<|recipient|>' + tool_call['function']['name'] + '\n<|content|>' + tool_call['function']['arguments'] %}
                {% if loop.index == 1 and contain_content == "no" %}
                    {{ prompt }}<br>
                {% else %}
                    {{ '\n' + prompt}}<br>
                {% endif %}
            {% endfor %}
        {% endif %}
        {{ '<|stop|>\n' }}<br>
    {% endif %}
    {% endfor %}
    {% if add_generation_prompt %}{{ '<|from|>assistant\n<|recipient|>' }}{% endif %}
    """
    chat_template = chat_template.replace("    ", "")
    chat_template = chat_template.replace("<br>\n", "")
    chat_template = chat_template.strip()
    return chat_template


def convert_message_to_prompt(message: Dict) -> str:
    role = message["role"]
    content = message.get("content", None)

    if role in [
        "system",
        "user",
    ]:  # <|from|>system\n<|recipient|>all\n<|content|>xxx
        return f"{from_token}{role}\n{recipient_token}all\n{content_token}{content}\n"

    if role == "tool":  # <|from|>tool_name\n<|recipient|>all\n<|content|>xxx
        tool_name = message["name"]
        return (
            f"{from_token}{tool_name}\n{recipient_token}all\n{content_token}{content}\n"
        )

    assert role == "assistant"
    tool_calls = message.get("tool_calls", [])
    if tool_calls is None:
        tool_calls = []
    if (
        len(tool_calls) == 0 and content is None
    ):  # for inference: <|from|> assistant\n<|recipient|>
        return f"{from_token}{role}\n{recipient_token}"

    if len(tool_calls) == 0:  # <|from|>assistant\n<|recipient|>all\n<|content|>xxx
        return f"{from_token}{role}\n{recipient_token}all\n{content_token}{content}{stop_token}\n"

    result = ""
    if content is not None:  # both text-response and function_call
        result += (
            f"{from_token}{role}\n{recipient_token}all\n{content_token}{content}\n"
        )

    for tool in tool_calls:
        func_name = tool["function"]["name"]
        arguments = tool["function"]["arguments"]
        #  <|from|>assistant\n<|recipient|>func_name\n<|content|>xxxx
        result += f"{from_token}{role}\n{recipient_token}{func_name}\n{content_token}{arguments}\n"

    result = result.strip() + f"{stop_token}\n"
    return result


def convert_data_type(param_type: str) -> str:
    """convert data_type to typescript data type

    Args:
        param_type (str): param_type

    Returns:
        str: param type in typescript
    """
    if param_type == "integer" or param_type == "float":
        return "number"
    return param_type


def get_enum_option_str(enum_options: List) -> str:
    """get enum option separated by: "|"

    Args:
        enum_options (List): list of options

    Returns:
        _type_: concatenation of options separated by "|"
    """
    # if each option is string --> add quote
    return " | ".join([f'"{v}"' if type(v) is str else str(v) for v in enum_options])


def get_array_typescript(
    param_name: Optional[str], param_dic: dict, depth: int = 0
) -> str:
    """recursive implementation for generating type script of array

    Args:
        param_name (Optional[str]): name of param, optional
        param_dic (dict): param_dic
        depth (int, optional): nested level. Defaults to 0.

    Returns:
        _type_: typescript of array
    """
    offset = ""
    if depth >= 1:
        offset = "".join(["    " for _ in range(depth)])
    items_info = param_dic.get("items", {})

    if len(items_info) == 0:
        if param_name is not None:
            return f"{offset}{param_name}: []"
        else:
            return "[]"
    array_type = get_param_type(items_info)
    if array_type == "object":
        info_lines = []
        child_lines = get_parameter_typescript(
            items_info.get("properties", {}), items_info.get("required", []), depth + 1
        )
        # if comment_info is not None:
        #    info_lines.append(f"{offset}{comment_info}")
        if param_name is not None:
            info_lines.append(f"{offset}{param_name}" + ": {")
        else:
            info_lines.append(f"{offset}" + "{")
        info_lines.extend(child_lines)
        info_lines.append(f"{offset}" + "}[]")
        return "\n".join(info_lines)

    elif array_type == "array":
        item_info = get_array_typescript(None, items_info, depth + 1)
        if param_name is None:
            return f"{item_info}[]"
        return f"{offset}{param_name}: {item_info.strip()}[]"

    else:
        if "enum" in items_info:
            item_type = get_enum_option_str(items_info["enum"])
            if param_name is None:
                return f"({item_type})[]"
            else:
                return f"{offset}{param_name}: ({item_type})[]"
        else:
            if param_name is None:
                return f"{array_type}[]"
            else:
                return f"{offset}{param_name}: {array_type}[],"


def get_param_type(param: Dict) -> str:
    """get param_type of parameter

    Args:
        param (Dict): param dict in properties

    Returns:
        str: _description_
    """
    param_type = "any"
    if "type" in param:
        raw_param_type = param["type"]
        if type(raw_param_type) is list:
            param_type = " | ".join(raw_param_type)
        else:
            param_type = raw_param_type

    else:  # in many cases, the json schema contains: oneOf instead of "type"
        if "oneOf" in param:
            one_of_types = []
            for item in param["oneOf"]:
                if "type" in item:
                    one_of_types.append(convert_data_type(item["type"]))
            one_of_types = list(set(one_of_types))
            param_type = " | ".join(one_of_types)
    return convert_data_type(param_type)


def get_format_param(param: Dict) -> Optional[str]:
    """Get "format" from param. There are cases where format is not directly in param but in oneOf

    Args:
        param (Dict): _description_

    Returns:
        Optional[str]: _description_
    """
    if "format" in param:
        return param["format"]
    if "oneOf" in param:
        formats = []
        for item in param["oneOf"]:
            if "format" in item:
                formats.append(item["format"])
        if len(formats) > 0:
            return " or ".join(formats)
    return None


def get_param_info(param: Dict) -> Optional[str]:
    """get additional information about parameter such as: format, default value, min, max, ...

    Args:
        param (Dict): _description_

    Returns:
        Optional[str]: _description_
    """
    param_type = param.get("type", "any")
    info_list = []
    if "description" in param:
        desc = param["description"]
        if not desc.endswith("."):
            desc += "."
        info_list.append(desc)

    if "default" in param:
        default_value = param["default"]
        if param_type == "string":
            default_value = f'"{default_value}"'  # if string --> add ""
        info_list.append(f"Default={default_value}.")

    format_param = get_format_param(param)
    if format_param is not None:
        info_list.append("Format=" + format_param)

    for field, field_name in [
        ("maximum", "Maximum"),
        ("minimum", "Minimum"),
        ("maxLength", "Maximum length"),
        ("minLength", "Minimum length"),
    ]:
        if field in param:
            info_list.append(f"{field_name}=" + str(param[field]))

    if len(info_list) > 0:
        result = "// " + " ".join(info_list)
        result = result.replace("\n", " ")
        return result
    return None


def append_new_param_info(
    info_list: List[str],
    param_declaration: str,
    comment_info: Optional[str],
    depth: int,
):
    """Append a new parameter with comment to the info_list

    Args:
        info_lines (List[str]): current info_list
        param_declaration (str): param: type
        comment_info (Optional[str]): information of comment
        depth (int): level of nested param
    """
    offset = ""
    if depth >= 1:
        offset = "".join(["    " for _ in range(depth)])
    if comment_info is not None:
        # if depth == 0:  # format: //comment\nparam: type
        info_list.append(f"{offset}{comment_info}")
        info_list.append(f"{offset}{param_declaration}")
    # else:  # format: param: type  // comment
    #     info_list.append(f"{offset}{param_declaration}    {comment_info}")
    else:
        info_list.append(f"{offset}{param_declaration}")


def get_parameter_typescript(properties, required_params, depth=0) -> List[str]:
    """Recursion, returning the information about parameters including data type, description and other information
    These kinds of information will be put into the prompt

    Args:
        properties (_type_): properties in parameters
        required_params (_type_): List of required parameters
        depth (int, optional): the depth of params (nested level). Defaults to 0.

    Returns:
        _type_: list of lines containing information about all parameters
    """
    tp_lines = []
    for param_name, param in properties.items():
        # Sometimes properties have "required" field as a list of string.
        # Even though its supposed to be not under properties. So we skip it
        if not isinstance(param, dict):
            continue
        # Param Description
        comment_info = get_param_info(param)
        # Param Name declaration
        param_declaration = f"{param_name}"
        if isinstance(required_params, list):
            if param_name not in required_params:
                param_declaration += "?"
        param_type = get_param_type(param)

        offset = ""
        if depth >= 1:
            offset = "".join(["    " for _ in range(depth)])

        if param_type == "object":  # param_type is object
            child_lines = get_parameter_typescript(
                param.get("properties", {}), param.get("required", []), depth + 1
            )
            if comment_info is not None:
                tp_lines.append(f"{offset}{comment_info}")

            param_declaration += ": {"
            tp_lines.append(f"{offset}{param_declaration}")
            tp_lines.extend(child_lines)
            tp_lines.append(f"{offset}" + "},")

        elif param_type == "array":  # param_type is an array
            item_info = param.get("items", {})
            if "type" not in item_info:  # don't know type of array
                param_declaration += ": [],"
                append_new_param_info(tp_lines, param_declaration, comment_info, depth)
            else:
                array_declaration = get_array_typescript(
                    param_declaration, param, depth
                )
                if not array_declaration.endswith(","):
                    array_declaration += ","
                if comment_info is not None:
                    tp_lines.append(f"{offset}{comment_info}")
                tp_lines.append(array_declaration)
        else:
            if "enum" in param:
                param_type = get_enum_option_str(param["enum"])
                # param_type = " | ".join([f'"{v}"' for v in param["enum"]])
            param_declaration += f": {param_type},"
            append_new_param_info(tp_lines, param_declaration, comment_info, depth)

    return tp_lines


def generate_schema_from_functions(functions: List, namespace="functions") -> str:
    """
    Convert functions schema to a schema that language models can understand.
    """

    schema = "// Supported function definitions that should be called when necessary.\n"
    schema += f"namespace {namespace} {{\n\n"

    for function in functions:
        # Convert a Function object to dict, if necessary
        if not isinstance(function, dict):
            function = function.model_dump()
        function_name = function.get("name", None)
        if function_name is None:
            continue

        description = function.get("description", "")
        schema += f"// {description}\n"
        schema += f"type {function_name}"

        parameters = function.get("parameters", None)
        if parameters is not None and parameters.get("properties") is not None:
            parameters = deepcopy(jsonref.JsonRef.replace_refs(parameters))
            schema += " = (_: {\n"
            required_params = parameters.get("required", [])
            tp_lines = get_parameter_typescript(
                parameters.get("properties"), required_params, 0
            )
            schema += "\n".join(tp_lines)
            schema += "\n}) => any;\n\n"
        else:
            # Doesn't have any parameters
            schema += " = () => any;\n\n"

    schema += f"}} // namespace {namespace}"

    return schema


def inject_system_messages_based_on_tools(
    messages: List[Dict], tools_or_functions: Optional[List[Dict]] = None
) -> List[Dict]:
    """This will be used to add Default system message, code-interpreter system message if needed

    Args:
        messages (List[Dict]): List of messages
        tools_or_functions (Optional[List[Dict]], optional): List of tools, functions. Defaults to None.

    Returns:
        List[Dict]: _description_
    """
    messages_clone = messages.copy()  # To avoid modifying the original list

    functions = []
    is_code_interpreter = False
    if tools_or_functions is not None:
        for item in tools_or_functions:
            if (
                "function" in item and item["function"] is not None
            ):  #  new data format: tools: [{"type": xx, "function": xxx}]
                functions.append(item["function"])
            elif "type" in item and item["type"] == "code_interpreter":
                is_code_interpreter = True
            else:
                functions.append(item)  #  old format

    messages_clone.insert(
        0, {"role": "system", "content": generate_schema_from_functions(functions)}
    )
    if is_code_interpreter:
        messages_clone.insert(1, {"role": "system", "content": PYTHON_RUN_SYS_MSG})
    else:
        messages_clone.insert(1, {"role": "system", "content": SYSTEM_MESSAGE})

    return messages_clone


tools = [  # For functionary-7b-v2 we use "tools"; for functionary-7b-v1.4 we use "functions" = [{"name": "get_current_weather", "description":..., "parameters": ....}]
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What is the weather for Istanbul?"}]


def generate_functionary_input(messages=messages, tools=tools):
    messages_clone = inject_system_messages_based_on_tools(messages, tools)

    full_text = ""
    for message in messages_clone:
        full_text += convert_message_to_prompt(message)
    return full_text.strip()


def fill_response_template(func_name: str, text: str) -> str:
    return f"\n<|from|>{func_name}\n<|recipient|>all\n<|content|>{text}"


if __name__ == "__main__":
    print(generate_functionary_input(messages, tools))
