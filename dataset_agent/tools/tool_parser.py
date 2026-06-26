import re
import uuid
import json
from langchain_core.messages import ToolCall

def tool_call_parser(tool_call: str) -> list:
    tool_calls = []
    func_call_matches = re.findall(r'\{\s*"name"[\s\S]*?\}\s*\}', tool_call)
    # Match -> possible tool call
    for match in func_call_matches:
        if match:
            try:
                tool_data = json.loads(match)
            except:
                pass
        else:
            pass
        # If the data was correclty organized from the JSON into the dictionary, we have a valid tool call
        try:
            if tool_data["name"] and tool_data["arguments"]:
                url = ""
                if tool_data["arguments"]["url"]:
                    url = tool_data["arguments"]["url"]
                print(f"{tool_data['name']} {url}")
                actual_call = ToolCall(
                        name=tool_data["name"],
                        args=tool_data["arguments"],
                        id=str(uuid.uuid4())
                )
                tool_calls.append(actual_call)
            else:#
                pass
        except:
            pass
    return tool_calls