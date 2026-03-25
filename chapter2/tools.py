import os
from operator import call
from typing import Any

import requests
from attr import dataclass


def search(input: str) -> str:
    try:
        api_key = "sk-188e3e867d124229ae9f9c74220e348c"
        if not api_key:
            raise ValueError("未找到 API 密钥，请确保在环境变量中设置了 BOCHAI_API_KEY")

        params = {"query": input, "freshness": "oneYear", "summary": True, "count": 9}
        header = dict({"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})

        resp = requests.post("https://api.bochaai.com/v1/web-search", data=params, headers=header)

        resp.raise_for_status()
        data = resp.json()

        print("搜索结果:", data)

    except requests.RequestException as e:
        return f"搜索请求失败: {e}"


# {
#     "type": "function",
#     "strict": True,
#     "function": {
#         "name": "get_weather",
#         "description": "查询指定城市的实时天气",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {
#                     "type": "string",
#                     "description": "城市名称。例如 '北京', 'Shanghai', 'New York'。如果用户提到‘首都’或‘魔都’，请转换为具体城市名。",
#                 }
#             },
#             "required": ["city"],
#         },
#     },
# }


@dataclass
class FunctionArgs:
    arg_name: str
    arg_type: str
    is_required: bool


@dataclass
class ToolFunctionInfo:
    name: str
    description: str

    function: callable

    args: list[FunctionArgs]


class ToolExecutor:
    def __init__(self):
        self.tools: dict[str, dict[str, any]] = {}

    def register_tool(self, name: str, desp: str, func: callable):
        if name in self.tools:
            print(f"工具 {name} 已存在，覆盖原有工具。")
        self.tools[name] = {"description": desp, "function": func}
        print(f"工具 {name} 注册成功。")

    def get_tool(self, name: str) -> callable:
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"工具 {name} 未找到。")
        return tool["function"]

    def getAvailableTools(self) -> list[str]:
        return [name for name in self.tools.keys()]


if __name__ == "__main__":
    user_input = "请推荐一些适合夏天旅游的城市"
    search(user_input)
