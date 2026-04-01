from typing import Any

from .base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True):
        if auto_expand and hasattr(tool, "expandable") and tool.expandable:
            expanded_tools = tool.get_expanded_tools()

            if expanded_tools:
                for sub_tool in expanded_tools:
                    if sub_tool in self._tools:
                        print(f"⚠️ 警告：工具 '{sub_tool.name}' 已存在，将被覆盖。")
                    self._tools[sub_tool.name] = sub_tool
                print(f"✅ 工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return
        if tool.name in self._tools:
            print(f"⚠️ 警告：工具 '{tool.name}' 已存在，将被覆盖。")

        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def get_tool(self, name: str) -> Tool | None:
        return self._tools[name]

    def execute_tool(self, name: str, input_text: str) -> str:
        if name in self._tools:
            tool = self._tools[name]

            try:
                return tool.run({input: input_text})
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"

        elif name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"
        else:
            return f"错误：未找到名为 '{name}' 的工具。"

    def get_tools_description(self) -> str:

        desc = []

        for tool in self._tools.values():
            desc.append(f"- {tool.name}: {tool.desc}")

        for name, info in self._functions.items():
            desc.append(f"- {name}: {info['description']}")

        return "\n".join(desc) if desc else "暂无可用工具"
