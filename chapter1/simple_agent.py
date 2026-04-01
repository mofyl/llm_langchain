import re

from tools.registry import ToolRegistry

from .agent.base import Agent
from .config import Config
from .message import Message
from .open_ai_provider import OpenAICompatibleClient


class SimpleAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: OpenAICompatibleClient,
        system_prompt: str | None,
        config: Config | None = None,
        tool_registry: ToolRegistry | None = None,
        enable_tool_calling: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)

        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    def _getenhanced_system_prompt(self) -> str:

        base_prompt = self.system_prompt or "你是一个AI助手"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()

        if not tools_description or tools_description == "暂无工具可用":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"

        tools_section += "### 参数格式说明\n"
        tools_section += "1. **多个参数**：使用 `key=value` 格式，用逗号分隔\n"
        tools_section += "   示例：`[TOOL_CALL:calculator_multiply:a=12,b=8]`\n"
        tools_section += "   示例：`[TOOL_CALL:filesystem_read_file:path=README.md]`\n\n"
        tools_section += "2. **单个参数**：直接使用 `key=value`\n"
        tools_section += "   示例：`[TOOL_CALL:search:query=Python编程]`\n\n"
        tools_section += "3. **简单查询**：可以直接传入文本\n"
        tools_section += "   示例：`[TOOL_CALL:search:Python编程]`\n\n"

        tools_section += "### 重要提示\n"
        tools_section += "- 参数名必须与工具定义的参数名完全匹配\n"
        tools_section += '- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a="12"`\n'
        tools_section += "- 文件路径等字符串参数直接写：`path=README.md`\n"
        tools_section += "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答\n"

        return base_prompt + tools_section

    async def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        message = []

        enhanced_system_prompt = self._getenhanced_system_prompt()

        message.append({"role": "sysmte", "content": enhanced_system_prompt})

        for msg in self._history:
            message.append({"role": msg.role, "content": msg.content})

        message.append({"role": "user", "content": input_text})

        if not self.enable_tool_calling:
            _, resp = await self.llm.generate(messages=message, system_prompt=enhanced_system_prompt, kwargs=kwargs)
            self.add_message(message=Message(input_text, "user"))
            self.add_message(message=Message(resp, "assistant"))
            return resp
            # 迭代处理，支持多轮工具调用
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            _, resp = await self.llm.generate(messages=message, system_prompt=enhanced_system_prompt, **kwargs)

            tool_calls = self._parse_tool_calls(resp)

            if tool_calls:
                tool_result = []

                clean_response = resp

                for call in tool_calls:
                    result = self._execute_tool_call(call["tool_name"], call["parameters"])
                    tool_result.append(result)

                    clean_response = clean_response.replace(call["original"], "")

                message.append({"role": "assistant", "content": clean_response})

                tool_result_text = "\n\n".join(tool_result)

                message.append(
                    {
                        "role": "user",
                        "content": f"工具执行结果为 \n{tool_result_text}\n\n请基于这些结果给出完整的回答。",
                    }
                )
                current_iteration += 1
                continue
            final_response = resp
            break

        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.generate(messages=message, system_prompt=enhanced_system_prompt, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用"""
        pattern = r"\[TOOL_CALL:([^:]+):([^\]]+)\]"
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append(
                {
                    "tool_name": tool_name.strip(),
                    "parameters": parameters.strip(),
                    "original": f"[TOOL_CALL:{tool_name}:{parameters}]",
                }
            )

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:

        if not self.tool_registry:
            return "未配置工具注册表"

        try:
            tool = self.tool_registry.get_tool(tool_name)

            if not tool:
                return f"未能找到工具{tool_name}"

            param_dict = self._parse_tool_parameters()
            result = tool.run(param_dict)

            return f"工具 {tool_name} 执行结果为：\n{result}"
        except Exception:
            return f"工具调用失败 {tool_name}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        import json

        param_dict = {}

        # 尝试解析JSON格式
        if parameters.strip().startswith("{"):
            try:
                param_dict = json.loads(parameters)
                # JSON解析成功，进行类型转换
                param_dict = self._convert_parameter_types(tool_name, param_dict)
                return param_dict
            except json.JSONDecodeError:
                # JSON解析失败，继续使用其他方式
                pass

        if "=" in parameters:
            # 格式: key=value 或 action=search,query=Python
            if "," in parameters:
                # 多个参数：action=search,query=Python,limit=3
                pairs = parameters.split(",")
                for pair in pairs:
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数：key=value
                key, value = parameters.split("=", 1)
                param_dict[key.strip()] = value.strip()

            # 类型转换
            param_dict = self._convert_parameter_types(tool_name, param_dict)

            # 智能推断action（如果没有指定）
            if "action" not in param_dict:
                param_dict = self._infer_action(tool_name, param_dict)
        else:
            # 直接传入参数，根据工具类型智能推断
            param_dict = self._infer_simple_parameters(tool_name, parameters)

        return param_dict

    def _convert_parameter_types(self, tool_name: str, param_dict: dict) -> dict:
        """
        根据工具的参数定义转换参数类型

        Args:
            tool_name: 工具名称
            param_dict: 参数字典

        Returns:
            类型转换后的参数字典
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        # 获取工具的参数定义
        try:
            tool_params = tool.get_parameters()
        except:
            return param_dict

        # 创建参数类型映射
        param_types = {}
        for param in tool_params:
            param_types[param.name] = param.type

        # 转换参数类型
        converted_dict = {}
        for key, value in param_dict.items():
            if key in param_types:
                param_type = param_types[key]
                try:
                    if param_type == "number" or param_type == "integer":
                        # 转换为数字
                        if isinstance(value, str):
                            converted_dict[key] = float(value) if param_type == "number" else int(value)
                        else:
                            converted_dict[key] = value
                    elif param_type == "boolean":
                        # 转换为布尔值
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ("true", "1", "yes")
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = value
                except (ValueError, TypeError):
                    # 转换失败，保持原值
                    converted_dict[key] = value
            else:
                converted_dict[key] = value

        return converted_dict
