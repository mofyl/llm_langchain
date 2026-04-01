import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel


def tool_action(name: str = None, description: str = None):
    """装饰器：标记一个方法为可展开的工具 action

    用法:
        @tool_action("memory_add", "添加新记忆")
        def _add_memory(self, content: str, importance: float = 0.5) -> str:
            '''添加记忆

            Args:
                content: 记忆内容
                importance: 重要性分数
            '''
            ...

    Args:
        name: 工具名称（如果不提供，从方法名自动生成）
        description: 工具描述（如果不提供，从 docstring 提取）
    """

    def decorator(func: Callable):
        func._is_tool_action = True
        func._tool_name = name
        func._tool_description = description
        return func

    return decorator


class ToolParameter(BaseModel):
    """工具参数定义"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """工具基类

    支持两种使用模式：
    1. 普通模式：工具作为单一实体使用
    2. 可展开模式：工具可以展开为多个独立的子工具（每个子工具对应一个功能）

    展开模式支持两种实现方式：
    - 手动定义子工具类（传统方式）
    - 使用 @tool_action 装饰器自动生成（推荐）
    """

    def __init__(self, name: str, desc: str, expandable: bool = False):
        self.name = name
        self.desc = desc
        self.expandable = expandable

    @abstractmethod
    def run(self, param: dict[str, Any]) -> str:
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        pass

    def get_expanded_tools(self) -> list["Tool"] | None:
        """获取展开后的子工具列表

        默认实现：自动从标记了 @tool_action 的方法生成子工具
        子类可以重写此方法提供自定义的展开逻辑

        Returns:
            如果工具支持展开，返回子工具列表；否则返回 None
        """

        if not self.expandable:
            return None

        tools = []

        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_tool_action"):
                tool = AutoGenTool(parent=self, method=method, name=method._tool_name, desc=method._tool_description)
                tools.append(tool)

        return tools if tools else None

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)


class AutoGenTool(Tool):
    """自动生成的工具 - 从方法签名和 docstring 自动提取参数"""

    def __init__(self, parent: Tool, method: Callable, name: str = None, desc: str = None):
        """初始化自动生成的工具

        Args:
            parent: 父工具实例
            method: 被装饰的方法
            name: 工具名称（如果为 None，从方法名生成）
            description: 工具描述（如果为 None，从 docstring 提取）
        """

        self.parent = parent
        self.method = method

        if name is None:
            method_name = method.__name__.lstrip("_")
            name = f"{parent.name}_{method_name}"

        if desc is None:
            desc = self._extract_description_from_docstring()

        super().__init__(name=name, desc=desc)

        self._parameters = self._parse_parameters()

    def _extract_description_from_docstring(self) -> str:
        doc = inspect.getdoc(self.method)
        if not doc:
            return f"执行 {self.method.__name__}"
        lines = doc.split("\n")

        for line in lines:
            line = line.strip()

            if line and not line.startswith("Args:") and not line.startswith("Returns:"):
                return line
        return f"执行 {self.method.__name__}"

    def _parse_parameters(self) -> list[ToolParameter]:
        """从方法签名和 docstring 自动提取参数"""
        sig = inspect.signature(self.method)
        type_hints = get_type_hints(self.method)
        docstring = inspect.getdoc(self.method) or ""

        # 从 docstring 解析参数描述
        param_descriptions = self._parse_param_descriptions(docstring)

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # 获取类型
            param_type_hint = type_hints.get(param_name, str)
            param_type = self._python_type_to_tool_type(param_type_hint)

            # 判断是否必需
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            # 获取描述
            description = param_descriptions.get(param_name, f"参数 {param_name}")

            parameters.append(
                ToolParameter(
                    name=param_name, type=param_type, description=description, required=required, default=default
                )
            )

        return parameters

    def _parse_param_descriptions(self, docstring: str) -> dict[str, str]:
        """从 docstring 解析参数描述

        支持格式:
            Args:
                param_name: 参数描述
                another_param: 另一个参数描述
        """
        descriptions = {}

        # 查找 Args: 部分
        args_match = re.search(r"Args:\s*\n(.*?)(?:\n\s*\n|Returns:|$)", docstring, re.DOTALL)
        if not args_match:
            return descriptions

        args_section = args_match.group(1)

        # 解析每个参数
        # 匹配格式: param_name: 描述 或 param_name (type): 描述
        param_pattern = r"^\s*(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=^\s*\w+\s*(?:\([^)]+\))?\s*:|$)"
        matches = re.finditer(param_pattern, args_section, re.MULTILINE | re.DOTALL)

        for match in matches:
            param_name = match.group(1).strip()
            param_desc = match.group(2).strip()
            # 清理描述中的多余空白
            param_desc = re.sub(r"\s+", " ", param_desc)
            descriptions[param_name] = param_desc

        return descriptions

    def _python_type_to_tool_type(self, py_type) -> str:
        """将 Python 类型转换为工具类型字符串"""
        # 处理泛型类型
        origin = getattr(py_type, "__origin__", None)
        if origin is not None:
            if origin is list:
                return "array"
            elif origin is dict:
                return "object"

        # 处理基本类型
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        return type_map.get(py_type, "string")
