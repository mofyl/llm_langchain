from importlib import import_module

from .base import AutoGenTool, Tool, ToolParameter, tool_action
from .memory_tool import MemoryConfig, MemoryManager, MemoryTool, MemoryType
from .registry import ToolRegistry
from .terminal_tool import TerminalTool


def __getattr__(name: str):
    if name in {"Tool", "ToolParameter", "tool_action", "AutoGenTool"}:
        module = import_module(".base", __name__)
        return getattr(module, name)
    if name == "ToolRegistry":
        return import_module(".registry", __name__).ToolRegistry
    if name == "MemoryTool":
        return import_module(".memory_tool", __name__).MemoryTool
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Tool",
    "ToolParameter",
    "tool_action",
    "AutoGenTool",
    "ToolRegistry",
    "MemoryTool",
    "MemoryManager",
    "MemoryConfig",
    "MemoryType",
    "TerminalTool",
]
