import os
from typing import Any

from fastmcp import FastMCP, FastMCPApp
from httpx import get

from protocols.mcp.client import MCPClient
from tools.base import Tool

# MCP服务器环境变量映射表
# 用于自动检测常见MCP服务器需要的环境变量
MCP_SERVER_ENV_MAP = {
    "server-github": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
    "server-slack": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
    "server-google-drive": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"],
    "server-postgres": ["POSTGRES_CONNECTION_STRING"],
    "server-sqlite": [],  # 不需要环境变量
    "server-filesystem": [],  # 不需要环境变量
}


class MCPTool(Tool):
    """MCP (Model Context Protocol) 工具

    连接到 MCP 服务器并调用其提供的工具、资源和提示词。

    功能：
    - 列出服务器提供的工具
    - 调用服务器工具
    - 读取服务器资源
    - 获取提示词模板

    使用示例:
        >>> from hello_agents.tools.builtin import MCPTool
        >>>
        >>> # 方式1: 使用内置演示服务器
        >>> tool = MCPTool()  # 自动创建内置服务器
        >>> result = tool.run({"action": "list_tools"})
        >>>
        >>> # 方式2: 连接到外部 MCP 服务器
        >>> tool = MCPTool(server_command=["python", "examples/mcp_example.py"])
        >>> result = tool.run({"action": "list_tools"})
        >>>
        >>> # 方式3: 使用自定义 FastMCP 服务器
        >>> from fastmcp import FastMCP
        >>> server = FastMCP("MyServer")
        >>> tool = MCPTool(server=server)

    注意：使用 fastmcp 库，已包含在依赖中
    """

    def __init__(
        self,
        name: str = "mcp",
        description: str | None = None,
        server_command: list[str] | None = None,
        server_args: list[str] | None = None,
        server: FastMCP | None = None,
        auto_expand: bool = True,
        env: dict[str, str] | None = None,
        env_keys: list[str] | None = None,
    ):
        """
        初始化 MCP 工具

        Args:
            name: 工具名称（默认为"mcp"，建议为不同服务器指定不同名称）
            description: 工具描述（可选，默认为通用描述）
            server_command: 服务器启动命令（如 ["python", "server.py"]）
            server_args: 服务器参数列表
            server: FastMCP 服务器实例（可选，用于内存传输）
            auto_expand: 是否自动展开为独立工具（默认True）
            env: 环境变量字典（优先级最高，直接传递给MCP服务器）
            env_keys: 要从系统环境变量加载的key列表（优先级中等）

        环境变量优先级（从高到低）：
            1. 直接传递的env参数
            2. env_keys指定的环境变量
            3. 自动检测的环境变量（根据server_command）

        注意：如果所有参数都为空，将创建内置演示服务器

        示例：
            >>> # 方式1：直接传递环境变量（优先级最高）
            >>> github_tool = MCPTool(
            ...     name="github",
            ...     server_command=["npx", "-y", "@modelcontextprotocol/server-github"],
            ...     env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"}
            ... )
            >>>
            >>> # 方式2：从.env文件加载指定的环境变量
            >>> github_tool = MCPTool(
            ...     name="github",
            ...     server_command=["npx", "-y", "@modelcontextprotocol/server-github"],
            ...     env_keys=["GITHUB_PERSONAL_ACCESS_TOKEN"]
            ... )
            >>>
            >>> # 方式3：自动检测（最简单，推荐）
            >>> github_tool = MCPTool(
            ...     name="github",
            ...     server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
            ...     # 自动从环境变量加载GITHUB_PERSONAL_ACCESS_TOKEN
            ... )
        """

        self.server_command = server_command
        self.server_args = server_args or []
        self.server = server
        self._client = None
        self._available_tools = []
        self.auto_expand = auto_expand
        self.prefix = f"{name}_" if auto_expand else ""

        # 环境变量处理（优先级：env > env_keys > 自动检测）
        self.env = self._prepare_env(env=env, env_keys=env_keys, server_command=server_command)

        if not server_command and not server:
            self.server = self._create_builtin_server()

        self._discover_tools()

        # if description is None :

    def _prepare_env(
        self,
        env: dict[str, str] | None = None,
        env_keys: list[str] | None = None,
        server_command: list[str] | None = None,
    ) -> dict[str, str]:
        """
        准备环境变量

        优先级：env > env_keys > 自动检测

        Args:
            env: 直接传递的环境变量字典
            env_keys: 要从系统环境变量加载的key列表
            server_command: 服务器命令（用于自动检测）

        Returns:
            合并后的环境变量字典
        """

        result_env = {}

        if server_command:
            server_name = None

            for part in server_command:
                if "server-" in part:
                    # 提取类似 "@modelcontextprotocol/server-github" 中的 "server-github"
                    server_name = part.split("/")[-1] if "/" in part else part
                    break

            if server_name and server_name in MCP_SERVER_ENV_MAP:
                auto_keys = MCP_SERVER_ENV_MAP[server_name]

                for key in auto_keys:
                    value = os.getenv(key)
                    if value:
                        result_env[key] = value
                        print(f"🔑 自动加载环境变量: {key}")
        if env_keys:
            for key in env_keys:
                value = os.getenv(key)
                if value:
                    result_env[key] = value
                    print(f"🔑 从env_keys加载环境变量: {key}")
                else:
                    print(f"⚠️  警告: 环境变量 {key} 未设置")

        # 3. 直接传递的env（优先级最高）
        if env:
            result_env.update(env)
            for key in env.keys():
                print(f"🔑 使用直接传递的环境变量: {key}")

        return result_env

    def _create_builtin_server(self):
        """创建内置演示服务器"""
        try:
            from fastmcp import FastMCP

            server = FastMCP("LR-BuiltinServer")

            @server.tool()
            def add(a: float, b: float) -> float:
                return a + b

            @server.tool()
            def subtract(a: float, b: float) -> float:
                """减法计算器"""
                return a - b

            @server.tool()
            def get_system_info() -> dict[str, Any]:
                """获取系统信息"""
                import platform
                import sys

                return {
                    "platform": platform.system(),
                    "python_version": sys.version,
                    "server_name": "LR-BuiltinServer",
                    "tools_count": 2,
                }

            return server

        except ImportError:
            raise ImportError("创建内置 MCP 服务器需要 fastmcp 库。请安装: pip install fastmcp")

    def _discover_tools(self):
        """发现MCP服务器提供的所有工具"""
        import asyncio

        async def discover():
            client_source = self.server if self.server else self.server_command

            if client_source is None:
                return []
            async with MCPClient(client_source, self.server_args, env=self.env) as client:
                tools = await client.list_tools()
                return tools

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(discover())
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                self._available_tools = future.result()
        except RuntimeError:
            self._available_tools = asyncio.run(discover())
        except Exception:
            # 工具发现失败不影响初始化
            self._available_tools = []

    def _generate_description(self) -> str:
        """生成增强的工具描述"""

        if not self._available_tools:
            return "连接到 MCP 服务器，调用工具、读取资源和获取提示词。支持内置服务器和外部服务器。"

        if self.auto_expand:
            # 展开模式：简单描述
            return f"MCP工具服务器，包含{len(self._available_tools)}个工具。这些工具会自动展开为独立的工具供Agent使用。"
        else:
            # 非展开模式：详细描述
            desc_parts = [f"MCP工具服务器，提供{len(self._available_tools)}个工具："]

            for tool in self._available_tools:
                tool_name = tool.get("name", "unknown")
                tool_desc = tool.get("description", "无描述")

                desc_parts.append(f"  • {tool_name}: {tool_desc}")

            # 添加调用格式说明
            desc_parts.append("\n调用格式：返回JSON格式的参数")
            desc_parts.append('{"action": "call_tool", "tool_name": "工具名", "arguments": {...}}')

            if self._available_tools:
                first_tool = self._available_tools[0]

                tool_name = first_tool.get("name", "example")

                desc_parts.append(
                    f'\n示例：{{"action"  :"call_tool" , "tool_name" :{tool_name} , arguments: {{...}} }}'
                )
            return "\n".join(desc_parts)

    def get_expanded_tools(self) -> list[Tool]:
        """
        获取展开的工具列表

        将MCP服务器的每个工具包装成独立的Tool对象

        Returns:
            Tool对象列表
        """
