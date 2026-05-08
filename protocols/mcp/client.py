"""
增强的 MCP 客户端实现

支持多种传输方式的 MCP 客户端，用于教学和实际应用。
这个实现展示了如何使用不同的传输方式连接到 MCP 服务器。

支持的传输方式：
1. Memory: 内存传输（用于测试，直接传递 FastMCP 实例）
2. Stdio: 标准输入输出传输（本地进程，Python/Node.js 脚本）
3. HTTP: HTTP 传输（远程服务器）
4. SSE: Server-Sent Events 传输（实时通信）

使用示例：
```python
# 1. 内存传输（测试）
from fastmcp import FastMCP
server = FastMCP("TestServer")
client = MCPClient(server)

# 2. Stdio 传输（本地脚本）
client = MCPClient("server.py")
client = MCPClient(["python", "server.py"])

# 3. HTTP 传输（远程服务器）
client = MCPClient("https://api.example.com/mcp")

# 4. SSE 传输（实时通信）
client = MCPClient("https://api.example.com/mcp", transport_type="sse")

# 5. 配置传输（高级用法）
config = {
    "transport": "stdio",
    "command": "python",
    "args": ["server.py"],
    "env": {"DEBUG": "1"}
}
client = MCPClient(config)
```
"""

from calendar import c
from os import sync
from typing import Any

from fastmcp import Client, FastMCP
from fastmcp.client.transports import (
    ClientTransport,
    PythonStdioTransport,
    SSETransport,
    StdioTransport,
    StreamableHttpTransport,
)

import mcp

PreparedServerSource = ClientTransport | FastMCP | dict[str, Any] | str


class MCPClient:
    def __init__(
        self,
        server_source: str | list[str] | FastMCP | dict[str, Any],
        server_args: list[str] | None = None,
        transport_type: str | None = None,
        env: dict[str, str] | None = None,
        **transport_kwargs,
    ):
        """
        初始化MCP 客户端

        Args:
            server_source: 服务器源，支持多种格式：
                - FastMCP 实例: 内存传输（用于测试）
                - 字符串路径: Python 脚本路径（如 "server.py"）
                - HTTP URL: 远程服务器（如 "https://api.example.com/mcp"）
                - 命令列表: 完整命令（如 ["python", "server.py"]）
                - 配置字典: 传输配置
            server_args: 服务器参数列表（可选）
            transport_type: 强制指定传输类型 ("stdio", "http", "sse", "memory")
            env: 环境变量字典（传递给MCP服务器进程）
            **transport_kwargs: 传输特定的额外参数

        Raises:
            ImportError: 如果 fastmcp 库未安装
        """

        self.server_args = server_args or []
        self.transport_type = transport_type
        self.env = env or {}
        self.transport_kwargs = transport_kwargs
        self.server_source: PreparedServerSource = self._prepare_server_source(server_source)
        self.client: Client | None = None
        self._context_manager = None

    def _prepare_server_source(self, server_source: str | list[str] | FastMCP | dict[str, Any]) -> PreparedServerSource:
        """准备服务器源，根据类型创建合适的传输配置"""

        if isinstance(server_source, FastMCP):
            print(f"🧠 使用内存传输: {server_source.name}")
            return server_source

        if isinstance(server_source, list):
            if not server_source:
                raise ValueError("server_source 命令列表不能为空")

            command, *command_args = server_source
            print(f"⚙️ 使用 Stdio 传输 (Command): {server_source}")
            return StdioTransport(
                command=command,
                args=[*command_args, *self.server_args],
                env=self.env if self.env else None,
                **self.transport_kwargs,
            )

        if isinstance(server_source, str) and (
            server_source.startswith("http://") or server_source.startswith("https://")
        ):
            transport_type = self.transport_type or "http"
            print(f"🌐 使用 {transport_type.upper()} 传输: {server_source}")
            if transport_type == "sse":
                return SSETransport(url=server_source, **self.transport_kwargs)
            else:
                return StreamableHttpTransport(url=server_source, **self.transport_kwargs)

        if isinstance(server_source, str) and server_source.endswith(".py"):
            print(f"🐍 使用 Stdio 传输 (Python): {server_source}")
            return PythonStdioTransport(
                script_path=server_source,
                args=self.server_args,
                env=self.env if self.env else None,
                **self.transport_kwargs,
            )

        # 6. 其他情况 - 直接返回，让 FastMCP 自动推断
        print(f"🔍 自动推断传输: {server_source}")
        return server_source

    # def _create_transport_from_config(self, config : dict[str, Any])

    async def __aenter__(self):
        """异步上下文管理器入口"""
        print("🔗 连接到 MCP 服务器...")
        self.client = Client(self.server_source)
        self._context_manager = self.client
        await self._context_manager.__aenter__()
        print("✅ 连接成功！")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """异步上下文管理器出口"""
        if self._context_manager:
            await self._context_manager.__aexit__(exc_type, exc, tb)
        self._context_manager = None
        self.client = None
        print("🔌 连接已断开")

    async def list_tools(self) -> list[dict[str, Any]]:
        """列出所有可用的工具"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_tools()

        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }
            for tool in result
        ]

    async def call_tools(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用 MCP 工具"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.call_tool(tool_name, arguments=arguments)

        def get_data(content: mcp.types.ContentBlock):
            if isinstance(content, mcp.types.TextContent):
                return content.text
            if isinstance(content, mcp.types.ImageContent) or isinstance(content, mcp.types.AudioContent):
                return content.data
            return ""

        if result.content:
            if len(result.content) == 1:
                content = result.content[0]
                return get_data(content)
            return [get_data(c) for c in result.content]
        return None

    async def list_resource(self) -> list[dict[str, Any]]:
        """列出所有可用的资源"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_resources()

        return [
            {
                "uri": resource.uri,
                "name": resource.name or "",
                "description": resource.description or "",
                "mime_type": resource.mimeType,
            }
            for resource in result
        ]

    async def read_resource(self, uri: str) -> Any:
        """读取资源内容"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.read_resource(uri=uri)

        # BlobResourceContents
        def get_content(content: mcp.types.TextResourceContents | mcp.types.BlobResourceContents):
            if isinstance(content, mcp.types.TextResourceContents):
                return content.text
            if isinstance(content, mcp.types.BlobResourceContents):
                return content.blob
            return ""

        if len(result) == 1:
            return get_content(result[0])

        return [get_content(c) for c in result]

    async def list_prompts(self) -> list[dict[str, Any]]:
        """列出所有可用的提示词模板"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_prompts()

        return [
            {"name": prompt.name, "description": prompt.description or "", "arguments": prompt.arguments}
            for prompt in result
        ]

    async def get_prompt(self, prompt_name: str, arguments: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """获取提示词内容"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.get_prompt(name=prompt_name, arguments=arguments or {})

        if isinstance(result, mcp.types.GetPromptRequest):
            return [{"role": msg.role, "content": msg.content} for msg in result.messages]

        return []

    async def ping(self) -> bool:
        """测试服务器连接"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        try:
            await self.client.ping()
            return True
        except Exception:
            return False
