import asyncio
import sys
from pathlib import Path

try:
    from chapter1.open_ai_provider import OpenAICompatibleClient
    from chapter1.simple_agent import SimpleAgent
    from memory.base import MemoryType
    from tools.memory_tool import MemoryTool
    from tools.registry import ToolRegistry
except ModuleNotFoundError:
    # 允许在 examples 目录直接执行: python3 memory_rag.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from chapter1.open_ai_provider import OpenAICompatibleClient
    from chapter1.simple_agent import SimpleAgent
    from memory.base import MemoryType
    from tools.memory_tool import MemoryTool
    from tools.registry import ToolRegistry


async def demo_simple_agent_with_memory():
    llm = OpenAICompatibleClient(mode="qwen3.5:0.8b")

    memory_tool = MemoryTool(user_id="demo_user_001", memory_type=[MemoryType.WORKINGMEMORY])

    tool_registry = ToolRegistry()
    tool_registry.register_tool(memory_tool)

    agent = SimpleAgent(
        name="记忆助手",
        llm=llm,
        tool_registry=tool_registry,
        system_prompt="""你是一个有记忆能力的AI助手。你能记住我们的对话历史和重要信息。

工具使用指南：
- 当用户提供个人信息时，使用 [TOOL_CALL:memory:store=信息内容] 存储
- 当需要回忆用户信息时，使用 [TOOL_CALL:memory:recall=查询关键词] 检索
- 当用户询问历史对话时，使用 [TOOL_CALL:memory:action=summary] 获取摘要

重要原则：
- 主动记录用户的重要信息（姓名、职业、兴趣等）
- 在回答时参考相关的历史记忆
- 提供个性化的建议和服务""",
    )

    # 模拟多轮对话
    conversations = [
        "你好！我叫李明，是一名软件工程师，专门做Python开发",
        "我最近在学习机器学习，特别对深度学习感兴趣",
        "你能推荐一些Python机器学习的库吗？",
        "你还记得我的名字和职业吗？请结合我的背景给我一些学习建议",
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n--- 对话轮次 {i} ---")
        print(f"👤 用户: {user_input}")

        # SimpleAgent会自动使用memory工具
        response = await agent.run(user_input)
        print(f"🤖 助手: {response}")

    # 显示记忆摘要
    print("\n📊 最终记忆系统状态:")
    summary = memory_tool.run({"action": "summary"})
    print(summary)

    return memory_tool


def main():
    asyncio.run(demo_simple_agent_with_memory())


# if __name__ == "__main__":
#     # main()

#     from django.core.management.utils import get_random_secret_key

#     print(get_random_secret_key())
