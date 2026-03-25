import ast
import re
from typing import Any

from main import OpenAICompatibleClient

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""


class Planner:
    def __init__(self, llm_client: OpenAICompatibleClient):
        self.llm_client = llm_client

    async def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        _, raw_content = await self.llm_client.generate(messages)
        if not raw_content:
            return []

        steps = self._parse_steps(raw_content)
        print("LLM 规划输出:", steps)
        return steps

    @staticmethod
    def _parse_steps(raw_content: str) -> list[str]:
        # 优先提取 markdown 代码块中的 Python 列表
        code_block_match = re.search(r"```python\s*(.*?)\s*```", raw_content, re.DOTALL)
        candidate = code_block_match.group(1) if code_block_match else raw_content

        try:
            parsed = ast.literal_eval(candidate.strip())
        except (ValueError, SyntaxError):
            return []

        if not isinstance(parsed, list):
            return []

        return [str(item) for item in parsed if str(item).strip()]


class FakeLLMClient:
    async def generate(self, messages: list[dict[str, Any]]) -> tuple[Any, str]:
        _ = messages
        return (
            None,
            """```python
["查询北京天气", "根据天气筛选景点", "给出推荐理由"]
```""",
        )


async def test_planner_with_mock() -> None:
    planner = Planner(FakeLLMClient())
    question = "请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    plan = await planner.plan(question)

    assert plan == ["查询北京天气", "根据天气筛选景点", "给出推荐理由"]
    print("test_planner_with_mock 通过")


if __name__ == "__main__":
    import asyncio

    async def main():
        await test_planner_with_mock()

    asyncio.run(main())
