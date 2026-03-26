import ast
import re
import sys
from os import system
from typing import Any

from open_ai_provider import OpenAICompatibleClient

PLANNER_PROMPT_TEMPLATE = """

# 角色
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。


# 要求
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
必须说明每个步骤的具体内容和目的，确保它们共同构成一个完整的解决方案。

一定要将步骤输出为一个 Python 列表，列表中的每个元素都是一个描述子任务的字符串。请勿输出任何除该列表以外的内容。
并且一定要有输出，禁止空输出。

# 输出格式
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。只需要一个列表，禁止输出任何额外的文本、解释或格式。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1:内容+目的", "步骤2:内容+目的", "步骤3:内容+目的", ...]
```
"""


class Planner:
    def __init__(self, llm_client: OpenAICompatibleClient):
        self.llm_client = llm_client

    async def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        _, raw_content = await self.llm_client.generate(messages, system_prompt=PLANNER_PROMPT_TEMPLATE)
        if not raw_content:
            return []

        steps = self._parse_steps(raw_content)
        print("LLM 规划输出:", steps)

        try:
            plan_str = raw_content.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)

            return plan if isinstance(plan, list) else []
        except (IndexError, ValueError, SyntaxError) as e:
            print(f"解析规划输出失败: {e}")
            print("原始输出:", raw_content)
            return []
        except Exception as e:
            print(f"未知错误: {e}")
            return []

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
