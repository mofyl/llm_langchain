from memory import Memory, RecordType
from open_ai_provider import OpenAICompatibleClient

INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""


REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class ReflectionAgent:
    def __init__(self, llm_client: OpenAICompatibleClient, max_iterations: int = 5):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.memory = Memory()

    async def run(self, task: str) -> str:

        # 初次尝试
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)

        initial_code = await self._get_llm_response(initial_prompt)
        self.memory.add_record(RecordType.execution, initial_code)

        # 循环迭代，进行反思和优化
        for i in range(self.max_iterations):
            print(f"\n----- 第{i + 1}/{self.max_iterations}轮----")

            last_code = self.memory.get_last_execution()
            # 第一次反馈
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = await self._get_llm_response(reflect_prompt)
            self.memory.add_record(RecordType.reflection, feedback)

            if "无需改进" in feedback:
                print("评审员认为代码已达到最优，无需进一步优化。")
                break

            print("\n-> 正在进行优化")
            # 根据反馈生成新版本代码
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(task=task, last_code_attempt=last_code, feedback=feedback)
            refined_code = await self._get_llm_response(refine_prompt)
            self.memory.add_record(RecordType.execution, refined_code)

        final_code = self.memory.get_last_execution()
        return final_code or "未能生成代码。"

    async def _get_llm_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        _, response = await self.llm_client.generate(messages, system_prompt=prompt)
        return response or ""
