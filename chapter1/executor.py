from open_ai_provider import OpenAICompatibleClient

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请结合上下文仅输出针对“当前步骤”的回答:
"""


class Executor:
    def __init__(self, llm_client: OpenAICompatibleClient):
        self.llm_client = llm_client

    async def execute(self, question: str, plan: list[str]) -> str:
        history = ""
        print(f"开始执行计划... , 共{len(plan)}步")

        for i, step in enumerate(plan):
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history,
                current_step=step,
            )
            messages = [{"role": "user", "content": prompt}]
            _, raw = await self.llm_client.generate(messages, system_prompt=EXECUTOR_PROMPT_TEMPLATE)

            history += f"步骤 {i + 1}: {step}\n结果: {raw}\n\n"

            print(f"步骤 {i + 1}: {step}\n结果: {raw}\n" + "=" * 40)

        # 循环结束后，最后一步的响应就是最终答案
        final_answer = raw
        return final_answer
