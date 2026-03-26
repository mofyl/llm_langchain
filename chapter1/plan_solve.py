from executor import Executor
from plan import Planner


class PlanAndSolveAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    async def run(self, question: str) -> str:

        plan = await self.planner.plan(question)
        print("生成的计划:", plan)

        if not plan:
            print("未能生成有效的计划，直接返回原始问题的答案。")
            return

        finnal_answer = await self.executor.execute(question, plan)

        print("最终答案:", finnal_answer)
        return finnal_answer
