import asyncio
import json

from open_ai_provider import LLMResponse, OpenAICompatibleClient, Usage
from plan import Planner
from plan_solve import PlanAndSolveAgent
from prompt import COMMON_PROMPT
from reflection_agent import ReflectionAgent
from tools import available_tools


def assistant_tool_message(llm_output: LLMResponse) -> dict:
    """把本轮解析出的工具调用还原成 API 要求的 assistant 消息（后续必须接 tool 消息）。"""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": st.tools_args.id,
                "type": "function",
                "function": {"name": st.tools_args.name, "arguments": st.tools_args.args},
            }
            for st in llm_output.subtasks
        ],
    }


async def run_with_tools(
    llm: OpenAICompatibleClient,
    user_content: str,
    *,
    max_tool_rounds: int = 16,
) -> tuple[LLMResponse, str | None]:
    """
    单轮用户输入下的多轮工具循环：每次模型返回 tool_calls → 本地执行 → 把 assistant+tool
    拼回 messages 再请求，直到不再要工具或达到上限。
    """
    messages: list[dict] = [{"role": "user", "content": user_content}]
    last_llm: LLMResponse | None = None
    last_content: str | None = None
    observations_log: list[str] = []

    for _ in range(max_tool_rounds):
        last_llm, last_content = await llm.generate(messages, system_prompt=COMMON_PROMPT)
        if not last_llm.subtasks:
            text = (last_content or "").strip()
            if not text and observations_log:
                text = "（模型未输出正文，以下为本次已执行工具的结果汇总）\n" + "\n".join(observations_log)
            return last_llm, text or None
        assistant_msg = assistant_tool_message(last_llm)
        tool_msgs: list[dict] = []
        for subtask in last_llm.subtasks:
            tool_args = subtask.tools_args
            if available_tools.get(tool_args.name):
                json_args = json.loads(tool_args.args)
                observation = available_tools[tool_args.name](**json_args)
            else:
                observation = f"错误: 未定义的工具 '{tool_args.name}'"
            obs = str(observation)
            observations_log.append(obs)
            tool_msgs.append({"role": "tool", "content": obs, "tool_call_id": tool_args.id})
            print(f"工具调用结果: {observation}\n" + "=" * 40)

        messages = [*messages, assistant_msg, *tool_msgs]

    exhausted = last_llm or LLMResponse(
        subtasks=[], execution_strategy="sequential", usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )
    tail = (last_content or "").strip()
    if not tail and observations_log:
        tail = "（模型未输出正文，以下为本次已执行工具的结果汇总）\n" + "\n".join(observations_log)
    return exhausted, tail or None


async def main_async():
    llm = OpenAICompatibleClient(mode="qwen3.5:0.8b", url="192.168.3.6:11434")

    user_input = "请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"

    prompt_history = [f"用户输入: {user_input}"]

    full_prompt = "\n".join(prompt_history)

    print(f"当前完整提示:\n{full_prompt}\n" + "=" * 40)
    llm_output, assistant_text = await run_with_tools(llm, full_prompt)

    print(f"模型输出 {llm_output}\n" + "=" * 40)
    if assistant_text:
        print(f"最终回复:\n{assistant_text}\n" + "=" * 40)
    elif not llm_output.subtasks:
        print("最终回复: （无）\n" + "=" * 40)

    if llm_output.subtasks:
        print("达到工具轮次上限或仍返回工具调用，请检查 max_tool_rounds 或模型行为")
    else:
        print("任务已经完成")
        if assistant_text:
            prompt_history.append(f"助手: {assistant_text}")


async def test_run_with_tools():
    llm = OpenAICompatibleClient(mode="qwen3.5:0.8b", url="127.0.0.1:11434")
    # plan = Planner(llm)

    # questoin = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"

    # resp = await plan.plan(
    #     "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    # )

    # resp = await PlanAndSolveAgent(llm).run(questoin)

    # print("规划结果:", resp)

    question = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"

    final_code = await ReflectionAgent(llm).run(question)

    print("最终代码:\n", final_code)


if __name__ == "__main__":
    # asyncio.run(main_async())

    asyncio.run(test_run_with_tools())
