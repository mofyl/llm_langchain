import asyncio
import json
import re
import time

from openai import AsyncOpenAI

from prompt import COMMON_PROMPT, LLMResponse, Usage, parse_llm_response
from tools import available_tools, tools


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


class OpenAICompatibleClient:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def __init__(self, mode: str, url: str):
        self.mode = mode
        self.url = url
        self.client = AsyncOpenAI(api_key="ollama", timeout=60, base_url=f"http://{url}/v1")

    async def generate(self, messages: list[dict]) -> tuple[LLMResponse, str | None]:
        date_prefix = f"The current date is {self.current_time}\n"
        system_prompt = date_prefix + COMMON_PROMPT
        # 系统提示只放开头一轮，避免多轮工具调用重复堆叠
        payload = [{"role": "system", "content": system_prompt}, *messages]

        # 与原生 tools 同时要求 json_object 时，部分后端（如 Ollama）末轮可能产生 token 但 content 为空
        api_request = {
            "model": self.mode,
            "messages": payload,
            "max_tokens": 1000,
            "tools": tools,
            "tool_choice": "auto",
        }

        try:
            response = await self.client.chat.completions.create(**api_request)
            llm_response = parse_llm_response(response)
            msg = response.choices[0].message
            raw = msg.content
            # if raw is None and not llm_response.subtasks:
            #     print("raw is None and not llm_response.subtasks ", response)
            #     # 少数实现把末轮正文放在其它字段，尽量兜底
            #     extra = getattr(msg, "reasoning_content", None) or getattr(msg, "refusal", None)
            #     if isinstance(extra, str) and extra.strip():
            #         raw = extra
            return llm_response, raw
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return (
                LLMResponse(
                    subtasks=[],
                    execution_strategy="sequential",
                    usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                ),
                None,
            )


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
        last_llm, last_content = await llm.generate(messages)
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

        # match = re.search(
        #     r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
        #     llm_output,
        #     re.DOTALL,
        # )
        # if match:
        #     truncated = match.group(1).strip()
        #     if truncated != llm_output.strip():
        #         llm_output = truncated
        #         print("已截断多余的 Thought-Action 对")
        # print(f"模型输出:\n{llm_output}\n")
        # prompt_history.append(llm_output)

        # # 3.3. 解析并执行行动
        # action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        # if not action_match:
        #     observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        #     observation_str = f"Observation: {observation}"
        #     print(f"{observation_str}\n" + "="*40)
        #     prompt_history.append(observation_str)
        #     continue
        # action_str = action_match.group(1).strip()

        # if action_str.startswith("Finish"):
        #     final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        #     print(f"任务完成，最终答案: {final_answer}")
        #     break

        # tool_name = re.search(r"(\w+)\(", action_str).group(1)
        # args_str = re.search(r"\((.*)\)", action_str).group(1)
        # kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        # if tool_name in available_tools:
        #     observation = available_tools[tool_name](**kwargs)
        # else:
        #     observation = f"错误:未定义的工具 '{tool_name}'"

        # 3.4. 记录观察结果
        # observation_str = f"Observation: {observation}"
        # print(f"{observation_str}\n" + "=" * 40)
        # prompt_history.append(observation_str)


if __name__ == "__main__":
    asyncio.run(main_async())
