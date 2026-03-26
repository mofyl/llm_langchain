import json
import time

from attr import dataclass
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall
from tools import tools


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ToolCall:
    id: str
    name: str
    args: str


@dataclass
class SubTask:
    description: str  # 对子任务的简要描述
    # 工具名与参数（JSON 字符串），与 OpenAI function.arguments 对齐
    tools_args: ToolCall


@dataclass
class LLMResponse:
    subtasks: list[SubTask]  # LLM生成的子任务列表
    execution_strategy: str  # 执行策略，表示如何执行subtask列表中的任务，sequential ， parallel.
    usage: Usage


def parse_llm_response(llm_output: ChatCompletion) -> LLMResponse:

    # 解析LLM输出，提取subtasks和usage信息
    # 这里需要根据实际的LLM输出格式进行解析，以下是一个示例解析逻辑
    subtasks = []
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    message = llm_output.choices[0].message
    # 示例解析逻辑（需要根据实际输出格式调整）
    try:
        if message.tool_calls:
            for function in message.tool_calls:
                if isinstance(function, ChatCompletionMessageFunctionToolCall):
                    subtasks.append(
                        SubTask(
                            description="function calls",
                            tools_args=ToolCall(
                                id=function.id, name=function.function.name, args=function.function.arguments
                            ),
                        )
                    )

        if llm_output.usage:
            usage_data = llm_output.usage
            usage.completion_tokens = usage_data.completion_tokens
            usage.prompt_tokens = usage_data.prompt_tokens
            usage.total_tokens = usage_data.total_tokens
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM output: {e}")
        # 处理解析错误，例如返回一个默认的LLMResponse或者抛出异常
        return LLMResponse(subtasks=[], execution_strategy="sequential", usage=usage)

    return LLMResponse(subtasks=subtasks, execution_strategy="sequential", usage=usage)


class OpenAICompatibleClient:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def __init__(self, mode: str, url: str):
        self.mode = mode
        self.url = url
        self.client = AsyncOpenAI(api_key="ollama", timeout=60, base_url=f"http://{url}/v1")

    async def generate(self, messages: list[dict], system_prompt: str) -> tuple[LLMResponse, str | None]:
        date_prefix = f"The current date is {self.current_time}\n"
        system_prompt = date_prefix + system_prompt
        # 系统提示只放开头一轮，避免多轮工具调用重复堆叠
        payload = [{"role": "system", "content": system_prompt}, *messages]

        # 与原生 tools 同时要求 json_object 时，部分后端（如 Ollama）末轮可能产生 token 但 content 为空
        api_request = {
            "model": self.mode,
            "messages": payload,
            "max_tokens": 1000,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
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
