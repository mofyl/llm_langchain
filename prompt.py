import json

from attr import dataclass
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageFunctionToolCall

COMMON_PROMPT = (
    "你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。\n"
    "# 可用工具：\n"
    "- `get_weather(city: str)`: 查询指定城市的实时天气。\n"
    "- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。\n"
    "# 输出格式要求:\n"
    "你的每次回复必须严格遵循以下格式:\n"
    "{\n"
    '  "subtasks": [\n'
    "    {\n"
    '      "description": "Task description",\n'
    '      "tools_args": {"name": "get_weather","args": {"city": "北京"} },\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "规则：\n"
    "- subTask: 每个子任务都应该是一个独立的、可执行的步骤，描述需要完成的具体任务。如果当前已经可以回答用户的问题，子任务返回[]\n"
    "- tools_args: 每个子任务都必须指定一个工具调用，包含工具名称和参数。\n"
    "例如：\n"
    "用户输入: 请帮我查询一下今天北京的天气\n"
    "你的回复:\n"
    "{\n"
    '  "subtasks": [\n'
    "    {\n"
    '      "description": "查询北京的天气",\n'
    '      "tools_args": {"name": "get_weather","args": {"city": "北京"} },\n'
    "    },\n"
    "  ]\n"
    "}\n"
)


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
    tools_args: ToolCall  # 需要调用的工具 以及工具所需要的参数 {"tool": "web_search", "query": "Apple stock AAPL trend analysis forecast"}


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
