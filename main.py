import asyncio
import re
import re
import time

from tools import tools , available_tools
from prompt import COMMON_PROMPT, LLMResponse, Usage, parse_llm_response
import openai
from openai import AsyncOpenAI




class OpenAICompatibleClient:

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    

    def __init__(self, mode: str ):
        self.mode = mode
        self.client = AsyncOpenAI(api_key="ollama", timeout=60 , base_url="http://127.0.0.1:11434/v1")


    async def generate(self, user_input : str)  -> LLMResponse:

        date_prefix = f"The current date is {self.current_time}\n"

        system_prompt = date_prefix + COMMON_PROMPT


        message = [
            {"role" : "user" , "content" : user_input},
            {"role" : "system" , "content" : system_prompt}
        ]

        api_request = {
            "model" : self.mode,
            "messages" : message,
            "max_tokens" : 1000,
            "tools" : tools,
            "tool_choice" : "auto",
            "response_format" : {"type" :"json_object"}
        }

        try:
            response = await self.client.chat.completions.create(**api_request)
           
            llm_response = parse_llm_response(response)
            return llm_response
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return LLMResponse(subtasks=[], execution_strategy="sequential", usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0))




if __name__ == "__main__":


    llm = OpenAICompatibleClient(mode="qwen3.5:0.8b")

    user_input = "请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"

    prompt_history = [f"用户输入: {user_input}"]

    for i in range(5) : 

        full_prompt = "\n".join(prompt_history)


        llm_output = asyncio.run( llm.generate(full_prompt))

        print(f"模型输出 {llm_output}\n" + "="*40)

        # match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
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

        # # 3.4. 记录观察结果
        # observation_str = f"Observation: {observation}"
        # print(f"{observation_str}\n" + "="*40)
        # prompt_history.append(observation_str)