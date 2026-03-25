import time
from gc import collect
from os import name

from anyio.lowlevel import T
from openai import AsyncOpenAI, ChatCompletion


class OpenAICompatibleClient:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def __init__(self, mode: str, url: str):
        self.mode = mode
        self.url = url
        self.client = AsyncOpenAI(api_key="ollama", timeout=60, base_url=f"http://{url}/v1")

    async def think(self, message: list[dict], temperature: float = 0.7) -> str:
        response = await self.client.chat.completions.create(
            model=self.mode, messages=message, temperature=temperature, stream=True
        )

        collect_response = []
        async for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            collect_response.append(content)
        print()  # 在流式输出结束后换行
        return "".join(collect_response)


async def asyncMain():
    try:
        llmClient = OpenAICompatibleClient(mode="qwen3.5:0.8b", url="127.0.0.1:11434")

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        print("--- 调用LLM ---")
        responseText = await llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(asyncMain())
