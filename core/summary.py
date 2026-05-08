from core.llm import HelloAgentsLLM


def summary_keep_structions(llm: HelloAgentsLLM, user_input: str) -> str:
    """
    使用模型对 user_input 进行总结，保持 user_input 的格式
    """
