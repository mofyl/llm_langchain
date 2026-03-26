import os


def _auto_detect_provider(api_key: str | None, base_url: str | None) -> str:

    if os.getenv("OLLAMA_API_KEY"):
        return "ollama"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    return "auto"
