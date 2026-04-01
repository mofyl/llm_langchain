from importlib import import_module


def __getattr__(name: str):
    if name == "OpenAICompatibleClient":
        return import_module(".open_ai_provider", __name__).OpenAICompatibleClient
    if name == "SimpleAgent":
        return import_module(".simple_agent", __name__).SimpleAgent
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["OpenAICompatibleClient", "SimpleAgent"]
