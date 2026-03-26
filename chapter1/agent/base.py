from abc import ABC, abstractmethod

from config import Config
from message import Message
from open_ai_provider import OpenAICompatibleClient


class Agent(ABC):
    # OpenAICompatibleClient
    def __init__(self, name: str, llm: OpenAICompatibleClient, system_prompt: str | None, config: Config | None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        pass

    def add_message(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> list[Message]:
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name} , provider=ollama)"
