import os
from typing import Any

from pydantic import BaseModel


class Config(BaseModel):
    default_model: str = "qwen3:0.8b"
    default_provider: str = "ollama"

    temperature: float = 0.7

    max_token: int | None = None

    debug: bool = False

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_token=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
