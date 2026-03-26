import datetime
import time
from enum import Enum

from pydantic import BaseModel


class RoleType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class Message(BaseModel):
    content: str
    role: RoleType
    timestamp: datetime = None
    metadata: dict[str, any] = None | None

    def __init__(self, content: str, role: RoleType, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, any]:
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
