from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class RoleType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class Message(BaseModel):
    content: str
    role: RoleType
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __init__(self, content: str, role: RoleType, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
