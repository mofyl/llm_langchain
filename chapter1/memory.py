from enum import Enum
from typing import Any


class RecordType(str, Enum):
    execution = "execution"
    reflection = "reflection"


class Memory:
    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def add_record(self, record_type: RecordType, content: Any):
        self.records.append({"type": record_type, "content": content})

    def get_trajectory(self) -> str:
        trajectory_parts = []

        for record in self.records:
            if record["type"] == RecordType.execution:
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record["type"] == RecordType.reflection:
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")

        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> str | None:
        for record in reversed(self.records):
            if record["type"] == RecordType.execution:
                return record["content"]
        return None
