import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, enum.Enum):
    WORKINGMEMORY = "working"
    LONGTERMMEMORY = "long_term"
    SENSORYMEMORY = "sensory"
    EPISODICMEMORY = "episodic"
    DOCUMENT = "document"


# @dataclass
# class MemoriesState:
#     count: int
#     forgotten_count: int
#     total_count: int
#     current_tokens: int
#     max_cap: int
#     max_tokens: int
#     max_age_min: int
#     session_duration_min: int
#     avg_importance: float
#     cap_usage: float
#     token_usage: float
#     memory_type: MemoryType


class MemoryItem(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    stroage_path: str = "./memory_data"

    max_cap: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    working_memory_cap: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_min: int = 120


class BaseMemoroy(ABC):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type: MemoryType = MemoryType(self.__class__.__name__.lower().replace("memory", ""))

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆项

        Args:
            memory_item : 记忆项对象

        Returns:
            记忆ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> list[MemoryItem]:
        """检索记忆项

        Args:
            query: 查询内容
            limit: 查询返回数量限制
            **kwargs: 其他检索参数

        Returns:
            相关记忆列表
        """
        pass

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆

        Args:
            memory_id: 记忆ID
            content: 新的记忆内容
            importance: 新的重要性
            metadata: 新的元数据
        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在

        Args:
            memory_id: 记忆ID

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """获取记忆统计信息

        Returns:
            统计信息字典
        """
        pass

    def _generate_id(self) -> str:
        import uuid

        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_impartance: float = 0.5) -> float:

        importance = base_impartance

        if len(content) > 100:
            importance += 0.1

        importance_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]

        if any(keyword in content for keyword in importance_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self):
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self):
        return self.__str__()
