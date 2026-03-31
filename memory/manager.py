import datetime
import logging
import uuid
from types import MemberDescriptorType

from memory.base import BaseMemoroy, MemoryConfig, MemoryItem, MemoryType
from memory.types.working import WorkingMemory


class MemoryManager:
    def __init__(
        self,
        config: MemoryConfig | None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = True,
    ):
        self.config = config or MemoryConfig()

        self.user_id = user_id

        self.memory_types = dict[MemoryType, BaseMemoroy]

        if enable_working:
            self.memory_types[MemoryType.WORKINGMEMORY] = WorkingMemory(self.config)

        logging.Logger.info("memory manager init finish")

    # MemoryType
    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float | None = None,
        metadata: dict[str, any] = None,
        auto_classify: bool = True,  # 自动给记忆分类
    ) -> str:

        memory_type = MemoryType.WORKINGMEMORY

        if importance is None:
            importance = self._calculate_importance(content=content, metadata=metadata)

        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {},
        )

        if memory_type in self.memory_types:
            memory_id = self.memory_types[memory_type].add(memory_item)
            logging.Logger.debug(f"添加记忆到 {memory_type} : {memory_id}")

        return ""

    def _calculate_importance(self, content: str, metadata: dict[str, any] | None) -> float:
        importance = 0.5

        if len(content) > 100:
            importance += 0.1

        # 基于关键词
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]

        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        if metadata:
            if metadata.get("priority") == "hight":
                importance += 0.3
            elif metadata.get("priority") == "low":
                importance -= 0.2
        return max(0.0, min(1.0, importance))
