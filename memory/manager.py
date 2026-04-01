import logging
import uuid
from datetime import datetime
from typing import Any

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

        self.memory_types: dict[MemoryType, BaseMemoroy] = {}

        if enable_working:
            self.memory_types[MemoryType.WORKINGMEMORY] = WorkingMemory(self.config)

        logging.getLogger(__name__).info("memory manager init finish")

    # MemoryType
    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
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
            logging.getLogger(__name__).debug(f"添加记忆到 {memory_type} : {memory_id}")
            return memory_id

        return ""

    def _calculate_importance(self, content: str, metadata: dict[str, Any] | None) -> float:
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

    def retrieve_memories(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
        min_importance: float = 0.8,
    ) -> list[MemoryItem]:
        """检索记忆

        Args:
            query: 查询内容
            memory_types: 要检索的记忆类型列表
            limit: 返回数量限制
            min_importance: 最小重要性阈值
            time_range: 时间范围 (start_time, end_time)

        Returns:
            检索到的记忆列表
        """
        if memory_types is None:
            memory_types = list(self.memory_types.keys())

        all_result = []

        per_type_limit = max(1, limit // max(1, len(memory_types)))

        for memory_type in memory_types:
            if memory_type in self.memory_types:
                memory_instance = self.memory_types[memory_type]

                try:
                    type_results = memory_instance.retrieve(
                        query=query, limit=per_type_limit, min_importance=min_importance, user_id=self.user_id
                    )
                    all_result.extend(type_results)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"检索 {memory_type} 记忆时出错：{e}")
                    continue

        all_result.sort(key=lambda x: x.importance, reverse=True)
        return all_result[:limit]
