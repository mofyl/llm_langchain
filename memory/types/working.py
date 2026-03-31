import heapq
from curses import meta, noecho
from datetime import datetime, timedelta
from itertools import count
from typing import Any

from ..base import BaseMemoroy, MemoriesState, MemoryConfig, MemoryItem, MemoryType


class WorkingMemory(BaseMemoroy):
    """工作记忆实现

    特点：
    - 容量有限（通常10-20条记忆）
    - 时效性强（会话级别）
    - 优先级管理
    - 自动清理过期记忆
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend=storage_backend)

        self.max_cap = self.config.working_memory_cap
        self.max_tokens = self.config.working_memory_tokens

        self.max_age_min = self.config.working_memory_ttl_min or 120

        self.current_tokens = 0

        self.session_start = datetime.now()

        self.memories: list[MemoryItem] = []

        self.memory_heap = []  # (prioriry , timestamp , memory_item)

    def add(self, memory_item: MemoryItem) -> str:

        self._expire_old_memories()

        priority = self._calculate_priority(memory_item)

        heapq.heappush(self.memory_heap, (-priority, memory_item.timestamp, memory_item))
        self.memories.append(memory_item)

        self.current_tokens += len(memory_item.content.split())

        self._enforce_cap_limit()

        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, user_id: str = None, **kwargs) -> list[MemoryItem]:
        self._expire_old_memories

        if not self.memories:
            return []

        active_memories = [m for m in self.memories if not m.metadata.get("forgotten", False)]

        if user_id:
            filtered_memories = [m for m in active_memories if m.user_id == user_id]

        if not filtered_memories:
            return []

        vector_scores = {}

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            documents = [query] + [m.content for m in filtered_memories]

            vectorizer = TfidfVectorizer(stop_words=None, lowercase=None)

            matrix = vectorizer.fit_transform(documents)

            query_vector = matrix[0:1]
            doc_vectors = matrix[1:]

            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            for i, memory in enumerate(filtered_memories):
                vector_scores[memory.id] = similarities[i]

        except Exception:
            vector_scores = {}

        query_lower = query.lower()

        scored_memories = []

        for memory in filtered_memories:
            content_lower = memory.content.lower()

            vector_score = vector_scores.get(memory.id, 0.0)

            keyword_score = 0.0

            if query_lower in content_lower:
                # 这里是在处理关键词匹配 , 关键词越多，得分越高
                keyword_score = len(query_lower) / len(content_lower)
            else:
                # 分词匹配
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())

                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection) / len(query_words.union(content_words)) * 0.8

            if vector_score > 0:
                base_lelevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_lelevance = keyword_score

            time_decay = self._calculate_time_decay(memory.timestamp)
            base_lelevance *= time_decay

            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_lelevance * importance_weight

            if final_score > 0:
                scored_memories.append((final_score, memory))

        scored_memories.sort(key=lambda x: x[0], reverse=True)

        return [memory for _, memory in scored_memories[:limit]]

    def update(
        self, memory_id: str, content: str = None, importatce: float = None, metadate: dict[str, Any] = None
    ) -> bool:

        for memory in self.memories:
            if memory.id != memory_id:
                continue

            old_tokens = len(memory.content.split())

            if content is not None:
                memory.content = content
                new_tokens = len(content.split())
                self.current_tokens = self.current_tokens - old_tokens + new_tokens

            if importatce is not None:
                memory.importance = importatce

            if metadate is not None:
                memory.metadata.update(metadate)

            self._update_heap_prioryty(memory=memory)
            return True
        return False

    def remove(self, memory_id: str) -> bool:
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                remove_memory = self.memories.pop(i)

                self.current_tokens -= len(remove_memory.content.split())
                self.current_tokens = max(0, self.current_tokens)

                return True
        return False

    def has_memory(self, memory_id: str) -> bool:
        return any(memory.id == memory_id for memory in self.memorys)

    def clear(self):
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0

    # MemoriesState
    def get_stats(self) -> MemoriesState:

        self._expire_old_memories()

        active_memories = self.memories
        m = MemoriesState(
            count=len(active_memories),
            forgotten_count=0,
            total_count=len(self.memories),
            current_tokens=self.current_tokens,
            max_cap=self.max_cap,
            max_tokens=self.max_tokens,
            max_age_min=self.max_age_min,
            session_duration_min=(datetime.now() - self.session_start).total_seconds() / 60,
            avg_importance=sum(m.importance for m in active_memories) / len(active_memories)
            if active_memories
            else 0.0,
            cap_usage=len(active_memories) / self.max_cap if self.max_cap > 0 else 0.0,
            token_usage=self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            memory_type=MemoryType.WORKINGMEMORY,
        )
        return m

    def get_recent(self, limit: int = 10) -> list[MemoryItem]:
        sorted_memories = sorted(self.memories, key=lambda x: x.timestamp, reverse=True)
        return sorted_memories[:limit]

    def get_important(self, limit: int = 10) -> list[MemoryItem]:
        sorted_memories = sorted(self.memories, key=lambda x: x.importance, reverse=True)
        return sorted_memories[:limit]

    def get_all(self) -> list[MemoryItem]:
        return self.memories.copy()

    def get_context_summary(self, max_length: int = 500) -> str:
        if not self.memories:
            return "No working memories available."

        sorted_memories = sorted(
            self.memories,
            key=lambda m: (m.importance, m.timestamp),
            reverse=True,
        )

        summary_parts = []
        current_length = 0

        for memory in sorted_memories:
            content = memory.content

            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
            else:
                ramaining = max_length - current_length

                if ramaining > 50:
                    summary_parts.append(content[:ramaining] + "...")
                break
        return "Working Memory Context:\n" + "\n".join(summary_parts)

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 1) -> int:
        forgotten_count = 0
        current_time = datetime.now()

        to_remove = []

        cutoff_ttl = current_time - timedelta(minutes=self.max_age_min)

        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                to_remove.append(memory.id)

        if strategy == "importance_based":
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.id)

        if strategy == "time_based":
            cutoff_time = current_time - timedelta(hours=max_age_days * 24)
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    to_remove.append(memory.id)

        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1
        return forgotten_count

    def _enforce_cap_limit(self):
        # 检查记忆数量
        while len(self.memories) > self.max_cap:
            self._remo_lowest_priority_memory()
        # 检查 tokens
        while self.current_tokens > self.max_tokens:
            self._remo_lowest_priority_memory()

    def remove(self, id: str) -> bool:
        for i, mem in enumerate(self.memories):
            if mem.id == id:
                removed_memory = self.memories.pop(i)
                # 这里对于 堆中的内容没有删除，因为在add 的时候会去自动删除
                self.current_tokens -= len(removed_memory.content.split())
                self.current_tokens = max(0, self.current_tokens)  # 防止减为复数
                return True
        return False

    def _update_heap_prioryty(self, memory: MemoryItem):

        self.memory_heap = [(-self._calculate_priority(mem), mem.timestamp, mem) for mem in self.self.memories]

        heapq.heapify(self.memory_heap)

    def _remo_lowest_priority_memory(self):
        if not self.memories:
            return

        # 找到优先级最低的记忆
        lowest_priority = float("inf")
        lowest_memory = None

        for memory in self.memories:
            priority = self._calculate_priority(memory)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = memory

        self.remove(lowest_memory.id)

    def _expire_old_memories(self):

        if not self.memories:
            return

        curoff_time = datetime.now() - datetime.timedelta(minutes=self.max_age_min)

        kept: list[MemoryItem] = []

        removed_token_sum = 0

        for m in self.memories:
            if m.timestamp >= curoff_time:
                kept.append(m)
            else:
                removed_token_sum += len(m.content.split())

        if len(kept) == len(self.memories):
            # 没有过期的
            return

        self.memories = kept

        self.current_tokens = max(0, self.current_tokens - removed_token_sum)
        self.memory_heap = []

        for mem in self.memories:
            priority = self._calculate_priority(mem.timestamp)
            # 这里 heapq 是固定的最小堆， 存入了三个元素，比较的时候 按顺序依次比较
            # priority 是0~1 的数字，取负值后 就变成了最大堆，
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))

    def _calculate_priority(self, memory: MemoryItem) -> float:

        priority = memory.importance

        time_decay = self._calculate_time_decay(memory.timestamp)

        priority *= time_decay

        return priority

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        time_diff = datetime.now() - timestamp
        hour_passed = time_diff.total_seconds() / 3600
        # 保证 时间越近的 指数越大，优先级越高
        decay_factor = self.config.decay_factor ** (hour_passed / 6)

        return max(1.0, decay_factor)
