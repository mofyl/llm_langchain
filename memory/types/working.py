import heapq
from datetime import datetime, timedelta

from ..base import BaseMemoroy, MemoryConfig, MemoryItem, MemoryType


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
