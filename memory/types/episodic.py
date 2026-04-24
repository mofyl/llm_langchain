import logging
import os
from ast import Dict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest import result

from numpy import sort
from sympy import true

logger = logging.getLogger(__name__)

from memory.base import BaseMemoroy, MemoryConfig, MemoryItem, MemoryType
from memory.embedding import get_dimension, get_text_embedder
from memory.storage.document_store import SQLiteDocumentStore


class Episode:
    """情景记忆中的单个情景"""

    def __init__(
        self,
        episode_id: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        content: str,
        context: dict[str, Any],
        outcome: str | None = None,
        importance: float = 0.5,
    ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance


class EpisodicMemory(BaseMemoroy):
    """情景记忆实现

    特点：
    - 存储具体的交互事件
    - 包含丰富的上下文信息
    - 按时间序列组织
    - 支持模式识别和回溯
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        self.episodes: list[Episode] = []
        self.sessoin: dict[str, list[str]] = {}  # session_id -> episode_ids

        self.patterns_cache = {}
        self.last_pattern_analysis = None

        db_dir = self.config.stroage_path if hasattr(self.config, "stroage_path") else "./memory_data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)

        self.embedder = get_text_embedder()

        from ..storage.qdrant_store import QdrantConnectionManager

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        collection_name = os.getenv("QDRANT_COLLECTION", "hello_agents_vectors")
        dimension = get_dimension(getattr(self.embedder, "dimension", 384))
        distance = os.getenv("QDRANT_DISTANCE", "cosine")
        self.vector_store = QdrantConnectionManager.get_instance(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            vector_size=dimension,
            distance=distance,
        )

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""

        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        participants = memory_item.metadata.get("participants", [])
        tags = memory_item.metadata.get("tags", [])

        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance,
        )

        self.episodes.append(episode)
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type=MemoryType.EPISODICMEMORY,
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "session_id": session_id,
                "context": context,
                "outcome": outcome,
                "participants": participants,
                "tags": tags,
            },
        )
        return ""

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> list[MemoryItem]:
        """检索情景记忆（结构化过滤 + 语义向量检索）"""
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        time_range: tuple[datetime, datetime] | None = kwargs.get("time_range")
        importance_threshold: float | None = kwargs.get("importance_threshold")

        candidate_ids: set | None = None

        if time_range is not None or importance_threshold is not None:
            start_ts = int(time_range[0].timestamp()) if time_range else None
            end_ts = int(time_range[1].timestamp()) if time_range else None

            docs = self.doc_store.search_memories(
                user_id=user_id,
                memory_type=MemoryType.EPISODICMEMORY,
                start_time=start_ts,
                end_time=end_ts,
                importance_threshold=importance_threshold,
                limit=1000,
            )
            candidate_ids = {d["memory_id"] for d in docs}

        try:
            query_vec = self.embedder.encode(query)
            if hasattr(query_vec, "tolist"):
                query_vec = query_vec.tolist()

            where = {"memory": "episodic"}
            if user_id:
                where["user_id"] = user_id
            hits = self.vector_store.search_similar(query_vector=query_vec, limit=max(limit * 5, 20), where=where)
        except Exception:
            hits = []

        now_ts = int(datetime.now().timestamp())

        results: list[tuple[float, MemoryItem]] = []

        seen = set()

        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = hit.get("memory_id")

            if not mem_id or mem_id in seen:
                continue

            episode = next((e for e in self.episodes if e.episode_id == mem_id), None)

            if episode and episode.context.get("forgotten", False):
                continue
            if session_id and meta.get("session_id") != session_id:
                continue

            doc = self.doc_store.get_memory(memory_id=mem_id)

            if not doc:
                continue
            # 计算综合分数：向量0.6 + 近因0.2 + 重要性0.2
            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(doc["timestamp"])) / 86400.0)
            recency_score = 1.0 / (1.0 + age_days)
            imp = float(doc.get("importance", 0.5))

            # 新评分算法：向量检索纯基于相似度，重要性作为加权因子

            # 基础相似度得分（不受重要性影响）
            base_relevance = vec_score * 0.8 + recency_score * 0.2
            # 重要性作为乘法加权因子，范围 [0.8, 1.2]
            importance_weight = 0.8 + (imp * 0.4)

            combined = base_relevance * importance_weight

            item = MemoryItem(
                id=doc["memory_id"],
                content=doc["content"],
                memory_type=doc["memory_type"],
                user_id=doc["user_id"],
                timestamp=datetime.fromtimestamp(doc["timestampe"]),
                importance=doc.get("importance", 0.5),
                metadata={
                    **doc.get("properties", {}),
                    "relevance_score": combined,
                    "vector_score": vec_score,
                    "recency_score": recency_score,
                },
            )
            results.append((combined, item))
            seen.add(mem_id)

        if not results:
            fallback = super()._generate_id()
            query_lower = query.lower()

            for ep in self._filter_episodes(user_id=user_id, session_id=session_id, time_range=time_range):
                if query_lower in ep.content.lower():
                    recency_score = 1.0 / (1.0 + max(0.0, (now_ts - int(ep.timestamp.timestamp()))) / 86400.0)

                    # 回退匹配：新评分算法
                    keyword_score = 0.5  # 简单关键词匹配的基础分数

                    base_relevance = keyword_score * 0.8 + recency_score * 0.2
                    #  [0.8 , 1.2]
                    importance_weight = 0.8 + (ep.importance * 0.4)
                    combined = base_relevance * importance_weight

                    item = MemoryItem(
                        id=ep.episode_id,
                        content=ep.content,
                        memory_type=MemoryType.EPISODICMEMORY,
                        user_id=ep.user_id,
                        timestamp=ep.timestamp,
                        importance=ep.importance,
                        metadata={
                            "session_id": ep.session_id,
                            "context": ep.context,
                            "outcome": ep.outcome,
                            "relevance_score": combined,
                        },
                    )
                    results.append((combined, item))
        results.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in results[:limit]]

    def _filter_episodes(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[Episode]:
        """过滤情景"""
        filtered = self.episodes

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id]

        if time_range:
            start_time, end_time = time_range
            filtered = [e for e in filtered if start_time <= e.timestamp <= end_time]

        return filtered

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新情景记忆（SQLite为权威，Qdrant按需重嵌入）"""
        updated = False

        for ep in self.episodes:
            if ep.episode_id == memory_id:
                if content is not None:
                    ep.content = content
                if importance is not None:
                    ep.importance = importance
                if metadata is not None:
                    ep.context.update(metadata.get("context", {}))
                    if "outcome" in metadata:
                        ep.outcome = metadata["outcome"]
                updated = True
                break

        doc_updated = self.doc_store.update_memory(
            memory_id=memory_id, content=content, importance=importance, properties=metadata
        )
        if content is not None:
            try:
                embedding = self.embedder.encode(content)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                doc = self.doc_store.get_memory(memory_id=memory_id)

                payload = {
                    "memory_id": memory_id,
                    "user_id": doc["user_id"] if doc else "",
                    "memory_type": "episodic",
                    "importance": (doc.get("importance") if doc else importance) or 0.5,
                    "session_id": (doc.get("properties", {}) or {}).get("session_id") if doc else "",
                    "content": content,
                }

                self.vector_store.add_vectors(vectors=[embedding], metadata=[payload], ids=[memory_id])
            except Exception:
                pass

        return updated or doc_updated

    def remove(self, memory_id: str) -> bool:
        """删除情景记忆（SQLite + Qdrant）"""

        removed = False

        for i, ep in enumerate(self.episodes):
            if ep.episode_id == memory_id:
                removed_ep = self.episodes.pop(i)

                session_id = removed_ep.session_id
                if session_id in self.sessoin:
                    self.sessoin[session_id].remove(memory_id)
                    # 如果session 已经为空
                    if not self.sessoin[session_id]:
                        del self.sessoin[session_id]
                    removed = True
                    break
        doc_deleted = self.doc_store.delete_memory(memory_id=memory_id)
    
        try : 
            self.vector_store.delete_

    # def has_memory(self, memory_id: str) -> bool:
    #     """检查记忆是否存在

    #     Args:
    #         memory_id: 记忆ID

    #     Returns:
    #         是否存在
    #     """

    # def clear(self):
    #     """清空所有记忆"""

    # def get_stats(self) -> dict[str, Any]:
    #     """获取记忆统计信息

    #     Returns:
    #         统计信息字典
    #     """
    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> int:
        """情景记忆遗忘机制（硬删除）"""
        forgotten_count = 0
        current_time = datetime.now()

        to_remove = []

        for ep in self.episodes:
            should_forget = False

            if strategy == "importance_based":
                if ep.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                cutoff_time = current_time - timedelta(days=max_age_days)
                if ep.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                # 基于容量遗忘（保留最重要的）
                if len(self.episodes) > self.config.max_cap:
                    sorted_ep = sorted(self.episodes, key=lambda e: e.importance)
                    execess_count = self.config.max_cap - len(self.episodes)
                    # 如果在前面 那么就需要遗忘
                    if ep in sorted_ep[:execess_count]:
                        should_forget = True

            if should_forget:
                to_remove.append(ep.episode_id)

        for ep_id in to_remove:
            if self.remove(ep_id):
                forgotten_count += 1
                logger.info(f"情景记忆硬删除: {ep_id[:8]}... (策略: {strategy})")

        return forgotten_count
