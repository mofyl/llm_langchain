import datetime
import os

from memory.base import BaseMemoroy, MemoryConfig, MemoryItem
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
        context: dict[str, any],
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
        super.__init__(config, storage_backend)

        self.episodes: list[Episode] = []
        self.sessoin = dict[str, list[str]] = {}  # session_id -> episode_ids

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
            memory_type="episodic",
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
