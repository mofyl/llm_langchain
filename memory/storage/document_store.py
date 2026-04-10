"""文档存储实现

支持多种文档数据库后端：
- SQLite: 轻量级关系型数据库
- PostgreSQL: 企业级关系型数据库（可扩展）
"""

import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from concurrent.futures import thread
from tracemalloc import start
from typing import Any, Optional

from click import Option
from numpy import where

from memory.base import MemoryType


class DocumentStore(ABC):
    """文档存储基类"""

    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: MemoryType,
        timestamp: int,
        importantce: float,
        properties: dict[str, Any] = None,
    ) -> str:
        """添加记忆"""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """获取单个记忆"""
        pass

    @abstractmethod
    def search_memories(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        importance_threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """搜索记忆"""
        pass

    @abstractmethod
    def update_memory(
        self, memory_id: str, content: str = None, importance: float = None, properties: dict[str, Any] = None
    ) -> bool:
        """更新记忆"""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    def get_database_stats(self) -> dict[str, Any]:
        """获取数据库统计信息"""
        pass

    @abstractmethod
    def add_document(self, content: str, metadata: dict[str, Any] = None) -> str:
        """添加文档"""
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """获取文档"""
        pass


class SQLiteDocumentStore(DocumentStore):
    """SQLite文档存储实现"""

    _instance = {}  # 存储已创建的实例
    _initialized_dbs = set()  # 存储已初始化的数据库路径
    _initialized = False

    def __new__(cls, db_path: str = "./memory.db"):
        abs_path = os.path.abspath(db_path)

        if abs_path not in cls._instance:
            instance = super().__new__(cls)
            cls._instance[abs_path] = instance
        return cls._instance[abs_path]

    def __init__(self, db_path: str = "./memory.db"):

        if self._initialized:
            return

        self.db_path = db_path
        self.local = threading.local()
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        abs_path = os.path.abspath(db_path)

        if abs_path not in self._initialized_dbs:
            self._init_database()
            self._initialized_dbs.add(abs_path)
            print(f"[OK] SQLite 文档存储初始化完成: {db_path}")

        self._initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row
        return self.local.connection

    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()

        cursor = conn.cursor()

        # 创建用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 创建记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                importance REAL NOT NULL,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # 创建概念表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建记忆-概念关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_concepts (
                memory_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (memory_id, concept_id),
                FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
                FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
        """)

        # 创建概念关系表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concept_relationships (
                from_concept_id TEXT NOT NULL,
                to_concept_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_concept_id, to_concept_id, relationship_type),
                FOREIGN KEY (from_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                FOREIGN KEY (to_concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
        """)

        # 创建索引
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts (concept_id)",
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

        conn.commit()
        print("[OK] SQLite 数据库表和索引创建完成")

    def add_memory(self, memory_id, user_id, content, memory_type, timestamp, importance, properties=None) -> str:
        """添加记忆"""
        conn = self._get_connection()

        cursor = conn.cursor()

        cursor.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, user_id))

        # 插入记忆
        cursor.execute(
            """
            INSERT OR REPLACE INTO memories 
            (id, user_id, content, memory_type, timestamp, importance, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                memory_id,
                user_id,
                content,
                memory_type,
                timestamp,
                importance,
                json.dumps(properties) if properties else None,
            ),
        )

        conn.commit()
        return memory_id

    def get_memory(self, memory_id) -> dict[str, Any] | None:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            WHERE id = ?
        """,
            (memory_id,),
        )

        row = cursor.fetchone()

        if not row:
            return None

        return {
            "memory_id": row["id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "memory_type": row["menoty_type"],
            "timestamp": row["timestamp"],
            "importance": row["importance"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
            "created_at": row["created_at"],
        }

    def search_memories(
        self, user_id=None, memory_type=None, start_time=None, end_time=None, importance_threshold=None, limit=10
    ) -> list[dict[str, Any]]:
        """搜索记忆"""

        conn = self._get_connection()
        cursor = conn.cursor()

        where_cond = []
        params = []

        if user_id:
            where_cond.append("user_id = ?")
            params.append(user_id)

        if memory_type:
            where_cond.append("memory_type = ?")
            params.append(memory_type)

        if start_time:
            where_cond.append("start_time = ?")
            params.append(start_time)

        if end_time:
            where_cond.append("end_time = ?")
            params.append(end_time)

        if importance_threshold:
            where_cond.append("importance >= ?")
            params.append(importance_threshold)

        where_clause = ""

        if where_cond:
            where_clause = "WHERE " + " AND ".join(where_cond)

        cursor.execute(
            f"""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            {where_clause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """,
            params + [limit],
        )

        memories = []

        for row in cursor.fetchall():
            memories.append(
                {
                    "memory_id": row["id"],
                    "user_id": row["user_id"],
                    "content": row["content"],
                    "memory_type": row["memory_type"],
                    "timestamp": row["timestamp"],
                    "importance": row["importance"],
                    "properties": json.loads(row["properties"]) if row["properties"] else {},
                    "created_at": row["created_at"],
                }
            )

        return memories
