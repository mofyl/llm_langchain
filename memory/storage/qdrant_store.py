import logging
import threading
import uuid
from ast import Nonlocal
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SearchRequest,
    VectorParams,
)
from requests import api

logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    """Qdrant连接管理器 - 防止重复连接和初始化"""

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        url: str = None | None,
        api_key: str = None | None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs,
    ) -> "QdrantVectorStore":
        """获取或创建Qdrant实例（单例模式）"""

        key = (url or "local", collection_name)

        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    logger.debug(f"🔄 创建新的Qdrant连接: {collection_name}")
                    cls._instances[key] = QdrantVectorStore(
                        url=url,
                        api_key=api_key,
                        collectoin_name=collection_name,
                        vector_size=vector_size,
                        distance=distance,
                        timeout=timeout,
                        kwargs=kwargs,
                    )
                else:
                    logger.debug(f"♻️ 复用现有Qdrant连接: {collection_name}")
        else:
            logger.debug(f"♻️ 复用现有Qdrant连接: {collection_name}")

        return cls._instances[key]


class QdrantVectorStore:
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collectoin_name: str = "hello_agent_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs,
    ):
        """
        初始化Qdrant向量存储 (支持云API)

        Args:
            url: Qdrant云服务URL (如果为None则使用本地)
            api_key: Qdrant云服务API密钥
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离度量方式 (cosine, dot, euclidean)
            timeout: 连接超时时间
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collectoin_name
        self.vector_size = vector_size
        self.timeout = timeout

        self.hnsw_m = 32
        self.hnsw_ef_construct = 256

        self.search_ef = 128

        self.search_exact = "1"

        distance_map = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclidean": Distance.EUCLID}

        self.distance = distance_map.get(distance.lower(), Distance.COSINE)

        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            self.client = QdrantClient(host="localhost", port=6333, time=self.timeout)
            self.client.get_collections()
            logger.info("✅ 成功连接到本地Qdrant服务: localhost:6333")
            self._ensure_collection()
        except Exception as e:
            logger.error(f"❌ Qdrant连接失败: {e}")

    def _ensure_collection(self):
        """确保集合存在，不存在则创建"""
        try:
            collections = self.client.get_collections().collections

            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                hnsw_cfg = None

                try:
                    hnsw_cfg = models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct)
                except Exception:
                    hnsw_cfg = None

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
                    hnsw_config=hnsw_cfg,
                )
                logger.info(f"✅ 创建Qdrant集合: {self.collection_name}")
            else:
                logger.info(f"✅ 使用现有Qdrant集合: {self.collection_name}")
                # 尝试更新 HNSW 配置
                try:
                    self.client.update_collection(
                        collection_name=self.collection_name,
                        hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct),
                    )
                except Exception as ie:
                    logger.debug(f"跳过更新HNSW配置: {ie}")
            # 确保必要的payload索引
            self._ensure_payload_indexes()
        except Exception as e:
            logger.error(f"❌ 集合初始化失败: {e}")
            raise

    def _ensure_payload_indexes(self):
        try:
            index_fields = [
                ("memory_type", models.PayloadSchemaType.KEYWORD),
                ("user_id", models.PayloadSchemaType.KEYWORD),
                ("memory_id", models.PayloadSchemaType.KEYWORD),
                ("timestamp", models.PayloadSchemaType.INTEGER),
                ("modality", models.PayloadSchemaType.KEYWORD),  # 感知记忆模态筛选
                ("source", models.PayloadSchemaType.KEYWORD),
                ("external", models.PayloadSchemaType.BOOL),
                ("namespace", models.PayloadSchemaType.KEYWORD),
                # RAG相关字段索引
                ("is_rag_data", models.PayloadSchemaType.BOOL),
                ("rag_namespace", models.PayloadSchemaType.KEYWORD),
                ("data_source", models.PayloadSchemaType.KEYWORD),
            ]

            for field_name, schema_type in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name, field_name=field_name, field_schema=schema_type
                    )
                except Exception as ie:
                    # 索引已存在会报错，忽略
                    logger.debug(f"索引 {field_name} 已存在或创建失败: {ie}")
        except Exception as e:
            logger.debug(f"创建payload索引时出错: {e}")

    def add_vectors(
        self, vectors: list[list[float]], metadata: list[dict[str, Any]], ids: list[str] | None = None
    ) -> bool:
        """
        添加向量到Qdrant

        Args:
            vectors: 向量列表
            metadata: 元数据列表
            ids: 可选的ID列表

        Returns:
            bool: 是否成功
        """

        try:
            if not vectors:
                logger.warning("⚠️ 向量列表为空")
                return False

            if ids is None:
                ids = [f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}" for i in range(len(vectors))]

            # 构建点数据
            logger.info(
                f"[Qdrant] add_vectors start: n_vectors={len(vectors)} n_meta={len(metadata)} collection={self.collection_name}"
            )

            points = []
            time_now = datetime.now().timestamp
            for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
                try:
                    vlen = len(vector)
                except Exception:
                    logger.error(f"[Qdrant] 非法向量类型: index={i} type={type(vector)} value={vector}")
                    continue
                if vlen != self.vector_size:
                    logger.warning(f"⚠️ 向量维度不匹配: 期望{self.vector_size}, 实际{len(vector)}")
                    continue

                meta_with_timestamp = meta.copy()
                meta_with_timestamp["timestamp"] = int(time_now)
                meta_with_timestamp["added_at"] = int(time_now)
                if "external" in meta_with_timestamp and not isinstance(meta_with_timestamp.get("external"), bool):
                    val = meta_with_timestamp.get("external")
                    meta_with_timestamp["external"] = True if str(val).lower() in ("1", "true", "yes") else False

                safe_id: Any

                if isinstance(point_id, int):
                    safe_id = point_id
                elif isinstance(point_id, str):
                    try:
                        uuid.UUID(point_id)
                        safe_id = point_id
                    except Exception:
                        safe_id = str(uuid.uuid4())
                else:
                    safe_id = str(uuid.uuid4())

                point = PointStruct(id=safe_id, vector=vector, payload=meta_with_timestamp)
                points.append(point)
            if not points:
                logger.warning("⚠️ 没有有效的向量点")
                return False
            # 批量插入
            logger.info(f"[Qdrant] upsert begin: points={len(points)}")

            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            logger.info("[Qdrant] upsert done")

            logger.info(f"✅ 成功添加 {len(points)} 个向量到Qdrant")
            return True

        except Exception as e:
            logger.error(f"❌ 添加向量失败: {e}")
            return False

    def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = None | None,
        where: dict[str, Any] = None | None,
    ) -> list[dict[str, Any]]:
        """
        搜索相似向量

        Args:
            query_vector: 查询向量
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            where: 过滤条件

        Returns:
            List[Dict]: 搜索结果
        """
        try:
            if len(query_vector) != self.vector_size:
                logger.error(f"❌ 查询向量维度错误: 期望{self.vector_size}, 实际{len(query_vector)}")
                return []

            query_filter = None

            if where:
                conditions = []

                for key, value in where.items():
                    if isinstance(value, (str, int, float, bool)):
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

                if conditions:
                    query_filter = Filter(must=conditions)

            search_params = None

            try:
                search_params = models.SearchParams(hnsw_ef=self.search_ef, exact=self.search_exact)
            except Exception:
                search_params = None

            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                )
                search_result = response.points
            except AttributeError:
                # 回退到旧API (qdrant-client < 1.16.0)
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                )

            results = []

            for hit in search_result:
                result = {"id": hit.id, "score": hit.score, "metadata": hit.payload or {}}
                results.append(result)
            logger.debug(f"🔍 Qdrant搜索返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def delete_vectors(self, ids: list[str]) -> bool:
        """
        删除向量

        Args:
            ids: 要删除的向量ID列表

        Returns:
            bool: 是否成功
        """

        try:
            if not ids:
                return True

            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids,
                ),
                wait=True,
            )

            logger.info(f"✅ 成功删除 {len(ids)} 个向量")
            return True

        except Exception as e:
            logger.error(f"❌ 删除向量失败: {e}")
            return False

    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except:
                pass
