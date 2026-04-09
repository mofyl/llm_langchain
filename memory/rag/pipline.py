import hashlib
import os
from ast import Nonlocal
from types import Any
from typing import Optional

from attr.validators import get_disabled
from ollama import embed
from sympy import EX, limit

from chapter1.open_ai_provider import OpenAICompatibleClient
from memory.storage.qdrant_store import QdrantVectorStore

from ..embedding import get_dimension, get_text_embedder


def _get_markitdown_instance():
    """
    Get a configured MarkItDown instance for document conversion.
    """
    try:
        from markitdown import MarkItDown

        return MarkItDown()
    except ImportError:
        print("[WARNING] MarkItDown not available. Install with: pip install markitdown")
        return None


def _fallback_text_reader(path: str) -> str:

    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        try:
            with open(path, encoding="latin-1", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


def _detect_lang(sample: str) -> str:
    try:
        from langdetect import detect

        return detect(sample[:1000]) if sample else "unknown"
    except Exception:
        return "unknown"


def _split_paragraphs_with_headings(text: str) -> list[dict]:
    lines = text.splitlines()

    heading_stack: list[str] = []
    paragraphs: list[dict] = []
    buf: list[str] = []
    char_pos = 0

    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()

        if not content:
            return

        paragraphs.append(
            {
                "content": content,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos,
            }
        )

    for ln in lines:
        raw = ln

        if raw.strip().startswith("#"):
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip("#"))

            title = raw.lstrip("#").strip()

            if level <= 0:
                level = 1

            if level <= len(heading_stack):
                heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            char_pos += len(raw) + 1
            continue

        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)

        char_pos += len(raw) + 1

    flush_buf(char_pos)
    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
    return paragraphs


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _approx_token_len(text: str) -> int:
    # 近似估计：CJK字符按1 token，其他按空白分词
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens


def _chunk_pargaraphs(paragraphs: list[dict], chunk_tokens: int, overlap_tokens: int) -> list[dict]:
    chunks = list[dict] = []

    cur: list[dict] = []
    cur_tokens = 0
    i = 0

    while i < len(paragraphs):
        p = paragraphs[i]

        p_tokens = _approx_token_len(p["content"]) or 1

        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            content = "\n\n".join(x["content"] for x in cur)

            start = cur[0]["start"]
            end = cur[-1]["end"]

            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)

            chunks.append(
                {
                    "content": content,
                    "start": start,
                    "end": end,
                    "heading_path": heading_path,
                }
            )

            if overlap_tokens > 0 and cur:
                kept: list[dict] = []
                kept_tokens = 0

                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    # 由于 切分的chunk 之间会有重复，这里就是在处理重复，overlap_tokens 表示重复的token 需要有多少
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0

    if cur:
        # 还剩了点
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)

        chunks.append(
            {
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            }
        )

    return chunks


def load_and_chunk_texts(
    paths: list[str],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    namespace: str | None = None,
    source_label: str = "rag",
) -> list[dict]:
    """
    Universal document loader and chunker using MarkItDown.
    Converts all supported formats to markdown, then chunks intelligently.
    """
    print(
        f"[RAG] Universal loader start: files={len(paths)} chunk_size={chunk_size} overlap={chunk_overlap} ns={namespace or 'default'}"
    )

    chunks: list[dict] = []

    seen_hashes = set()

    for path in paths:
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue

        print(f"[RAG] Processing: {path}")
        ext = (os.path.splitext(path)[1] or "").lower()

        markdown_text = _convert_to_markdown(path=path)

        if not markdown_text.strip():
            print(f"[WARNING] No content extracted from: {path}")
            continue

        lang = _detect_lang(markdown_text)
        doc_id = hashlib.md5(f"{path}|{len(markdown_text)}".encode()).hexdigest()

        # jiang
        para = _split_paragraphs_with_headings(markdown_text)

        token_chunks = _chunk_pargaraphs(
            paragraphs=para, chunk_tokens=max(1, chunk_size), overlap_tokens=max(0, chunk_overlap)
        )

        for ch in token_chunks:
            content: str = ch["content"]
            start = ch.get("start", 0)
            end = ch.get("end", 0)
            norm = content.strip()

            if not norm:
                continue

            content_hash = hashlib.md5(norm.encode("utf-8")).hexdigest()

            if content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)

            chunk_id = hashlib.md5(f"{doc_id}|{start}|{end}|{content_hash}".encode()).hexdigest()

            chunks.append(
                {
                    "id": chunk_id,
                    "content": content,
                    "metadata": {
                        "source_path": path,
                        "file_ext": ext,
                        "doc_id": doc_id,
                        "lang": lang,
                        "start": start,
                        "end": end,
                        "content_hash": content_hash,
                        "namespace": namespace or "default",
                        "source": source_label,
                        "external": True,
                        "heading_path": ch.get("heading_path"),
                        "format": "markdown",
                    },
                }
            )
    print(f"[RAG] Universal loader done: total_chunks={len(chunks)}")
    return chunks


def _convert_to_markdown(path: str) -> str:

    if not os.path.exists(path):
        return ""

    ext = (os.path.splitext(path)[1] or "").lower()

    md_instance = _get_markitdown_instance()

    if md_instance is None:
        return _fallback_text_reader()

    try:
        result = md_instance.convert(path)
        text = getattr(result, "text_content", None)
        if isinstance(text, str) and text.strip():
            return text
        return ""
    except Exception as e:
        print(f"[WARNING] MarkItDown failed for {path}: {e}")
        return _fallback_text_reader(path)


def _create_default_vector_store(dimension: int = None) -> QdrantVectorStore:

    if dimension is None:
        dimension = get_dimension(384)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # 使用连接管理器
    from ..storage.qdrant_store import QdrantConnectionManager

    return QdrantConnectionManager.get_instance(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name="hello_agents_rag_vectors",
        vector_size=dimension,
        distance="cosine",
    )


def _preprocess_markdown_for_embedding(text: str) -> str:
    """
    Preprocess markdown text for better embedding quality.
    Removes excessive markup while preserving semantic content.
    """
    import re

    # Remove markdown headers symbols but keep the text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove markdown links but keep the text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove markdown emphasis markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # italic
    text = re.sub(r"`([^`]+)`", r"\1", text)  # inline code

    # Remove markdown code blocks but keep content
    text = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)

    # Remove excessive whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def index_chunks(
    store=None,
    chunks: list[dict] = None,
    cache_db: str | None = None,
    batch_size: int = 64,
    rag_namespace: str = "default",
) -> None:
    """
    Index markdown chunks with unified embedding and Qdrant storage.
    Uses百炼 API with fallback to sentence-transformers.
    """
    if not chunks:
        print("[RAG] No chunks to index")
        return

    # Use unified embedding from embedding module
    embedder = get_text_embedder()
    dimension = get_dimension(384)

    # Create default Qdrant store if not provided
    if store is None:
        store = _create_default_vector_store(dimension)
        print(f"[RAG] Created default Qdrant store with dimension {dimension}")

    # Preprocess markdown texts for better embeddings
    processed_texts = []
    for c in chunks:
        raw_content = c["content"]
        processed_content = _preprocess_markdown_for_embedding(raw_content)
        processed_texts.append(processed_content)

    print(f"[RAG] Embedding start: total_texts={len(processed_texts)} batch_size={batch_size}")

    # Batch encoding with unified embedder
    vecs: list[list[float]] = []
    for i in range(0, len(processed_texts), batch_size):
        part = processed_texts[i : i + batch_size]
        try:
            # Use unified embedder directly (handles caching internally)
            part_vecs = embedder.encode(part)

            # Normalize to List[List[float]]
            if not isinstance(part_vecs, list):
                # 单个numpy数组转为列表中的列表
                if hasattr(part_vecs, "tolist"):
                    part_vecs = [part_vecs.tolist()]
                else:
                    part_vecs = [list(part_vecs)]
            else:
                # 检查是否是嵌套列表
                if part_vecs and not isinstance(part_vecs[0], (list, tuple)) and hasattr(part_vecs[0], "__len__"):
                    # numpy数组列表 -> 转换每个数组
                    normalized_vecs = []
                    for v in part_vecs:
                        if hasattr(v, "tolist"):
                            normalized_vecs.append(v.tolist())
                        else:
                            normalized_vecs.append(list(v))
                    part_vecs = normalized_vecs
                elif part_vecs and not isinstance(part_vecs[0], (list, tuple)):
                    # 单个向量被误判为列表，实际应该包装成[[...]]
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                    else:
                        part_vecs = [list(part_vecs)]

            for v in part_vecs:
                try:
                    # 确保向量是float列表
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    v_norm = [float(x) for x in v]
                    if len(v_norm) != dimension:
                        print(f"[WARNING] 向量维度异常: 期望{dimension}, 实际{len(v_norm)}")
                        # 用零向量填充或截断
                        if len(v_norm) < dimension:
                            v_norm.extend([0.0] * (dimension - len(v_norm)))
                        else:
                            v_norm = v_norm[:dimension]
                    vecs.append(v_norm)
                except Exception as e:
                    print(f"[WARNING] 向量转换失败: {e}, 使用零向量")
                    vecs.append([0.0] * dimension)

        except Exception as e:
            print(f"[WARNING] Batch {i} encoding failed: {e}")
            print(f"[RAG] Retrying batch {i} with smaller chunks...")

            # 尝试重试：将批次分解为更小的块
            success = False
            for j in range(0, len(part), 8):  # 更小的批次
                small_part = part[j : j + 8]
                try:
                    import time

                    time.sleep(2)  # 等待2秒避免频率限制

                    small_vecs = embedder.encode(small_part)
                    # Normalize to List[List[float]]
                    if isinstance(small_vecs, list) and small_vecs and not isinstance(small_vecs[0], list):
                        small_vecs = [small_vecs]

                    for v in small_vecs:
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        try:
                            v_norm = [float(x) for x in v]
                            if len(v_norm) != dimension:
                                print(f"[WARNING] 向量维度异常: 期望{dimension}, 实际{len(v_norm)}")
                                if len(v_norm) < dimension:
                                    v_norm.extend([0.0] * (dimension - len(v_norm)))
                                else:
                                    v_norm = v_norm[:dimension]
                            vecs.append(v_norm)
                            success = True
                        except Exception as e2:
                            print(f"[WARNING] 小批次向量转换失败: {e2}")
                            vecs.append([0.0] * dimension)
                except Exception as e2:
                    print(f"[WARNING] 小批次 {j // 8} 仍然失败: {e2}")
                    # 为这个小批次创建零向量
                    for _ in range(len(small_part)):
                        vecs.append([0.0] * dimension)

            if not success:
                print(f"[ERROR] 批次 {i} 完全失败，使用零向量")

        print(f"[RAG] Embedding progress: {min(i + batch_size, len(processed_texts))}/{len(processed_texts)}")

    # Prepare metadata with RAG tags
    metas: list[dict] = []
    ids: list[str] = []
    for ch in chunks:
        meta = {
            "memory_id": ch["id"],
            "user_id": "rag_user",
            "memory_type": "rag_chunk",
            "content": ch["content"],  # Keep original markdown content
            "data_source": "rag_pipeline",  # RAG identification tag
            "rag_namespace": rag_namespace,
            "is_rag_data": True,  # Clear RAG data marker
        }
        # Merge chunk metadata
        meta.update(ch.get("metadata", {}))
        metas.append(meta)
        ids.append(ch["id"])

    print(f"[RAG] Qdrant upsert start: n={len(vecs)}")
    success = store.add_vectors(vectors=vecs, metadata=metas, ids=ids)
    if success:
        print(f"[RAG] Qdrant upsert done: {len(vecs)} vectors indexed")
    else:
        print("[RAG] Qdrant upsert failed")
        raise RuntimeError("Failed to index vectors to Qdrant")


def embed_query(query: str) -> list[float]:
    embedder = get_text_embedder()
    dimension = get_dimension(384)

    try:
        vec = embedder.encode(query)

        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        # 处理嵌套列表情况
        if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
            vec = vec[0]  # Extract first vector if nested

        result = [float(x) for x in vec]

        if len(result) != dimension:
            print(f"[WARNING] Query向量维度异常: 期望{dimension}, 实际{len(result)}")
            # 用零向量填充或截断
            if len(result) < dimension:
                result.extend([0.0] * (dimension - len(result)))
            else:
                result = result[:dimension]

        return result
    except Exception as e:
        print(f"[WARNING] Query embedding failed: {e}")
        # Return zero vector as fallback
        return [0.0] * dimension


def search_vectors(
    store: QdrantVectorStore | None,
    query: str = "",
    top_k: int = 8,
    rag_namespace: str = None | None,
    only_rag_data: bool = True,
    score_threshold: float = None | None,
) -> list[dict]:
    """
    Search RAG vectors using unified embedding and Qdrant.
    """

    if not query:
        return []

    if store is None:
        store = _create_default_vector_store()

    qv = embed_query(query)

    where = {"memory_type": "rag_chunk"}

    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace

    try:
        return store.search_similar(query_vector=qv, limit=limit, score_threshold=score_threshold, where=where)
    except Exception as e:
        print(f"[WARNING] RAG search failed: {e}")
        return []


def _prompt_mqe(query: str, n: int) -> list[str]:
    try:
        llm = OpenAICompatibleClient(mode="qwen3.5:0.8b")
        prompt = [
            {"role": "user", "content": f"原始查询{query}\n 请给出{n}个不同表述的查询，但是语义需要相同，每行一个"}
        ]

        _, text = llm.generate(
            prompt,
            "你是检索查询扩展助手。生成语义等价或互补的多样化查询。必须和用户输入的语言一致，简短精炼，避免标点。",
        )

        lines = [ln.strip("- \t") for ln in (text or "").splitlines()]

        outs = [ln for ln in lines if ln]

        return outs[:n] or [query]
    except Exception:
        return [query]


def search_vectors_expanded(
    store: QdrantVectorStore = None,
    query: str = "",
    top_k: int = 8,
    rag_namespace: str | None = None,
    only_rag_data: bool = True,
    score_threshold: float | None = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
) -> list[dict]:
    """
    Search with query expansion using unified embedding and Qdrant.
    """

    if not query:
        return []

    if store is None:
        store = _create_default_vector_store()

    expansions = list[str] = [query]

    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(query, mqe_expansions))

    if enable_hyde:
        text = _prompt_hyde(query=query)
        if text:
            expansions.append(text)

    uniq: list[str] = []

    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)

    expansions = uniq[: max(1, len(uniq))]

    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, expansions))

    where = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace

    agg = dict[str, dict] = {}

    for q in expansions:
        qv = embed_query(q)

        hits = store.search_similar(query_vector=qv, limit=per, score_threshold=score_threshold, where=where)

        for h in hits:
            mid = h.get("metadata", {}).get("memory_id", h.get("id"))
            s = float(h.get("score", 0.0))

            if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                agg[mid] = h

    merged = list(agg.values())

    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    return merged[:top_k]


def _prompt_hyde(query: str) -> str | None:
    try:
        llm = OpenAICompatibleClient(mode="qwen3.5:0.8b")

        prompt = [{"role": "user", "content": f"问题：{query}\n 请直接写一段中等长度、客观、包含关键术语的段落。"}]

        _, text = llm.generate(
            messages=prompt,
            system_prompt="根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。",
        )

        return text
    except Exception:
        return None


def create_rag_pipeline(
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "hello_agents_rag_vectors",
    rag_namespace: str = "default",
) -> dict[str, Any]:
    """
    Create a complete RAG pipeline with Qdrant and unified embedding.

    Returns:
        Dict containing store, namespace, and helper functions
    """
    dimension = get_dimension(384)

    store = QdrantVectorStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collectoin_name=collection_name,
        vector_size=dimension,
        distance="cosine",
    )

    def add_documents(file_path: list[str], chunk_size: int = 800, chunk_overlap: int = 100) -> int:
        chunks = load_and_chunk_texts(
            paths=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=rag_namespace,
            source_label="rag",
        )

        index_chunks(chunks=chunks, store=store, rag_namespace=rag_namespace)

        return len(chunks)

    def search(query: str, top_k: int = 8, score_threshold: float = None | None):
        return search_vectors(
            store=store, query=query, top_k=top_k, rag_namespace=rag_namespace, score_threshold=score_threshold
        )

    def search_advanced(
        query: str,
        top_k: int = 8,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: float | None = None,
    ):
        """Advanced search with query expansion"""
        return search_vectors_expanded(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            enable_hyde=enable_hyde,
            enable_mqe=enable_mqe,
            score_threshold=score_threshold,
        )

    def get_stats():
        """Get pipeline statistics"""
        return store.get_collection_stats()

    return {
        "store": store,
        "namespace": rag_namespace,
        "add_documents": add_documents,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats,
    }
