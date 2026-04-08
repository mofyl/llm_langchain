import hashlib
import os
from ast import Nonlocal
from types import Any
from typing import Optional

from attr.validators import get_disabled
from sympy import EX

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
    store=None, chunks: list[dict] = None, cache_db: str = None, batch_size: int = 64, rag_namespace: str = "default"
):
    if not chunks:
        return

    embedder = get_text_embedder()
    dimension = get_dimension(384)

    if store is None:
        store = _create_default_vector_store(dimension=dimension)

    processed_texts = []

    for c in chunks:
        raw_content = c["content"]

        processed_content = _preprocess_markdown_for_embedding(raw_content)
        processed_texts.append(processed_content)


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

    def add_documents(file_path: list[str], chunk_size: int = 800, chunk_overlap: int = 100):
        chunks = load_and_chunk_texts(
            paths=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=rag_namespace,
            source_label="rag",
        )

    return {
        "store": store,
        "namespace": rag_namespace,
    }
