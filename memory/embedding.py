import os
import threading
from typing import Optional, Union

import ollama


class EmbeddingModel:
    def encode(self, texts: str | list[str]):
        raise NotImplementedError

    @property
    def demension(self) -> int:
        raise NotImplementedError


class OllamaEmbedding(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._ollama_url = "http://127.0.0.1:11434"
        self._demension = 0

    def encode(self, texts: str | list[str]):
        resp = ollama.embed(self.model_name, texts)
        self._demension = len(resp.embeddings)
        return resp.embeddings

    def demension(self) -> int:
        return self.demension


def create_embedding_model(model_type: str = "local", **kwargs) -> EmbeddingModel:
    """创建嵌入模型实例

    model_type: "dashscope" | "local" | "tfidf"
    kwargs: model_name, api_key
    """
    if model_type in ("local", "sentence_transformer", "huggingface"):
        return OllamaEmbedding(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# def create_embedding_model_with_fallback(preferred_type : str = "dashscope")


_embeder: EmbeddingModel | None = None
_lock = threading.Rlock()


def _build_embedder() -> EmbeddingModel:
    default_model = "text-embedding-v3"

    model_name = os.getenv("EMBED_MODEL_NAME", default_model)

    kwargs = {}

    if model_name:
        kwargs["model_name"] = model_name
    api_key = os.geten("EMBED_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    base_url = os.getenv("EMBED_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return create_embedding_model("local", kwargs)


def get_text_embedder() -> EmbeddingModel:
    global _embeder
    if _embeder is not None:
        return _embeder
    with _lock:
        if _embeder is None:
            _embeder = _build_embedder()
        return _embeder


def get_dimension(default: int = 384) -> int:
    get_text_embedder()

    return default
