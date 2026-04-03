from types import Any

from ..embedding import get_dimension, get_text_embedder


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

    store = 
