import os
from types import Any

from .base import Tool


class RAGTool(Tool):
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        expandable: bool = True,
    ):
        super().__init__(
            name="rag", description="RAG工具 - 支持多格式文档检索增强生成，提供智能问答能力", expandable=expandable
        )

        self.knowledge_base_path = knowledge_base_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        self._piplines: dict[str, dict[str, Any]] = {}

        os.makedirs(self.knowledge_base_path , exist_ok=True)

    
    def _init_components(self) : 

        try : 
            
