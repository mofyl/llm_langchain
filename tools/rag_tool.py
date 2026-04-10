import os
import time
from types import Any
from typing import Optional

from openai import file_from_path

from chapter1.open_ai_provider import OpenAICompatibleClient
from memory.rag.pipline import create_rag_pipeline

from .base import Tool, ToolParameter, tool_action


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

        os.makedirs(self.knowledge_base_path, exist_ok=True)

    def _init_components(self):

        try:
            default_pipline = create_rag_pipeline(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                rag_namespace=self.rag_namespace,
            )
            self._piplines[self.rag_namespace] = default_pipline

            self.llm = OpenAICompatibleClient(mode="qwen3.5:0.8b")

            self.initialized = True
            print(f"✅ RAG工具初始化成功: namespace={self.rag_namespace}, collection={self.collection_name}")

        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"❌ RAG工具初始化失败: {e}")

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：add_document(添加文档), add_text(添加文本), ask(智能问答), search(搜索), stats(统计), clear(清空)",
                required=True,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="文档文件路径（支持PDF、Word、Excel、PPT、图片、音频等多种格式）",
                required=False,
            ),
            ToolParameter(
                name="text",
                type="string",
                description="要添加的文本内容",
                required=False,
            ),
            ToolParameter(name="question", type="string", description="用户问题（用于智能问答）", required=False),
            ToolParameter(name="query", type="string", description="搜索查询词（用于基础搜索）", required=False),
            # 可选配置参数
            ToolParameter(
                name="namespace",
                type="string",
                description="知识库命名空间（用于隔离不同项目，默认：default）",
                required=False,
                default="default",
            ),
            ToolParameter(
                name="limit", type="integer", description="返回结果数量（默认：5）", required=False, default=5
            ),
            ToolParameter(
                name="include_citations",
                type="boolean",
                description="是否包含引用来源（默认：true）",
                required=False,
                default=True,
            ),
        ]

    def _get_pipline(self, namespace: str | None = None) -> dict[str, Any]:
        """获取指定命名空间的 RAG 管道，若不存在则自动创建"""
        target_ns = namespace or self.rag_namespace

        if target_ns in self._piplines:
            return self._piplines[target_ns]

        pipline = create_rag_pipeline(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            rag_namespace=target_ns,
        )

        self._piplines[target_ns] = pipline
        return pipline

    @tool_action("rag_add_document", "添加文档到知识库（支持PDF、Word）")
    def _add_document(
        self,
        file_path: str,
        document_id: str = None,
        namespace: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> str:
        """添加文档到知识库

        Args:
            file_path: 文档文件路径
            document_id: 文档ID（可选）
            namespace: 知识库命名空间（用于隔离不同项目）
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            执行结果
        """

        try:
            if not file_path or not os.path.exists(file_path):
                return f"❌ 文件不存在: {file_path}"

            pipeline = self._get_pipline()

            t0 = time.time()

            chunks_added = pipeline["add_documents"](
                file_paths=[file_path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            t1 = time.time()

            process_ms = int((t0 - t1) * 1000)

            if chunks_added == 0:
                return f"⚠️ 未能从文件解析内容: {os.path.basename(file_path)}"

            return (
                f"✅ 文档已添加到知识库: {os.path.basename(file_path)}\n"
                f"📊 分块数量: {chunks_added}\n"
                f"⏱️ 处理时间: {process_ms}ms\n"
                f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}"
            )

        except Exception as e:
            return f"❌ 添加文档失败: {str(e)}"

    @tool_action("rag_add_text", "添加文本到知识库")
    def _add_text(
        self,
        text: str,
        document_id: str = None,
        namespace: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> str:
        """添加文本到知识库

        Args:
            text: 要添加的文本内容
            document_id: 文档ID（可选）
            namespace: 知识库命名空间
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            执行结果
        """

        try:
            if not text or not text.strip():
                return "❌ 文本内容不能为空"

            document_id = document_id or f"text_{abs(hash(text)) % 100000}"

            tmp_path = os.path.join(self.knowledge_base_path, f"{document_id}.md")

            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(text)

                return self._add_document(
                    file_path=tmp_path,
                    document_id=document_id,
                    namespace=namespace,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            return f"❌ 添加文本失败: {str(e)}"

    @tool_action("rag_search", "搜索知识库中的相关内容")
    def _search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
        enable_advanced_search: bool = True,
        max_chars: int = 1200,
        include_citations: bool = True,
        namespace: str = "default",
    ) -> str:
        """搜索知识库

        Args:
            query: 搜索查询词
            limit: 返回结果数量
            min_score: 最低相关度分数
            enable_advanced_search: 是否启用高级搜索（MQE、HyDE）
            max_chars: 每个结果最大字符数
            include_citations: 是否包含引用来源
            namespace: 知识库命名空间

        Returns:
            搜索结果
        """

        try:
            if not query or not query.strip():
                return "❌ 搜索查询不能为空"

            pipeline = self._get_pipline(namespace=namespace)

            if enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=query,
                    top_k=limit,
                    enable_mqe=True,
                    enable_hyde=True,
                    score_threshold=min_score if min_score > 0 else None,
                )
            else:
                results = pipeline["search"](
                    query=query, top_k=limit, score_threshold=min_score if min_score > 0 else None
                )
            if not results:
                return f"🔍 未找到与 '{query}' 相关的内容"

            search_result = ["搜索结果："]

            for i, result in enumerate(results):
                meta = result.get("metadata", {})
                score = result.get("score", 0.0)
                content = meta.get("content", "")[:200] + "..."
                source = meta.get("source_path", "unknown")

                def clean_text(text):
                    try:
                        return str(text).encode("utf-8", errors="ignore").decode("utf-8")
                    except Exception:
                        return str(text)

                clean_content = clean_text(content)

                clean_source = clean_text(source)

                search_result.append(f"\n{i}. 文档: **{clean_source}** (相似度: {score:.3f})")
                search_result.append(f"   {clean_content}")

                if include_citations and meta.get("heading_path"):
                    clean_heading = clean_text(str(meta.get("heading_path")))

                    search_result.append(f"   章节: {clean_heading}")

            return "\n".join(search_result)

        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"

    def _clean_content_for_context(self, content: str) -> str:
        """清理内容用于上下文"""
        # 移除过多的换行和空格
        content = " ".join(content.split())
        # 截断过长内容
        if len(content) > 300:
            content = content[:300] + "..."
        return content

    def _smart_truncate_context(self, context: str, max_chars: int) -> str:
        """智能截断上下文，保持段落完整性"""
        if len(context) <= max_chars:
            return context

        # 寻找最近的段落分隔符
        truncated = context[:max_chars]
        last_break = truncated.rfind("\n\n")

        if last_break > max_chars * 0.7:  # 如果断点位置合理
            return truncated[:last_break] + "\n\n[...更多内容被截断]"
        else:
            return truncated[: max_chars - 20] + "...[内容被截断]"

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return (
            "你是一个专业的知识助手，具备以下能力：\n"
            "1. 📖 精准理解：仔细理解用户问题的核心意图\n"
            "2. 🎯 可信回答：严格基于提供的上下文信息回答，不编造内容\n"
            "3. 🔍 信息整合：从多个片段中提取关键信息，形成完整答案\n"
            "4. 💡 清晰表达：用简洁明了的语言回答，适当使用结构化格式\n"
            "5. 🚫 诚实表达：如果上下文不足以回答问题，请坦诚说明\n\n"
            "回答格式要求：\n"
            "• 直接回答核心问题\n"
            "• 必要时使用要点或步骤\n"
            "• 引用关键原文时使用引号\n"
            "• 避免重复和冗余"
        )

    def _build_user_prompt(self, question: str, context: str) -> str:
        """构建用户提示词"""
        return (
            f"请基于以下上下文信息回答问题：\n\n"
            f"【问题】{question}\n\n"
            f"【相关上下文】\n{context}\n\n"
            f"【要求】请提供准确、有帮助的回答。如果上下文信息不足，请说明需要什么额外信息。"
        )

    @tool_action("rag_ask", "基于知识库进行智能问答")
    def _ask(
        self,
        question: str,
        limit: int = 5,
        enable_advanced_search: bool = True,
        include_citations: bool = True,
        max_chars: int = 1200,
        namespace: str = "default",
    ) -> str:
        """智能问答：检索 → 上下文注入 → LLM生成答案

        Args:
            question: 用户问题
            limit: 检索结果数量
            enable_advanced_search: 是否启用高级搜索
            include_citations: 是否包含引用来源
            max_chars: 每个结果最大字符数
            namespace: 知识库命名空间

        Returns:
            智能问答结果

        核心流程:
        1. 解析用户问题
        2. 智能检索相关内容
        3. 构建上下文和提示词
        4. LLM生成准确答案
        5. 添加引用来源
        """
        try:
            if not question or not question.strip():
                return "❌ 请提供要询问的问题"

            user_question = question.strip()
            print(f"🔍 智能问答: {user_question}")

            pipeline = self._get_pipline(namespace=namespace)
            search_start = time.time()

            if enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=user_question,
                    top_k=limit,
                    enable_mqe=True,
                    enable_hyde=True,
                )
            else:
                results = pipeline["search"](query=user_question, top_k=limit)

            search_time = int((time.time() - search_start) * 1000)

            if not results:
                return (
                    f"🤔 抱歉，我在知识库中没有找到与「{user_question}」相关的信息。\n\n"
                    f"💡 建议：\n"
                    f"• 尝试使用更简洁的关键词\n"
                    f"• 检查是否已添加相关文档\n"
                    f"• 使用 stats 操作查看知识库状态"
                )

            context_parts = []

            citations = []
            total_score = 0

            for i, result in enumerate(results):
                meta = result.get("metadata", {})
                content = meta.get("content", "").strip()
                source = meta.get("source_path", "unknown")
                score = result.get("score", 0.0)
                total_score += score

                if content:
                    cleaned_content = self._clean_content_for_context(content=content)
                    context_parts.append(f"片段 {i + 1}：{cleaned_content}")

                    if include_citations:
                        citations.append({"index": i + 1, "source": os.path.basename(source), "score": score})

            context = "\n\n".join(context_parts)

            if len(context) > max_chars:
                context = self._smart_truncate_context(context, max_chars)

            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(user_question, context=content)

            # enhanced_prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            llm_start = time.time()
            answer = self.llm.generate([user_prompt], system_prompt)
            llm_time = int((time.time() - llm_start) * 1000)

            if not answer or not answer.strip():
                return "❌ LLM未能生成有效答案，请稍后重试"

            return self._format_final_answer(
                question=question,
                answer=answer,
                citations=citations,
                search_time=search_time,
                llm_time=llm_time,
                avg_score=total_score / len(results) if results else 0,
            )

        except Exception as e:
            return f"❌ 智能问答失败: {str(e)}\n💡 请检查知识库状态或稍后重试"

    def _format_final_answer(
        self,
        question: str,
        answer: str,
        citations: list[dict] | None = None,
        search_time: int = 0,
        llm_time: int = 0,
        avg_score: float = 0,
    ) -> str:
        """格式化最终答案"""
        result = ["🤖 **智能问答结果**\n"]

        result.append(answer)

        if citations:
            result.append("\n\n📚 **参考来源**")
            for citation in citations:
                score_emoji = "🟢" if citation["score"] > 0.8 else "🟡" if citation["score"] > 0.6 else "🔵"
                result.append(
                    f"{score_emoji} [{citation['index']}] {citation['source']} (相似度: {citation['score']:.3f})"
                )
        # 添加性能信息（调试模式）
        result.append(f"\n⚡ 检索: {search_time}ms | 生成: {llm_time}ms | 平均相似度: {avg_score:.3f}")

        return "\n".join(result)

    def run(self, parameters: dict[str, Any]) -> str:
        """执行工具（非展开模式）

        Args:
            parameters: 工具参数字典，必须包含action参数

        Returns:
            执行结果字符串
        """
        if not self.validate_parameters(parameters=parameters):
            return "❌ 参数验证失败：缺少必需的参数"

        if not self.initialized:
            return f"❌ RAG工具未正确初始化，请检查配置: {getattr(self, 'init_error', '未知错误')}"

        action = parameters.get("action")

        try:
            if action == "add_document":
                self._add_document(
                    file_path=parameters.get("file_path"),
                    document_id=parameters.get("document_id"),
                    namespace=parameters.get("namespace", "default"),
                    chunk_size=parameters.get("chunk_size", 800),
                    chunk_over_lap=parameters.get("chunk_over_lap", 100),
                )
            elif action == "add_text":
                return self._add_text(
                    text=parameters.get("text"),
                    document_id=parameters.get("document_id"),
                    namespace=parameters.get("namespace", "default"),
                    chunk_size=parameters.get("chunk_size", 800),
                    chunk_overlap=parameters.get("chunk_overlap", 100),
                )
            elif action == "ask":
                question = parameters.get("question") or parameters.get("query")
                return self._ask(
                    question=question,
                    limit=parameters.get("limit", 5),
                    enable_advanced_search=parameters.get("enable_advanced_search", True),
                    include_citations=parameters.get("include_citations", True),
                    max_chars=parameters.get("max_chars", 1200),
                    namespace=parameters.get("namespace", "default"),
                )
            elif action == "search":
                return self._search(
                    query=parameters.get("query") or parameters.get("question"),
                    limit=parameters.get("limit", 5),
                    min_score=parameters.get("min_score", 0.1),
                    enable_advanced_search=parameters.get("enable_advanced_search", True),
                    max_chars=parameters.get("max_chars", 1200),
                    include_citations=parameters.get("include_citations", True),
                    namespace=parameters.get("namespace", "default"),
                )
            elif action == "stats":
                return self._get_stats(namespace=parameters.get("namespace", "default"))
            else:
                return f"❌ 不支持的操作: {action}"
        except Exception as e:
            return f"❌ 执行操作 '{action}' 时发生错误: {str(e)}"
