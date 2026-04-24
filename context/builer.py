import asyncio
import math
import select
from dataclasses import dataclass, field
from datetime import datetime
from struct import pack
from typing import Any, List

import tiktoken
from anyio.lowlevel import current_token
from click import Context
from sympy import EX

import context
from chapter1.open_ai_provider import OpenAICompatibleClient
from core.message import Message
from tools.memory_tool import MemoryTool
from tools.rag_tool import RAGTool


@dataclass
class ContextPacket:
    """上下文信息包"""

    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0  # 0.0 - 1.0

    def __post_init__(self):
        """自动计算token数"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """上下文构建配置"""

    max_tokens: int = 8000  # 总预算
    reserve_ratio: float = 0.15  # 生成余量（10-20%）
    min_relevance: float = 0.3  # 最小相关性阈值
    enable_mmr: bool = True  # 启用最大边际相关性（多样性）
    mmr_lambda: float = 0.7  # MMR平衡参数（0=纯多样性, 1=纯相关性）
    system_prompt_template: str = ""  # 系统提示模板
    enable_compression: bool = True  # 启用压缩

    def get_avaliable_tokens(self) -> int:
        """获取可用token预算（扣除余量）"""
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """上下文构建器 - GSSC流水线

    用法示例：
    ```python
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )

    context = builder.build(
        user_query="用户问题",
        conversation_history=[...],
        system_instructions="系统指令"
    )
    ```
    """

    def __init__(
        self,
        memory_tool: MemoryTool | None = None,
        rag_tool: RAGTool | None = None,
        config: ContextConfig | None = None,
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self.llm = OpenAICompatibleClient(mode="qwen3.5:0.8b")

    def build(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        additional_packets: list[ContextPacket] | None = None,
    ):
        """构建完整上下文

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            system_instructions: 系统指令
            additional_packets: 额外的上下文包

        Returns:
            结构化上下文字符串
        """
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or [],
        )

        selected_packets = self._select(packets=packets, user_query=user_query)

        structured_context = self._structure(
            selected_packets=selected_packets, user_query=user_query, system_instructions=system_instructions
        )

    def _gather(
        self,
        user_query: str,
        conversation_history: list[Message],
        system_instructions: str | None,
        additional_packets: list[ContextPacket],
    ) -> list[ContextPacket]:
        """Gather: 收集候选信息"""
        packets = []
        # P0: 系统指令（强约束）
        if system_instructions:
            packets.append(ContextPacket(content=system_instructions, metadata={"type": "instructions"}))

        # P1: 从记忆中获取任务状态与关键结论
        if self.memory_tool:
            try:
                state_results = self.memory_tool.run(
                    param={
                        "action": "search",
                        "query": "(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                        "min_importance": 0.7,
                        "limit": 5,
                    }
                )

                if state_results and "未找到" not in state_results:
                    packets.append(
                        ContextPacket(content=state_results, metadata={"type": "task_state", "importance": "hight"})
                    )
                # 搜索与当前查询相关的记忆
                related_results = self.memory_tool.run(
                    param={
                        "action": "search",
                        "query": user_query,
                        "limit": 5,
                    }
                )
                if related_results and "未找到" not in related_results:
                    packets.append(ContextPacket(content=related_results, metadata={"type": "related_memory"}))
            except Exception as e:
                print(f"⚠️ 记忆检索失败: {e}")
        # P2: 从RAG中获取事实证据
        if self.rag_tool:
            try:
                rag_result = self.rag_tool.run({"action": "search", "query": user_query, "limit": 5})
                if rag_result and "未找到" not in rag_result:
                    packets.append(ContextPacket(content=rag_result, metadata={"type": "knowledge_base"}))
            except Exception as e:
                print(f"⚠️ RAG检索失败: {e}")
        # P3: 对话历史（辅助材料）
        if conversation_history:
            # 只保留最近N条
            recent_history = conversation_history[-10:]
            history_text = "\n".join([f"[{msg.role}] {msg.content}" for msg in recent_history])
            packets.append(
                ContextPacket(content=history_text, metadata={"type": "history", "count": len(recent_history)})
            )
        packets.extend(additional_packets)

        return packets

    def _select(self, packets: list[ContextPacket], user_query: str) -> list[ContextPacket]:
        """Select: 基于分数与预算的筛选"""

        # 1) 计算相关性（关键词重叠）
        query_tokens = set(user_query.lower().split())
        for packet in packets:
            content_tokens = set(packet.content.lower().split())

            if len(query_tokens):
                overlap = len(query_tokens & content_tokens)
                packet.relevance_score = overlap / len(query_tokens)
            else:
                packet.relevance_score = 0.0

        # 2) 计算新近性（指数衰减）
        def recency_socre(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600
            return math.exp(-delta / tau)

        score_packet: list[tuple[float, ContextPacket]] = []

        for p in packets:
            rec = recency_socre(p.timestamp)
            score = 0.7 * p.relevance_score + 0.3 * rec
            score_packet.append((score, p))

        system_packets = [p for (_, p) in score_packet if p.metadata.get("type") == "instructions"]

        remaining = [
            p
            for (_, p) in sorted(score_packet, key=lambda x: x[0], reverse=True)
            if p.metadata.get("type") != "instructions"
        ]

        filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]

        available_tokens = self.config.get_avaliable_tokens()
        selected: list[ContextPacket] = []
        used_tokens = 0

        for p in system_packets:
            if used_tokens + p.token_count <= available_tokens:
                selected.append(p)
                used_tokens += p.token_count

        # 再按分数加入其余
        for p in filtered:
            if used_tokens + p.token_count > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token_count

        return selected

    def _structure(
        self, selected_packets: list[ContextPacket], user_query: str, system_instructions: str | None = None
    ) -> str:
        """Structure: 组织成结构化上下文模板"""
        sections = []

        # [Role & Policies] - 系统指令
        p0_packets = [p for p in selected_packets if p.metadata.get("type") == "instructions"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)
        # [Task] - 当前任务
        sections.append(f"[Task]\n用户问题：{user_query}")

        # [State] - 任务状态
        p1_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if p1_packets:
            state_section = "[State]\n关键进展与未决问题：\n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)

        # [Evidence] - 事实证据
        p2_packets = [
            p
            for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\n事实与引用：\n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)

        # [Context] - 辅助材料（历史等）
        p3_packets = [p for p in selected_packets if p.metadata.get("type") == "history"]

        if p3_packets:
            context_section = "[Context]\n对话历史与背景：\n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)

        output_section = """[Output]
                            请按以下格式回答：
                            1. 结论（简洁明确）
                            2. 依据（列出支撑证据及来源）
                            3. 风险与假设（如有）
                            4. 下一步行动建议(如适用)"""
        sections.append(output_section)
        return "\n\n".join(sections)

    def _compress(self, context: str) -> str:
        """Compress: 压缩与规范化"""
        if not self.config.enable_compression:
            return context

        current_tokens = count_tokens(context)
        available_tokens = self.config.get_avaliable_tokens()

        if current_tokens <= available_tokens:
            return context
        result = context
        print(f"⚠️ 上下文超预算 ({current_tokens} > {available_tokens})，执行截断")
        system_prompt = (
            "你是一个专门用于文本总结的智能助手。你的任务是在不破坏原文结构的前提下，对给定内容进行总结。"
            "请严格遵守以下规则："
            "1.保留原有结构：你应当按照原文的段落、章节、条目或逻辑层次进行总结。如果原文有条目（如1. 2. 3. 或 - 项目符号），请在总结中保持相同条目结构。"
            "2.逐段/逐节提炼：对每一段或每一小节分别进行总结，而不是将全文内容混在一起概括。每个部分总结后，尽量保持与原文顺序一致。"
            "3.不增删结构化元素：不要合并段落、不要移除或新增小标题、不要打乱条目顺序。不要将多个条目合并成一段文字。"
            "4.语言精简但信息完整：在保持结构的前提下，压缩冗余表达，保留关键信息、核心观点、数据或结论。"
            "5.输出格式与原文对齐：如果原文使用 Markdown、编号、列表等方式组织内容，请在总结中也使用相同的格式。"
            "6.禁止重写结构：不要将原文改写成一篇全新的文章，也不要添加原文没有的结构（如“总结如下：”之类的前言）。"
            f"7.输出总 token 数不超过 {available_tokens}"
        )

        for i in range(4):
            _, result = asyncio.run(self.llm.generate(messages=[{"user": result}], system_prompt=system_prompt))
            if result is None:
                return context
            cur_tokens = count_tokens(result)

            if cur_tokens <= available_tokens:
                return result
        return context


def count_tokens(text: str) -> int:
    """计算文本token数（使用tiktoken）"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text=text))
    except Exception:
        # 降级方案：粗略估算（1 token ≈ 4 字符）
        return len(text) // 4
