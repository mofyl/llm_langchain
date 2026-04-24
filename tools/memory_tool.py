from datetime import datetime
from typing import Any

from memory import MemoryConfig, MemoryType
from memory.manager import MemoryManager

from .base import Tool, ToolParameter, tool_action


class MemoryTool(Tool):
    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig | None = None,
        memory_type: list[MemoryType] | None = None,
        memoty_type: list[MemoryType] | None = None,
        expandable: bool = False,
    ):
        super().__init__(name="memory", desc="记忆工具 - 可以存储和检索对话历史、知识和经验", expandable=expandable)

        self.memory_config = memory_config or MemoryConfig()
        selected_memory_types = memory_type if memory_type is not None else memoty_type
        self.memory_types = selected_memory_types or [MemoryType.WORKINGMEMORY]

        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working=MemoryType.WORKINGMEMORY in self.memory_types,
            enable_episodic=False,
            enable_semantic=False,
            enable_perceptual=False,
        )

        self.current_session_id = None
        self.conversation_count = 0

    def run(self, param: dict[str, Any]) -> str:
        if not self.validate_parameters(parameters=param):
            return "参数验证失败"

        action = param.get("action")

        if action == "add":
            return self._add_memory(
                content=param.get("content", ""),
                memory_type=param.get("memory_type", "working"),
                importance=param.get("importance", 0.5),
                file_path=param.get("file_path"),
                modality=param.get("modality"),
            )
        elif action == "search":
            return self._search_memory(
                query=param.get("query"),
                limit=param.get("limit", 5),
                memory_type=param.get("memory_type"),
                min_importance=param.get("min_importance", 0.1),
            )
        elif action == "summary":
            return self._get_summary(limit=param.get("limit", 10))
        else:
            return f"❌ 不支持的操作: {action}"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "要执行的操作："
                    "add(添加记忆), search(搜索记忆), summary(获取摘要), stats(获取统计), "
                    "update(更新记忆), remove(删除记忆), forget(遗忘记忆), consolidate(整合记忆), clear_all(清空所有记忆)"
                ),
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="记忆内容（add/update时可用；感知记忆可作描述）",
                required=False,
            ),
            ToolParameter(name="query", type="string", description="搜索查询（search时可用）", required=False),
            ToolParameter(
                name="memory_type",
                type="string",
                description="记忆类型：working, episodic, semantic, perceptual（默认：working）",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="importance", type="number", description="重要性分数，0.0-1.0（add/update时可用）", required=False
            ),
            ToolParameter(
                name="limit", type="integer", description="搜索结果数量限制（默认：5）", required=False, default=5
            ),
            ToolParameter(
                name="memory_id", type="string", description="目标记忆ID（update/remove时必需）", required=False
            ),
            ToolParameter(
                name="file_path", type="string", description="感知记忆：本地文件路径（image/audio）", required=False
            ),
            ToolParameter(
                name="modality",
                type="string",
                description="感知记忆模态：text/image/audio（不传则按扩展名推断）",
                required=False,
            ),
            ToolParameter(
                name="strategy",
                type="string",
                description="遗忘策略：importance_based/time_based/capacity_based（forget时可用）",
                required=False,
                default="importance_based",
            ),
            ToolParameter(
                name="threshold",
                type="number",
                description="遗忘阈值（forget时可用，默认0.1）",
                required=False,
                default=0.1,
            ),
            ToolParameter(
                name="max_age_days",
                type="integer",
                description="最大保留天数（forget策略为time_based时可用）",
                required=False,
                default=30,
            ),
            ToolParameter(
                name="from_type",
                type="string",
                description="整合来源类型（consolidate时可用，默认working）",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="to_type",
                type="string",
                description="整合目标类型（consolidate时可用，默认episodic）",
                required=False,
                default="episodic",
            ),
            ToolParameter(
                name="importance_threshold",
                type="number",
                description="整合重要性阈值（默认0.7）",
                required=False,
                default=0.7,
            ),
        ]

    @tool_action("memory_add", "添加新的记忆到记忆系统中")
    def _add_memory(
        self,
        content: str = "",
        memory_type: MemoryType = MemoryType.WORKINGMEMORY,
        importance: float = 0.5,
        file_path: str | None = None,
        modality: str | None = None,
    ):
        """添加记忆

        Args:
            content: 记忆内容
            memory_type: 记忆类型：working(工作记忆), episodic(情景记忆), semantic(语义记忆), perceptual(感知记忆)
            importance: 重要性分数，0.0-1.0
            file_path: 感知记忆：本地文件路径（image/audio）
            modality: 感知记忆模态：text/image/audio（不传则按扩展名推断）

        Returns:
            执行结果
        """
        metadata = {}

        try:
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 感知记忆文件支持：注入 raw_data 与模态
            # if memory_type == "perceptual" and file_path:
            #     inferred = modality or self._infer_modality(file_path)
            #     metadata.setdefault("modality", inferred)
            #     metadata.setdefault("raw_data", file_path)

            metadata.update({"session_id": self.current_session_id, "timestamp": datetime.now().isoformat()})

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata,
                auto_classify=False,
            )

            return f"记忆已经添加. ID:{memory_id}"
        except Exception as e:
            return f"❌ 添加记忆失败: {str(e)}"

    @tool_action("memory_search", "搜索相关记忆")
    def _search_memory(
        self, query: str, limit: int = 5, memory_type: MemoryType | None = None, min_importance: float = 0.1
    ) -> str:
        """搜索记忆

        Args:
            query: 搜索查询内容
            limit: 搜索结果数量限制
            memory_type: 限定记忆类型：working/episodic/semantic/perceptual
            min_importance: 最低重要性阈值

        Returns:
            搜索结果
        """
        try:
            memory_types = [memory_type] if memory_type else None

            result = self.memory_manager.retrieve_memories(
                query=query, limit=limit, memory_types=memory_types, min_importance=min_importance
            )

            if not result:
                return f"未找到与 '{query}' 相关的记忆"

            formatted_results = []

            formatted_results.append(f"找到 {len(result)} 条相关记忆")

            for i, memory in enumerate(result, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆",
                }.get(memory.memory_type, memory.memory_type)

                content_preview = memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
                formatted_results.append(
                    f"{i}. [{memory_type_label}] {content_preview} (重要性: {memory.importance:.2f})"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"搜索记忆失败: {str(e)}"

    @tool_action("memory_summary", "获取记忆系统摘要（包含重要记忆和统计信息）")
    def _get_summary(self, limit: int = 10) -> str:
        """获取记忆摘要

        Args:
            limit: 显示的重要记忆数量

        Returns:
            记忆摘要
        """
        try:
            stats = self.memory_manager.get_memory_stats()

            summary_parts = [
                "📊 记忆系统摘要",
                f"总记忆数: {stats['total_memories']}",
                f"当前会话: {self.current_session_id or '未开始'}",
                f"对话轮次: {self.conversation_count}",
            ]

            # 各类型记忆统计
            if stats["memories_by_type"]:
                summary_parts.append("\n📋 记忆类型分布:")
                for memory_type, type_stats in stats["memories_by_type"].items():
                    count = type_stats.get("count", 0)
                    avg_importance = type_stats.get("avg_importance", 0)
                    type_label = {
                        "working": "工作记忆",
                        "episodic": "情景记忆",
                        "semantic": "语义记忆",
                        "perceptual": "感知记忆",
                    }.get(memory_type, memory_type)

                    summary_parts.append(f"  • {type_label}: {count} 条 (平均重要性: {avg_importance:.2f})")

            # 获取重要记忆 - 修复重复问题
            important_memories = self.memory_manager.retrieve_memories(
                query="",
                memory_types=None,  # 从所有类型中检索
                limit=limit * 3,  # 获取更多候选，然后去重
                min_importance=0.5,  # 降低阈值以获取更多记忆
            )

            if important_memories:
                # 去重：使用记忆ID和内容双重去重
                seen_ids = set()
                seen_contents = set()
                unique_memories = []

                for memory in important_memories:
                    # 使用ID去重
                    if memory.id in seen_ids:
                        continue

                    # 使用内容去重（防止相同内容的不同记忆）
                    content_key = memory.content.strip().lower()
                    if content_key in seen_contents:
                        continue

                    seen_ids.add(memory.id)
                    seen_contents.add(content_key)
                    unique_memories.append(memory)

                # 按重要性排序
                unique_memories.sort(key=lambda x: x.importance, reverse=True)
                summary_parts.append(f"\n⭐ 重要记忆 (前{min(limit, len(unique_memories))}条):")

                for i, memory in enumerate(unique_memories[:limit], 1):
                    content_preview = memory.content[:60] + "..." if len(memory.content) > 60 else memory.content
                    summary_parts.append(f"  {i}. {content_preview} (重要性: {memory.importance:.2f})")

            return "\n".join(summary_parts)

        except Exception as e:
            return f"❌ 获取摘要失败: {str(e)}"
