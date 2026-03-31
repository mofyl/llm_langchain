import datetime

from ..memory import MemoryConfig, MemoryType
from ..memory.manager import MemoryManager
from .base import Tool, tool_action


class MemoryTool(Tool):
    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig = None,
        memoty_type: list[MemoryType] = None,
        expandable: bool = False,
    ):
        super().__init__(name="memory", desc="记忆工具 - 可以存储和检索对话历史、知识和经验", expandable=expandable)

        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memoty_type or [MemoryType.WORKINGMEMORY]

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

    def run(self, parameters: dict[str, any]) -> str:
        if not self.validate_parameters(parameters=parameters):
            return "参数验证失败"

        action = parameters.get("action")

        if action == "add":
            return self._add_memory(
                content=parameters.get("content", ""),
                memory_type=parameters.get("memory_type", "working"),
                importance=parameters.get("importance", 0.5),
                file_path=parameters.get("file_path"),
                modality=parameters.get("modality"),
            )
        # if action == "search" :

    @tool_action("memory_add", "添加新的记忆到记忆系统中")
    def _add_memory(
        self,
        content: str = "",
        memory_type: MemoryType = MemoryType.WORKINGMEMORY,
        importance: float = 0.5,
        file_path: str = None,
        modality: str = None,
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
        self, query: str, limit: int = 5, memory_type: MemoryType = None, min_importance: float = 0.1
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
