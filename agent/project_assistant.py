from datetime import datetime
from typing import Any

from agent.simple_agent import SimpleAgent
from context.builer import ContextBuilder, ContextConfig, ContextPacket
from core.llm import HelloAgentsLLM
from core.message import Message, RoleType
from memory import rag
from tools.memory_tool import MemoryTool
from tools.note_tool import NoteTool
from tools.rag_tool import RAGTool


class ProjectAssistant(SimpleAgent):
    def __init__(self, name: str, project: str, **kwargs):
        super().__init__(name=name, llm=HelloAgentsLLM(), **kwargs)

        self.project_name = project

        self.memory_tool = MemoryTool(user_id=project)
        self.rag_tool = RAGTool(knowledge_base_path=f"./{project}_kb")
        self.note_tool = NoteTool(workspace=f"./{project}_notes")

        self.context_builder = ContextBuilder(
            memory_tool=self.memory_tool, rag_tool=self.rag_tool, config=ContextConfig(max_tokens=4000)
        )

        self.conversation_history = []

    def _retrieve_relevant_notes(self, user_input: str, limit: int = 3) -> list[dict]:
        try:
            blockers = self.note_tool.list_notes(
                note_type="blocker",
                limit=2,
            )
            search_results = self.note_tool.search_note(
                query=user_input,
                limit=limit,
            )

            all_notes = {note["note_id"]: note for note in blockers + search_results}

            return list(all_notes.values())[:limit]
        except Exception as e:
            print(f"[WARNING] 笔记检索失败: {e}")
            return []

    def _note_to_packets(self, notes: list[dict[str, Any]]) -> list[ContextPacket]:
        """将笔记转换为上下文包"""
        packets = []

        for note in notes:
            content = f"[笔记:{note['title']}]\n{note['content']}"

            packets.append(
                ContextPacket(
                    content=content,
                    timestamp=datetime.fromisoformat(note["updated_at"]),
                    token_count=len(content),
                    relevance_score=0.75,
                    metadata={"type": "note", "note_type": note["type"], "note_id": note["note_id"]},
                )
            )
        return packets

    def _build_system_instructions(self) -> str:
        """构建系统指令"""
        return f"""你是 {self.project_name} 项目的长期助手。
            你的职责:
            1. 基于历史笔记提供连贯的建议
            2. 追踪项目进展和待解决问题
            3. 在回答时引用相关的历史笔记
            4. 提供具体、可操作的下一步建议

            注意:
            - 优先关注标记为 blocker 的问题
            - 在建议中说明依据来源(笔记、记忆或知识库)
            - 保持对项目整体进度的认识"""

    def _save_as_note(self, user_input: str, response: str):
        """将交互保存为笔记"""

        try:
            # 判断应该保存为什么类型的笔记
            if "问题" in user_input or "阻塞" in user_input:
                note_type = "blocker"
            elif "计划" in user_input or "下一步" in user_input:
                note_type = "action"
            else:
                note_type = "conclusion"

            self.note_tool.run(
                {
                    "action": "create",
                    "title": f"{user_input[:30]}...",
                    "content": f"## 问题\n{user_input}\n\n## 分析\n{response}",
                    "note_type": note_type,
                    "tags": [self.project_name, "auto_generated"],
                }
            )
        except Exception as e:
            print(f"[WARNING] 保存笔记失败: {e}")

    def _update_history(self, user_input: str, response: str):
        """更新对话历史"""

        self.conversation_history.append(Message(content=user_input, role=RoleType.USER, timestamp=datetime.now()))

        self.conversation_history.append(Message(content=response, role=RoleType.ASSISTANT, timestamp=datetime.now()))

        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[:10]

    def run(self, input_text: str, **kwargs) -> str:

        relevant_notes = self._retrieve_relevant_notes(input_text)

        note_packets = self._note_to_packets(relevant_notes)

        context = self.context_builder.build(
            user_query=input_text,
            conversation_history=self.conversation_history,
            system_instructions=self._build_system_instructions(),
            additional_packets=note_packets,
        )

        response = self.llm.invoke([{"role": "user", "content": input_text}, {"role": "system", "content": context}])

        self._save_as_note(user_input=input_text, response=response)

        self._update_history(user_input=input_text, response=response)

        return response
