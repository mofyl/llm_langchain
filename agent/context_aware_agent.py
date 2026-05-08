from agent.simple_agent import SimpleAgent
from context.builer import ContextBuilder, ContextConfig
from core.llm import HelloAgentsLLM
from core.message import RoleType
from tools.memory_tool import MemoryTool
from tools.rag_tool import RAGTool


class ContextAwareAgent(SimpleAgent):
    def __init__(self, name: str, llm: HelloAgentsLLM, **kwargs):
        super().__init__(name=name, llm=llm, system_prompt=kwargs.get("system_prompt", ""))
        self.memory_tool = MemoryTool(user_id=kwargs.get("user_id", ""))
        self.rag_tool = RAGTool(knowledge_base_path=kwargs.get("knowledge_base_path", "./kb"))

        self.context_builder = ContextBuilder(
            memory_tool=self.memory_tool, rag_tool=self.rag_tool, config=ContextConfig(max_tokens=4000)
        )

        self.conversation_history = []

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent,自动构建优化的上下文"""

        opt_context = self.context_builder.build(
            user_query=input_text,
            conversation_history=self.conversation_history,
            system_instructions=self.system_prompt,
        )

        message = [{"role": "user", "content": input_text}, {"role": "system", "content": opt_context}]
        response = self.llm.invoke(message)

        from datetime import datetime

        from core import Message

        time_now = datetime.now()

        self.conversation_history.append(Message(role=RoleType.USER, content=input_text, timestamp=time_now))
        self.conversation_history.append(
            Message(
                role=RoleType.ASSISTANT,
                content=response,
                timestamp=time_now,
            )
        )

        self.memory_tool.run(
            {
                "action": "add",
                "content": f"用户输入：{input_text} 。 llm响应：{response}",
                "memory_type": "episodic",
                "importance": 0.6,
            }
        )
        return response
