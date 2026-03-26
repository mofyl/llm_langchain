from memory.base import MemoryConfig


class MemoryManager:
    def __init__(
        self,
        config: MemoryConfig | None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = True,
    ):
        self.config = config or MemoryConfig()

        self.user_id = user_id

        self.memory_types = {}

        if enable_working : 
            self.memory_types['working'] = 