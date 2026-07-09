"""Memory module for Company AGI."""

from .shared_memory import SharedMemory
from .agent_memory import AgentMemory
from .learning import AgentLearning, Lesson, Pattern
from .session import (
    Session,
    SessionState,
    SessionManager,
    Message,
    ToolExecution,
    Checkpoint,
    get_session_manager,
)
from .context_manager import (
    ContextManager,
    ContextMessage,
    CompactionResult,
    TokenUsage,
    TokenizerType,
    get_context_manager,
    reset_context_manager,
)

__all__ = [
    # Existing memory
    "SharedMemory",
    "AgentMemory",
    "AgentLearning",
    "Lesson",
    "Pattern",
    # Session management
    "Session",
    "SessionState",
    "SessionManager",
    "Message",
    "ToolExecution",
    "Checkpoint",
    "get_session_manager",
    # Context management (auto-compact)
    "ContextManager",
    "ContextMessage",
    "CompactionResult",
    "TokenUsage",
    "TokenizerType",
    "get_context_manager",
    "reset_context_manager",
]
