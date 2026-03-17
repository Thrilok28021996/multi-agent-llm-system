"""UI module for Company AGI."""

from .console import CompanyConsole, console, AgentStatus
from .logger import ConversationLogger, logger
from .streaming import (
    StreamingOutput,
    StreamEvent,
    StreamEventType,
    StreamStats,
    StreamBuffer,
    ProgressIndicator,
    get_streaming_output,
    reset_streaming_output,
)

from .interactive import (
    AskUserQuestion,
    Question,
    QuestionOption,
    QuestionResult,
    QuestionType,
    get_ask_user_question,
    reset_ask_user_question,
    ask_user,
)

__all__ = [
    # Console
    "CompanyConsole",
    "console",
    "AgentStatus",
    "ConversationLogger",
    "logger",
    # Streaming
    "StreamingOutput",
    "StreamEvent",
    "StreamEventType",
    "StreamStats",
    "StreamBuffer",
    "ProgressIndicator",
    "get_streaming_output",
    "reset_streaming_output",
    # Interactive
    "AskUserQuestion",
    "Question",
    "QuestionOption",
    "QuestionResult",
    "QuestionType",
    "get_ask_user_question",
    "reset_ask_user_question",
    "ask_user",
]
