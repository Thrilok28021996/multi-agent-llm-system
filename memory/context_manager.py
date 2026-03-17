"""
Context Manager for Company AGI.

Provides Claude Code-style context management with:
- Token counting and tracking
- Automatic context summarization (auto-compact)
- Intelligent message pruning
- Context window optimization
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib


class TokenizerType(Enum):
    """Supported tokenizer types."""
    SIMPLE = "simple"  # Word-based estimation
    TIKTOKEN = "tiktoken"  # OpenAI tiktoken
    SENTENCEPIECE = "sentencepiece"  # For Llama models


@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Add token usage from another instance."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost=self.estimated_cost + other.estimated_cost,
        )


@dataclass
class ContextMessage:
    """A message in the context window."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    importance: float = 1.0  # 0.0 to 1.0
    is_summary: bool = False
    original_messages: List[str] = field(default_factory=list)  # IDs of summarized messages
    message_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.message_id:
            self.message_id = hashlib.md5(
                f"{self.role}{self.content}{self.timestamp}".encode()
            ).hexdigest()[:12]


@dataclass
class CompactionResult:
    """Result of a context compaction operation."""
    success: bool
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    summary_created: bool
    error: Optional[str] = None


class SimpleTokenizer:
    """Simple word-based token estimator."""

    # Average characters per token (approximation)
    CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        """Estimate token count from text."""
        # Simple estimation: ~4 chars per token
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            # Add overhead for message structure
            total += 4  # Role tokens
            total += self.count(msg.get("content", ""))
        return total


class ContextManager:
    """
    Manages conversation context with automatic compaction.

    Features:
    - Token counting and tracking
    - Automatic summarization when context fills
    - Importance-based message retention
    - Configurable compaction strategies
    """

    # Default context limits (can be overridden)
    DEFAULT_MAX_TOKENS = 100000  # 100k default
    COMPACTION_THRESHOLD = 0.8  # Compact at 80% capacity
    MIN_MESSAGES_TO_KEEP = 5  # Always keep recent messages
    SUMMARY_TARGET_RATIO = 0.3  # Target 30% of original size

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        compaction_threshold: float = COMPACTION_THRESHOLD,
        auto_compact: bool = True,
        tokenizer_type: TokenizerType = TokenizerType.SIMPLE,
        summarizer: Optional[Any] = None,  # LLM for summarization
    ):
        self.max_tokens = max_tokens
        self.compaction_threshold = compaction_threshold
        self.auto_compact = auto_compact
        self.tokenizer = self._create_tokenizer(tokenizer_type)
        self.summarizer = summarizer

        self.messages: List[ContextMessage] = []
        self.total_tokens: int = 0
        self.usage_history: List[TokenUsage] = []
        self.compaction_history: List[CompactionResult] = []

        # System message (always kept)
        self.system_message: Optional[ContextMessage] = None

    def _create_tokenizer(self, tokenizer_type: TokenizerType) -> SimpleTokenizer:
        """Create appropriate tokenizer."""
        # For now, always use simple tokenizer
        # Can be extended to support tiktoken, sentencepiece
        _ = tokenizer_type  # Reserved for future tokenizer selection
        return SimpleTokenizer()

    def set_system_message(self, content: str) -> None:
        """Set the system message (always kept in context)."""
        self.system_message = ContextMessage(
            role="system",
            content=content,
            token_count=self.tokenizer.count(content),
            importance=1.0,
        )

    def add_message(
        self,
        role: str,
        content: str,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        """Add a message to the context."""
        token_count = self.tokenizer.count(content)

        message = ContextMessage(
            role=role,
            content=content,
            token_count=token_count,
            importance=importance,
            metadata=metadata or {},
        )

        self.messages.append(message)
        self.total_tokens += token_count

        # Check if compaction is needed
        if self.auto_compact and self.should_compact():
            self.compact()

        return message

    def get_token_count(self) -> int:
        """Get current total token count."""
        system_tokens = self.system_message.token_count if self.system_message else 0
        return system_tokens + sum(m.token_count for m in self.messages)

    def get_usage_percentage(self) -> float:
        """Get percentage of context window used."""
        return self.get_token_count() / self.max_tokens

    def should_compact(self) -> bool:
        """Check if context should be compacted."""
        return self.get_usage_percentage() >= self.compaction_threshold

    def compact(self, force: bool = False) -> CompactionResult:
        """
        Compact the context by summarizing older messages.

        Strategy:
        1. Keep system message
        2. Keep recent messages (MIN_MESSAGES_TO_KEEP)
        3. Summarize older messages
        4. Keep high-importance messages
        """
        if not force and not self.should_compact():
            return CompactionResult(
                success=True,
                messages_before=len(self.messages),
                messages_after=len(self.messages),
                tokens_before=self.get_token_count(),
                tokens_after=self.get_token_count(),
                tokens_saved=0,
                summary_created=False,
            )

        tokens_before = self.get_token_count()
        messages_before = len(self.messages)

        try:
            # Separate messages to keep vs summarize
            keep_count = max(self.MIN_MESSAGES_TO_KEEP, len(self.messages) // 4)
            recent_messages = self.messages[-keep_count:]
            older_messages = self.messages[:-keep_count]

            # Also keep high-importance messages from older set
            high_importance = [m for m in older_messages if m.importance >= 0.8]
            to_summarize = [m for m in older_messages if m.importance < 0.8]

            if not to_summarize:
                return CompactionResult(
                    success=True,
                    messages_before=messages_before,
                    messages_after=len(self.messages),
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                    tokens_saved=0,
                    summary_created=False,
                )

            # Create summary
            summary = self._create_summary(to_summarize)

            if summary:
                summary_message = ContextMessage(
                    role="system",
                    content=f"[Previous conversation summary]\n{summary}",
                    token_count=self.tokenizer.count(summary),
                    importance=0.9,
                    is_summary=True,
                    original_messages=[m.message_id for m in to_summarize],
                )

                # Rebuild message list
                self.messages = [summary_message] + high_importance + recent_messages
            else:
                # Fallback: just keep recent + high importance
                self.messages = high_importance + recent_messages

            # Recalculate total tokens
            self.total_tokens = sum(m.token_count for m in self.messages)

            tokens_after = self.get_token_count()

            result = CompactionResult(
                success=True,
                messages_before=messages_before,
                messages_after=len(self.messages),
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_saved=tokens_before - tokens_after,
                summary_created=summary is not None,
            )

            self.compaction_history.append(result)
            return result

        except Exception as e:
            return CompactionResult(
                success=False,
                messages_before=messages_before,
                messages_after=len(self.messages),
                tokens_before=tokens_before,
                tokens_after=self.get_token_count(),
                tokens_saved=0,
                summary_created=False,
                error=str(e),
            )

    def _create_summary(self, messages: List[ContextMessage]) -> Optional[str]:
        """Create a summary of messages."""
        if not messages:
            return None

        # If we have a summarizer (LLM), use it
        if self.summarizer:
            return self._llm_summarize(messages)

        # Otherwise, use extractive summarization
        return self._extractive_summarize(messages)

    def _llm_summarize(self, messages: List[ContextMessage]) -> Optional[str]:
        """Use LLM to create summary."""
        if not self.summarizer:
            return None

        # Format messages for summarization
        conversation = "\n".join([
            f"{m.role.upper()}: {m.content}"
            for m in messages
        ])

        prompt = f"""Summarize the following conversation in a concise way,
preserving key decisions, code changes, and important context:

{conversation}

Summary:"""

        try:
            # Call summarizer (assumes it has a generate method)
            if hasattr(self.summarizer, 'generate'):
                result = self.summarizer.generate(prompt)
                return str(result) if result else None
            elif callable(self.summarizer):
                result = self.summarizer(prompt)
                return str(result) if result else None
        except Exception:
            pass  # Fall through to extractive summarization

        return self._extractive_summarize(messages)

    def _extractive_summarize(self, messages: List[ContextMessage]) -> str:
        """Create extractive summary (key points extraction)."""
        summary_parts = []

        # Group by role
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]

        # Extract key points from user messages
        if user_messages:
            user_topics = self._extract_topics(user_messages)
            if user_topics:
                summary_parts.append(f"User requested: {', '.join(user_topics[:5])}")

        # Extract key actions from assistant messages
        if assistant_messages:
            actions = self._extract_actions(assistant_messages)
            if actions:
                summary_parts.append(f"Actions taken: {', '.join(actions[:5])}")

        # Add message count
        summary_parts.append(
            f"({len(messages)} messages summarized)"
        )

        return "\n".join(summary_parts)

    def _extract_topics(self, messages: List[ContextMessage]) -> List[str]:
        """Extract main topics from messages."""
        topics = []

        # Look for common patterns
        patterns = [
            r"(?:implement|create|add|build|make)\s+(\w+(?:\s+\w+)?)",
            r"(?:fix|debug|solve)\s+(\w+(?:\s+\w+)?)",
            r"(?:help\s+(?:me\s+)?(?:with\s+)?)?(\w+\s+(?:issue|problem|error|bug))",
        ]

        for msg in messages:
            content = msg.content.lower()
            for pattern in patterns:
                matches = re.findall(pattern, content)
                topics.extend(matches)

        # Deduplicate while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        return unique_topics

    def _extract_actions(self, messages: List[ContextMessage]) -> List[str]:
        """Extract main actions from assistant messages."""
        actions = []

        # Look for action patterns
        patterns = [
            r"(?:I(?:'ll| will)?|Let me)\s+(\w+(?:\s+\w+){0,3})",
            r"(?:Created|Updated|Fixed|Added|Implemented)\s+(\w+(?:\s+\w+)?)",
            r"(?:File|Code|Function)\s+(\w+)\s+(?:created|updated|modified)",
        ]

        for msg in messages:
            content = msg.content
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                actions.extend(matches[:2])  # Limit per message

        # Deduplicate
        seen = set()
        unique_actions = []
        for action in actions:
            action_lower = action.lower()
            if action_lower not in seen:
                seen.add(action_lower)
                unique_actions.append(action)

        return unique_actions

    def get_context_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for API call."""
        result = []

        # Add system message if present
        if self.system_message:
            result.append({
                "role": "system",
                "content": self.system_message.content,
            })

        # Add conversation messages
        for msg in self.messages:
            result.append({
                "role": msg.role,
                "content": msg.content,
            })

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            "total_messages": len(self.messages),
            "total_tokens": self.get_token_count(),
            "max_tokens": self.max_tokens,
            "usage_percentage": self.get_usage_percentage() * 100,
            "compactions_performed": len(self.compaction_history),
            "tokens_saved_total": sum(c.tokens_saved for c in self.compaction_history),
            "auto_compact_enabled": self.auto_compact,
        }

    def clear(self) -> None:
        """Clear all messages (keeps system message)."""
        self.messages = []
        self.total_tokens = 0

    def save_state(self, path: str) -> bool:
        """Save context state to file."""
        try:
            state = {
                "system_message": {
                    "role": self.system_message.role,
                    "content": self.system_message.content,
                } if self.system_message else None,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp.isoformat(),
                        "token_count": m.token_count,
                        "importance": m.importance,
                        "is_summary": m.is_summary,
                        "message_id": m.message_id,
                    }
                    for m in self.messages
                ],
                "stats": self.get_stats(),
            }

            Path(path).write_text(json.dumps(state, indent=2))
            return True
        except Exception:
            return False

    def load_state(self, path: str) -> bool:
        """Load context state from file."""
        try:
            state = json.loads(Path(path).read_text())

            if state.get("system_message"):
                self.set_system_message(state["system_message"]["content"])

            self.messages = []
            for m in state.get("messages", []):
                msg = ContextMessage(
                    role=m["role"],
                    content=m["content"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                    token_count=m["token_count"],
                    importance=m["importance"],
                    is_summary=m.get("is_summary", False),
                    message_id=m["message_id"],
                )
                self.messages.append(msg)

            self.total_tokens = sum(m.token_count for m in self.messages)
            return True
        except Exception:
            return False


# Singleton instance
_context_manager: Optional[ContextManager] = None


def get_context_manager(
    max_tokens: int = ContextManager.DEFAULT_MAX_TOKENS,
    auto_compact: bool = True,
) -> ContextManager:
    """Get or create the global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(
            max_tokens=max_tokens,
            auto_compact=auto_compact,
        )
    return _context_manager


def reset_context_manager() -> None:
    """Reset the global context manager."""
    global _context_manager
    _context_manager = None
