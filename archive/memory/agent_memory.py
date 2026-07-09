"""Individual agent memory for maintaining context and learning."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ui.console import console
from utils.file_lock import atomic_write_json, safe_read_json


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experience:
    """An experience that the agent can learn from."""

    action: str
    context: str
    outcome: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    lessons_learned: List[str] = field(default_factory=list)


class AgentMemory:
    """
    Individual memory for each agent.
    Maintains conversation history, experiences, and learned patterns.
    """

    def __init__(
        self,
        agent_name: str,
        max_conversation_history: int = 50,
        max_experiences: int = 100,
        persist_dir: str = None
    ):
        self.agent_name = agent_name
        self.max_conversation_history = max_conversation_history
        self.max_experiences = max_experiences

        # Conversation memory (short-term)
        self.conversation_history: deque = deque(maxlen=max_conversation_history)

        # Experience memory (long-term learning)
        self.experiences: List[Experience] = []

        # Working memory (current task context)
        self.working_memory: Dict[str, Any] = {}

        # Learned patterns and preferences
        self.learned_patterns: Dict[str, Any] = {
            "successful_approaches": [],
            "failed_approaches": [],
            "preferences": {},
            "frequently_used_tools": {},
        }

        # Persistence
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load agent memory from disk with file locking."""
        if not self.persist_dir:
            return

        memory_file = self.persist_dir / f"{self.agent_name}_memory.json"
        if memory_file.exists():
            try:
                # Use safe_read_json with file locking to prevent race conditions
                data = safe_read_json(memory_file, default={})
                if data:
                    self.learned_patterns = data.get("learned_patterns", self.learned_patterns)
                    # Load experiences
                    for exp_data in data.get("experiences", []):
                        exp_data["timestamp"] = datetime.fromisoformat(exp_data["timestamp"])
                        self.experiences.append(Experience(**exp_data))
            except Exception as e:
                console.warning(f"Could not load agent memory: {e}")

    def _save_to_disk(self) -> None:
        """Save agent memory to disk with file locking and atomic writes."""
        if not self.persist_dir:
            return

        memory_file = self.persist_dir / f"{self.agent_name}_memory.json"
        data = {
            "agent_name": self.agent_name,
            "learned_patterns": self.learned_patterns,
            "experiences": [
                {
                    "action": e.action,
                    "context": e.context,
                    "outcome": e.outcome,
                    "success": e.success,
                    "timestamp": e.timestamp.isoformat(),
                    "lessons_learned": e.lessons_learned
                }
                for e in self.experiences[-self.max_experiences:]
            ]
        }
        # Use atomic_write_json with file locking to prevent race conditions
        # and ensure data integrity even if process crashes during write
        atomic_write_json(memory_file, data, indent=2)

    # ============================================================
    #  CONVERSATION MEMORY
    # ============================================================

    def add_to_conversation(
        self,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add a turn to conversation history."""
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.conversation_history.append(turn)

    def compact(self) -> None:
        """Compact memory by trimming old conversation history to half capacity."""
        if len(self.conversation_history) > self.max_conversation_history // 2:
            keep = self.max_conversation_history // 2
            recent = list(self.conversation_history)[-keep:]
            self.conversation_history.clear()
            self.conversation_history.extend(recent)
        # Trim old experiences beyond limit
        if len(self.experiences) > self.max_experiences:
            self.experiences = self.experiences[-self.max_experiences:]

    def get_conversation_history(self, limit: int = None) -> List[Dict[str, str]]:
        """Get recent conversation history formatted for LLM."""
        history = list(self.conversation_history)
        if limit:
            history = history[-limit:]
        return [{"role": t.role, "content": t.content} for t in history]

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history."""
        if not self.conversation_history:
            return "No conversation history."

        total_turns = len(self.conversation_history)
        recent = list(self.conversation_history)[-5:]
        summary_parts = [f"Total turns: {total_turns}", "Recent exchanges:"]
        for turn in recent:
            summary_parts.append(f"  [{turn.role}]: {turn.content}")

        return "\n".join(summary_parts)

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    # ============================================================
    #  WORKING MEMORY
    # ============================================================

    def set_context(self, key: str, value: Any) -> None:
        """Set a value in working memory."""
        self.working_memory[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from working memory."""
        return self.working_memory.get(key, default)

    def update_context(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in working memory."""
        self.working_memory.update(updates)

    def clear_context(self) -> None:
        """Clear working memory."""
        self.working_memory.clear()

    def get_full_context(self) -> Dict[str, Any]:
        """Get the entire working memory."""
        return self.working_memory.copy()

    # ============================================================
    #  EXPERIENCE MEMORY
    # ============================================================

    def record_experience(
        self,
        action: str,
        context: str,
        outcome: str,
        success: bool,
        lessons: List[str] = None
    ) -> None:
        """Record an experience for learning."""
        experience = Experience(
            action=action,
            context=context,
            outcome=outcome,
            success=success,
            lessons_learned=lessons or []
        )
        self.experiences.append(experience)

        # Update learned patterns
        if success:
            self.learned_patterns["successful_approaches"].append({
                "action": action,
                "context_summary": context,
                "timestamp": datetime.now().isoformat()
            })
        else:
            self.learned_patterns["failed_approaches"].append({
                "action": action,
                "context_summary": context,
                "timestamp": datetime.now().isoformat()
            })

        # Keep lists bounded
        self.learned_patterns["successful_approaches"] = \
            self.learned_patterns["successful_approaches"][-50:]
        self.learned_patterns["failed_approaches"] = \
            self.learned_patterns["failed_approaches"][-50:]

        self._save_to_disk()

    def get_similar_experiences(self, context: str, limit: int = 5) -> List[Experience]:
        """Find experiences similar to the current context."""
        # Simple keyword-based similarity for now
        context_words = set(context.lower().split())
        scored_experiences = []

        for exp in self.experiences:
            exp_words = set(exp.context.lower().split())
            overlap = len(context_words & exp_words)
            if overlap > 0:
                scored_experiences.append((overlap, exp))

        scored_experiences.sort(reverse=True, key=lambda x: x[0])
        return [exp for _, exp in scored_experiences[:limit]]

    def get_successful_approaches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent successful approaches."""
        return self.learned_patterns["successful_approaches"][-limit:]

    def get_failed_approaches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failed approaches to avoid."""
        return self.learned_patterns["failed_approaches"][-limit:]

    # ============================================================
    #  TOOL USAGE TRACKING
    # ============================================================

    def record_tool_usage(self, tool_name: str, success: bool) -> None:
        """Record usage of a tool."""
        if tool_name not in self.learned_patterns["frequently_used_tools"]:
            self.learned_patterns["frequently_used_tools"][tool_name] = {
                "total_uses": 0,
                "successes": 0
            }

        self.learned_patterns["frequently_used_tools"][tool_name]["total_uses"] += 1
        if success:
            self.learned_patterns["frequently_used_tools"][tool_name]["successes"] += 1

        self._save_to_disk()

    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about tool usage."""
        stats = {}
        for tool, data in self.learned_patterns["frequently_used_tools"].items():
            total = data["total_uses"]
            successes = data["successes"]
            stats[tool] = {
                "total_uses": total,
                "success_rate": successes / total if total > 0 else 0
            }
        return stats

    # ============================================================
    #  REFLECTION
    # ============================================================

    def generate_self_reflection(self) -> str:
        """Generate a reflection on past performance."""
        total_experiences = len(self.experiences)
        successful = sum(1 for e in self.experiences if e.success)
        success_rate = successful / total_experiences if total_experiences > 0 else 0

        recent_successes = self.get_successful_approaches(5)
        recent_failures = self.get_failed_approaches(5)

        reflection = [
            f"=== Self Reflection for {self.agent_name} ===",
            f"Total experiences: {total_experiences}",
            f"Success rate: {success_rate:.1%}",
            "",
            "Recent successful approaches:",
        ]

        for approach in recent_successes:
            reflection.append(f"  - {approach['action']}")

        reflection.extend(["", "Approaches to avoid:"])
        for approach in recent_failures:
            reflection.append(f"  - {approach['action']}")

        # Tool usage insights
        tool_stats = self.get_tool_stats()
        if tool_stats:
            reflection.extend(["", "Tool usage:"])
            for tool, stats in sorted(
                tool_stats.items(),
                key=lambda x: x[1]["total_uses"],
                reverse=True
            )[:5]:
                reflection.append(
                    f"  - {tool}: {stats['total_uses']} uses, "
                    f"{stats['success_rate']:.1%} success"
                )

        return "\n".join(reflection)

    # ============================================================
    #  EXPORT/IMPORT
    # ============================================================

    def export_state(self) -> Dict[str, Any]:
        """Export the complete memory state."""
        return {
            "agent_name": self.agent_name,
            "conversation_history": [
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in self.conversation_history
            ],
            "working_memory": self.working_memory,
            "experiences_count": len(self.experiences),
            "learned_patterns": self.learned_patterns
        }

    def reset(self) -> None:
        """Reset all memory (fresh start)."""
        self.conversation_history.clear()
        self.experiences.clear()
        self.working_memory.clear()
        self.learned_patterns = {
            "successful_approaches": [],
            "failed_approaches": [],
            "preferences": {},
            "frequently_used_tools": {},
        }
        self._save_to_disk()
