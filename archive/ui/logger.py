"""Conversation logger for saving full agent interactions."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, field, asdict


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    event_type: str  # agent_message, agent_thinking, agent_action, meeting, decision, phase, error
    agent: str = ""
    recipient: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationLogger:
    """
    Logs all agent conversations and interactions to file.
    Provides full, untruncated access to all communications.
    """

    def __init__(self, log_dir: str = "output/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create session log file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"conversation_{self.session_id}.json"
        self.transcript_file = self.log_dir / f"transcript_{self.session_id}.txt"

        self.entries: List[LogEntry] = []
        self.current_phase = "initialization"

    def _timestamp(self) -> str:
        return datetime.now().isoformat()

    def log_phase(self, phase: str) -> None:
        """Log a workflow phase change."""
        self.current_phase = phase
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="phase",
            content=phase
        )
        self.entries.append(entry)
        self._append_to_transcript(f"\n{'='*60}\n  PHASE: {phase}\n{'='*60}\n")

    def log_agent_thinking(self, agent: str, thought: str) -> None:
        """Log agent thinking (full content)."""
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="agent_thinking",
            agent=agent,
            content=thought,
            metadata={"phase": self.current_phase}
        )
        self.entries.append(entry)
        self._append_to_transcript(f"\n[{agent} THINKING]\n{thought}\n")

    def log_agent_action(self, agent: str, action: str, details: str = "") -> None:
        """Log agent action."""
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="agent_action",
            agent=agent,
            content=action,
            metadata={"details": details, "phase": self.current_phase}
        )
        self.entries.append(entry)
        self._append_to_transcript(f"\n[{agent} ACTION] {action}\n{details}\n")

    def log_agent_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        message_type: str = "general"
    ) -> None:
        """Log inter-agent communication (full message)."""
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="agent_message",
            agent=from_agent,
            recipient=to_agent,
            content=message,
            metadata={"message_type": message_type, "phase": self.current_phase}
        )
        self.entries.append(entry)
        self._append_to_transcript(f"\n[{from_agent} -> {to_agent}] ({message_type})\n{message}\n")

    def log_decision(self, agent: str, decision: str, reasoning: str = "") -> None:
        """Log a decision made by an agent."""
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="decision",
            agent=agent,
            content=decision,
            metadata={"reasoning": reasoning, "phase": self.current_phase}
        )
        self.entries.append(entry)
        self._append_to_transcript(f"\n[{agent} DECISION]\nDecision: {decision}\nReasoning: {reasoning}\n")

    def log_meeting(
        self,
        topic: str,
        participants: List[str],
        transcript: List[Dict[str, str]],
        outcome: Dict[str, Any]
    ) -> None:
        """Log a complete meeting."""
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="meeting",
            content=topic,
            metadata={
                "participants": participants,
                "transcript": transcript,
                "outcome": outcome,
                "phase": self.current_phase
            }
        )
        self.entries.append(entry)

        # Write detailed transcript
        self._append_to_transcript(f"\n{'='*60}\n  MEETING: {topic}\n{'='*60}\n")
        self._append_to_transcript(f"Participants: {', '.join(participants)}\n\n")
        for msg in transcript:
            speaker = msg.get("speaker", "Unknown")
            content = msg.get("content", msg.get("message", ""))
            self._append_to_transcript(f"[{speaker}]\n{content}\n\n")
        self._append_to_transcript(f"OUTCOME: {json.dumps(outcome, indent=2)}\n")

    def log_error(self, error: str, context: str = "") -> None:
        """Log an error."""
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="error",
            content=error,
            metadata={"context": context, "phase": self.current_phase}
        )
        self.entries.append(entry)
        self._append_to_transcript(f"\n[ERROR] {error}\nContext: {context}\n")

    def log_problem(self, problem: Dict[str, Any]) -> None:
        """Log a discovered/provided problem."""
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="problem",
            content=problem.get("description", ""),
            metadata=problem
        )
        self.entries.append(entry)
        self._append_to_transcript(f"\n{'='*60}\n  PROBLEM\n{'='*60}\n")
        self._append_to_transcript(f"{json.dumps(problem, indent=2)}\n")

    def log_solution(self, solution: Dict[str, Any]) -> None:
        """Log a generated solution."""
        entry = LogEntry(
            timestamp=self._timestamp(),
            event_type="solution",
            content=str(solution.get("description", "")),
            metadata=solution
        )
        self.entries.append(entry)
        self._append_to_transcript(f"\n{'='*60}\n  SOLUTION\n{'='*60}\n")
        self._append_to_transcript(f"{json.dumps(solution, indent=2, default=str)}\n")

    def _append_to_transcript(self, text: str) -> None:
        """Append text to the transcript file."""
        with open(self.transcript_file, "a", encoding="utf-8") as f:
            f.write(text)

    def save(self) -> str:
        """Save all logs to JSON file and return the file path."""
        data = {
            "session_id": self.session_id,
            "started_at": self.entries[0].timestamp if self.entries else None,
            "ended_at": self._timestamp(),
            "total_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries]
        }

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return str(self.log_file)

    def get_transcript_path(self) -> str:
        """Get the path to the human-readable transcript."""
        return str(self.transcript_file)

    def get_full_transcript(self) -> str:
        """Get the full transcript as a string."""
        if self.transcript_file.exists():
            return self.transcript_file.read_text(encoding="utf-8")
        return ""

    def get_agent_messages(self, agent: str) -> List[LogEntry]:
        """Get all messages from a specific agent."""
        return [e for e in self.entries if e.agent == agent]

    def get_phase_entries(self, phase: str) -> List[LogEntry]:
        """Get all entries from a specific phase."""
        return [e for e in self.entries if e.metadata.get("phase") == phase]


# Global logger instance
logger = ConversationLogger()
