"""
Session Persistence for Company AGI - Claude Code style session management.

Supports:
- Session creation, save, resume
- Full conversation history persistence
- Tool execution state preservation
- Session forking and checkpointing
- Automatic cleanup of old sessions
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionState(Enum):
    """Session lifecycle states."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    FORKED = "forked"


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ToolExecution:
    """Record of a tool execution."""
    tool_name: str
    input: Dict[str, Any]
    output: Any
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "input": self.input,
            "output": str(self.output) if self.output else None,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolExecution":
        return cls(
            tool_name=data["tool_name"],
            input=data["input"],
            output=data["output"],
            success=data["success"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_ms=data.get("duration_ms"),
            error=data.get("error"),
        )


@dataclass
class Checkpoint:
    """A checkpoint in the session for rewind capability."""
    id: str
    name: str
    message_index: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "message_index": self.message_index,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            id=data["id"],
            name=data["name"],
            message_index=data["message_index"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """A session containing conversation history and state."""
    id: str
    name: Optional[str] = None
    state: SessionState = SessionState.ACTIVE
    messages: List[Message] = field(default_factory=list)
    tool_executions: List[ToolExecution] = field(default_factory=list)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    workflow_state: Dict[str, Any] = field(default_factory=dict)
    agent_states: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parent_session_id: Optional[str] = None  # For forked sessions
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a message to the session."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = datetime.now()
        return msg

    def add_tool_execution(
        self,
        tool_name: str,
        input: Dict[str, Any],
        output: Any,
        success: bool,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> ToolExecution:
        """Record a tool execution."""
        exec = ToolExecution(
            tool_name=tool_name,
            input=input,
            output=output,
            success=success,
            duration_ms=duration_ms,
            error=error,
        )
        self.tool_executions.append(exec)
        self.updated_at = datetime.now()
        return exec

    def create_checkpoint(self, name: str, metadata: Optional[Dict] = None) -> Checkpoint:
        """Create a checkpoint at the current state."""
        cp = Checkpoint(
            id=str(uuid.uuid4())[:8],
            name=name,
            message_index=len(self.messages),
            metadata=metadata or {},
        )
        self.checkpoints.append(cp)
        self.updated_at = datetime.now()
        return cp

    def rewind_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rewind session to a checkpoint."""
        for cp in self.checkpoints:
            if cp.id == checkpoint_id:
                self.messages = self.messages[:cp.message_index]
                # Remove checkpoints after this one
                self.checkpoints = [c for c in self.checkpoints if c.message_index <= cp.message_index]
                self.updated_at = datetime.now()
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "messages": [m.to_dict() for m in self.messages],
            "tool_executions": [t.to_dict() for t in self.tool_executions],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "workflow_state": self.workflow_state,
            "agent_states": self.agent_states,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parent_session_id": self.parent_session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            id=data["id"],
            name=data.get("name"),
            state=SessionState(data.get("state", "active")),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            tool_executions=[ToolExecution.from_dict(t) for t in data.get("tool_executions", [])],
            checkpoints=[Checkpoint.from_dict(c) for c in data.get("checkpoints", [])],
            workflow_state=data.get("workflow_state", {}),
            agent_states=data.get("agent_states", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            parent_session_id=data.get("parent_session_id"),
            metadata=data.get("metadata", {}),
        )


class SessionManager:
    """Manages session persistence and lifecycle."""

    def __init__(
        self,
        sessions_dir: str = "output/sessions",
        max_age_days: int = 30,
        auto_save: bool = True,
        auto_save_interval: int = 60,  # seconds
    ):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        self.current_session: Optional[Session] = None
        self._last_save_time: Optional[datetime] = None

    def create_session(self, name: Optional[str] = None) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:12]
        session = Session(id=session_id, name=name)
        self.current_session = session
        self._save_session(session)
        return session

    def resume_session(self, session_id: Optional[str] = None, name: Optional[str] = None) -> Optional[Session]:
        """Resume an existing session by ID or name."""
        if session_id:
            session = self._load_session(session_id)
        elif name:
            session = self._find_session_by_name(name)
        else:
            # Resume most recent
            session = self._get_most_recent_session()

        if session:
            session.state = SessionState.ACTIVE
            self.current_session = session
            self._save_session(session)

        return session

    def fork_session(self, source_session_id: Optional[str] = None) -> Optional[Session]:
        """Fork a session to create a new branch."""
        source = None
        if source_session_id:
            source = self._load_session(source_session_id)
        elif self.current_session:
            source = self.current_session

        if not source:
            return None

        # Create new session as fork
        new_session = Session(
            id=str(uuid.uuid4())[:12],
            name=f"fork-{source.name or source.id}",
            messages=source.messages.copy(),
            tool_executions=source.tool_executions.copy(),
            checkpoints=source.checkpoints.copy(),
            workflow_state=source.workflow_state.copy(),
            agent_states=source.agent_states.copy(),
            parent_session_id=source.id,
        )

        # Mark source as forked
        source.state = SessionState.FORKED
        self._save_session(source)

        self.current_session = new_session
        self._save_session(new_session)

        return new_session

    def save_current(self) -> bool:
        """Save the current session."""
        if not self.current_session:
            return False
        self._save_session(self.current_session)
        self._last_save_time = datetime.now()
        return True

    def maybe_auto_save(self) -> bool:
        """Auto-save if interval has passed."""
        if not self.auto_save or not self.current_session:
            return False

        if self._last_save_time is None:
            return self.save_current()

        elapsed = (datetime.now() - self._last_save_time).total_seconds()
        if elapsed >= self.auto_save_interval:
            return self.save_current()

        return False

    def end_session(self, state: SessionState = SessionState.COMPLETED) -> bool:
        """End the current session."""
        if not self.current_session:
            return False

        self.current_session.state = state
        self._save_session(self.current_session)
        self.current_session = None
        return True

    def list_sessions(
        self,
        limit: int = 20,
        state: Optional[SessionState] = None,
        include_forked: bool = False,
    ) -> List[Dict[str, Any]]:
        """List available sessions."""
        sessions = []

        for session_file in sorted(
            self.sessions_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        ):
            try:
                with open(session_file) as f:
                    data = json.load(f)

                session_state = SessionState(data.get("state", "active"))

                if state and session_state != state:
                    continue

                if not include_forked and session_state == SessionState.FORKED:
                    continue

                sessions.append({
                    "id": data["id"],
                    "name": data.get("name"),
                    "state": session_state.value,
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "message_count": len(data.get("messages", [])),
                })

                if len(sessions) >= limit:
                    break

            except (json.JSONDecodeError, KeyError):
                continue

        return sessions

    def cleanup_old_sessions(self) -> int:
        """Remove sessions older than max_age_days."""
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        removed = 0

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)

                updated_at = datetime.fromisoformat(data["updated_at"])
                if updated_at < cutoff:
                    session_file.unlink()
                    removed += 1

            except (json.JSONDecodeError, KeyError, OSError):
                continue

        return removed

    def export_session(self, session_id: str, output_path: str) -> bool:
        """Export a session to a file."""
        session = self._load_session(session_id)
        if not session:
            return False

        with open(output_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        return True

    def import_session(self, input_path: str) -> Optional[Session]:
        """Import a session from a file."""
        try:
            with open(input_path) as f:
                data = json.load(f)

            session = Session.from_dict(data)
            # Assign new ID to avoid conflicts
            session.id = str(uuid.uuid4())[:12]
            self._save_session(session)
            return session

        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a session without loading full history."""
        session = self._load_session(session_id)
        if not session:
            return None

        return {
            "id": session.id,
            "name": session.name,
            "state": session.state.value,
            "message_count": len(session.messages),
            "tool_execution_count": len(session.tool_executions),
            "checkpoint_count": len(session.checkpoints),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "duration_minutes": (session.updated_at - session.created_at).total_seconds() / 60,
            "parent_session_id": session.parent_session_id,
        }

    def _save_session(self, session: Session) -> None:
        """Save a session to disk."""
        session.updated_at = datetime.now()
        session_file = self.sessions_dir / f"{session.id}.json"

        with open(session_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def _load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from disk."""
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file) as f:
                data = json.load(f)
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def _find_session_by_name(self, name: str) -> Optional[Session]:
        """Find a session by name."""
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                if data.get("name") == name:
                    return Session.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def _get_most_recent_session(self) -> Optional[Session]:
        """Get the most recently updated session."""
        session_files = sorted(
            self.sessions_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        for session_file in session_files:
            try:
                with open(session_file) as f:
                    data = json.load(f)
                # Skip completed/failed sessions
                state = SessionState(data.get("state", "active"))
                if state in (SessionState.ACTIVE, SessionState.PAUSED):
                    return Session.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                continue

        return None


# Global session manager instance
_global_session_manager: Optional[SessionManager] = None


def get_session_manager(sessions_dir: str = "output/sessions") -> SessionManager:
    """Get or create the global session manager."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager(sessions_dir)
    return _global_session_manager


def reset_session_manager() -> None:
    """Reset the global session manager."""
    global _global_session_manager
    _global_session_manager = None
