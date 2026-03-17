"""Enhanced Interactive Session - Claude Code-style conversational interface."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ConversationMessage:
    """A message in the conversation history."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Full context of a conversation session."""
    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    workspace_root: str = "."
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history."""
        msg = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(msg)

    def get_recent_context(self, n: int = 5) -> List[ConversationMessage]:
        """Get last N messages for context."""
        return self.messages[-n:]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in self.messages
            ],
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "workspace_root": self.workspace_root,
            "metadata": self.metadata
        }

    def save(self, path: str):
        """Save conversation to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ConversationContext':
        """Load conversation from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        context = cls(session_id=data['session_id'])
        context.files_modified = data.get('files_modified', [])
        context.files_created = data.get('files_created', [])
        context.workspace_root = data.get('workspace_root', '.')
        context.metadata = data.get('metadata', {})

        for msg_data in data.get('messages', []):
            context.messages.append(ConversationMessage(
                role=msg_data['role'],
                content=msg_data['content'],
                timestamp=datetime.fromisoformat(msg_data['timestamp']),
                metadata=msg_data.get('metadata', {})
            ))

        return context


class EnhancedInteractiveSession:
    """
    Enhanced interactive session with multi-turn conversation support.

    Features:
    - Conversation history and context
    - Progress indicators
    - Better tool result presentation
    - Session persistence
    """

    def __init__(
        self,
        workspace_root: str = ".",
        session_id: Optional[str] = None
    ):
        self.workspace_root = Path(workspace_root).resolve()

        # Create or load conversation context
        if session_id:
            self.context = self._load_session(session_id)
        else:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.context = ConversationContext(
                session_id=session_id,
                workspace_root=str(self.workspace_root)
            )

        self.progress_callback = None

    def _load_session(self, session_id: str) -> ConversationContext:
        """Load existing session."""
        session_file = Path(f".claude_sessions/{session_id}.json")

        if session_file.exists():
            return ConversationContext.load(str(session_file))
        else:
            return ConversationContext(
                session_id=session_id,
                workspace_root=str(self.workspace_root)
            )

    def save_session(self):
        """Save current session."""
        session_dir = Path(".claude_sessions")
        session_dir.mkdir(exist_ok=True)

        session_file = session_dir / f"{self.context.session_id}.json"
        self.context.save(str(session_file))

    def set_progress_callback(self, callback):
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _report_progress(self, stage: str, message: str):
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(stage, message)

    async def chat(
        self,
        user_message: str,
        agent: Any,
        task_type: str = "implement_feature",
        additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Multi-turn conversational interface.

        Args:
            user_message: User's message/request
            agent: Agent to handle the request
            task_type: Type of task to execute
            additional_context: Additional context for the task

        Returns:
            Response with results and conversation context
        """
        # Add user message to history
        self.context.add_message("user", user_message)

        # Build context from conversation history
        recent_messages = self.context.get_recent_context(n=5)
        conversation_context = "\n\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in recent_messages[:-1]  # Exclude current message
        ])

        # Report progress
        self._report_progress("analyzing", "Understanding your request...")

        # Build task with full context
        task = {
            "type": task_type,
            "description": user_message,
            "specification": user_message,
            "conversation_history": conversation_context,
            "files_modified": self.context.files_modified,
            "files_created": self.context.files_created,
            "workspace_root": str(self.workspace_root),
            **(additional_context or {})
        }

        self._report_progress("executing", f"Executing {task_type}...")

        # Execute task
        result = await agent.execute_task(task)

        # Process result
        response_content = self._format_response(result)

        # Add assistant response to history
        self.context.add_message(
            "assistant",
            response_content,
            metadata={"task_type": task_type, "success": result.success}
        )

        # Update file tracking
        if result.success and isinstance(result.output, dict):
            files_written = result.output.get("files_written", [])
            files_created = result.output.get("files_created", [])

            self.context.files_modified.extend(files_written)
            self.context.files_created.extend(files_created)

        self._report_progress("complete", "Done!")

        # Save session
        self.save_session()

        return {
            "success": result.success,
            "response": response_content,
            "output": result.output,
            "conversation_id": self.context.session_id,
            "message_count": len(self.context.messages)
        }

    def _format_response(self, result: Any) -> str:
        """Format agent response for display."""
        if not result.success:
            return f"❌ Task failed: {result.output}"

        if isinstance(result.output, dict):
            parts = []

            # Files created/modified
            if result.output.get("files_written"):
                files = result.output["files_written"]
                parts.append(f"✓ Modified {len(files)} file(s):")
                for file in files[:10]:  # Show first 10
                    parts.append(f"  - {file}")

            if result.output.get("files_created"):
                files = result.output["files_created"]
                parts.append(f"✓ Created {len(files)} file(s):")
                for file in files[:10]:
                    parts.append(f"  - {file}")

            # Implementation summary
            if result.output.get("implementation"):
                impl = result.output["implementation"]
                if isinstance(impl, str) and len(impl) < 500:
                    parts.append(f"\nImplementation:\n{impl}")

            # Review summary
            if result.output.get("review"):
                review = result.output["review"]
                if isinstance(review, str) and len(review) < 1000:
                    parts.append(f"\nReview:\n{review}")

            # Issue summary
            if result.output.get("issues"):
                issues = result.output["issues"]
                if isinstance(issues, list):
                    parts.append(f"\n⚠️  Found {len(issues)} issue(s)")
                    for issue in issues[:5]:
                        if isinstance(issue, dict):
                            severity = issue.get('severity', 'unknown')
                            desc = issue.get('description', str(issue))
                            parts.append(f"  - [{severity}] {desc}")

            return "\n".join(parts) if parts else "✓ Task completed successfully"

        return str(result.output)

    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        lines = [
            f"Session ID: {self.context.session_id}",
            f"Messages: {len(self.context.messages)}",
            f"Files Modified: {len(self.context.files_modified)}",
            f"Files Created: {len(self.context.files_created)}",
            "",
            "Recent conversation:"
        ]

        for msg in self.context.get_recent_context(n=3):
            time_str = msg.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{time_str}] {msg.role}: {msg.content}")

        return "\n".join(lines)

    def list_modified_files(self) -> List[str]:
        """Get list of all modified files in this session."""
        return list(set(self.context.files_modified + self.context.files_created))


class ProgressIndicator:
    """Simple progress indicator for terminal."""

    def __init__(self):
        self.current_stage = None
        self.current_message = None

    def update(self, stage: str, message: str):
        """Update progress."""
        self.current_stage = stage
        self.current_message = message

        # Print progress
        icons = {
            "analyzing": "🔍",
            "executing": "⚙️",
            "writing": "✍️",
            "reading": "📖",
            "complete": "✅",
            "error": "❌"
        }

        icon = icons.get(stage, "•")
        print(f"\r{icon} {stage.capitalize()}: {message}", end="", flush=True)

        if stage == "complete" or stage == "error":
            print()  # New line when done

    def clear(self):
        """Clear progress indicator."""
        print("\r" + " " * 80 + "\r", end="", flush=True)


# Convenience function for quick interactions
async def quick_chat(
    user_message: str,
    agent: Any,
    workspace_root: str = ".",
    show_progress: bool = True
) -> str:
    """
    Quick chat interaction without full session management.

    Args:
        user_message: User's message
        agent: Agent to use
        workspace_root: Working directory
        show_progress: Whether to show progress

    Returns:
        Response text
    """
    session = EnhancedInteractiveSession(workspace_root=workspace_root)

    if show_progress:
        progress = ProgressIndicator()
        session.set_progress_callback(progress.update)

    result = await session.chat(user_message, agent)

    return result["response"]
