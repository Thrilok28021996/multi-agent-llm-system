"""
Hooks System for Company AGI - Claude Code style lifecycle hooks.

Supports:
- PreToolUse: Validate/block tool calls before execution
- PostToolUse: React to tool results (logging, auto-formatting)
- PermissionRequest: Custom approval logic
- SessionStart/SessionEnd: Session lifecycle
- AgentStart/AgentStop: Agent lifecycle
"""

import asyncio
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HookEvent(Enum):
    """Types of hook events."""
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    PERMISSION_REQUEST = "permission_request"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    WORKFLOW_PHASE_CHANGE = "workflow_phase_change"
    ERROR = "error"


class HookDecision(Enum):
    """Decision returned by hooks."""
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    ASK_USER = "ask_user"


@dataclass
class HookContext:
    """Context passed to hooks."""
    event: HookEvent
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    workflow_phase: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event.value,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": str(self.tool_output) if self.tool_output else None,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "workflow_phase": self.workflow_phase,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class HookResult:
    """Result from a hook execution."""
    decision: HookDecision
    message: Optional[str] = None
    modified_input: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Hook(ABC):
    """Abstract base class for hooks."""

    def __init__(
        self,
        name: str,
        events: List[HookEvent],
        priority: int = 100,
        enabled: bool = True
    ):
        self.name = name
        self.events = events
        self.priority = priority  # Lower = higher priority
        self.enabled = enabled

    @abstractmethod
    async def execute(self, context: HookContext) -> HookResult:
        """Execute the hook with given context."""
        pass

    def matches_event(self, event: HookEvent) -> bool:
        """Check if hook handles this event type."""
        return event in self.events


class CommandHook(Hook):
    """Hook that executes a shell command."""

    def __init__(
        self,
        name: str,
        command: str,
        events: List[HookEvent],
        timeout: int = 30,
        priority: int = 100,
        enabled: bool = True,
        tool_matcher: Optional[str] = None,  # Regex pattern to match tool names
    ):
        super().__init__(name, events, priority, enabled)
        self.command = command
        self.timeout = timeout
        self.tool_matcher = tool_matcher

    async def execute(self, context: HookContext) -> HookResult:
        """Execute shell command with context as JSON stdin."""
        import re

        # Check tool matcher if specified
        if self.tool_matcher and context.tool_name:
            if not re.match(self.tool_matcher, context.tool_name):
                return HookResult(decision=HookDecision.ALLOW)

        try:
            # Pass context as JSON to stdin
            process = await asyncio.create_subprocess_shell(
                self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            context_json = json.dumps(context.to_dict())
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=context_json.encode()),
                timeout=self.timeout
            )

            exit_code = process.returncode

            # Exit code interpretation (Claude Code style):
            # 0 = allow
            # 2 = block
            # Other = allow with warning
            if exit_code == 0:
                # Try to parse JSON output for modifications
                try:
                    output = json.loads(stdout.decode())
                    if "decision" in output:
                        decision = HookDecision(output["decision"])
                        return HookResult(
                            decision=decision,
                            message=output.get("message"),
                            modified_input=output.get("modified_input"),
                            metadata=output.get("metadata", {})
                        )
                except (json.JSONDecodeError, ValueError):
                    pass
                return HookResult(decision=HookDecision.ALLOW)

            elif exit_code == 2:
                message = stderr.decode().strip() or stdout.decode().strip()
                return HookResult(
                    decision=HookDecision.BLOCK,
                    message=message or f"Blocked by hook: {self.name}"
                )
            else:
                # Non-zero exit (not 2) = allow with warning
                return HookResult(
                    decision=HookDecision.ALLOW,
                    message=f"Hook {self.name} exited with code {exit_code}"
                )

        except asyncio.TimeoutError:
            return HookResult(
                decision=HookDecision.ALLOW,
                message=f"Hook {self.name} timed out after {self.timeout}s"
            )
        except Exception as e:
            return HookResult(
                decision=HookDecision.ALLOW,
                message=f"Hook {self.name} error: {str(e)}"
            )


class CallableHook(Hook):
    """Hook that executes a Python callable."""

    def __init__(
        self,
        name: str,
        callback: Callable[[HookContext], Union[HookResult, bool, None, Awaitable[HookResult]]],
        events: List[HookEvent],
        priority: int = 100,
        enabled: bool = True,
    ):
        super().__init__(name, events, priority, enabled)
        self.callback = callback

    async def execute(self, context: HookContext) -> HookResult:
        """Execute the callback."""
        try:
            if inspect.iscoroutinefunction(self.callback):
                result = await self.callback(context)
            else:
                result = self.callback(context)

            if result is None or result is True:
                return HookResult(decision=HookDecision.ALLOW)
            elif result is False:
                return HookResult(decision=HookDecision.BLOCK)
            elif isinstance(result, HookResult):
                return result
            else:
                return HookResult(decision=HookDecision.ALLOW)

        except Exception as e:
            return HookResult(
                decision=HookDecision.ALLOW,
                message=f"Hook {self.name} error: {str(e)}"
            )


class ToolValidatorHook(Hook):
    """Hook for validating specific tool usage patterns."""

    def __init__(
        self,
        name: str,
        tool_name: str,
        validator: Callable[[Dict[str, Any]], bool],
        block_message: str = "Validation failed",
        priority: int = 50,
        enabled: bool = True,
    ):
        super().__init__(name, [HookEvent.PRE_TOOL_USE], priority, enabled)
        self.tool_name = tool_name
        self.validator = validator
        self.block_message = block_message

    async def execute(self, context: HookContext) -> HookResult:
        """Validate tool input."""
        if context.tool_name != self.tool_name:
            return HookResult(decision=HookDecision.ALLOW)

        try:
            if self.validator(context.tool_input or {}):
                return HookResult(decision=HookDecision.ALLOW)
            else:
                return HookResult(
                    decision=HookDecision.BLOCK,
                    message=self.block_message
                )
        except Exception as e:
            return HookResult(
                decision=HookDecision.BLOCK,
                message=f"Validation error: {str(e)}"
            )


class FileProtectionHook(Hook):
    """Hook that protects certain files from modification."""

    def __init__(
        self,
        name: str = "file_protection",
        protected_patterns: Optional[List[str]] = None,
        priority: int = 10,
        enabled: bool = True,
    ):
        super().__init__(name, [HookEvent.PRE_TOOL_USE], priority, enabled)
        self.protected_patterns = protected_patterns or [
            "*.env",
            "*.pem",
            "*.key",
            "**/secrets/*",
            "**/.git/*",
            "**/node_modules/*",
        ]

    async def execute(self, context: HookContext) -> HookResult:
        """Check if tool is trying to modify protected files."""
        import fnmatch

        # Only check write/edit tools
        write_tools = {"write", "edit", "multi_edit", "bash"}
        if context.tool_name not in write_tools:
            return HookResult(decision=HookDecision.ALLOW)

        # Get file path from tool input
        file_path = None
        if context.tool_input:
            file_path = context.tool_input.get("path") or context.tool_input.get("file_path")

        if not file_path:
            return HookResult(decision=HookDecision.ALLOW)

        # Check against protected patterns
        for pattern in self.protected_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return HookResult(
                    decision=HookDecision.BLOCK,
                    message=f"File '{file_path}' is protected (matches pattern: {pattern})"
                )

        return HookResult(decision=HookDecision.ALLOW)


class AuditLogHook(Hook):
    """Hook that logs all tool usage for audit purposes."""

    def __init__(
        self,
        name: str = "audit_log",
        log_file: Optional[str] = None,
        priority: int = 1000,  # Run last
        enabled: bool = True,
    ):
        super().__init__(
            name,
            [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE],
            priority,
            enabled
        )
        self.log_file = log_file or "output/logs/audit.jsonl"

    async def execute(self, context: HookContext) -> HookResult:
        """Log the tool usage."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            **context.to_dict(),
            "hook": self.name,
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return HookResult(decision=HookDecision.ALLOW)


class HooksManager:
    """Manages and executes hooks."""

    def __init__(self, config_file: Optional[str] = None):
        self.hooks: List[Hook] = []
        self.config_file = config_file
        self._load_config()

    def _load_config(self) -> None:
        """Load hooks from config file."""
        if not self.config_file:
            return

        config_path = Path(self.config_file)
        if not config_path.exists():
            return

        try:
            with open(config_path) as f:
                config = json.load(f)

            for hook_config in config.get("hooks", []):
                hook = self._create_hook_from_config(hook_config)
                if hook:
                    self.hooks.append(hook)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load hooks config: %s", e)

    def _create_hook_from_config(self, config: Dict[str, Any]) -> Optional[Hook]:
        """Create a hook from configuration."""
        hook_type = config.get("type", "command")
        name = config.get("name", "unnamed")
        events = [HookEvent(e) for e in config.get("events", [])]
        priority = config.get("priority", 100)
        enabled = config.get("enabled", True)

        if hook_type == "command":
            return CommandHook(
                name=name,
                command=config.get("command", ""),
                events=events,
                timeout=config.get("timeout", 30),
                priority=priority,
                enabled=enabled,
                tool_matcher=config.get("tool_matcher"),
            )
        elif hook_type == "file_protection":
            return FileProtectionHook(
                name=name,
                protected_patterns=config.get("protected_patterns"),
                priority=priority,
                enabled=enabled,
            )
        elif hook_type == "audit_log":
            return AuditLogHook(
                name=name,
                log_file=config.get("log_file"),
                priority=priority,
                enabled=enabled,
            )

        return None

    def register(self, hook: Hook) -> None:
        """Register a new hook."""
        self.hooks.append(hook)
        # Sort by priority (lower = higher priority)
        self.hooks.sort(key=lambda h: h.priority)

    def unregister(self, name: str) -> bool:
        """Unregister a hook by name."""
        original_len = len(self.hooks)
        self.hooks = [h for h in self.hooks if h.name != name]
        return len(self.hooks) < original_len

    def enable(self, name: str) -> bool:
        """Enable a hook by name."""
        for hook in self.hooks:
            if hook.name == name:
                hook.enabled = True
                return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a hook by name."""
        for hook in self.hooks:
            if hook.name == name:
                hook.enabled = False
                return True
        return False

    async def execute(self, context: HookContext) -> HookResult:
        """Execute all matching hooks for an event."""
        matching_hooks = [
            h for h in self.hooks
            if h.enabled and h.matches_event(context.event)
        ]

        for hook in matching_hooks:
            result = await hook.execute(context)

            # If any hook blocks, stop immediately
            if result.decision == HookDecision.BLOCK:
                return result

            # If hook modifies input, update context
            if result.decision == HookDecision.MODIFY and result.modified_input:
                context.tool_input = result.modified_input

        return HookResult(decision=HookDecision.ALLOW)

    async def pre_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> HookResult:
        """Execute pre-tool-use hooks."""
        context = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name=tool_name,
            tool_input=tool_input,
            agent_name=agent_name,
            session_id=session_id,
        )
        return await self.execute(context)

    async def post_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> HookResult:
        """Execute post-tool-use hooks."""
        context = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            agent_name=agent_name,
            session_id=session_id,
        )
        return await self.execute(context)

    async def session_start(self, session_id: str) -> HookResult:
        """Execute session start hooks."""
        context = HookContext(
            event=HookEvent.SESSION_START,
            session_id=session_id,
        )
        return await self.execute(context)

    async def session_end(self, session_id: str) -> HookResult:
        """Execute session end hooks."""
        context = HookContext(
            event=HookEvent.SESSION_END,
            session_id=session_id,
        )
        return await self.execute(context)

    async def agent_start(self, agent_name: str, session_id: Optional[str] = None) -> HookResult:
        """Execute agent start hooks."""
        context = HookContext(
            event=HookEvent.AGENT_START,
            agent_name=agent_name,
            session_id=session_id,
        )
        return await self.execute(context)

    async def agent_stop(self, agent_name: str, session_id: Optional[str] = None) -> HookResult:
        """Execute agent stop hooks."""
        context = HookContext(
            event=HookEvent.AGENT_STOP,
            agent_name=agent_name,
            session_id=session_id,
        )
        return await self.execute(context)


# Convenience function for creating common hooks
def create_bash_validator(
    allowed_commands: Optional[List[str]] = None,
    blocked_commands: Optional[List[str]] = None,
) -> ToolValidatorHook:
    """Create a Bash command validator hook."""
    blocked = blocked_commands or ["rm -rf /", "sudo rm", ":(){ :|:& };:"]
    allowed = allowed_commands

    def validator(input: Dict[str, Any]) -> bool:
        command = input.get("command", "")

        # Check blocked patterns
        for blocked_cmd in blocked:
            if blocked_cmd in command:
                return False

        # If allowed list specified, check against it
        if allowed:
            cmd_start = command.split()[0] if command.split() else ""
            return cmd_start in allowed

        return True

    return ToolValidatorHook(
        name="bash_validator",
        tool_name="bash",
        validator=validator,
        block_message="Command not allowed by security policy",
    )


def create_auto_formatter_hook(
    formatters: Optional[Dict[str, str]] = None
) -> CallableHook:
    """Create a post-tool hook that auto-formats files after writes."""
    import subprocess

    default_formatters = {
        ".py": "black {file}",
        ".js": "prettier --write {file}",
        ".ts": "prettier --write {file}",
        ".json": "prettier --write {file}",
    }
    formatters = formatters or default_formatters

    async def format_callback(context: HookContext) -> HookResult:
        if context.tool_name not in ("write", "edit"):
            return HookResult(decision=HookDecision.ALLOW)

        file_path = context.tool_input.get("path") if context.tool_input else None
        if not file_path:
            return HookResult(decision=HookDecision.ALLOW)

        # Get file extension
        ext = Path(file_path).suffix
        if ext not in formatters:
            return HookResult(decision=HookDecision.ALLOW)

        # Run formatter (quote path to prevent command injection)
        import shlex
        cmd = formatters[ext].format(file=shlex.quote(file_path))
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
        except Exception:
            pass  # Ignore formatting failures

        return HookResult(decision=HookDecision.ALLOW)

    return CallableHook(
        name="auto_formatter",
        callback=format_callback,
        events=[HookEvent.POST_TOOL_USE],
        priority=500,
    )


# Global hooks manager instance
_global_hooks_manager: Optional[HooksManager] = None


def get_hooks_manager(config_file: Optional[str] = None) -> HooksManager:
    """Get or create the global hooks manager."""
    global _global_hooks_manager
    if _global_hooks_manager is None:
        _global_hooks_manager = HooksManager(config_file)
    return _global_hooks_manager


def reset_hooks_manager() -> None:
    """Reset the global hooks manager."""
    global _global_hooks_manager
    _global_hooks_manager = None
