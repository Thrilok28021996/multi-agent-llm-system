"""
Permission System for Company AGI - Granular access control.

Provides:
- Hierarchical permission levels (managed > project > user)
- Tool-specific permissions (allow/deny/ask)
- Path-based access control
- Agent-specific permissions
- Audit logging of permission decisions
"""

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission decision levels."""
    ALLOW = "allow"  # Always allow
    DENY = "deny"  # Always deny
    ASK = "ask"  # Ask user for permission
    ALLOW_ONCE = "allow_once"  # Allow this time only
    DENY_ONCE = "deny_once"  # Deny this time only


class PermissionScope(Enum):
    """Scope of permission rules."""
    MANAGED = "managed"  # Enterprise-level, cannot be overridden
    PROJECT = "project"  # Project-level settings
    USER = "user"  # User-level settings
    SESSION = "session"  # Session-only settings


@dataclass
class PermissionRule:
    """A permission rule for access control."""
    id: str
    scope: PermissionScope
    tool: Optional[str] = None  # Tool name or "*" for all
    path_pattern: Optional[str] = None  # File path pattern
    agent: Optional[str] = None  # Agent name or "*" for all
    action: Optional[str] = None  # Specific action (read, write, execute)
    decision: PermissionLevel = PermissionLevel.ASK
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        tool: Optional[str] = None,
        path: Optional[str] = None,
        agent: Optional[str] = None,
        action: Optional[str] = None,
    ) -> bool:
        """Check if this rule matches the given context."""
        # Check expiration
        if self.expires_at and datetime.now() > self.expires_at:
            return False

        # Check tool
        if self.tool and self.tool != "*":
            if tool != self.tool:
                return False

        # Check path
        if self.path_pattern:
            if not path or not fnmatch.fnmatch(path, self.path_pattern):
                return False

        # Check agent
        if self.agent and self.agent != "*":
            if agent != self.agent:
                return False

        # Check action
        if self.action:
            if action != self.action:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope.value,
            "tool": self.tool,
            "path_pattern": self.path_pattern,
            "agent": self.agent,
            "action": self.action,
            "decision": self.decision.value,
            "reason": self.reason,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionRule":
        return cls(
            id=data["id"],
            scope=PermissionScope(data["scope"]),
            tool=data.get("tool"),
            path_pattern=data.get("path_pattern"),
            agent=data.get("agent"),
            action=data.get("action"),
            decision=PermissionLevel(data.get("decision", "ask")),
            reason=data.get("reason"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class PermissionRequest:
    """A request for permission."""
    tool: str
    action: str
    path: Optional[str] = None
    agent: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionDecision:
    """The result of a permission check."""
    allowed: bool
    level: PermissionLevel
    rule_id: Optional[str] = None
    reason: Optional[str] = None
    needs_user_input: bool = False


class PermissionAuditLog:
    """Logs permission decisions for audit purposes."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or "output/logs/permissions.jsonl"
        self.entries: List[Dict[str, Any]] = []

    def log(
        self,
        request: PermissionRequest,
        decision: PermissionDecision,
        user_response: Optional[str] = None,
    ) -> None:
        """Log a permission decision."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": request.tool,
            "action": request.action,
            "path": request.path,
            "agent": request.agent,
            "allowed": decision.allowed,
            "level": decision.level.value,
            "rule_id": decision.rule_id,
            "reason": decision.reason,
            "user_response": user_response,
        }

        self.entries.append(entry)

        # Write to file
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class PermissionManager:
    """Manages permission rules and decisions."""

    def __init__(
        self,
        config_file: Optional[str] = None,
        ask_callback: Optional[Callable[[PermissionRequest], bool]] = None,
    ):
        self.rules: List[PermissionRule] = []
        self.config_file = config_file
        self.ask_callback = ask_callback
        self.audit_log = PermissionAuditLog()
        self._session_grants: Dict[str, PermissionLevel] = {}
        self._rule_counter = 0

        self._load_config()

    def _load_config(self) -> None:
        """Load permission rules from config files."""
        config_paths = []

        # Project config
        if self.config_file:
            config_paths.append(Path(self.config_file))
        else:
            config_paths.append(Path(".claude/permissions.json"))

        # User config
        config_paths.append(Path.home() / ".claude" / "permissions.json")

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    for rule_data in config.get("rules", []):
                        rule = PermissionRule.from_dict(rule_data)
                        self.rules.append(rule)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Failed to load permissions from %s: %s", config_path, e)

        # Sort rules by scope (managed first, then project, then user)
        scope_order = {
            PermissionScope.MANAGED: 0,
            PermissionScope.PROJECT: 1,
            PermissionScope.USER: 2,
            PermissionScope.SESSION: 3,
        }
        self.rules.sort(key=lambda r: scope_order.get(r.scope, 99))

    def add_rule(self, rule: PermissionRule) -> None:
        """Add a permission rule."""
        self.rules.append(rule)
        # Re-sort
        scope_order = {
            PermissionScope.MANAGED: 0,
            PermissionScope.PROJECT: 1,
            PermissionScope.USER: 2,
            PermissionScope.SESSION: 3,
        }
        self.rules.sort(key=lambda r: scope_order.get(r.scope, 99))

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a permission rule by ID."""
        original_len = len(self.rules)
        self.rules = [r for r in self.rules if r.id != rule_id]
        return len(self.rules) < original_len

    def create_rule(
        self,
        tool: Optional[str] = None,
        path_pattern: Optional[str] = None,
        agent: Optional[str] = None,
        action: Optional[str] = None,
        decision: PermissionLevel = PermissionLevel.ASK,
        scope: PermissionScope = PermissionScope.SESSION,
        reason: Optional[str] = None,
    ) -> PermissionRule:
        """Create and add a new permission rule."""
        self._rule_counter += 1
        rule = PermissionRule(
            id=f"rule_{self._rule_counter}",
            scope=scope,
            tool=tool,
            path_pattern=path_pattern,
            agent=agent,
            action=action,
            decision=decision,
            reason=reason,
        )
        self.add_rule(rule)
        return rule

    def check(self, request: PermissionRequest) -> PermissionDecision:
        """Check if an action is permitted."""
        # Check session grants first
        session_key = self._get_session_key(request)
        if session_key in self._session_grants:
            level = self._session_grants[session_key]
            allowed = level in (PermissionLevel.ALLOW, PermissionLevel.ALLOW_ONCE)
            if level in (PermissionLevel.ALLOW_ONCE, PermissionLevel.DENY_ONCE):
                del self._session_grants[session_key]
            return PermissionDecision(
                allowed=allowed,
                level=level,
                reason="Session grant",
            )

        # Find matching rule
        for rule in self.rules:
            if rule.matches(
                tool=request.tool,
                path=request.path,
                agent=request.agent,
                action=request.action,
            ):
                if rule.decision == PermissionLevel.ALLOW:
                    decision = PermissionDecision(
                        allowed=True,
                        level=rule.decision,
                        rule_id=rule.id,
                        reason=rule.reason,
                    )
                elif rule.decision == PermissionLevel.DENY:
                    decision = PermissionDecision(
                        allowed=False,
                        level=rule.decision,
                        rule_id=rule.id,
                        reason=rule.reason or "Denied by permission rule",
                    )
                else:  # ASK
                    decision = PermissionDecision(
                        allowed=False,
                        level=rule.decision,
                        rule_id=rule.id,
                        needs_user_input=True,
                        reason="User approval required",
                    )

                self.audit_log.log(request, decision)
                return decision

        # Default: ask
        decision = PermissionDecision(
            allowed=False,
            level=PermissionLevel.ASK,
            needs_user_input=True,
            reason="No matching rule, user approval required",
        )
        self.audit_log.log(request, decision)
        return decision

    async def check_and_ask(self, request: PermissionRequest) -> PermissionDecision:
        """Check permission and ask user if needed."""
        decision = self.check(request)

        if decision.needs_user_input and self.ask_callback:
            # Ask user
            user_allowed = self.ask_callback(request)

            # Update decision
            decision = PermissionDecision(
                allowed=user_allowed,
                level=PermissionLevel.ALLOW_ONCE if user_allowed else PermissionLevel.DENY_ONCE,
                reason="User decision",
            )

            # Log with user response
            self.audit_log.log(
                request, decision,
                user_response="allowed" if user_allowed else "denied"
            )

        return decision

    def grant_session(
        self,
        tool: Optional[str] = None,
        path: Optional[str] = None,
        agent: Optional[str] = None,
        action: Optional[str] = None,
        level: PermissionLevel = PermissionLevel.ALLOW,
    ) -> None:
        """Grant a session-level permission."""
        key = self._get_session_key(PermissionRequest(
            tool=tool or "*",
            action=action or "*",
            path=path,
            agent=agent,
        ))
        self._session_grants[key] = level

    def revoke_session(
        self,
        tool: Optional[str] = None,
        path: Optional[str] = None,
        agent: Optional[str] = None,
        action: Optional[str] = None,
    ) -> bool:
        """Revoke a session-level permission."""
        key = self._get_session_key(PermissionRequest(
            tool=tool or "*",
            action=action or "*",
            path=path,
            agent=agent,
        ))
        if key in self._session_grants:
            del self._session_grants[key]
            return True
        return False

    def clear_session_grants(self) -> None:
        """Clear all session-level grants."""
        self._session_grants.clear()

    def _get_session_key(self, request: PermissionRequest) -> str:
        """Generate a key for session grants."""
        return f"{request.tool}:{request.action}:{request.path}:{request.agent}"

    def get_rules_for_tool(self, tool: str) -> List[PermissionRule]:
        """Get all rules that apply to a tool."""
        return [r for r in self.rules if r.tool in (tool, "*", None)]

    def get_rules_for_agent(self, agent: str) -> List[PermissionRule]:
        """Get all rules that apply to an agent."""
        return [r for r in self.rules if r.agent in (agent, "*", None)]

    def save_rules(self, config_file: Optional[str] = None) -> bool:
        """Save current rules to config file."""
        path = Path(config_file or self.config_file or ".claude/permissions.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Only save project and user scope rules (not managed or session)
            saveable_rules = [
                r for r in self.rules
                if r.scope in (PermissionScope.PROJECT, PermissionScope.USER)
            ]

            config = {
                "rules": [r.to_dict() for r in saveable_rules],
            }

            with open(path, "w") as f:
                json.dump(config, f, indent=2)

            return True
        except Exception:
            return False


# Convenience functions for common permission patterns

def create_readonly_permissions() -> List[PermissionRule]:
    """Create rules for read-only access."""
    return [
        PermissionRule(
            id="readonly_allow_read",
            scope=PermissionScope.SESSION,
            tool="read",
            decision=PermissionLevel.ALLOW,
        ),
        PermissionRule(
            id="readonly_allow_glob",
            scope=PermissionScope.SESSION,
            tool="glob",
            decision=PermissionLevel.ALLOW,
        ),
        PermissionRule(
            id="readonly_allow_grep",
            scope=PermissionScope.SESSION,
            tool="grep",
            decision=PermissionLevel.ALLOW,
        ),
        PermissionRule(
            id="readonly_deny_write",
            scope=PermissionScope.SESSION,
            tool="write",
            decision=PermissionLevel.DENY,
            reason="Read-only mode",
        ),
        PermissionRule(
            id="readonly_deny_edit",
            scope=PermissionScope.SESSION,
            tool="edit",
            decision=PermissionLevel.DENY,
            reason="Read-only mode",
        ),
        PermissionRule(
            id="readonly_deny_bash",
            scope=PermissionScope.SESSION,
            tool="bash",
            decision=PermissionLevel.DENY,
            reason="Read-only mode",
        ),
    ]


def create_development_permissions(workspace: str) -> List[PermissionRule]:
    """Create rules for development work."""
    return [
        # Allow all file operations in workspace
        PermissionRule(
            id="dev_allow_workspace",
            scope=PermissionScope.PROJECT,
            path_pattern=f"{workspace}/**",
            decision=PermissionLevel.ALLOW,
        ),
        # Block sensitive files
        PermissionRule(
            id="dev_block_env",
            scope=PermissionScope.PROJECT,
            path_pattern="**/.env*",
            decision=PermissionLevel.DENY,
            reason="Environment files are protected",
        ),
        PermissionRule(
            id="dev_block_secrets",
            scope=PermissionScope.PROJECT,
            path_pattern="**/secrets/**",
            decision=PermissionLevel.DENY,
            reason="Secrets directory is protected",
        ),
        # Allow common dev commands
        PermissionRule(
            id="dev_allow_git",
            scope=PermissionScope.PROJECT,
            tool="bash",
            action="git *",
            decision=PermissionLevel.ALLOW,
        ),
        PermissionRule(
            id="dev_allow_npm",
            scope=PermissionScope.PROJECT,
            tool="bash",
            action="npm *",
            decision=PermissionLevel.ALLOW,
        ),
        PermissionRule(
            id="dev_allow_python",
            scope=PermissionScope.PROJECT,
            tool="bash",
            action="python *",
            decision=PermissionLevel.ALLOW,
        ),
    ]


def create_agent_permissions(agent_name: str, allowed_tools: List[str]) -> List[PermissionRule]:
    """Create permission rules for a specific agent."""
    rules = []

    for tool in allowed_tools:
        rules.append(PermissionRule(
            id=f"{agent_name}_allow_{tool}",
            scope=PermissionScope.SESSION,
            agent=agent_name,
            tool=tool,
            decision=PermissionLevel.ALLOW,
        ))

    # Deny all other tools for this agent
    rules.append(PermissionRule(
        id=f"{agent_name}_deny_others",
        scope=PermissionScope.SESSION,
        agent=agent_name,
        tool="*",
        decision=PermissionLevel.DENY,
        reason=f"Tool not allowed for {agent_name}",
    ))

    return rules


# Global permission manager instance
_global_permission_manager: Optional[PermissionManager] = None


def get_permission_manager(
    config_file: Optional[str] = None,
    ask_callback: Optional[Callable[[PermissionRequest], bool]] = None,
) -> PermissionManager:
    """Get or create the global permission manager."""
    global _global_permission_manager
    if _global_permission_manager is None:
        _global_permission_manager = PermissionManager(config_file, ask_callback)
    return _global_permission_manager


def reset_permission_manager() -> None:
    """Reset the global permission manager."""
    global _global_permission_manager
    _global_permission_manager = None
