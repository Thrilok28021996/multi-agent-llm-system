"""Dynamic agent pool — tracks headcount, performance, and role changes."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from ui.console import console


@dataclass
class AgentRecord:
    """Record of an agent in the pool."""

    name: str
    role: str
    hired_at: datetime = field(default_factory=datetime.now)
    performance_score: float = 0.5  # 0.0-1.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    is_active: bool = True
    termination_reason: Optional[str] = None


class AgentPool:
    """Tracks available agents, their load, and performance.

    Currently logs headcount changes. Actual multi-instance spawning
    is planned for a future version.
    """

    def __init__(self):
        self._agents: Dict[str, AgentRecord] = {}
        self._history: List[Dict] = []

    def register(self, name: str, role: str) -> None:
        """Register an agent in the pool."""
        self._agents[name] = AgentRecord(name=name, role=role)

    def hire_agent(self, role: str, reason: str) -> str:
        """Log a hiring event. Returns the new agent name."""
        # Count existing agents in this role
        existing = [a for a in self._agents.values() if a.role == role and a.is_active]
        new_name = f"{role}_{len(existing) + 1}"

        self._agents[new_name] = AgentRecord(name=new_name, role=role)
        self._history.append({
            "action": "hire",
            "agent": new_name,
            "role": role,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        console.info(f"[Hiring] New {role} hired: {new_name}. Reason: {reason}")
        return new_name

    def fire_agent(self, agent_name: str, reason: str) -> bool:
        """Log a termination event."""
        if agent_name not in self._agents:
            return False

        record = self._agents[agent_name]
        record.is_active = False
        record.termination_reason = reason
        self._history.append({
            "action": "fire",
            "agent": agent_name,
            "role": record.role,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        console.warning(f"[Termination] {agent_name} terminated. Reason: {reason}")
        return True

    def rotate_role(self, agent_name: str, new_role: str, reason: str) -> bool:
        """Log a role rotation event."""
        if agent_name not in self._agents:
            return False

        old_role = self._agents[agent_name].role
        self._agents[agent_name].role = new_role
        self._history.append({
            "action": "rotate",
            "agent": agent_name,
            "old_role": old_role,
            "new_role": new_role,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        console.info(f"[Role Change] {agent_name}: {old_role} -> {new_role}. Reason: {reason}")
        return True

    def record_task_outcome(self, agent_name: str, success: bool) -> None:
        """Record a task outcome for performance tracking."""
        if agent_name not in self._agents:
            return
        record = self._agents[agent_name]
        if success:
            record.tasks_completed += 1
        else:
            record.tasks_failed += 1
        total = record.tasks_completed + record.tasks_failed
        record.performance_score = record.tasks_completed / total if total > 0 else 0.5

    def get_underperformers(self, threshold: float = 0.3, min_tasks: int = 3) -> List[AgentRecord]:
        """Get agents performing below threshold with enough data."""
        return [
            a for a in self._agents.values()
            if a.is_active
            and (a.tasks_completed + a.tasks_failed) >= min_tasks
            and a.performance_score < threshold
        ]

    def get_active_agents(self) -> List[AgentRecord]:
        """Get all active agents."""
        return [a for a in self._agents.values() if a.is_active]

    def get_headcount_summary(self) -> Dict[str, int]:
        """Get active headcount by role."""
        summary: Dict[str, int] = {}
        for agent in self._agents.values():
            if agent.is_active:
                summary[agent.role] = summary.get(agent.role, 0) + 1
        return summary
