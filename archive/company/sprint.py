"""
Sprint planning, execution tracking, and retrospective helpers.

All summaries are template-based -- no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import uuid


class SprintStatus(Enum):
    """Lifecycle states of a sprint or sprint task."""

    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SprintTask:
    """A single unit of work inside a sprint."""

    id: str
    name: str
    assigned_to: str
    status: SprintStatus = SprintStatus.PLANNED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Sprint:
    """A time-boxed iteration containing one or more tasks."""

    id: str
    name: str
    goal: str
    tasks: List[SprintTask] = field(default_factory=list)
    status: SprintStatus = SprintStatus.PLANNED
    duration_target_minutes: int = 30
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def completed_tasks(self) -> List[SprintTask]:
        return [t for t in self.tasks if t.status == SprintStatus.COMPLETED]

    @property
    def failed_tasks(self) -> List[SprintTask]:
        return [t for t in self.tasks if t.status == SprintStatus.FAILED]

    @property
    def active_tasks(self) -> List[SprintTask]:
        return [t for t in self.tasks if t.status == SprintStatus.ACTIVE]

    @property
    def planned_tasks(self) -> List[SprintTask]:
        return [t for t in self.tasks if t.status == SprintStatus.PLANNED]

    @property
    def progress_pct(self) -> float:
        if not self.tasks:
            return 0.0
        return len(self.completed_tasks) / len(self.tasks) * 100.0


class SprintManager:
    """Creates, tracks, and summarises sprints.

    All summary methods produce deterministic, template-based text.
    """

    def __init__(self) -> None:
        self._sprints: Dict[str, Sprint] = {}
        self._current_sprint_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_sprint(
        self,
        name: str,
        goal: str,
        tasks: List[SprintTask],
    ) -> Sprint:
        """Create a new sprint in PLANNED state and return it."""
        sprint_id = uuid.uuid4().hex[:12]
        sprint = Sprint(
            id=sprint_id,
            name=name,
            goal=goal,
            tasks=list(tasks),
            status=SprintStatus.PLANNED,
        )
        self._sprints[sprint_id] = sprint
        return sprint

    def start_sprint(self, sprint_id: str) -> None:
        """Transition a sprint from PLANNED to ACTIVE."""
        sprint = self._get_sprint(sprint_id)
        if sprint.status != SprintStatus.PLANNED:
            raise ValueError(
                f"Cannot start sprint '{sprint_id}': current status is "
                f"{sprint.status.value}, expected 'planned'."
            )
        sprint.status = SprintStatus.ACTIVE
        sprint.started_at = datetime.now(timezone.utc)
        self._current_sprint_id = sprint_id

    def complete_task(self, sprint_id: str, task_id: str) -> None:
        """Mark a task inside a sprint as COMPLETED."""
        sprint = self._get_sprint(sprint_id)
        task = self._get_task(sprint, task_id)
        task.status = SprintStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)

    def end_sprint(self, sprint_id: str) -> None:
        """End a sprint, marking it COMPLETED or FAILED based on task outcomes."""
        sprint = self._get_sprint(sprint_id)
        if sprint.status != SprintStatus.ACTIVE:
            raise ValueError(
                f"Cannot end sprint '{sprint_id}': current status is "
                f"{sprint.status.value}, expected 'active'."
            )

        sprint.completed_at = datetime.now(timezone.utc)

        # Mark any remaining planned/active tasks as failed.
        for task in sprint.tasks:
            if task.status in (SprintStatus.PLANNED, SprintStatus.ACTIVE):
                task.status = SprintStatus.FAILED

        all_completed = all(
            t.status == SprintStatus.COMPLETED for t in sprint.tasks
        )
        sprint.status = SprintStatus.COMPLETED if all_completed else SprintStatus.FAILED

        if self._current_sprint_id == sprint_id:
            self._current_sprint_id = None

    # ------------------------------------------------------------------
    # Summaries (template-based, no LLM)
    # ------------------------------------------------------------------

    def run_standup(self, sprint: Sprint) -> str:
        """Return a brief daily-standup summary for *sprint*."""
        total = len(sprint.tasks)
        done = len(sprint.completed_tasks)
        active = len(sprint.active_tasks)
        planned = len(sprint.planned_tasks)
        failed = len(sprint.failed_tasks)

        lines: List[str] = [
            f"=== Standup: {sprint.name} ===",
            f"Goal: {sprint.goal}",
            f"Status: {sprint.status.value}",
            f"Progress: {done}/{total} tasks completed ({sprint.progress_pct:.0f}%)",
        ]

        if active:
            lines.append(f"In-progress: {active} task(s)")
            for t in sprint.active_tasks:
                lines.append(f"  - {t.name} (assigned to {t.assigned_to})")

        if planned:
            lines.append(f"Remaining: {planned} task(s) not yet started")

        if failed:
            lines.append(f"Failed: {failed} task(s)")
            for t in sprint.failed_tasks:
                lines.append(f"  - {t.name} (assigned to {t.assigned_to})")

        return "\n".join(lines)

    def run_retrospective(self, sprint: Sprint) -> str:
        """Return a template-based retrospective summary for *sprint*."""
        total = len(sprint.tasks)
        done = len(sprint.completed_tasks)
        failed = len(sprint.failed_tasks)

        duration_str = "N/A"
        if sprint.started_at and sprint.completed_at:
            delta = sprint.completed_at - sprint.started_at
            minutes = delta.total_seconds() / 60.0
            duration_str = f"{minutes:.1f} min"

        lines: List[str] = [
            f"=== Retrospective: {sprint.name} ===",
            f"Goal: {sprint.goal}",
            f"Outcome: {sprint.status.value}",
            f"Duration: {duration_str} (target: {sprint.duration_target_minutes} min)",
            f"Tasks completed: {done}/{total}",
        ]

        if failed:
            lines.append("")
            lines.append("What went wrong:")
            for t in sprint.failed_tasks:
                lines.append(f"  - {t.name} ({t.assigned_to}) did not complete")

        lines.append("")
        if done == total:
            lines.append("All tasks completed successfully. Well done.")
        elif done > total // 2:
            lines.append(
                "Majority of tasks completed. Review failures to improve "
                "next sprint planning."
            )
        else:
            lines.append(
                "Less than half the tasks completed. Consider reducing scope "
                "or investigating blockers before the next sprint."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_current_sprint(self) -> Optional[Sprint]:
        """Return the currently active sprint, or ``None``."""
        if self._current_sprint_id is None:
            return None
        return self._sprints.get(self._current_sprint_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sprint(self, sprint_id: str) -> Sprint:
        try:
            return self._sprints[sprint_id]
        except KeyError:
            raise ValueError(f"Sprint '{sprint_id}' not found.") from None

    @staticmethod
    def _get_task(sprint: Sprint, task_id: str) -> SprintTask:
        for task in sprint.tasks:
            if task.id == task_id:
                return task
        raise ValueError(
            f"Task '{task_id}' not found in sprint '{sprint.id}'."
        )
