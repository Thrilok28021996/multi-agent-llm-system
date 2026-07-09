"""Task manager for assigning and tracking agent tasks."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """Status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """A task to be executed by an agent."""

    id: str
    type: str
    description: str
    assigned_to: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "dependencies": self.dependencies,
        }


class TaskManager:
    """
    Manages task creation, assignment, and tracking for the company workflow.
    """

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_history: List[Task] = []

    def create_task(
        self,
        task_type: str,
        description: str,
        assigned_to: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        params: Dict[str, Any] = None,
        dependencies: List[str] = None
    ) -> Task:
        """
        Create a new task.

        Args:
            task_type: Type of task
            description: Task description
            assigned_to: Agent to assign the task to
            priority: Task priority
            params: Additional task parameters
            dependencies: List of task IDs this task depends on
        """
        task_id = f"TASK-{uuid.uuid4().hex[:8].upper()}"

        task = Task(
            id=task_id,
            type=task_type,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            params=params or {},
            dependencies=dependencies or []
        )

        self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def start_task(self, task_id: str) -> bool:
        """Mark a task as started."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        # Check dependencies
        for dep_id in task.dependencies:
            dep = self.tasks.get(dep_id)
            if dep and dep.status != TaskStatus.COMPLETED:
                return False  # Dependency not completed

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        return True

    def complete_task(self, task_id: str, result: Dict[str, Any] = None) -> bool:
        """Mark a task as completed."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = result

        # Move to history
        self.task_history.append(task)

        return True

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.error = error

        # Move to history
        self.task_history.append(task)

        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status == TaskStatus.COMPLETED:
            return False  # Cannot cancel completed task

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()

        return True

    def get_pending_tasks(self, agent_name: str = None) -> List[Task]:
        """Get pending tasks, optionally filtered by agent."""
        pending = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.PENDING
        ]

        if agent_name:
            pending = [t for t in pending if t.assigned_to == agent_name]

        return sorted(pending, key=lambda t: t.priority.value, reverse=True)

    def get_in_progress_tasks(self, agent_name: str = None) -> List[Task]:
        """Get in-progress tasks, optionally filtered by agent."""
        in_progress = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.IN_PROGRESS
        ]

        if agent_name:
            in_progress = [t for t in in_progress if t.assigned_to == agent_name]

        return in_progress

    def get_next_task(self, agent_name: str) -> Optional[Task]:
        """Get the next task for an agent to work on."""
        pending = self.get_pending_tasks(agent_name)

        for task in pending:
            # Check if all dependencies are met
            deps_met = all(
                self.tasks.get(dep_id) and
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_met:
                return task

        return None

    def get_agent_workload(self) -> Dict[str, Dict[str, int]]:
        """Get workload statistics per agent."""
        workload = {}

        for task in self.tasks.values():
            agent = task.assigned_to
            if agent not in workload:
                workload[agent] = {
                    "pending": 0,
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0
                }

            workload[agent][task.status.value] = \
                workload[agent].get(task.status.value, 0) + 1

        return workload

    def get_task_stats(self) -> Dict[str, Any]:
        """Get overall task statistics."""
        total = len(self.tasks)
        by_status = {}

        for task in self.tasks.values():
            status = task.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total": total,
            "by_status": by_status,
            "history_count": len(self.task_history)
        }

    def clear_completed(self) -> int:
        """Clear completed tasks from active list, keeping history."""
        completed_ids = [
            t.id for t in self.tasks.values()
            if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]

        for task_id in completed_ids:
            del self.tasks[task_id]

        return len(completed_ids)

    def export_tasks(self) -> List[Dict[str, Any]]:
        """Export all tasks as dictionaries."""
        return [t.to_dict() for t in self.tasks.values()]

    def export_history(self) -> List[Dict[str, Any]]:
        """Export task history as dictionaries."""
        return [t.to_dict() for t in self.task_history]
