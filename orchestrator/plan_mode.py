"""
Plan Mode for Company AGI.

Provides Claude Code-style planning with:
- Interactive planning with user approval
- Plan file creation and management
- Permission pre-authorization
- Step-by-step execution tracking
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class PlanStatus(Enum):
    """Status of a plan."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan."""
    id: str
    title: str
    description: str
    status: StepStatus = StepStatus.PENDING
    order: int = 0
    dependencies: List[str] = field(default_factory=list)  # Step IDs
    estimated_tokens: int = 0
    actual_tokens: int = 0
    tools_required: List[str] = field(default_factory=list)
    permissions_required: List[Dict[str, str]] = field(default_factory=list)
    output: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "order": self.order,
            "dependencies": self.dependencies,
            "estimated_tokens": self.estimated_tokens,
            "actual_tokens": self.actual_tokens,
            "tools_required": self.tools_required,
            "permissions_required": self.permissions_required,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            status=StepStatus(data.get("status", "pending")),
            order=data.get("order", 0),
            dependencies=data.get("dependencies", []),
            estimated_tokens=data.get("estimated_tokens", 0),
            actual_tokens=data.get("actual_tokens", 0),
            tools_required=data.get("tools_required", []),
            permissions_required=data.get("permissions_required", []),
            output=data.get("output"),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class AllowedPrompt:
    """Pre-authorized permission for plan execution."""
    tool: str
    prompt: str  # Semantic description like "run tests", "install dependencies"
    granted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "prompt": self.prompt,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllowedPrompt":
        return cls(
            tool=data["tool"],
            prompt=data["prompt"],
            granted_at=datetime.fromisoformat(data["granted_at"]) if data.get("granted_at") else None,
        )


@dataclass
class Plan:
    """A complete execution plan."""
    id: str
    name: str
    description: str
    status: PlanStatus = PlanStatus.DRAFT
    steps: List[PlanStep] = field(default_factory=list)
    allowed_prompts: List[AllowedPrompt] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "agent"
    approved_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        title: str,
        description: str,
        tools_required: Optional[List[str]] = None,
        permissions_required: Optional[List[Dict[str, str]]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> PlanStep:
        """Add a step to the plan."""
        step = PlanStep(
            id=f"step_{len(self.steps) + 1}",
            title=title,
            description=description,
            order=len(self.steps),
            tools_required=tools_required or [],
            permissions_required=permissions_required or [],
            dependencies=dependencies or [],
        )
        self.steps.append(step)
        return step

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                # Check dependencies
                deps_met = all(
                    self.get_step(dep_id) and
                    self.get_step(dep_id).status == StepStatus.COMPLETED  # type: ignore
                    for dep_id in step.dependencies
                )
                if deps_met:
                    return step
        return None

    def get_progress(self) -> Dict[str, Any]:
        """Get plan progress statistics."""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        in_progress = sum(1 for s in self.steps if s.status == StepStatus.IN_PROGRESS)
        pending = sum(1 for s in self.steps if s.status == StepStatus.PENDING)

        return {
            "total_steps": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "progress_percentage": (completed / total * 100) if total > 0 else 0,
        }

    def to_markdown(self) -> str:
        """Convert plan to markdown format."""
        lines = [
            f"# {self.name}",
            "",
            f"**Status:** {self.status.value}",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Overview",
            self.description,
            "",
            "## Steps",
            "",
        ]

        for step in self.steps:
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }.get(step.status, "•")

            lines.append(f"{step.order + 1}. {status_icon} **{step.title}**")
            lines.append(f"   {step.description}")
            if step.tools_required:
                lines.append(f"   - Tools: {', '.join(step.tools_required)}")
            lines.append("")

        if self.allowed_prompts:
            lines.extend([
                "## Requested Permissions",
                "",
            ])
            for prompt in self.allowed_prompts:
                lines.append(f"- **{prompt.tool}**: {prompt.prompt}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "allowed_prompts": [p.to_dict() for p in self.allowed_prompts],
            "created_at": self.created_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by,
            "approved_by": self.approved_by,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        """Create from dictionary."""
        plan = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            status=PlanStatus(data.get("status", "draft")),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", "agent"),
            approved_by=data.get("approved_by"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )
        plan.steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        plan.allowed_prompts = [AllowedPrompt.from_dict(p) for p in data.get("allowed_prompts", [])]
        if data.get("approved_at"):
            plan.approved_at = datetime.fromisoformat(data["approved_at"])
        if data.get("completed_at"):
            plan.completed_at = datetime.fromisoformat(data["completed_at"])
        return plan


class PlanManager:
    """
    Manages plans and their execution.

    Features:
    - Plan creation and storage
    - User approval workflow
    - Permission pre-authorization
    - Execution tracking
    """

    def __init__(
        self,
        plans_dir: str = ".claude/plans",
        auto_save: bool = True,
    ):
        self.plans_dir = Path(plans_dir)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        self.current_plan: Optional[Plan] = None
        self.plans: Dict[str, Plan] = {}
        self._approval_callback: Optional[Callable[[Plan], bool]] = None
        self._load_existing_plans()

    def _load_existing_plans(self) -> None:
        """Load existing plans from storage."""
        for path in self.plans_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                plan = Plan.from_dict(data)
                self.plans[plan.id] = plan
            except Exception:
                continue

    def _generate_plan_name(self) -> str:
        """Generate a unique plan name."""
        adjectives = ["swift", "clever", "bright", "calm", "bold", "keen", "wise", "quick"]
        nouns = ["falcon", "tiger", "eagle", "wolf", "hawk", "fox", "owl", "bear"]
        import random
        return f"{random.choice(adjectives)}-{random.choice(nouns)}-{uuid.uuid4().hex[:4]}"

    def create_plan(
        self,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Plan:
        """Create a new plan."""
        plan = Plan(
            id=str(uuid.uuid4())[:12],
            name=name or self._generate_plan_name(),
            description=description,
            tags=tags or [],
        )
        self.plans[plan.id] = plan
        self.current_plan = plan

        if self.auto_save:
            self._save_plan(plan)

        return plan

    def _save_plan(self, plan: Plan) -> bool:
        """Save plan to file."""
        try:
            path = self.plans_dir / f"{plan.id}.json"
            path.write_text(json.dumps(plan.to_dict(), indent=2))

            # Also save markdown version
            md_path = self.plans_dir / f"{plan.id}.md"
            md_path.write_text(plan.to_markdown())

            return True
        except Exception:
            return False

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        return self.plans.get(plan_id)

    def get_plan_by_name(self, name: str) -> Optional[Plan]:
        """Get a plan by name."""
        for plan in self.plans.values():
            if plan.name == name:
                return plan
        return None

    def list_plans(
        self,
        status: Optional[PlanStatus] = None,
        limit: int = 10,
    ) -> List[Plan]:
        """List plans, optionally filtered by status."""
        plans = list(self.plans.values())
        if status:
            plans = [p for p in plans if p.status == status]
        plans.sort(key=lambda p: p.created_at, reverse=True)
        return plans[:limit]

    def set_approval_callback(self, callback: Callable[[Plan], bool]) -> None:
        """Set callback for approval requests."""
        self._approval_callback = callback

    def request_approval(self, plan: Optional[Plan] = None) -> bool:
        """Request user approval for a plan."""
        plan = plan or self.current_plan
        if not plan:
            return False

        plan.status = PlanStatus.PENDING_APPROVAL

        if self.auto_save:
            self._save_plan(plan)

        # If we have an approval callback, use it
        if self._approval_callback:
            approved = self._approval_callback(plan)
            if approved:
                self.approve_plan(plan)
            else:
                self.reject_plan(plan)
            return approved

        # Otherwise, plan stays in pending state for external approval
        return False

    def approve_plan(
        self,
        plan: Optional[Plan] = None,
        approved_by: str = "user",
    ) -> bool:
        """Approve a plan for execution."""
        plan = plan or self.current_plan
        if not plan:
            return False

        plan.status = PlanStatus.APPROVED
        plan.approved_at = datetime.now()
        plan.approved_by = approved_by

        # Grant all requested permissions
        for prompt in plan.allowed_prompts:
            prompt.granted_at = datetime.now()

        if self.auto_save:
            self._save_plan(plan)

        return True

    def reject_plan(
        self,
        plan: Optional[Plan] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """Reject a plan."""
        plan = plan or self.current_plan
        if not plan:
            return False

        plan.status = PlanStatus.REJECTED
        if reason:
            plan.metadata["rejection_reason"] = reason

        if self.auto_save:
            self._save_plan(plan)

        return True

    def start_execution(self, plan: Optional[Plan] = None) -> bool:
        """Start executing a plan."""
        plan = plan or self.current_plan
        if not plan or plan.status != PlanStatus.APPROVED:
            return False

        plan.status = PlanStatus.IN_PROGRESS

        if self.auto_save:
            self._save_plan(plan)

        return True

    def start_step(self, step_id: str, plan: Optional[Plan] = None) -> bool:
        """Mark a step as in progress."""
        plan = plan or self.current_plan
        if not plan:
            return False

        step = plan.get_step(step_id)
        if not step:
            return False

        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.now()

        if self.auto_save:
            self._save_plan(plan)

        return True

    def complete_step(
        self,
        step_id: str,
        output: Optional[str] = None,
        tokens_used: int = 0,
        plan: Optional[Plan] = None,
    ) -> bool:
        """Mark a step as completed."""
        plan = plan or self.current_plan
        if not plan:
            return False

        step = plan.get_step(step_id)
        if not step:
            return False

        step.status = StepStatus.COMPLETED
        step.completed_at = datetime.now()
        step.output = output
        step.actual_tokens = tokens_used

        # Check if plan is complete
        if all(s.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] for s in plan.steps):
            plan.status = PlanStatus.COMPLETED
            plan.completed_at = datetime.now()

        if self.auto_save:
            self._save_plan(plan)

        return True

    def fail_step(
        self,
        step_id: str,
        error: str,
        plan: Optional[Plan] = None,
    ) -> bool:
        """Mark a step as failed."""
        plan = plan or self.current_plan
        if not plan:
            return False

        step = plan.get_step(step_id)
        if not step:
            return False

        step.status = StepStatus.FAILED
        step.completed_at = datetime.now()
        step.error = error

        if self.auto_save:
            self._save_plan(plan)

        return True

    def check_permission(self, tool: str, action: str) -> bool:
        """Check if an action is pre-authorized by the plan."""
        if not self.current_plan:
            return False

        action_lower = action.lower()
        for prompt in self.current_plan.allowed_prompts:
            if prompt.tool == tool and prompt.granted_at:
                # Check if action matches the prompt semantically
                prompt_lower = prompt.prompt.lower()
                if any(word in action_lower for word in prompt_lower.split()):
                    return True
        return False

    def enter_plan_mode(self) -> Plan:
        """Enter plan mode by creating a new draft plan."""
        return self.create_plan()

    def exit_plan_mode(
        self,
        request_approval: bool = True,
    ) -> Optional[Plan]:
        """Exit plan mode, optionally requesting approval."""
        if not self.current_plan:
            return None

        if request_approval:
            self.request_approval()

        plan = self.current_plan
        self.current_plan = None
        return plan

    def is_in_plan_mode(self) -> bool:
        """Check if currently in plan mode."""
        return self.current_plan is not None and self.current_plan.status == PlanStatus.DRAFT


# Singleton instance
_plan_manager: Optional[PlanManager] = None


def get_plan_manager(plans_dir: str = ".claude/plans") -> PlanManager:
    """Get or create the global plan manager."""
    global _plan_manager
    if _plan_manager is None:
        _plan_manager = PlanManager(plans_dir=plans_dir)
    return _plan_manager


def reset_plan_manager() -> None:
    """Reset the global plan manager."""
    global _plan_manager
    _plan_manager = None
