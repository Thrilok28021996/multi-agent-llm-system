"""Project backlog management — priority queue for multi-problem workflows."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BacklogItem:
    """A single item in the project backlog."""
    id: str
    description: str
    domain: str = "software"
    severity: float = 0.5      # 0.0-1.0
    frequency: float = 0.5     # 0.0-1.0
    feasibility: float = 0.5   # 0.0-1.0
    market_size: float = 0.5   # 0.0-1.0
    status: str = "pending"    # pending, in_progress, completed, skipped
    added_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def priority_score(self) -> float:
        """Calculate priority: severity * frequency * feasibility * market_size."""
        return self.severity * self.frequency * self.feasibility * self.market_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "domain": self.domain,
            "priority_score": round(self.priority_score, 3),
            "severity": self.severity,
            "frequency": self.frequency,
            "feasibility": self.feasibility,
            "market_size": self.market_size,
            "status": self.status,
            "added_at": self.added_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


class ProjectBacklog:
    """Priority queue for managing multiple problems.

    Problems are scored by severity * frequency * feasibility * market_size.
    After solving one problem, the next highest-priority is automatically selected.
    """

    def __init__(self):
        self._items: Dict[str, BacklogItem] = {}

    def add(self, item: BacklogItem) -> None:
        """Add an item to the backlog."""
        self._items[item.id] = item

    def add_problem(
        self,
        problem_id: str,
        description: str,
        domain: str = "software",
        severity: float = 0.5,
        frequency: float = 0.5,
        feasibility: float = 0.5,
        market_size: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BacklogItem:
        """Add a problem to the backlog with scoring."""
        item = BacklogItem(
            id=problem_id,
            description=description,
            domain=domain,
            severity=severity,
            frequency=frequency,
            feasibility=feasibility,
            market_size=market_size,
            metadata=metadata or {},
        )
        self._items[item.id] = item
        return item

    def get_next(self) -> Optional[BacklogItem]:
        """Get the highest-priority pending item."""
        pending = [
            item for item in self._items.values()
            if item.status == "pending"
        ]
        if not pending:
            return None
        return max(pending, key=lambda x: x.priority_score)

    def mark_in_progress(self, item_id: str) -> None:
        """Mark an item as in progress."""
        if item_id in self._items:
            self._items[item_id].status = "in_progress"

    def mark_completed(self, item_id: str) -> None:
        """Mark an item as completed."""
        if item_id in self._items:
            self._items[item_id].status = "completed"
            self._items[item_id].completed_at = datetime.now()

    def mark_skipped(self, item_id: str, reason: str = "") -> None:
        """Mark an item as skipped."""
        if item_id in self._items:
            self._items[item_id].status = "skipped"
            self._items[item_id].metadata["skip_reason"] = reason

    def get_all(self, status: Optional[str] = None) -> List[BacklogItem]:
        """Get all items, optionally filtered by status."""
        items = list(self._items.values())
        if status:
            items = [i for i in items if i.status == status]
        return sorted(items, key=lambda x: x.priority_score, reverse=True)

    def size(self, status: Optional[str] = None) -> int:
        """Get count of items, optionally filtered by status."""
        if status:
            return sum(1 for i in self._items.values() if i.status == status)
        return len(self._items)

    def to_dict(self) -> Dict[str, Any]:
        """Export backlog state."""
        return {
            "total": len(self._items),
            "pending": self.size("pending"),
            "in_progress": self.size("in_progress"),
            "completed": self.size("completed"),
            "items": [item.to_dict() for item in self.get_all()],
        }
