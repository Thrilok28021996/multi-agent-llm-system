"""
Progress Tracking with ETA Calculation for Company AGI.

Provides terminal-friendly progress tracking:
- Phase-based progress bars
- ETA calculation based on historical data
- Real-time updates with ANSI escape codes
- Persistent metrics for better estimates
"""

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


class ProgressStatus(Enum):
    """Progress item status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseMetrics:
    """Historical metrics for a phase."""
    phase_name: str
    execution_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    last_duration_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.execution_count == 0:
            return 0.0
        return self.total_duration_ms / self.execution_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    def record_execution(self, duration_ms: float, success: bool = True) -> None:
        """Record a phase execution."""
        self.execution_count += 1
        self.total_duration_ms += duration_ms
        self.last_duration_ms = duration_ms

        if duration_ms < self.min_duration_ms:
            self.min_duration_ms = duration_ms
        if duration_ms > self.max_duration_ms:
            self.max_duration_ms = duration_ms

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseMetrics":
        return cls(**data)


@dataclass
class ProgressItem:
    """A single progress item."""
    id: str
    name: str
    description: str = ""
    status: ProgressStatus = ProgressStatus.PENDING
    progress_pct: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    sub_items: List["ProgressItem"] = field(default_factory=list)

    def start(self) -> None:
        """Mark item as started."""
        self.status = ProgressStatus.IN_PROGRESS
        self.started_at = datetime.now().isoformat()

    def complete(self, success: bool = True) -> None:
        """Mark item as completed."""
        self.status = ProgressStatus.COMPLETED if success else ProgressStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.progress_pct = 100.0

        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_ms = (end - start).total_seconds() * 1000

    def skip(self) -> None:
        """Mark item as skipped."""
        self.status = ProgressStatus.SKIPPED
        self.completed_at = datetime.now().isoformat()

    def fail(self, error_message: str) -> None:
        """Mark item as failed."""
        self.status = ProgressStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.error_message = error_message

        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_ms = (end - start).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "progress_pct": self.progress_pct,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "sub_items": [i.to_dict() for i in self.sub_items],
        }


class ProgressBar:
    """Terminal progress bar with ANSI escape codes."""

    def __init__(
        self,
        width: int = 40,
        fill_char: str = "=",
        empty_char: str = " ",
        tip_char: str = ">",
        color_enabled: bool = True,
    ):
        self.width = width
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.tip_char = tip_char
        self.color_enabled = color_enabled

    def render(
        self,
        progress: float,
        label: str = "",
        status: Optional[ProgressStatus] = None,
    ) -> str:
        """Render progress bar as string."""
        # Clamp progress
        progress = max(0.0, min(100.0, progress))

        # Calculate fill width
        fill_width = int((progress / 100.0) * self.width)
        empty_width = self.width - fill_width

        # Build bar
        if fill_width > 0 and fill_width < self.width:
            bar = self.fill_char * (fill_width - 1) + self.tip_char + self.empty_char * empty_width
        elif fill_width == self.width:
            bar = self.fill_char * fill_width
        else:
            bar = self.empty_char * self.width

        # Color based on status
        colors = {
            ProgressStatus.PENDING: "\033[90m",      # Gray
            ProgressStatus.IN_PROGRESS: "\033[36m",  # Cyan
            ProgressStatus.COMPLETED: "\033[32m",    # Green
            ProgressStatus.FAILED: "\033[31m",       # Red
            ProgressStatus.SKIPPED: "\033[33m",      # Yellow
        }
        reset = "\033[0m"

        if not self.color_enabled:
            colors = {k: "" for k in colors}
            reset = ""

        default_color = "\033[36m"
        color = colors.get(status, default_color) if (self.color_enabled and status) else ""

        # Build output
        parts = []
        if label:
            parts.append(f"{label:20}")
        parts.append(f"[{color}{bar}{reset}]")
        parts.append(f"{progress:5.1f}%")

        return " ".join(parts)


class ProgressTracker:
    """
    Progress tracker with ETA calculation.

    Features:
    - Multi-phase progress tracking
    - Historical metrics for ETA estimation
    - Terminal-friendly output
    - Persistent metrics storage
    """

    def __init__(
        self,
        output: Optional[TextIO] = None,
        metrics_file: Optional[Path] = None,
        color_enabled: bool = True,
    ):
        self.output = output or sys.stderr
        self.metrics_file = metrics_file
        self.color_enabled = color_enabled

        self._items: Dict[str, ProgressItem] = {}
        self._order: List[str] = []
        self._metrics: Dict[str, PhaseMetrics] = {}
        self._start_time: Optional[float] = None
        self._current_phase: Optional[str] = None

        self._progress_bar = ProgressBar(color_enabled=color_enabled)

        # Load historical metrics
        if metrics_file and metrics_file.exists():
            self._load_metrics()

    def _load_metrics(self) -> None:
        """Load historical metrics from file."""
        if self.metrics_file and self.metrics_file.exists():
            try:
                with open(self.metrics_file) as f:
                    data = json.load(f)
                    for phase_name, metrics_data in data.items():
                        self._metrics[phase_name] = PhaseMetrics.from_dict(metrics_data)
            except Exception as e:
                import sys
                print(f"[ProgressTracker] Could not load metrics: {e}", file=sys.stderr)

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        if self.metrics_file:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.metrics_file, "w") as f:
                    json.dump(
                        {k: v.to_dict() for k, v in self._metrics.items()},
                        f,
                        indent=2,
                    )
            except Exception as e:
                import sys
                print(f"[ProgressTracker] Could not save metrics: {e}", file=sys.stderr)

    def add_phase(
        self,
        phase_id: str,
        name: str,
        description: str = "",
    ) -> ProgressItem:
        """Add a phase to track."""
        item = ProgressItem(
            id=phase_id,
            name=name,
            description=description,
        )
        self._items[phase_id] = item
        self._order.append(phase_id)

        # Initialize metrics if not exists
        if phase_id not in self._metrics:
            self._metrics[phase_id] = PhaseMetrics(phase_name=phase_id)

        return item

    def add_phases(self, phases: List[Dict[str, str]]) -> None:
        """Add multiple phases."""
        for phase in phases:
            self.add_phase(
                phase_id=phase.get("id", phase["name"].lower().replace(" ", "_")),
                name=phase["name"],
                description=phase.get("description", ""),
            )

    def start(self) -> None:
        """Start tracking progress."""
        self._start_time = time.time()

    def start_phase(self, phase_id: str) -> None:
        """Start a specific phase."""
        if phase_id in self._items:
            self._items[phase_id].start()
            self._current_phase = phase_id
            self._render()

    def update_phase(
        self,
        phase_id: str,
        progress: float,
        message: Optional[str] = None,
    ) -> None:
        """Update phase progress."""
        if phase_id in self._items:
            self._items[phase_id].progress_pct = progress
            if message:
                self._items[phase_id].description = message
            self._render()

    def complete_phase(self, phase_id: str, success: bool = True) -> None:
        """Complete a phase."""
        if phase_id in self._items:
            item = self._items[phase_id]
            item.complete(success)

            # Record metrics
            if phase_id in self._metrics:
                self._metrics[phase_id].record_execution(
                    duration_ms=item.duration_ms,
                    success=success,
                )

            self._current_phase = None
            self._render()
            self._save_metrics()

    def fail_phase(self, phase_id: str, error_message: str) -> None:
        """Mark phase as failed."""
        if phase_id in self._items:
            self._items[phase_id].fail(error_message)

            # Record metrics
            if phase_id in self._metrics:
                self._metrics[phase_id].record_execution(
                    duration_ms=self._items[phase_id].duration_ms,
                    success=False,
                )

            self._current_phase = None
            self._render()
            self._save_metrics()

    def skip_phase(self, phase_id: str) -> None:
        """Skip a phase."""
        if phase_id in self._items:
            self._items[phase_id].skip()
            self._render()

    @contextmanager
    def phase(self, phase_id: str):
        """Context manager for tracking a phase."""
        self.start_phase(phase_id)
        try:
            yield
            self.complete_phase(phase_id)
        except Exception as e:
            self.fail_phase(phase_id, str(e))
            raise

    def get_overall_progress(self) -> float:
        """Calculate overall progress percentage."""
        if not self._items:
            return 0.0

        total_weight = len(self._items)
        completed_weight = 0.0

        for item in self._items.values():
            if item.status == ProgressStatus.COMPLETED:
                completed_weight += 1.0
            elif item.status == ProgressStatus.IN_PROGRESS:
                completed_weight += item.progress_pct / 100.0
            elif item.status == ProgressStatus.SKIPPED:
                completed_weight += 1.0

        return (completed_weight / total_weight) * 100.0

    def get_eta_ms(self) -> Optional[float]:
        """Estimate time remaining in milliseconds."""
        remaining_phases = [
            phase_id for phase_id in self._order
            if self._items[phase_id].status in [
                ProgressStatus.PENDING,
                ProgressStatus.IN_PROGRESS,
            ]
        ]

        if not remaining_phases:
            return 0.0

        total_eta_ms = 0.0
        for phase_id in remaining_phases:
            metrics = self._metrics.get(phase_id)
            if metrics and metrics.avg_duration_ms > 0:
                # Use historical average
                avg_ms = metrics.avg_duration_ms
                # Adjust for current progress if in progress
                if self._items[phase_id].status == ProgressStatus.IN_PROGRESS:
                    progress = self._items[phase_id].progress_pct
                    avg_ms *= (100.0 - progress) / 100.0
                total_eta_ms += avg_ms
            else:
                # No historical data - use a default estimate
                return None

        return total_eta_ms

    def format_duration(self, ms: float) -> str:
        """Format milliseconds as human-readable duration."""
        seconds = ms / 1000.0
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _render(self) -> None:
        """Render progress to output."""
        # Clear previous output (move cursor up)
        lines_to_clear = len(self._items) + 3  # phases + header + summary
        if self.color_enabled:
            # Move cursor up and clear lines
            self.output.write(f"\033[{lines_to_clear}A\033[J")

        # Header
        overall = self.get_overall_progress()
        eta_ms = self.get_eta_ms()
        eta_str = f" (ETA: {self.format_duration(eta_ms)})" if eta_ms else ""

        header = f"Progress: {overall:.1f}%{eta_str}"
        self.output.write(f"{header}\n")
        self.output.write("-" * 60 + "\n")

        # Phase bars
        for phase_id in self._order:
            item = self._items[phase_id]
            bar = self._progress_bar.render(
                progress=item.progress_pct,
                label=item.name,
                status=item.status,
            )

            # Add status indicator
            status_indicators = {
                ProgressStatus.PENDING: "   ",
                ProgressStatus.IN_PROGRESS: " * ",
                ProgressStatus.COMPLETED: " + ",
                ProgressStatus.FAILED: " - ",
                ProgressStatus.SKIPPED: " ~ ",
            }
            indicator = status_indicators.get(item.status, "   ")

            self.output.write(f"{indicator}{bar}\n")

        # Summary
        completed = sum(1 for i in self._items.values() if i.status == ProgressStatus.COMPLETED)
        failed = sum(1 for i in self._items.values() if i.status == ProgressStatus.FAILED)
        total = len(self._items)

        self.output.write(f"\n{completed}/{total} complete")
        if failed:
            self.output.write(f", {failed} failed")
        self.output.write("\n")

        self.output.flush()

    def render_once(self) -> str:
        """Render progress as a string (without clearing)."""
        lines = []

        # Header
        overall = self.get_overall_progress()
        eta_ms = self.get_eta_ms()
        eta_str = f" (ETA: {self.format_duration(eta_ms)})" if eta_ms else ""
        lines.append(f"Progress: {overall:.1f}%{eta_str}")
        lines.append("-" * 60)

        # Phase bars
        for phase_id in self._order:
            item = self._items[phase_id]
            bar = self._progress_bar.render(
                progress=item.progress_pct,
                label=item.name,
                status=item.status,
            )
            status_indicators = {
                ProgressStatus.PENDING: "   ",
                ProgressStatus.IN_PROGRESS: " * ",
                ProgressStatus.COMPLETED: " + ",
                ProgressStatus.FAILED: " - ",
                ProgressStatus.SKIPPED: " ~ ",
            }
            indicator = status_indicators.get(item.status, "   ")
            lines.append(f"{indicator}{bar}")

        return "\n".join(lines)

    def get_report(self) -> Dict[str, Any]:
        """Get full progress report."""
        elapsed_ms = 0.0
        if self._start_time:
            elapsed_ms = (time.time() - self._start_time) * 1000

        return {
            "overall_progress": self.get_overall_progress(),
            "elapsed_ms": elapsed_ms,
            "eta_ms": self.get_eta_ms(),
            "phases": {k: v.to_dict() for k, v in self._items.items()},
            "phase_order": self._order,
            "metrics": {k: v.to_dict() for k, v in self._metrics.items()},
        }


# Global progress tracker instance
_progress_tracker: Optional[ProgressTracker] = None


def get_progress_tracker(
    metrics_file: Optional[Path] = None,
) -> ProgressTracker:
    """Get or create the global progress tracker."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker(
            metrics_file=metrics_file or Path("output/metrics/phase_metrics.json"),
        )
    return _progress_tracker


def reset_progress_tracker() -> None:
    """Reset the global progress tracker."""
    global _progress_tracker
    _progress_tracker = None
