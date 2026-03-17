"""
Agent performance tracking and KPI aggregation.

All logic is pure computation -- no LLM calls, no network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentOKR:
    """Objective and Key Results for an agent."""
    objective: str
    key_results: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0.0 - 1.0


DEFAULT_OKRS: Dict[str, AgentOKR] = {
    "ceo": AgentOKR(
        objective="Ensure every shipped solution solves the stated problem",
        key_results=["Approval based on evidence not assumption", "< 20% rejection rate"]
    ),
    "cto": AgentOKR(
        objective="Design architectures that developers can build on first try",
        key_results=["< 2 redesigns per project", "Developer asks 0 clarifying questions"]
    ),
    "product_manager": AgentOKR(
        objective="Define requirements so clear that there is zero ambiguity",
        key_results=["Acceptance criteria are testable commands", "< 1 rescope per project"]
    ),
    "researcher": AgentOKR(
        objective="Find real problems backed by data from multiple sources",
        key_results=["> 3 sources per problem", "Cross-validation score > 0.7"]
    ),
    "developer": AgentOKR(
        objective="Write code that passes QA on the first attempt",
        key_results=["< 20% rework rate", "All files are runnable"]
    ),
    "qa_engineer": AgentOKR(
        objective="Catch real bugs, not style issues",
        key_results=["< 10% false positive rate", "Every FAIL has a specific blocking issue"]
    ),
    "devops_engineer": AgentOKR(
        objective="Every solution is deployable with one command",
        key_results=["Entry point exists", "Dependencies are pinned"]
    ),
    "data_analyst": AgentOKR(
        objective="Provide unbiased cross-validation of research findings",
        key_results=["Detect > 80% of biased sources", "Confidence scores within 0.1 of actual"]
    ),
    "security_engineer": AgentOKR(
        objective="Identify real security vulnerabilities, not false alarms",
        key_results=["Zero false positives on placeholder values", "Catch all hardcoded real secrets"]
    ),
}


@dataclass
class AgentKPIs:
    """Composite key-performance indicators for a single agent.

    All rate/ratio fields are in the range [0, 1].
    ``calculate_score`` produces a normalised 0-100 composite.
    """

    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time_ms: float = 0.0
    approval_rate: float = 0.0        # 0-1
    rework_rate: float = 0.0          # 0-1
    bug_detection_rate: float = 0.0   # 0-1, meaningful for QA agents
    research_accuracy: float = 0.0    # 0-1, meaningful for Researcher / DataAnalyst
    tokens_used: int = 0
    cost_efficiency: float = 0.0

    def calculate_score(self) -> float:
        """Return a composite performance score in the range [0, 100].

        Weighting
        ---------
        - Completion ratio   : 30 %
        - Approval rate      : 25 %
        - Low rework bonus   : 20 %
        - Response-time bonus: 15 %  (< 5 s considered excellent)
        - Specialist bonus   : 10 %  (bug_detection_rate or research_accuracy)

        The formula is intentionally simple and deterministic so that
        callers can reason about the score without hidden complexity.
        """
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 0.0

        # --- completion ratio (0-1) ---
        completion_ratio = self.tasks_completed / total_tasks

        # --- response-time factor (0-1, 5000 ms = 0, 0 ms = 1) ---
        response_factor = max(0.0, 1.0 - self.avg_response_time_ms / 5000.0)

        # --- specialist factor (0-1) ---
        specialist = max(self.bug_detection_rate, self.research_accuracy)

        score = (
            completion_ratio * 30.0
            + self.approval_rate * 25.0
            + (1.0 - self.rework_rate) * 20.0
            + response_factor * 15.0
            + specialist * 10.0
        )

        return round(max(0.0, min(100.0, score)), 2)


@dataclass
class RoleSpecificKPIs:
    """Extended KPIs tailored to specific agent roles."""

    # CEO-specific
    decision_quality: float = 0.0      # Rejections that led to better solutions / total rejections
    time_to_decision_ms: float = 0.0   # Average time to make a decision

    # Developer-specific
    first_attempt_pass_rate: float = 0.0  # Code that passes QA on first try
    fix_velocity_ms: float = 0.0          # Average time to fix a bug

    # QA-specific
    bug_detection_rate: float = 0.0       # Bugs caught before CEO review
    false_positive_rate: float = 0.0      # Issues flagged that were not real

    # Researcher-specific
    problem_validation_rate: float = 0.0  # Problems that led to successful solutions


# ------------------------------------------------------------------
# Internal bookkeeping structures
# ------------------------------------------------------------------

@dataclass
class _AgentRecord:
    """Mutable accumulator used internally by ``PerformanceTracker``."""

    tasks_completed: int = 0
    tasks_failed: int = 0
    total_response_time_ms: float = 0.0
    total_tasks: int = 0
    total_tokens: int = 0
    approvals: int = 0
    approval_checks: int = 0
    reworks: int = 0


class PerformanceTracker:
    """Collects raw events and produces ``AgentKPIs`` snapshots on demand.

    Usage::

        tracker = PerformanceTracker()
        tracker.record_task("Developer", success=True, response_time_ms=1200, tokens=350)
        tracker.record_approval("Developer", approved=True)
        kpis = tracker.get_kpis("Developer")
        print(kpis.calculate_score())
    """

    def __init__(self) -> None:
        self._records: Dict[str, _AgentRecord] = {}

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _ensure(self, agent_name: str) -> _AgentRecord:
        if agent_name not in self._records:
            self._records[agent_name] = _AgentRecord()
        return self._records[agent_name]

    def record_task(
        self,
        agent_name: str,
        success: bool,
        response_time_ms: float,
        tokens: int,
    ) -> None:
        """Record the outcome of a single task execution."""
        rec = self._ensure(agent_name)
        if success:
            rec.tasks_completed += 1
        else:
            rec.tasks_failed += 1
        rec.total_response_time_ms += response_time_ms
        rec.total_tasks += 1
        rec.total_tokens += tokens

    def record_approval(self, agent_name: str, approved: bool) -> None:
        """Record whether an agent's output was approved by a reviewer."""
        rec = self._ensure(agent_name)
        rec.approval_checks += 1
        if approved:
            rec.approvals += 1

    def record_rework(self, agent_name: str) -> None:
        """Record that an agent's output required rework."""
        rec = self._ensure(agent_name)
        rec.reworks += 1

    # ------------------------------------------------------------------
    # KPI materialisation
    # ------------------------------------------------------------------

    def get_kpis(self, agent_name: str) -> AgentKPIs:
        """Materialise the current KPIs for *agent_name*.

        Returns a zero-valued ``AgentKPIs`` if no data has been recorded.
        """
        rec = self._records.get(agent_name)
        if rec is None:
            return AgentKPIs()

        total = rec.tasks_completed + rec.tasks_failed
        avg_rt = rec.total_response_time_ms / rec.total_tasks if rec.total_tasks else 0.0
        approval_rate = rec.approvals / rec.approval_checks if rec.approval_checks else 0.0
        rework_rate = rec.reworks / total if total else 0.0

        return AgentKPIs(
            tasks_completed=rec.tasks_completed,
            tasks_failed=rec.tasks_failed,
            avg_response_time_ms=avg_rt,
            approval_rate=approval_rate,
            rework_rate=rework_rate,
            tokens_used=rec.total_tokens,
        )

    def get_all_kpis(self) -> Dict[str, AgentKPIs]:
        """Return a KPI snapshot for every tracked agent."""
        return {name: self.get_kpis(name) for name in self._records}

    def get_recent_scores(self, agent_name: str, last_n: int = 5) -> List[float]:
        """Get the most recent composite scores for an agent.

        Args:
            agent_name: The agent to query.
            last_n: Number of recent scores to return.

        Returns:
            List of recent composite scores (may be shorter than last_n).
        """
        rec = self._records.get(agent_name)
        if rec is None:
            return []
        history = getattr(rec, '_score_history', [])
        return list(history[-last_n:])

    def apply_performance_consequences(self, agent_name: str) -> Dict[str, Any]:
        """Apply consequences based on recent performance.

        Args:
            agent_name: The agent to evaluate.

        Returns:
            Dict with 'status', 'supervision', and 'note' keys.
        """
        recent = self.get_recent_scores(agent_name, last_n=5)
        avg = sum(recent) / len(recent) if recent else 50.0

        if avg < 40:
            return {
                "status": "probation",
                "supervision": "thorough",
                "note": "Performance below threshold — extra review required",
            }
        elif avg > 80:
            return {
                "status": "autonomous",
                "supervision": "light",
                "note": "Strong track record — reduced oversight",
            }
        return {
            "status": "standard",
            "supervision": "standard",
            "note": "",
        }

    def get_underperformers(self, threshold: float = 40.0) -> List[str]:
        """Return names of agents whose composite score is below *threshold*."""
        result: List[str] = []
        for name in sorted(self._records):
            kpis = self.get_kpis(name)
            if kpis.calculate_score() < threshold:
                result.append(name)
        return result

    def performance_trend(self, agent_name: str, window: int = 5) -> Dict[str, Any]:
        """Track KPI trends across recent task recordings.

        Analyzes the last `window` recordings to identify improving
        or declining performance.

        Args:
            agent_name: The agent to analyze.
            window: Number of recent data points to consider.

        Returns:
            Dict with trend direction and details.
        """
        rec = self._records.get(agent_name)
        if rec is None or rec.total_tasks < 2:
            return {"trend": "insufficient_data", "detail": "Not enough data points"}

        kpis = self.get_kpis(agent_name)
        score = kpis.calculate_score()

        # Compare current score against stored historical scores
        history = getattr(rec, '_score_history', [])
        history.append(score)
        rec._score_history = history[-window:]  # Keep only last N

        if len(rec._score_history) < 2:
            return {"trend": "insufficient_data", "detail": "Need at least 2 data points"}

        recent = rec._score_history[-min(3, len(rec._score_history)):]
        older = rec._score_history[:-len(recent)] if len(rec._score_history) > len(recent) else recent

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)

        if avg_recent > avg_older + 5:
            trend = "improving"
        elif avg_recent < avg_older - 5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "current_score": score,
            "recent_avg": round(avg_recent, 2),
            "older_avg": round(avg_older, 2),
            "history": rec._score_history,
            "detail": f"{agent_name}: {trend} (recent={avg_recent:.1f}, older={avg_older:.1f})"
        }

    def get_all_trends(self) -> Dict[str, Dict[str, Any]]:
        """Get performance trends for all tracked agents."""
        return {name: self.performance_trend(name) for name in self._records}

    def evaluate_okrs(self) -> Dict[str, Dict]:
        """Evaluate OKR progress for each agent based on actual KPIs.

        Maps measurable key results to actual KPI metrics:
        - Rejection/rework rate: measured from rework_rate and approval_rate
        - Redesign count: estimated from reworks
        - First-pass rate: 1 - rework_rate

        Returns a dict of agent_name -> {objective, progress, details}.
        """
        results = {}
        for agent_name, okr in DEFAULT_OKRS.items():
            kpis = self.get_kpis(agent_name)
            total = kpis.tasks_completed + kpis.tasks_failed
            if total == 0:
                results[agent_name] = {
                    "objective": okr.objective,
                    "progress": 0.0,
                    "details": "No tasks recorded yet",
                }
                continue

            # Score each key result (0.0 = not met, 1.0 = fully met)
            kr_scores = []
            details = []
            for kr in okr.key_results:
                score, detail = self._evaluate_key_result(kr, kpis)
                kr_scores.append(score)
                details.append(detail)

            progress = sum(kr_scores) / len(kr_scores) if kr_scores else 0.0
            okr.progress = progress

            results[agent_name] = {
                "objective": okr.objective,
                "progress": round(progress, 2),
                "details": "; ".join(details),
            }
        return results

    @staticmethod
    def _evaluate_key_result(kr: str, kpis: AgentKPIs) -> tuple:
        """Evaluate a single key result string against actual KPIs.

        Returns (score: float 0-1, detail: str).
        """
        kr_lower = kr.lower()

        # Rework rate checks: "< 20% rework rate"
        if "rework" in kr_lower and "%" in kr:
            target = float(kr.split("%")[0].split()[-1]) / 100.0
            actual = kpis.rework_rate
            met = actual <= target
            return (1.0 if met else max(0.0, 1.0 - (actual - target) / target), f"Rework: {actual:.0%} (target <{target:.0%})")

        # Rejection rate checks: "< 20% rejection rate"
        if "rejection" in kr_lower and "%" in kr:
            target = float(kr.split("%")[0].split()[-1]) / 100.0
            actual = 1.0 - kpis.approval_rate
            met = actual <= target
            return (1.0 if met else max(0.0, 1.0 - (actual - target) / target), f"Rejection: {actual:.0%} (target <{target:.0%})")

        # Redesign/rescope count: "< 2 redesigns per project"
        if "redesign" in kr_lower or "rescope" in kr_lower:
            reworks = kpis.rework_rate * (kpis.tasks_completed + kpis.tasks_failed)
            target = 2.0
            met = reworks <= target
            return (1.0 if met else max(0.0, 1.0 - (reworks - target) / target), f"Reworks: {reworks:.1f} (target <{target:.0f})")

        # False positive rate: "< 10% false positive rate"
        if "false positive" in kr_lower and "%" in kr:
            # Approximate: high rework_rate for QA suggests false positives
            target = float(kr.split("%")[0].split()[-1]) / 100.0
            actual = kpis.rework_rate  # Best proxy
            met = actual <= target
            return (1.0 if met else max(0.0, 1.0 - (actual - target) / target), f"FP proxy: {actual:.0%} (target <{target:.0%})")

        # Source count checks: "> 3 sources per problem"
        if "sources" in kr_lower:
            return (kpis.research_accuracy, f"Research accuracy: {kpis.research_accuracy:.0%}")

        # Cross-validation score
        if "cross-validation" in kr_lower or "confidence" in kr_lower:
            return (kpis.research_accuracy, f"Research accuracy: {kpis.research_accuracy:.0%}")

        # Default: check approval rate as general quality proxy
        return (kpis.approval_rate, f"Approval: {kpis.approval_rate:.0%}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Persist performance records to disk as JSON."""
        import json
        from pathlib import Path as _P
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, rec in self._records.items():
            data[name] = {
                "tasks_completed": rec.tasks_completed,
                "tasks_failed": rec.tasks_failed,
                "total_response_time_ms": rec.total_response_time_ms,
                "total_tasks": rec.total_tasks,
                "total_tokens": rec.total_tokens,
                "approvals": rec.approvals,
                "approval_checks": rec.approval_checks,
                "reworks": rec.reworks,
            }
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path) -> "PerformanceTracker":
        """Load performance records from disk. Returns empty tracker on failure."""
        import json
        from pathlib import Path as _P
        tracker = cls()
        p = _P(path)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                for name, rec_data in data.items():
                    tracker._records[name] = _AgentRecord(**rec_data)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        return tracker
