"""Escalation manager for the company workflow.

Controls escalation logic when QA failures and CEO rejections occur.
Uses a strategy rotation system that NEVER gives up — it cycles through
increasingly creative approaches until the problem is solved.

    Rounds 1-3:  Developer fixes the code (DEVELOPER_FIX)
    Round  4:    Break problem into sub-problems (DECOMPOSE_PROBLEM)
    Round  5:    CTO redesigns the technical approach (CTO_REDESIGN)
    Round  6:    PM re-scopes the requirements (PM_RESCOPE)
    Round  7:    All agents brainstorm solutions (TEAM_BRAINSTORM)
    Round  8:    Try a completely different approach (PIVOT_APPROACH)
    Round  9+:   Cycle back to DEVELOPER_FIX with accumulated context

This module is pure deterministic logic -- no LLM calls, no I/O.
It is imported and called by ``workflow.py`` to decide the next action
after each failed QA or CEO-rejection round.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import random


# ---------------------------------------------------------------------------
# Public enum
# ---------------------------------------------------------------------------

class EscalationAction(Enum):
    """Action to take after a QA failure or CEO rejection.

    HARD_STOP is removed — the system never gives up.
    """

    DEVELOPER_FIX = "developer_fix"
    CTO_REDESIGN = "cto_redesign"
    PM_RESCOPE = "pm_rescope"
    TEAM_BRAINSTORM = "team_brainstorm"
    DECOMPOSE_PROBLEM = "decompose_problem"
    PIVOT_APPROACH = "pivot_approach"
    FRESH_START = "fresh_start"
    SIMPLIFY = "simplify"
    ALTERNATIVE_STACK = "alternative_stack"


# ---------------------------------------------------------------------------
# Strategy memory
# ---------------------------------------------------------------------------

@dataclass
class StrategyAttempt:
    """Record of a single strategy attempt."""
    approach: str
    round: int
    outcome: str  # "failed", "partial", "abandoned"
    feedback: str = ""


@dataclass
class StrategyMemory:
    """Track what strategies have been tried and their outcomes.

    This allows the system to avoid repeating failed approaches
    and to provide context for the next attempt.
    """

    tried_approaches: List[StrategyAttempt] = field(default_factory=list)
    tried_architectures: List[str] = field(default_factory=list)
    tried_scopes: List[str] = field(default_factory=list)

    def record_attempt(
        self,
        approach: str,
        round: int,
        outcome: str,
        feedback: str = ""
    ) -> None:
        """Record that a strategy was tried."""
        self.tried_approaches.append(StrategyAttempt(
            approach=approach, round=round, outcome=outcome, feedback=feedback
        ))

    def record_architecture(self, architecture: str) -> None:
        """Record a tried architecture."""
        if architecture and architecture not in self.tried_architectures:
            self.tried_architectures.append(architecture)

    def record_scope(self, scope: str) -> None:
        """Record a tried scope."""
        if scope and scope not in self.tried_scopes:
            self.tried_scopes.append(scope)

    def get_untried_strategies(self) -> List[str]:
        """Return strategy types not yet attempted."""
        tried = {a.approach for a in self.tried_approaches}
        all_strategies = {
            "developer_fix", "cto_redesign", "pm_rescope",
            "team_brainstorm", "decompose_problem", "pivot_approach",
            "fresh_start", "simplify", "alternative_stack"
        }
        return sorted(all_strategies - tried)

    def summarize_for_next_attempt(self) -> str:
        """Generate context for the next attempt: what failed and why."""
        if not self.tried_approaches:
            return "No previous attempts."

        lines = ["Previous attempts and their outcomes:"]
        for attempt in self.tried_approaches:
            lines.append(
                f"  Round {attempt.round}: {attempt.approach} -> {attempt.outcome}"
            )
            if attempt.feedback:
                lines.append(f"    Feedback: {attempt.feedback}")

        if self.tried_architectures:
            lines.append(f"  Tried architectures: {', '.join(self.tried_architectures)}")

        if self.tried_scopes:
            lines.append(f"  Tried scopes: {', '.join(self.tried_scopes)}")

        untried = self.get_untried_strategies()
        if untried:
            lines.append(f"  Untried strategies: {', '.join(untried)}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all memory."""
        self.tried_approaches.clear()
        self.tried_architectures.clear()
        self.tried_scopes.clear()


# ---------------------------------------------------------------------------
# Internal data-classes
# ---------------------------------------------------------------------------

@dataclass
class EscalationRecord:
    """A single entry in the escalation history."""

    round: int
    action: EscalationAction
    qa_verdict: str
    dev_fix_count: int
    repeated_failure_detected: bool


# ---------------------------------------------------------------------------
# Escalation manager
# ---------------------------------------------------------------------------

# The full strategy cycle. After round 11, it wraps back to the beginning
# with fresh context from StrategyMemory.
_STRATEGY_CYCLE = [
    EscalationAction.DEVELOPER_FIX,      # Rounds 1, 12, 23, ...
    EscalationAction.DEVELOPER_FIX,      # Rounds 2, 13, 24, ...
    EscalationAction.DEVELOPER_FIX,      # Rounds 3, 14, 25, ...
    EscalationAction.DECOMPOSE_PROBLEM,  # Rounds 4, 15, 26, ...
    EscalationAction.CTO_REDESIGN,       # Rounds 5, 16, 27, ...
    EscalationAction.PM_RESCOPE,         # Rounds 6, 17, 28, ...
    EscalationAction.SIMPLIFY,           # Rounds 7, 18, 29, ... — NEW: reduce to bare minimum
    EscalationAction.TEAM_BRAINSTORM,    # Rounds 8, 19, 30, ...
    EscalationAction.ALTERNATIVE_STACK,  # Rounds 9, 20, 31, ... — NEW: try different tech
    EscalationAction.PIVOT_APPROACH,     # Rounds 10, 21, 32, ...
    EscalationAction.FRESH_START,        # Rounds 11, 22, 33, ... — NEW: start completely over
]


class EscalationManager:
    """Decides the appropriate recovery action based on workflow history.

    The manager never triggers a hard stop. Instead, it cycles through
    increasingly creative strategies. The only stop conditions are:
    - Time limit (user-configurable, checked by workflow.py)
    - Token budget (checked by workflow.py)
    - User abort

    Class-level constants control the thresholds:

    * ``MAX_DEVELOPER_RETRIES`` -- how many rounds the developer gets to fix
      issues before the problem is escalated upward.
    * ``MAX_TOTAL_ROUNDS`` -- soft budget for display purposes only
      (the system warns but does NOT stop).
    """

    MAX_DEVELOPER_RETRIES: int = 3
    MAX_TOTAL_ROUNDS: int = 999  # Soft limit — never enforced as a stop

    # Number of identical QA feedback hashes that trigger strategy rotation.
    _REPEATED_FAILURE_THRESHOLD: int = 3

    def __init__(self) -> None:
        self._history: List[EscalationRecord] = []
        self.strategy_memory = StrategyMemory()

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def should_escalate(
        self,
        round: int,
        qa_verdict: str,
        dev_fix_count: int,
        failure_history: List[str],
    ) -> EscalationAction:
        """Return the next action the workflow should take.

        The system NEVER returns a hard stop. Instead:
        - If repeated failures are detected, it skips to the next
          strategy in the cycle (avoiding the strategy that's stuck).
        - After round 8, it cycles back to round 1 with accumulated
          context from StrategyMemory.

        Args:
            round: Current approval-loop round (1-indexed).
            qa_verdict: The latest QA verdict string.
            dev_fix_count: How many developer fix attempts so far.
            failure_history: List of raw QA feedback strings.

        Returns:
            The ``EscalationAction`` indicating what to do next.
        """

        # If the same failure keeps repeating, skip ahead in the cycle
        if self.detect_repeated_failures(failure_history):
            action = self._get_rotation_action(round)
            self._record(round, action, qa_verdict, dev_fix_count, repeated=True)
            # Record the repeated failure in strategy memory
            self.strategy_memory.record_attempt(
                approach="repeated_failure_detected",
                round=round,
                outcome="stuck",
                feedback=failure_history[-1] if failure_history else ""
            )
            return action

        # Check for recurring failure categories
        category = self.detect_failure_category(failure_history)
        if category:
            self.strategy_memory.record_attempt(
                approach=f"category_{category}_detected",
                round=round,
                outcome="recurring",
                feedback=self.get_meta_analysis_prompt(category)
            )

        # Normal graduated escalation — cycles through strategy list
        cycle_index = (round - 1) % len(_STRATEGY_CYCLE)
        action = _STRATEGY_CYCLE[cycle_index]

        self._record(round, action, qa_verdict, dev_fix_count, repeated=False)
        return action

    def _get_rotation_action(self, round: int) -> EscalationAction:
        """When stuck on repeated failures, jump to the next non-developer strategy.

        Tries untried strategies first. If all have been tried, picks the
        least-recently-tried one.
        """
        untried = self.strategy_memory.get_untried_strategies()

        # Prefer untried high-impact strategies
        priority_order = [
            "fresh_start", "pivot_approach", "alternative_stack",
            "simplify", "decompose_problem", "team_brainstorm",
            "pm_rescope", "cto_redesign"
        ]
        for strategy_name in priority_order:
            if strategy_name in untried:
                return EscalationAction(strategy_name)

        # All strategies tried — cycle based on round with creative constraint
        # Skip developer_fix since that's what was repeating
        non_dev_strategies = [
            a for a in _STRATEGY_CYCLE
            if a != EscalationAction.DEVELOPER_FIX
        ]
        idx = (round - 1) % len(non_dev_strategies)
        action = non_dev_strategies[idx]

        # Inject a creative constraint to break out of cycles
        constraint = self.get_creative_constraint()
        self.strategy_memory.record_attempt(
            approach=action.value,
            round=round,
            outcome="retrying_with_constraint",
            feedback=f"Creative constraint: {constraint}"
        )

        return action

    # ------------------------------------------------------------------
    # Creative constraints
    # ------------------------------------------------------------------

    CREATIVE_CONSTRAINTS = [
        "Try implementing the entire solution in a single file",
        "Use only Python standard library — no external dependencies",
        "Try a completely different algorithm or data structure",
        "Simplify to the absolute minimum: 1 function, <50 lines",
        "Start from the test case and work backwards",
        "Use a different design pattern (functional instead of OO, or vice versa)",
        "Focus only on the happy path — ignore all edge cases for now",
        "Rewrite from scratch using a different approach than any previously tried",
    ]

    def get_creative_constraint(self) -> str:
        """Get a random creative constraint to break out of cycles."""
        return random.choice(self.CREATIVE_CONSTRAINTS)

    # ------------------------------------------------------------------
    # Repeated-failure detection
    # ------------------------------------------------------------------

    def detect_repeated_failures(self, failure_history: List[str]) -> bool:
        """Return ``True`` if the same QA feedback appears 3+ times.

        Feedback strings are normalised to lowercase, stripped of
        whitespace, and then SHA-256 hashed.  If any single hash appears
        at least ``_REPEATED_FAILURE_THRESHOLD`` times the workflow is
        considered stuck.
        """

        if len(failure_history) < self._REPEATED_FAILURE_THRESHOLD:
            return False

        counts: Dict[str, int] = {}
        for feedback in failure_history:
            digest = self._hash_feedback(feedback)
            counts[digest] = counts.get(digest, 0) + 1
            if counts[digest] >= self._REPEATED_FAILURE_THRESHOLD:
                return True

        return False

    def detect_failure_category(self, failure_history: List[str]) -> Optional[str]:
        """Detect the category of recurring failures.

        Instead of just checking exact text matches, categorize failures
        into types: import_error, logic_error, missing_file, syntax_error,
        runtime_error, test_failure. If the same CATEGORY appears 3+ times
        across different strategies, returns the category name.
        """
        if len(failure_history) < 3:
            return None

        categories: Dict[str, int] = {}
        category_keywords = {
            "import_error": ["import", "modulenotfound", "no module named", "cannot import"],
            "syntax_error": ["syntaxerror", "syntax error", "invalid syntax", "unexpected token"],
            "missing_file": ["file not found", "no such file", "filenotfound", "missing file"],
            "runtime_error": ["runtime", "exception", "traceback", "error:", "crashed"],
            "logic_error": ["wrong output", "incorrect", "does not match", "expected", "assertion"],
            "test_failure": ["test failed", "tests failed", "fail", "assertion error"],
        }

        for feedback in failure_history:
            feedback_lower = feedback.lower()
            for category, keywords in category_keywords.items():
                if any(kw in feedback_lower for kw in keywords):
                    categories[category] = categories.get(category, 0) + 1

        for category, count in categories.items():
            if count >= 3:
                return category

        return None

    def get_meta_analysis_prompt(self, category: str) -> str:
        """Generate a meta-analysis prompt for a recurring failure category.

        When the same failure category keeps appearing, this prompt forces
        the team to address the root cause rather than symptoms.
        """
        prompts = {
            "import_error": (
                "RECURRING ISSUE: Import errors keep appearing across multiple fix attempts. "
                "The root cause is likely: (1) missing dependency in requirements.txt, "
                "(2) incorrect module path, or (3) circular import. "
                "Address the ROOT CAUSE: verify every import resolves, check requirements.txt is complete."
            ),
            "syntax_error": (
                "RECURRING ISSUE: Syntax errors persist. The root cause is likely: "
                "incomplete code generation or mismatched brackets/quotes. "
                "Address the ROOT CAUSE: regenerate the entire file from scratch rather than patching."
            ),
            "missing_file": (
                "RECURRING ISSUE: Missing files keep being reported. The root cause is likely: "
                "file paths in code do not match actual file locations. "
                "Address the ROOT CAUSE: list all files that should exist, verify each one, create any missing ones."
            ),
            "runtime_error": (
                "RECURRING ISSUE: Runtime errors persist across fix attempts. The root cause is likely: "
                "a fundamental logic flaw, not a surface-level bug. "
                "Address the ROOT CAUSE: trace the execution path from entry to crash point."
            ),
            "logic_error": (
                "RECURRING ISSUE: Logic errors keep appearing. The output is consistently wrong. "
                "Address the ROOT CAUSE: re-examine the algorithm or data transformation logic from first principles."
            ),
            "test_failure": (
                "RECURRING ISSUE: Tests keep failing. The root cause is likely: "
                "mismatch between test expectations and actual implementation behavior. "
                "Address the ROOT CAUSE: verify both the test and the implementation against the requirements."
            ),
        }
        return prompts.get(category, (
            f"RECURRING ISSUE: The '{category}' failure category has appeared 3+ times. "
            "Stop patching symptoms. Identify and fix the root cause."
        ))

    # ------------------------------------------------------------------
    # Post-mortem report (still useful for status reporting, not termination)
    # ------------------------------------------------------------------

    def generate_post_mortem(
        self,
        workflow_state: Dict[str, Any],
        failure_history: List[str],
    ) -> Dict[str, Any]:
        """Build a structured status report for a struggling workflow.

        Unlike the old version, this does NOT imply termination — it's
        informational, summarizing what's been tried so far.
        """

        total_rounds = len(self._history)
        repeated = self.detect_repeated_failures(failure_history)
        unique_hashes = set(self._hash_feedback(f) for f in failure_history)

        escalation_timeline = [
            {
                "round": record.round,
                "action": record.action.value,
                "qa_verdict": record.qa_verdict,
                "dev_fix_count": record.dev_fix_count,
                "repeated_failure_detected": record.repeated_failure_detected,
            }
            for record in self._history
        ]

        failure_pattern = "repeated" if repeated else "progressive"

        recommendations = self._build_recommendations(
            total_rounds=total_rounds,
            repeated=repeated,
            unique_failure_count=len(unique_hashes),
            failure_history=failure_history,
            workflow_state=workflow_state,
        )

        return {
            "total_rounds": total_rounds,
            "failure_pattern": failure_pattern,
            "unique_failure_count": len(unique_hashes),
            "repeated_failure_detected": repeated,
            "escalation_history": escalation_timeline,
            "last_qa_feedback": failure_history[-1] if failure_history else None,
            "recommendations": recommendations,
            "strategy_summary": self.strategy_memory.summarize_for_next_attempt(),
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[EscalationRecord]:
        """Read-only access to the escalation history."""
        return list(self._history)

    def reset(self) -> None:
        """Clear internal history.  Useful between workflow runs."""
        self._history.clear()
        self.strategy_memory.reset()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        round: int,
        action: EscalationAction,
        qa_verdict: str,
        dev_fix_count: int,
        repeated: bool,
    ) -> None:
        """Append a record to the internal escalation history."""
        self._history.append(
            EscalationRecord(
                round=round,
                action=action,
                qa_verdict=qa_verdict,
                dev_fix_count=dev_fix_count,
                repeated_failure_detected=repeated,
            )
        )

    @staticmethod
    def _hash_feedback(feedback: str) -> str:
        """Produce a stable SHA-256 hex digest for a feedback string."""
        normalised = feedback.strip().lower()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def save_strategy_memory(self, path) -> None:
        """Persist strategy memory to disk for cross-run learning."""
        import json
        from pathlib import Path as _P
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tried_approaches": [
                {"approach": a.approach, "round": a.round,
                 "outcome": a.outcome, "feedback": a.feedback}
                for a in self.strategy_memory.tried_approaches
            ],
            "tried_architectures": self.strategy_memory.tried_architectures,
            "tried_scopes": self.strategy_memory.tried_scopes,
        }
        p.write_text(json.dumps(data, indent=2))

    def load_strategy_memory(self, path) -> None:
        """Load strategy memory from disk. Silently ignores errors."""
        import json
        from pathlib import Path as _P
        p = _P(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            for a in data.get("tried_approaches", []):
                self.strategy_memory.tried_approaches.append(
                    StrategyAttempt(**a)
                )
            self.strategy_memory.tried_architectures = data.get(
                "tried_architectures", []
            )
            self.strategy_memory.tried_scopes = data.get(
                "tried_scopes", []
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    @staticmethod
    def _build_recommendations(
        total_rounds: int,
        repeated: bool,
        unique_failure_count: int,
        failure_history: List[str],
        workflow_state: Dict[str, Any],
    ) -> List[str]:
        """Generate human-readable recommendations."""

        recommendations: List[str] = []

        if repeated:
            recommendations.append(
                "The same failure appeared multiple times. Strategy rotation "
                "has been triggered to try a fundamentally different approach."
            )

        if unique_failure_count == 1 and total_rounds >= 3:
            recommendations.append(
                "Only one distinct failure was observed across all rounds. "
                "This suggests a single blocking issue. Consider problem "
                "decomposition to isolate the root cause."
            )
        elif unique_failure_count > 3:
            recommendations.append(
                "Multiple distinct failures were observed. The technical "
                "design may need to be revisited from scratch via pivot."
            )

        if total_rounds >= 8:
            recommendations.append(
                "Multiple strategy cycles completed. The problem may "
                "require breaking into smaller sub-problems."
            )

        phase = workflow_state.get("phase", "")
        if isinstance(phase, str) and phase == "failed":
            recommendations.append(
                "The workflow reached a failed state. Review the full "
                "escalation history to identify where the process broke down."
            )

        if not recommendations:
            recommendations.append(
                "No specific pattern detected. Review the QA feedback "
                "in detail to determine next steps."
            )

        return recommendations
