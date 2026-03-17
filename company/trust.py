"""Inter-agent trust tracking.

Tracks trust between agents based on collaboration quality.
Trust affects meeting dynamics, review depth, and escalation speed.

Pure logic — no LLM calls, no I/O (except optional disk persistence).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class TrustTracker:
    """Track trust between agents based on collaboration quality.

    trust_scores[reviewer][reviewed] = float (0.0-1.0)

    Trust updates:
    - If agent_b's work passes agent_a's review: trust goes up
    - If agent_b's work fails agent_a's review: trust goes down
    - Trust changes are small and incremental (0.05 per event)

    Trust effects:
    - High trust (>0.8) -> light review
    - Medium trust (0.4-0.8) -> standard review
    - Low trust (<0.4) -> thorough review
    """

    DEFAULT_TRUST = 0.6  # Start with moderate trust
    TRUST_INCREMENT = 0.05
    TRUST_DECREMENT = 0.08  # Losing trust is faster than gaining it

    def __init__(self) -> None:
        self._scores: Dict[str, Dict[str, float]] = {}

    def _ensure(self, reviewer: str, reviewed: str) -> None:
        """Ensure the trust entry exists."""
        if reviewer not in self._scores:
            self._scores[reviewer] = {}
        if reviewed not in self._scores[reviewer]:
            self._scores[reviewer][reviewed] = self.DEFAULT_TRUST

    def get_trust(self, reviewer: str, reviewed: str) -> float:
        """Get trust score from reviewer toward reviewed agent."""
        self._ensure(reviewer, reviewed)
        return self._scores[reviewer][reviewed]

    def update_trust(self, reviewer: str, reviewed: str, positive: bool) -> float:
        """Update trust based on a review outcome.

        Args:
            reviewer: The agent doing the review
            reviewed: The agent whose work was reviewed
            positive: True if work passed review, False if it failed

        Returns:
            The new trust score.
        """
        self._ensure(reviewer, reviewed)
        current = self._scores[reviewer][reviewed]

        if positive:
            new_score = min(1.0, current + self.TRUST_INCREMENT)
        else:
            new_score = max(0.0, current - self.TRUST_DECREMENT)

        self._scores[reviewer][reviewed] = round(new_score, 3)
        return self._scores[reviewer][reviewed]

    def get_review_threshold(self, reviewer: str, reviewed: str) -> str:
        """Determine review depth based on trust level.

        Returns:
            "light", "standard", or "thorough"
        """
        trust = self.get_trust(reviewer, reviewed)
        if trust >= 0.8:
            return "light"
        elif trust >= 0.4:
            return "standard"
        return "thorough"

    def get_all_scores(self) -> Dict[str, Dict[str, float]]:
        """Get all trust scores."""
        return {
            reviewer: dict(scores)
            for reviewer, scores in self._scores.items()
        }

    def get_agent_reputation(self, agent_name: str) -> float:
        """Get the average trust others have toward this agent."""
        scores = []
        for reviewer, reviewed_scores in self._scores.items():
            if agent_name in reviewed_scores and reviewer != agent_name:
                scores.append(reviewed_scores[agent_name])
        if not scores:
            return self.DEFAULT_TRUST
        return round(sum(scores) / len(scores), 3)

    def reset(self) -> None:
        """Clear all trust scores."""
        self._scores.clear()

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {"scores": self._scores}

    @classmethod
    def from_dict(cls, data: Dict) -> "TrustTracker":
        """Deserialize from dict."""
        tracker = cls()
        tracker._scores = data.get("scores", {})
        return tracker

    def save(self, path: Path) -> None:
        """Persist trust scores to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "TrustTracker":
        """Load trust scores from disk."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass
        return cls()
