"""Agent personality traits and experience accumulation.

Gives each agent a distinct personality that affects their behavior,
and tracks career progression across workflow runs.

Pure logic — no LLM calls, no I/O (except optional disk persistence).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class AgentPersonality:
    """Personality traits that affect agent behavior.

    Each trait is a float from 0.0 to 1.0:
    - risk_tolerance: 0=conservative, 1=adventurous
    - thoroughness:   0=quick, 1=meticulous
    - creativity:     0=conventional, 1=innovative
    - assertiveness:  0=agreeable, 1=pushback
    """

    risk_tolerance: float = 0.5
    thoroughness: float = 0.5
    creativity: float = 0.5
    assertiveness: float = 0.5

    def adjust_temperature(self, base_temp: float) -> float:
        """Adjust LLM temperature based on personality.

        More creative -> higher temp, more thorough -> lower temp.
        """
        adjustment = (self.creativity - 0.5) * 0.3 - (self.thoroughness - 0.5) * 0.2
        return max(0.1, min(1.5, base_temp + adjustment))

    def get_prompt_modifier(self) -> str:
        """Return personality-based prompt additions.

        Uses concrete behavioral instructions that 7-8B LLMs can follow.
        """
        modifiers = []

        if self.assertiveness > 0.7:
            modifiers.append(
                "If you see a problem, say 'I disagree because [reason]' before continuing."
            )
        elif self.assertiveness < 0.3:
            modifiers.append(
                "Start by acknowledging the team's input before adding your perspective."
            )

        if self.thoroughness > 0.7:
            modifiers.append(
                "After your main response, add a RISKS section with 1-3 things that could go wrong."
            )
        elif self.thoroughness < 0.3:
            modifiers.append(
                "Give your top recommendation first. Skip detailed analysis unless asked."
            )

        if self.risk_tolerance > 0.7:
            modifiers.append(
                "Include one unconventional approach, clearly marked as [BOLD IDEA]."
            )
        elif self.risk_tolerance < 0.3:
            modifiers.append(
                "Prefer proven, safe approaches. Flag any untested idea with [RISK]."
            )

        if self.creativity > 0.7:
            modifiers.append(
                "Suggest at least one non-obvious alternative the team hasn't considered."
            )

        return " ".join(modifiers) if modifiers else ""


# Default personalities per role
DEFAULT_PERSONALITIES: Dict[str, AgentPersonality] = {
    "ceo": AgentPersonality(
        risk_tolerance=0.6, thoroughness=0.5,
        creativity=0.5, assertiveness=0.8
    ),
    "cto": AgentPersonality(
        risk_tolerance=0.3, thoroughness=0.8,
        creativity=0.6, assertiveness=0.6
    ),
    "product_manager": AgentPersonality(
        risk_tolerance=0.5, thoroughness=0.6,
        creativity=0.6, assertiveness=0.5
    ),
    "researcher": AgentPersonality(
        risk_tolerance=0.4, thoroughness=0.8,
        creativity=0.5, assertiveness=0.4
    ),
    "developer": AgentPersonality(
        risk_tolerance=0.4, thoroughness=0.6,
        creativity=0.7, assertiveness=0.4
    ),
    "qa_engineer": AgentPersonality(
        risk_tolerance=0.1, thoroughness=0.9,
        creativity=0.3, assertiveness=0.7
    ),
    "devops_engineer": AgentPersonality(
        risk_tolerance=0.2, thoroughness=0.8,
        creativity=0.4, assertiveness=0.5
    ),
    "data_analyst": AgentPersonality(
        risk_tolerance=0.3, thoroughness=0.9,
        creativity=0.4, assertiveness=0.4
    ),
    "security_engineer": AgentPersonality(
        risk_tolerance=0.0, thoroughness=0.9,
        creativity=0.3, assertiveness=0.9
    ),
}


@dataclass
class AgentExperience:
    """Tracks an agent's career progression across runs.

    Level thresholds:
    - <10 tasks:   junior
    - 10-50 tasks: mid
    - 50-200:      senior
    - 200+:        lead
    """

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    domain_experience: Dict[str, int] = field(default_factory=dict)

    @property
    def level(self) -> str:
        """Calculate current level based on total experience."""
        if self.total_tasks >= 200:
            return "lead"
        elif self.total_tasks >= 50:
            return "senior"
        elif self.total_tasks >= 10:
            return "mid"
        return "junior"

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def add_experience(self, task_type: str, success: bool) -> None:
        """Record experience and update counts."""
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1

        # Track domain-specific experience
        self.domain_experience[task_type] = (
            self.domain_experience.get(task_type, 0) + 1
        )

    def get_prompt_adjustment(self) -> str:
        """Get prompt adjustment based on experience level.

        Junior agents get more detailed instructions.
        Senior agents get more autonomy.
        """
        level = self.level
        if level == "junior":
            return (
                "You are relatively new. Follow instructions carefully. "
                "Ask clarifying questions when uncertain. "
                "Double-check your work before submitting."
            )
        elif level == "mid":
            return ""  # No adjustment for mid-level
        elif level == "senior":
            return (
                "You are experienced. Use your judgment for ambiguous cases. "
                "Provide insights beyond what was asked when relevant."
            )
        else:  # lead
            return (
                "You are a lead with extensive experience. "
                "Mentor others by explaining your reasoning. "
                "Challenge assumptions when appropriate."
            )

    def to_dict(self) -> Dict:
        """Serialize to dict for persistence."""
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "domain_experience": self.domain_experience,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentExperience":
        """Deserialize from dict."""
        return cls(
            total_tasks=data.get("total_tasks", 0),
            successful_tasks=data.get("successful_tasks", 0),
            failed_tasks=data.get("failed_tasks", 0),
            domain_experience=data.get("domain_experience", {}),
        )

    def save(self, path: Path) -> None:
        """Persist experience to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "AgentExperience":
        """Load experience from disk."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass
        return cls()
