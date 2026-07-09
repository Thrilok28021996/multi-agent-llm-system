"""Structured meeting types for real company simulation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class MeetingMinutes:
    """Record of a meeting."""

    meeting_type: str
    attendees: List[str]
    agenda: str
    outcomes: List[str] = field(default_factory=list)
    action_items: List[Dict[str, str]] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.meeting_type,
            "attendees": self.attendees,
            "agenda": self.agenda,
            "outcomes": self.outcomes,
            "action_items": self.action_items,
            "blockers": self.blockers,
            "timestamp": self.timestamp.isoformat(),
        }


class StandupManager:
    """Manages daily standup meetings."""

    def __init__(self):
        self._standup_history: List[MeetingMinutes] = []

    def generate_standup_prompt(self, agent_name: str, role: str, current_phase: str, blockers: List[str] = None) -> str:
        """Generate a standup prompt for an agent."""
        blocker_text = ""
        if blockers:
            blocker_text = f"\nKnown blockers: {', '.join(blockers)}"
        return (
            f"DAILY STANDUP for {agent_name} ({role}):\n"
            f"Current phase: {current_phase}\n"
            f"Report briefly:\n"
            f"1. DONE: What did you complete since last standup?\n"
            f"2. DOING: What are you working on now?\n"
            f"3. BLOCKED: What is preventing progress? Name specific agents or resources needed.\n"
            f"{blocker_text}"
        )

    def record_standup(self, attendees: List[str], phase: str, reports: Dict[str, str]) -> MeetingMinutes:
        """Record standup results."""
        blockers = []
        for agent_name, report in reports.items():
            report_lower = report.lower()
            if "blocked" in report_lower or "waiting" in report_lower or "need" in report_lower:
                blockers.append(f"{agent_name}: {report}")

        minutes = MeetingMinutes(
            meeting_type="standup",
            attendees=attendees,
            agenda=f"Daily standup during {phase}",
            outcomes=[f"{name}: {report[:200]}" for name, report in reports.items()],
            blockers=blockers,
        )
        self._standup_history.append(minutes)
        return minutes


class DesignReviewManager:
    """Manages design review meetings."""

    def generate_review_prompt(self, design: str, reviewers: List[str]) -> str:
        """Generate a design review prompt."""
        return (
            f"DESIGN REVIEW:\n"
            f"Reviewers: {', '.join(reviewers)}\n\n"
            f"Design to review:\n{design}\n\n"
            f"For each reviewer, answer:\n"
            f"1. Does this design address the root cause?\n"
            f"2. What is the biggest risk?\n"
            f"3. What would you change?\n"
            f"4. APPROVE or REQUEST_CHANGES?"
        )


class IncidentResponseManager:
    """Manages incident response meetings for critical failures."""

    def generate_incident_prompt(self, failure_description: str, failure_history: List[str], round_number: int) -> str:
        """Generate an incident response meeting prompt."""
        history_text = "\n".join(f"  Round {i+1}: {f[:200]}" for i, f in enumerate(failure_history[-5:]))
        return (
            f"INCIDENT RESPONSE MEETING — Round {round_number}\n\n"
            f"Current failure:\n{failure_description}\n\n"
            f"Recent failure history:\n{history_text}\n\n"
            f"ALL AGENTS: Analyze this incident:\n"
            f"1. What is the ROOT CAUSE (not symptoms)?\n"
            f"2. Why did previous fixes not work?\n"
            f"3. What is the simplest possible fix?\n"
            f"4. Who should own the fix?\n"
            f"5. What should we STOP doing that is not helping?"
        )
