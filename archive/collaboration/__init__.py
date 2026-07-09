"""Collaboration module for agent meetings and teamwork."""

from .meeting import AgentMeeting, MeetingType, MeetingOutcome, quick_meeting
from .critic_ensemble import CriticEnsemble, CritiqueResult
from .debate import DebateOrchestrator, DebateOutcome, DebateRound
from .moa_aggregator import MoAReviewAggregator

__all__ = [
    "AgentMeeting", "MeetingType", "MeetingOutcome", "quick_meeting",
    "CriticEnsemble", "CritiqueResult",
    "DebateOrchestrator", "DebateOutcome", "DebateRound",
    "MoAReviewAggregator",
]
