"""
Token Cost Tracking and Dashboard for Company AGI.

Provides comprehensive cost tracking:
- Per-model pricing configuration
- Per-agent cost tracking
- Session cost aggregation
- Cost dashboard and reporting
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelProvider(Enum):
    """Supported model providers."""
    LOCAL = "local"      # Ollama (local models)
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelPricing:
    """Pricing configuration for a model."""
    model_name: str
    provider: ModelProvider
    input_cost_per_1k: float = 0.0   # Cost per 1000 input tokens
    output_cost_per_1k: float = 0.0  # Cost per 1000 output tokens
    currency: str = "USD"

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate total cost for given token counts."""
        input_cost = (input_tokens / 1000.0) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_cost_per_1k
        return input_cost + output_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "provider": self.provider.value,
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            "currency": self.currency,
        }


@dataclass
class UsageRecord:
    """A single usage record."""
    timestamp: str
    model: str
    agent: Optional[str]
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentCostSummary:
    """Cost summary for a single agent."""
    agent_name: str
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration_ms: float = 0.0

    def add_usage(self, record: UsageRecord) -> None:
        """Add a usage record to the summary."""
        self.total_requests += 1
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_tokens += record.total_tokens
        self.total_cost += record.cost
        self.total_duration_ms += record.duration_ms

    @property
    def avg_tokens_per_request(self) -> float:
        """Average tokens per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_tokens / self.total_requests

    @property
    def avg_cost_per_request(self) -> float:
        """Average cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_duration_ms": self.total_duration_ms,
            "avg_tokens_per_request": self.avg_tokens_per_request,
            "avg_cost_per_request": self.avg_cost_per_request,
        }


@dataclass
class SessionCostSummary:
    """Cost summary for a session."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    agents: Dict[str, AgentCostSummary] = field(default_factory=dict)
    models_used: Dict[str, int] = field(default_factory=dict)

    def add_record(self, record: UsageRecord) -> None:
        """Add a usage record to the session summary."""
        self.total_requests += 1
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_tokens += record.total_tokens
        self.total_cost += record.cost

        # Track by agent
        agent_name = record.agent or "unknown"
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentCostSummary(agent_name=agent_name)
        self.agents[agent_name].add_usage(record)

        # Track models used
        if record.model not in self.models_used:
            self.models_used[record.model] = 0
        self.models_used[record.model] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "models_used": self.models_used,
        }


# Default pricing for common models
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # Local models via Ollama (no cost)
    "goekdenizguelmez/JOSIEFIED-Qwen3": ModelPricing("goekdenizguelmez/JOSIEFIED-Qwen3", ModelProvider.LOCAL, 0.0, 0.0),
    "ministral-3": ModelPricing("ministral-3", ModelProvider.LOCAL, 0.0, 0.0),
    "thealxlabs/lumen": ModelPricing("thealxlabs/lumen", ModelProvider.LOCAL, 0.0, 0.0),
    # OpenAI models
    "gpt-4": ModelPricing("gpt-4", ModelProvider.OPENAI, 0.03, 0.06),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", ModelProvider.OPENAI, 0.01, 0.03),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", ModelProvider.OPENAI, 0.0005, 0.0015),
    # Anthropic models
    "claude-3-opus": ModelPricing("claude-3-opus", ModelProvider.ANTHROPIC, 0.015, 0.075),
    "claude-3-sonnet": ModelPricing("claude-3-sonnet", ModelProvider.ANTHROPIC, 0.003, 0.015),
    "claude-3-haiku": ModelPricing("claude-3-haiku", ModelProvider.ANTHROPIC, 0.00025, 0.00125),
}


class CostTracker:
    """
    Token cost tracking and dashboard.

    Features:
    - Per-model pricing configuration
    - Per-agent cost tracking
    - Session cost aggregation
    - Cost dashboard and reporting
    - Persistent cost history
    """

    def __init__(
        self,
        pricing: Optional[Dict[str, ModelPricing]] = None,
        history_file: Optional[Path] = None,
        default_model_cost: float = 0.0,
    ):
        self.pricing = pricing or DEFAULT_PRICING.copy()
        self.history_file = history_file
        self.default_model_cost = default_model_cost

        self._records: List[UsageRecord] = []
        self._current_session: Optional[SessionCostSummary] = None
        self._all_sessions: List[SessionCostSummary] = []

        # Load history if file exists
        if history_file and history_file.exists():
            self._load_history()

    def _load_history(self) -> None:
        """Load cost history from file."""
        if self.history_file and self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                    # Load sessions
                    for session_data in data.get("sessions", []):
                        session = SessionCostSummary(
                            session_id=session_data["session_id"],
                            started_at=session_data["started_at"],
                            ended_at=session_data.get("ended_at"),
                            total_requests=session_data.get("total_requests", 0),
                            total_input_tokens=session_data.get("total_input_tokens", 0),
                            total_output_tokens=session_data.get("total_output_tokens", 0),
                            total_tokens=session_data.get("total_tokens", 0),
                            total_cost=session_data.get("total_cost", 0.0),
                        )
                        self._all_sessions.append(session)
            except Exception as e:
                import sys
                print(f"[CostTracker] Could not load cost history: {e}", file=sys.stderr)

    def _save_history(self) -> None:
        """Save cost history to file."""
        if self.history_file:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = {
                    "sessions": [s.to_dict() for s in self._all_sessions],
                    "last_updated": datetime.now().isoformat(),
                }
                with open(self.history_file, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                import sys
                print(f"[CostTracker] Could not save cost history: {e}", file=sys.stderr)

    def set_budget(self, max_tokens: int) -> None:
        """Set a token budget limit. Workflow should check is_over_budget() periodically."""
        self._budget_tokens = max_tokens

    def is_over_budget(self) -> bool:
        """Check if current session has exceeded the token budget."""
        budget = getattr(self, '_budget_tokens', None)
        if budget is None:
            return False
        if self._current_session:
            return self._current_session.total_tokens >= budget
        return False

    def extend_budget(self, fraction: float = 0.3) -> None:
        """Extend the token budget by a fraction of the current budget.

        Args:
            fraction: Fraction to extend by (e.g., 0.3 = 30% more tokens).
        """
        budget = getattr(self, '_budget_tokens', None)
        if budget is not None:
            extension = int(budget * fraction)
            self._budget_tokens = budget + extension
            import sys
            print(f"[CostTracker] Budget extended by {extension} tokens to {self._budget_tokens}", file=sys.stderr)

    def get_current_session(self) -> Optional["SessionCostSummary"]:
        """Get the current active session."""
        return self._current_session

    def set_pricing(self, model: str, pricing: ModelPricing) -> None:
        """Set pricing for a model."""
        self.pricing[model] = pricing

    def get_pricing(self, model: str) -> ModelPricing:
        """Get pricing for a model."""
        if model in self.pricing:
            return self.pricing[model]

        # Try partial match
        for key, pricing in self.pricing.items():
            if model.startswith(key.split(":")[0]):
                return pricing

        # Return default (free for local models)
        return ModelPricing(
            model_name=model,
            provider=ModelProvider.LOCAL,
            input_cost_per_1k=self.default_model_cost,
            output_cost_per_1k=self.default_model_cost,
        )

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new cost tracking session."""
        import uuid

        if self._current_session:
            self.end_session()

        session_id = session_id or str(uuid.uuid4())[:8]
        self._current_session = SessionCostSummary(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
        )

        return session_id

    def end_session(self) -> Optional[SessionCostSummary]:
        """End the current session."""
        if self._current_session:
            self._current_session.ended_at = datetime.now().isoformat()
            self._all_sessions.append(self._current_session)
            self._save_history()
            session = self._current_session
            self._current_session = None
            return session
        return None

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent: Optional[str] = None,
        operation: str = "inference",
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a token usage."""
        pricing = self.get_pricing(model)
        cost = pricing.calculate_cost(input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            agent=agent,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        self._records.append(record)

        # Add to current session
        if self._current_session:
            self._current_session.add_record(record)

        return record

    def get_session_summary(self) -> Optional[SessionCostSummary]:
        """Get current session summary."""
        return self._current_session

    def get_all_time_summary(self) -> Dict[str, Any]:
        """Get all-time cost summary."""
        total_requests = sum(s.total_requests for s in self._all_sessions)
        total_tokens = sum(s.total_tokens for s in self._all_sessions)
        total_cost = sum(s.total_cost for s in self._all_sessions)

        if self._current_session:
            total_requests += self._current_session.total_requests
            total_tokens += self._current_session.total_tokens
            total_cost += self._current_session.total_cost

        return {
            "total_sessions": len(self._all_sessions) + (1 if self._current_session else 0),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }

    def get_dashboard(self, color: bool = True) -> str:
        """Get cost dashboard as formatted string."""
        colors = {
            "header": "\033[1;36m",   # Bold Cyan
            "label": "\033[33m",      # Yellow
            "value": "\033[32m",      # Green
            "cost": "\033[1;33m",     # Bold Yellow
            "reset": "\033[0m",
            "bold": "\033[1m",
        }

        if not color:
            colors = {k: "" for k in colors}

        c = colors
        lines = [
            f"{c['bold']}Cost Dashboard{c['reset']}",
            "=" * 50,
        ]

        # Current session
        if self._current_session:
            s = self._current_session
            lines.extend([
                f"\n{c['header']}Current Session ({s.session_id}){c['reset']}",
                f"  {c['label']}Requests:{c['reset']}      {c['value']}{s.total_requests}{c['reset']}",
                f"  {c['label']}Input Tokens:{c['reset']}  {c['value']}{s.total_input_tokens:,}{c['reset']}",
                f"  {c['label']}Output Tokens:{c['reset']} {c['value']}{s.total_output_tokens:,}{c['reset']}",
                f"  {c['label']}Total Tokens:{c['reset']}  {c['value']}{s.total_tokens:,}{c['reset']}",
                f"  {c['label']}Cost:{c['reset']}          {c['cost']}${s.total_cost:.4f}{c['reset']}",
            ])

            # Per-agent breakdown
            if s.agents:
                lines.append(f"\n  {c['header']}By Agent:{c['reset']}")
                for agent_name, agent_summary in sorted(s.agents.items()):
                    lines.append(
                        f"    {agent_name:15} {c['value']}{agent_summary.total_tokens:>8,}{c['reset']} tokens  "
                        f"{c['cost']}${agent_summary.total_cost:.4f}{c['reset']}"
                    )

            # Models used
            if s.models_used:
                lines.append(f"\n  {c['header']}Models Used:{c['reset']}")
                for model, count in sorted(s.models_used.items()):
                    lines.append(f"    {model:25} {c['value']}{count:>5}{c['reset']} requests")

        else:
            lines.append(f"\n{c['label']}No active session{c['reset']}")

        # All-time summary
        all_time = self.get_all_time_summary()
        lines.extend([
            f"\n{c['header']}All Time{c['reset']}",
            f"  {c['label']}Sessions:{c['reset']}      {c['value']}{all_time['total_sessions']}{c['reset']}",
            f"  {c['label']}Requests:{c['reset']}      {c['value']}{all_time['total_requests']:,}{c['reset']}",
            f"  {c['label']}Total Tokens:{c['reset']}  {c['value']}{all_time['total_tokens']:,}{c['reset']}",
            f"  {c['label']}Total Cost:{c['reset']}    {c['cost']}${all_time['total_cost']:.4f}{c['reset']}",
        ])

        return "\n".join(lines)

    def get_recent_records(self, count: int = 10) -> List[UsageRecord]:
        """Get recent usage records."""
        return self._records[-count:]

    def save_report(self, path: Path) -> None:
        """Save cost report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now().isoformat(),
            "current_session": self._current_session.to_dict() if self._current_session else None,
            "all_time_summary": self.get_all_time_summary(),
            "recent_records": [r.to_dict() for r in self._records[-100:]],
            "all_sessions": [s.to_dict() for s in self._all_sessions],
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker(
    history_file: Optional[Path] = None,
) -> CostTracker:
    """Get or create the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker(
            history_file=history_file or Path("output/costs/cost_history.json"),
        )
    return _cost_tracker


def reset_cost_tracker() -> None:
    """Reset the global cost tracker."""
    global _cost_tracker
    if _cost_tracker:
        _cost_tracker.end_session()
    _cost_tracker = None


def track_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    agent: Optional[str] = None,
    **kwargs: Any,
) -> UsageRecord:
    """Convenience function to track token usage."""
    tracker = get_cost_tracker()
    return tracker.record_usage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        agent=agent,
        **kwargs,
    )
