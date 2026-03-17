"""
Thinking Mode for Company AGI.

Provides Claude Code-style extended thinking with:
- Structured reasoning before responses
- Thinking block output
- Configurable thinking depth
- Reasoning traces for debugging
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ThinkingDepth(Enum):
    """Depth levels for thinking."""
    MINIMAL = "minimal"  # Quick, surface-level reasoning
    STANDARD = "standard"  # Normal reasoning depth
    DEEP = "deep"  # Extended reasoning
    EXHAUSTIVE = "exhaustive"  # Maximum reasoning depth


class ThinkingPhase(Enum):
    """Phases of the thinking process."""
    UNDERSTAND = "understand"  # Understanding the problem
    ANALYZE = "analyze"  # Analyzing constraints and context
    EXPLORE = "explore"  # Exploring possible approaches
    EVALUATE = "evaluate"  # Evaluating options
    DECIDE = "decide"  # Making a decision
    PLAN = "plan"  # Planning execution


@dataclass
class ThinkingStep:
    """A single step in the thinking process."""
    phase: ThinkingPhase
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0
    confidence: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class ThinkingBlock:
    """A complete thinking block."""
    id: str
    query: str
    steps: List[ThinkingStep] = field(default_factory=list)
    conclusion: Optional[str] = None
    depth: ThinkingDepth = ThinkingDepth.STANDARD
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_duration_ms: int = 0
    token_count: int = 0

    def add_step(
        self,
        phase: ThinkingPhase,
        content: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThinkingStep:
        """Add a thinking step."""
        step = ThinkingStep(
            phase=phase,
            content=content,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step

    def complete(self, conclusion: str) -> None:
        """Complete the thinking block."""
        self.conclusion = conclusion
        self.completed_at = datetime.now()
        self.total_duration_ms = int(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )

    def to_markdown(self) -> str:
        """Convert thinking block to markdown."""
        lines = [
            "<thinking>",
        ]

        for step in self.steps:
            phase_icon = {
                ThinkingPhase.UNDERSTAND: "🔍",
                ThinkingPhase.ANALYZE: "📊",
                ThinkingPhase.EXPLORE: "🗺️",
                ThinkingPhase.EVALUATE: "⚖️",
                ThinkingPhase.DECIDE: "✅",
                ThinkingPhase.PLAN: "📋",
            }.get(step.phase, "•")

            lines.append(f"\n{phase_icon} **{step.phase.value.title()}**")
            lines.append(step.content)

        if self.conclusion:
            lines.append(f"\n**Conclusion:** {self.conclusion}")

        lines.append("\n</thinking>")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "depth": self.depth.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "token_count": self.token_count,
        }


class ThinkingConfig:
    """Configuration for thinking behavior."""

    # Phase configurations by depth
    DEPTH_PHASES = {
        ThinkingDepth.MINIMAL: [
            ThinkingPhase.UNDERSTAND,
            ThinkingPhase.DECIDE,
        ],
        ThinkingDepth.STANDARD: [
            ThinkingPhase.UNDERSTAND,
            ThinkingPhase.ANALYZE,
            ThinkingPhase.DECIDE,
        ],
        ThinkingDepth.DEEP: [
            ThinkingPhase.UNDERSTAND,
            ThinkingPhase.ANALYZE,
            ThinkingPhase.EXPLORE,
            ThinkingPhase.EVALUATE,
            ThinkingPhase.DECIDE,
        ],
        ThinkingDepth.EXHAUSTIVE: [
            ThinkingPhase.UNDERSTAND,
            ThinkingPhase.ANALYZE,
            ThinkingPhase.EXPLORE,
            ThinkingPhase.EVALUATE,
            ThinkingPhase.DECIDE,
            ThinkingPhase.PLAN,
        ],
    }

    # Phase prompts
    PHASE_PROMPTS = {
        ThinkingPhase.UNDERSTAND: (
            "Break this down to fundamentals. What is ACTUALLY being asked? "
            "Strip away assumptions. What are the raw inputs and desired outputs? "
            "What constraints are absolute vs. negotiable?"
        ),
        ThinkingPhase.ANALYZE: (
            "What are the underlying principles that govern this domain? "
            "What invariants must hold? What are the failure modes? "
            "What does the evidence actually show vs. what we are assuming?"
        ),
        ThinkingPhase.EXPLORE: (
            "Generate at least 3 fundamentally different approaches. "
            "For each: What principle does it rely on? What is the strongest argument against it? "
            "What is the simplest version?"
        ),
        ThinkingPhase.EVALUATE: (
            "For each approach: What is the worst case? What is the most likely case? "
            "What assumptions must hold for this to work? "
            "Which approach requires the fewest assumptions?"
        ),
        ThinkingPhase.DECIDE: (
            "Choose the approach with the strongest evidence base and fewest assumptions. "
            "State explicitly: what could prove this decision wrong?"
        ),
        ThinkingPhase.PLAN: (
            "Sequence the work to get feedback earliest. "
            "What is the smallest thing we can build to validate our approach "
            "before committing fully?"
        ),
    }

    def __init__(
        self,
        default_depth: ThinkingDepth = ThinkingDepth.STANDARD,
        max_thinking_time_ms: int = 30000,
        show_thinking: bool = True,
        save_thinking: bool = True,
    ):
        self.default_depth = default_depth
        self.max_thinking_time_ms = max_thinking_time_ms
        self.show_thinking = show_thinking
        self.save_thinking = save_thinking

    def get_phases(self, depth: ThinkingDepth) -> List[ThinkingPhase]:
        """Get phases for a given depth."""
        return self.DEPTH_PHASES.get(depth, self.DEPTH_PHASES[ThinkingDepth.STANDARD])


class ThinkingEngine:
    """
    Engine for structured thinking.

    Features:
    - Multi-phase reasoning process
    - Configurable thinking depth
    - First principles integration
    - Reasoning traces
    """

    def __init__(
        self,
        config: Optional[ThinkingConfig] = None,
        llm: Optional[Any] = None,
        first_principles: Optional[List[str]] = None,
    ):
        self.config = config or ThinkingConfig()
        self.llm = llm
        self.first_principles = first_principles or []

        self._thinking_history: List[ThinkingBlock] = []
        self._current_block: Optional[ThinkingBlock] = None
        self._callbacks: List[Callable[[ThinkingStep], None]] = []

    def add_callback(self, callback: Callable[[ThinkingStep], None]) -> None:
        """Add callback for thinking steps."""
        self._callbacks.append(callback)

    def _emit_step(self, step: ThinkingStep) -> None:
        """Emit a thinking step to callbacks."""
        for callback in self._callbacks:
            try:
                callback(step)
            except Exception:
                pass

    def think(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        depth: Optional[ThinkingDepth] = None,
    ) -> ThinkingBlock:
        """
        Perform structured thinking on a query.

        Returns a ThinkingBlock with reasoning steps.
        """
        import uuid

        thinking_depth = depth or self.config.default_depth
        phases = self.config.get_phases(thinking_depth)

        block = ThinkingBlock(
            id=str(uuid.uuid4())[:12],
            query=query,
            depth=thinking_depth,
        )
        self._current_block = block

        start_time = time.time()

        for phase in phases:
            # Check time limit
            elapsed_ms = int((time.time() - start_time) * 1000)
            if elapsed_ms > self.config.max_thinking_time_ms:
                break

            step_start = time.time()

            # Generate thinking for this phase
            content = self._think_phase(phase, query, context, block.steps)

            step_duration = int((time.time() - step_start) * 1000)

            step = block.add_step(
                phase=phase,
                content=content,
                metadata={"duration_ms": step_duration},
            )
            step.duration_ms = step_duration

            self._emit_step(step)

        # Generate conclusion
        conclusion = self._generate_conclusion(block)
        block.complete(conclusion)

        self._thinking_history.append(block)
        self._current_block = None

        return block

    def _think_phase(
        self,
        phase: ThinkingPhase,
        query: str,
        context: Optional[Dict[str, Any]],
        previous_steps: List[ThinkingStep],
    ) -> str:
        """Generate thinking content for a phase."""
        # If we have an LLM, use it
        if self.llm:
            return self._llm_think(phase, query, context, previous_steps)

        # Otherwise, use template-based thinking
        return self._template_think(phase, query, context, previous_steps)

    def _llm_think(
        self,
        phase: ThinkingPhase,
        query: str,
        context: Optional[Dict[str, Any]],
        previous_steps: List[ThinkingStep],
    ) -> str:
        """Use LLM to generate thinking."""
        # Build prompt
        prompt_parts = [
            f"You are in the {phase.value} phase of reasoning.",
            f"Query: {query}",
        ]

        if context:
            prompt_parts.append(f"Context: {context}")

        if previous_steps:
            prev_thoughts = "\n".join([
                f"- {s.phase.value}: {s.content}"
                for s in previous_steps[-3:]
            ])
            prompt_parts.append(f"Previous thoughts:\n{prev_thoughts}")

        if self.first_principles:
            principles = "\n".join([f"- {p}" for p in self.first_principles])
            prompt_parts.append(f"Apply these first principles:\n{principles}")

        prompt_parts.append(f"\n{ThinkingConfig.PHASE_PROMPTS.get(phase, '')}")
        prompt_parts.append("\nProvide your reasoning for this phase:")

        prompt = "\n\n".join(prompt_parts)

        try:
            if self.llm is not None:
                if hasattr(self.llm, 'generate'):
                    result = self.llm.generate(prompt)  # type: ignore[union-attr]
                    return str(result) if result else self._template_think(phase, query, context, previous_steps)
                elif callable(self.llm):
                    result = self.llm(prompt)
                    return str(result) if result else self._template_think(phase, query, context, previous_steps)
        except Exception:
            pass  # Fall through to template-based thinking

        return self._template_think(phase, query, context, previous_steps)

    def _template_think(
        self,
        phase: ThinkingPhase,
        query: str,
        context: Optional[Dict[str, Any]],
        previous_steps: List[ThinkingStep],
    ) -> str:
        """Generate template-based thinking."""
        _ = previous_steps  # Reserved for future use

        templates = {
            ThinkingPhase.UNDERSTAND: self._template_understand,
            ThinkingPhase.ANALYZE: self._template_analyze,
            ThinkingPhase.EXPLORE: self._template_explore,
            ThinkingPhase.EVALUATE: self._template_evaluate,
            ThinkingPhase.DECIDE: self._template_decide,
            ThinkingPhase.PLAN: self._template_plan,
        }

        template_fn = templates.get(phase, lambda q, c: f"Processing: {q}")
        return template_fn(query, context)

    def _template_understand(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Template for understanding phase."""
        parts = [f"The request is: {query}"]

        if context:
            if "files" in context:
                parts.append(f"Relevant files: {len(context['files'])} files")
            if "constraints" in context:
                parts.append(f"Constraints: {context['constraints']}")

        # Apply first principles
        if self.first_principles:
            parts.append(f"Key principle to consider: {self.first_principles[0]}")

        return " ".join(parts)

    def _template_analyze(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Template for analysis phase."""
        parts = ["Analyzing the problem space."]

        # Identify key aspects
        keywords = ["implement", "fix", "add", "remove", "change", "update"]
        action = next((kw for kw in keywords if kw in query.lower()), "process")
        parts.append(f"Primary action: {action}")

        if context and "complexity" in context:
            parts.append(f"Complexity level: {context['complexity']}")

        return " ".join(parts)

    def _template_explore(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Template for exploration phase."""
        _ = query, context  # Reserved for future use
        parts = ["Exploring possible approaches:"]

        # Generate generic approaches
        approaches = [
            "1. Direct implementation",
            "2. Iterative refinement",
            "3. Test-driven approach",
        ]
        parts.extend(approaches)

        return "\n".join(parts)

    def _template_evaluate(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Template for evaluation phase."""
        _ = query, context  # Reserved for future use
        parts = ["Evaluating options:"]

        # Apply first principles for evaluation
        if self.first_principles:
            for i, principle in enumerate(self.first_principles[:3], 1):
                parts.append(f"- Principle {i}: {principle}")

        parts.append("Considering trade-offs between simplicity and completeness.")

        return "\n".join(parts)

    def _template_decide(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Template for decision phase."""
        _ = context  # Reserved for future use
        return f"Based on analysis, proceeding with the most appropriate approach for: {query}"

    def _template_plan(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Template for planning phase."""
        _ = query, context  # Reserved for future use
        parts = ["Execution plan:"]

        # Generate generic steps
        steps = [
            "1. Gather necessary context",
            "2. Implement core changes",
            "3. Verify results",
            "4. Handle edge cases if needed",
        ]
        parts.extend(steps)

        return "\n".join(parts)

    def _generate_conclusion(self, block: ThinkingBlock) -> str:
        """Generate conclusion from thinking steps."""
        if not block.steps:
            return "Proceeding with request."

        # Use the last few steps to form conclusion
        last_steps = block.steps[-2:]
        key_points = [s.content for s in last_steps]

        if len(key_points) == 1:
            return key_points[0]

        return f"After consideration: {key_points[-1]}"

    def get_thinking_summary(self) -> Dict[str, Any]:
        """Get summary of thinking history."""
        return {
            "total_blocks": len(self._thinking_history),
            "total_duration_ms": sum(b.total_duration_ms for b in self._thinking_history),
            "average_steps": (
                sum(len(b.steps) for b in self._thinking_history) / len(self._thinking_history)
                if self._thinking_history else 0
            ),
            "depths_used": list(set(b.depth.value for b in self._thinking_history)),
        }

    def clear_history(self) -> None:
        """Clear thinking history."""
        self._thinking_history = []


# Singleton instance
_thinking_engine: Optional[ThinkingEngine] = None


def get_thinking_engine(
    config: Optional[ThinkingConfig] = None,
    first_principles: Optional[List[str]] = None,
) -> ThinkingEngine:
    """Get or create the global thinking engine."""
    global _thinking_engine
    if _thinking_engine is None:
        _thinking_engine = ThinkingEngine(
            config=config,
            first_principles=first_principles,
        )
    return _thinking_engine


def reset_thinking_engine() -> None:
    """Reset the global thinking engine."""
    global _thinking_engine
    _thinking_engine = None
