"""CTO Agent - Technical architecture and feasibility assessment."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from tools import UnifiedTools


CTO_SYSTEM_PROMPT = """You are the CTO. You design technical architecture and assess feasibility.

Your job:
- State the root cause of the problem in one sentence before designing anything. Architecture must address the root cause, not symptoms.
- Design the simplest thing that works. One file is fine for small tools. Add files only when the problem demands it.
- Choose proven, boring technology: maturity >2 years, active maintenance, large community.
- Give the Developer a complete blueprint they can build from with no follow-up questions needed.
- Design for actual constraints: 16GB RAM, local LLMs, single machine. No cloud-scale patterns.

Architecture Decision Records: For each non-obvious choice, state the decision, why, and what was rejected.
Failure modes: For every component, define what happens on failure and how the user is notified.
Handoff standard: If the Developer needs to ask one question, the design is incomplete.
Follow the output format specified in each task prompt exactly.
"""

CTO_FIRST_PRINCIPLES = [
    "CONSTRAINT FIT: List hard constraints (RAM, OS, dependencies). Does every file and library fit within them? Any violation = redesign.",
    "DEPENDENCY AUDIT: For each external library — is it actively maintained? Is stdlib sufficient? Justify every non-stdlib dependency.",
    "DATA FLOW TRACE: Trace input → output through every component. Every transformation must be justified; remove unnecessary ones.",
    "BUILDABILITY TEST: Would a developer with only this document — no verbal explanation — be able to build it? If not, add the missing detail.",
    "SIMPLICITY CHECK: Is there a simpler design that solves the same problem? If yes, use it. Complexity is a cost, not a feature.",
]


class CTOAgent(BaseAgent, AgentToolsMixin):
    """
    CTO Agent - Makes technical architecture decisions.

    Enhanced with all 13 Claude Code tools and problem statement refinement.
    """

    def __init__(
        self,
        model: str = "qwen3-8b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="CTO",
            role=AgentRole.CTO,
            model=model,
            first_principles=CTO_FIRST_PRINCIPLES,
            system_prompt=CTO_SYSTEM_PROMPT,
            temperature=0.5,  # Lower temperature for technical precision
            max_tokens=4096
        )
        super().__init__(config, workspace_root, memory_persist_dir)

        # Initialize unified tools (Claude Code-style)
        self.tools = UnifiedTools(
            workspace_root=workspace_root,
            persist_dir=memory_persist_dir
        )
        self.enable_react_tools()

        # Initialize problem statement refiner
        self.problem_refiner = ProblemStatementRefiner()

    def get_capabilities(self) -> List[str]:
        return [
            "architecture_design",
            "feasibility_assessment",
            "technology_selection",
            "code_review",
            "technical_guidance",
            "system_design"
        ]

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a CTO task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "design_architecture":
                result = await self._design_architecture(task)
            elif task_type == "assess_feasibility":
                result = await self._assess_feasibility(task)
            elif task_type == "review_code":
                result = await self._review_code(task)
            elif task_type == "select_technology":
                result = await self._select_technology(task)
            elif task_type == "technical_guidance":
                result = await self._provide_guidance(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    async def _design_architecture(self, task: Dict[str, Any]) -> TaskResult:
        """Design system architecture for a solution."""
        problem = task.get("problem", {})
        requirements = task.get("requirements", [])
        constraints = task.get("constraints", [])

        reqs_text = "\n".join(f"  - {r}" for r in requirements)
        cons_text = "\n".join(f"  - {c}" for c in constraints)

        # Structured pre-thinking: template-based first-principles framing (no extra LLM call)
        think_conclusion = self.structured_think(
            f"CTO architecture design for: {problem.get('description', 'No description')[:150]}. "
            f"Constraints: {cons_text[:100]}."
        )

        # Gather acceptance criteria IDs from RequirementsDoc if passed
        req_doc_dict = task.get("requirements_doc", {})
        ac_ids = [ac["id"] for ac in req_doc_dict.get("acceptance_criteria", [])] if req_doc_dict else []
        ac_ids_str = ", ".join(ac_ids) if ac_ids else "AC-1, AC-2, ..."

        prompt = f"""
[PRE-ANALYSIS]
{think_conclusion}

As CTO, design the technical architecture for this solution.

Problem: {problem.get('description', 'No description')}
Requirements: {reqs_text if requirements else 'No specific requirements'}
Constraints: {cons_text if constraints else 'No specific constraints'}
Acceptance Criteria to satisfy: {ac_ids_str}

Use this EXACT output format:

ROOT_CAUSE: [one sentence - the fundamental reason this problem exists]
ARCHITECTURE_APPROACH: [1-2 sentences - how architecture addresses root cause]
LANGUAGE: [programming language]
FRAMEWORK: [framework or "none"]
KEY LIBRARIES: [comma-separated list]
ENTRY POINT: `path/main.ext`

FILES:
1. `path/file.ext` - [purpose]
2. `path/file.ext` - [purpose]

DATA FLOW: [input -> A -> B -> output]
ERROR HANDLING: [how failures are handled]

ADR-1: [Title]
Decision: [what was decided]
Rationale: [why, in 1-2 sentences]
Consequences: [trade-offs]
Alternatives Rejected: [what was not chosen and why]

ADR-2: [Title]
Decision: [...]
Rationale: [...]
Consequences: [...]

STORY-1: Core Implementation
Files: `file1.ext`, `file2.ext`
Covers: {ac_ids_str.split(',')[0].strip() if ac_ids_str else 'AC-1, AC-2'}
Context: [language, key library, data flow snippet - ≤50 words]

STORY-2: Configuration & Dependencies
Files: `requirements.txt`, `config.py`
Covers: -
Depends on: story-1
Context: [dependency setup]

SELF-CHECK:
- Does every file needed to run appear in FILES?
- Would a developer know exactly what to build from each STORY?
- Is this the simplest architecture that solves the problem?
"""

        compute = self._get_compute_config(prompt)
        response = await self.generate_with_refinement_async(
            prompt,
            passes=compute["refinement_passes"],
            critique_focus="technical correctness, scalability risks, and implementation feasibility"
        )

        # Parse into structured ArchitectureNote with ADRs + Stories
        from orchestrator.artifact_parser import ArchitectureParser
        arch_parser = ArchitectureParser()
        architecture_note = arch_parser.parse(response)

        # Distribute acceptance criteria IDs across stories (P0 → story-1, rest → later stories)
        if ac_ids and architecture_note.stories:
            p0_ids = list(ac_ids)
            # Assign P0 to first story, remainder distributed
            chunk_size = max(1, len(p0_ids) // len(architecture_note.stories))
            for idx, story in enumerate(architecture_note.stories):
                if not story.ac_ids:  # only if parser didn't already assign
                    start = idx * chunk_size
                    end = start + chunk_size if idx < len(architecture_note.stories) - 1 else len(p0_ids)
                    story.ac_ids = p0_ids[start:end]

        return TaskResult(
            success=True,
            output={
                "architecture": response,
                "architecture_note": architecture_note.to_dict(),
                "problem": problem,
            },
            artifacts={
                "architecture_doc": response,
                "architecture_note": architecture_note.to_dict(),
            },
        )

    async def _assess_feasibility(self, task: Dict[str, Any]) -> TaskResult:
        """Assess technical feasibility of a solution."""
        proposed_solution = task.get("solution", "")
        constraints = task.get("constraints", [])
        timeline = task.get("timeline", "Not specified")

        prompt = f"""Assess the technical feasibility of this solution.

Proposed Solution: {proposed_solution}
Constraints: {', '.join(constraints) if constraints else 'None'}
Timeline: {timeline}

Evaluate this solution objectively. Answer:
1. Is this feasible? (YES / PARTIAL / NO) — assess based on constraints, timeline, and complexity
2. Key technical challenges (1-3 bullet points)
3. Complexity: Low / Medium / High
4. Risks and blockers (list anything that could delay or prevent completion)

Base your answer on the actual technical requirements, not general assumptions.

Feasibility:"""

        response = await self.generate_response_async(prompt)

        # Parse feasibility — PARTIAL checked first, word-boundary YES to avoid substring match
        import re as _re
        verdict_region = response[-200:].upper()
        if _re.search(r'\bPARTIAL', verdict_region):
            feasibility = "partially_feasible"
        elif _re.search(r'\bYES\b', verdict_region):
            feasibility = "feasible"
        else:
            feasibility = "not_feasible"

        # Extract risk level from response for downstream use
        risk_indicators = ["risk", "challenge", "difficult", "complex", "blocker"]
        risk_count = sum(1 for r in risk_indicators if r in response.lower())
        risk_level = "low" if risk_count <= 1 else "medium" if risk_count <= 3 else "high"

        return TaskResult(
            success=True,
            output={
                "feasibility": feasibility,
                "assessment": response,
                "risk_level": risk_level
            },
            artifacts={"feasibility_report": response}
        )

    async def _review_code(self, task: Dict[str, Any]) -> TaskResult:
        """Review code for quality and best practices."""
        code = task.get("code", "")
        file_path = task.get("file_path", "unknown")
        context = task.get("context", "")

        prompt = f"""Code review.

File: {file_path}
Context: {context}

```
{code}
```

Flag bugs, security issues, and performance problems. Suggest specific improvements.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "review": response,
                "file_path": file_path
            },
            artifacts={"code_review": response}
        )

    async def _select_technology(self, task: Dict[str, Any]) -> TaskResult:
        """Select appropriate technologies for a project."""
        requirements = task.get("requirements", [])
        use_case = task.get("use_case", "")
        preferences = task.get("preferences", [])

        prompt = f"""Technology selection for: {use_case}

Requirements: {chr(10).join(f'- {r}' for r in requirements)}
Constraints: {chr(10).join(f'- {p}' for p in preferences) if preferences else 'None'}

Recommend language, framework, database. Justify each choice in one sentence. Prefer proven, low-complexity options.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "recommendations": response,
                "use_case": use_case
            },
            artifacts={"tech_stack": response}
        )

    async def _provide_guidance(self, task: Dict[str, Any]) -> TaskResult:
        """Provide technical guidance on a topic."""
        question = task.get("question", "")
        context = task.get("context", "")

        prompt = f"""Technical guidance needed.

Question: {question}
Context: {context}

Explain the concept, best practices, and common pitfalls. Be specific and concise.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={"guidance": response}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general CTO tasks."""
        description = task.get("description", "")

        response = await self.generate_response_async(
            f"CTO task: {description}"
        )

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  CTO-SPECIFIC METHODS
    # ============================================================

    def create_technical_spec(self, feature_description: str) -> str:
        """Create a technical specification for a feature."""
        prompt = f"""Write a technical specification for this feature:

{feature_description}

Cover: requirements, implementation approach, API design (if any), data models, and testing strategy.
"""
        return self.generate_response(prompt)

    def estimate_complexity(self, task_description: str) -> Dict[str, Any]:
        """Estimate the complexity of a technical task."""
        prompt = f"""Estimate complexity of this task:

{task_description}

State: Complexity level (Low/Medium/High/Very High), key factors, risks, and dependencies.
"""
        response = self.generate_response(prompt)

        # Parse complexity
        complexity = "medium"
        if "LOW" in response.upper():
            complexity = "low"
        elif "VERY HIGH" in response.upper():
            complexity = "very_high"
        elif "HIGH" in response.upper():
            complexity = "high"

        return {
            "complexity": complexity,
            "analysis": response
        }

    def review_architecture_decision(self, decision: str, context: str) -> str:
        """Review an architectural decision."""
        prompt = f"""Review this architectural decision.

Decision: {decision}
Context: {context}

Is this sound? State implications, better alternatives if any, and key risks to watch.
"""
        return self.generate_response(prompt, use_first_principles=True)
