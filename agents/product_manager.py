"""Product Manager Agent - Product strategy and prioritization."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from tools import UnifiedTools


PM_SYSTEM_PROMPT = """You are the Product Manager. You translate problems into clear, buildable requirements.

Your job:
- Produce 5-7 requirements the Developer can implement and QA can verify. Not fewer, not more.
- Every requirement must be testable: specific input → expected output → exact command to verify.
- Scope ruthlessly to MVP. If a feature does not directly serve the core user job, cut it.

Required output format:
JOB-TO-BE-DONE: When [situation], I want [motivation], so I can [outcome].
USER: [who they are, technical level, what frustrates them]

REQUIREMENT [N]: [one specific, testable sentence]
GIVEN: [precondition]  WHEN: [action]  THEN: [observable outcome]
VERIFY: [exact command and expected output]
PRIORITY: P0 | P1 | P2

OUT OF SCOPE: [features explicitly not being built]
FAILURE MODE: [what would make a user abandon this on first use]

Requirement quality bar:
- GOOD: "CLI accepts a filename argument and prints word count to stdout."
- BAD: "The tool should be user-friendly and handle various inputs gracefully." (untestable)

When rescoping: cut features, never cut quality. One thing done perfectly beats three things done poorly.
"""

PM_FIRST_PRINCIPLES = [
    "USER STORY CHECK: Does the requirement describe one specific thing? If it needs 'and' to describe, split it.",
    "TESTABILITY GATE: Can you write the exact terminal command and expected output right now? If not, rewrite the requirement until you can.",
    "SCOPE KNIFE: For each feature, ask: does this directly serve the core job? If no, cut it — every feature has a maintenance cost.",
    "ANTI-BLOAT: If you have >7 requirements, rank by user impact and cut the lowest until you reach 7.",
    "FAILURE MODE: What would make the user abandon this after first use? Ensure every P0 requirement prevents that.",
]


class ProductManagerAgent(BaseAgent, AgentToolsMixin):
    """
    Product Manager Agent - Defines product strategy and requirements.

    Enhanced with all 13 Claude Code tools and problem statement refinement.
    """

    def __init__(
        self,
        model: str = "ministral-8b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="ProductManager",
            role=AgentRole.PRODUCT_MANAGER,
            model=model,
            first_principles=PM_FIRST_PRINCIPLES,
            system_prompt=PM_SYSTEM_PROMPT,
            temperature=0.6,
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
            "requirements_definition",
            "feature_prioritization",
            "user_story_creation",
            "problem_analysis",
            "solution_validation",
            "roadmap_planning"
        ]

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a Product Manager task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "analyze_problem":
                result = await self._analyze_problem(task)
            elif task_type == "define_requirements":
                result = await self._define_requirements(task)
            elif task_type == "prioritize_features":
                result = await self._prioritize_features(task)
            elif task_type == "create_user_stories":
                result = await self._create_user_stories(task)
            elif task_type == "validate_solution":
                result = await self._validate_solution(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    async def _analyze_problem(self, task: Dict[str, Any]) -> TaskResult:
        """Analyze a user problem in depth."""
        problem = task.get("problem", {})
        research_data = task.get("research_data", "")

        prompt = f"""Problem analysis.

Problem: {problem.get('description', 'No description')}
Source: {problem.get('source', 'Unknown')} | Severity: {problem.get('severity', 'Unknown')}
Research: {research_data}

State: root cause, target users, pain level, frequency, current workarounds, and whether it is worth solving. 5-7 sentences.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "analysis": response,
                "problem": problem
            },
            artifacts={"problem_analysis": response}
        )

    async def _define_requirements(self, task: Dict[str, Any]) -> TaskResult:
        """Define product requirements for a solution, producing a structured RequirementsDoc."""
        problem = task.get("problem", {})
        target_users = task.get("target_users", "")
        constraints = task.get("constraints", [])
        problem_desc = problem.get("description", "No description")

        prompt = f"""
As Product Manager, define MVP requirements for this problem.

Problem: {problem_desc}
Target Users: {target_users}
Constraints: {', '.join(constraints) if constraints else 'None specified'}

Use this EXACT format (required for parsing):

JOB-TO-BE-DONE: When [situation], I want [motivation], so I can [outcome]

REQUIREMENT [1]: [specific, testable requirement - one sentence]
GIVEN: [precondition / system state]
WHEN: [action the user takes]
THEN: [observable outcome that proves it works]
MEASURABLE_CRITERIA: [exact terminal command + expected output]
PRIORITY: P0

REQUIREMENT [2]: [specific, testable requirement - one sentence]
GIVEN: [precondition / system state]
WHEN: [action the user takes]
THEN: [observable outcome that proves it works]
MEASURABLE_CRITERIA: [exact terminal command + expected output]
PRIORITY: P0|P1|P2

(Up to 7 requirements. P0=must have, P1=should have, P2=nice to have.)

OUT OF SCOPE:
- [feature NOT being built in this MVP]
- [feature NOT being built in this MVP]

GOOD example:
REQUIREMENT [1]: CLI accepts a filename argument and prints word count to stdout.
GIVEN: a text file exists at the given path
WHEN: user runs `python main.py myfile.txt`
THEN: stdout shows "Words: 42" and exits with code 0
MEASURABLE_CRITERIA: python main.py sample.txt | grep "Words:"
PRIORITY: P0
"""

        response = await self.generate_response_async(prompt)

        # Parse into structured RequirementsDoc
        from orchestrator.artifact_parser import RequirementsParser
        parser = RequirementsParser()
        requirements_doc = parser.parse(response, problem_summary=problem_desc[:200])

        return TaskResult(
            success=True,
            output={
                "requirements": response,
                "requirements_doc": requirements_doc.to_dict(),
                "problem": problem,
            },
            artifacts={
                "prd": response,
                "requirements_doc": requirements_doc.to_dict(),
            },
        )

    async def _prioritize_features(self, task: Dict[str, Any]) -> TaskResult:
        """Prioritize a list of features."""
        features = task.get("features", [])
        criteria = task.get("criteria", ["impact", "effort", "risk"])

        features_text = "\n".join(
            f"{i+1}. {f.get('name', 'Unknown')}: {f.get('description', '')}"
            for i, f in enumerate(features)
        )

        prompt = f"""Prioritize these features using {', '.join(criteria)}.

{features_text}

Score each on each criterion (1-5), rank highest to lowest, note quick wins and dependencies.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "prioritization": response,
                "features": features
            },
            artifacts={"priority_matrix": response}
        )

    async def _create_user_stories(self, task: Dict[str, Any]) -> TaskResult:
        """Create user stories for a feature."""
        feature = task.get("feature", {})
        personas = task.get("personas", [])

        prompt = f"""Write 3-7 user stories for this feature.

Feature: {feature.get('name', 'Unknown')} — {feature.get('description', '')}
Personas: {', '.join(personas) if personas else 'General users'}

Format each as:
"As a [user type], I want [goal] so that [benefit]"
Acceptance: Given/When/Then
Priority: High|Medium|Low | Complexity: S|M|L|XL
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "user_stories": response,
                "feature": feature
            },
            artifacts={"user_stories": response}
        )

    async def _validate_solution(self, task: Dict[str, Any]) -> TaskResult:
        """Validate that a solution meets requirements."""
        solution = task.get("solution", {})
        requirements = task.get("requirements", "")
        original_problem = task.get("problem", {})

        prompt = f"""
As Product Manager, I need to validate this solution:

Original Problem:
{original_problem.get('description', 'No description')}

Requirements:
{requirements}

Solution Delivered:
{solution.get('description', 'No description')}

Implementation Details:
{solution.get('implementation', '')}

Please validate:
1. Does the solution address the original problem?
2. Does it meet the must-have requirements?
3. What requirements are met/not met?
4. Would users be satisfied with this solution?
5. What's missing or could be improved?
6. Overall verdict: PASS / PASS_WITH_ISSUES / FAIL

Provide concise validation feedback.
"""

        response = await self.generate_response_async(prompt)

        # Parse verdict — use word-boundary regex to avoid "NOT PASS" false positive
        import re
        verdict_region = response[-200:].upper()
        if re.search(r'\bPASS_WITH_ISSUES\b', verdict_region):
            verdict = "pass_with_issues"
        elif re.search(r'\bPASS\b', verdict_region) and not re.search(r'\bFAIL\b', verdict_region):
            verdict = "pass"
        else:
            verdict = "fail"

        return TaskResult(
            success=True,
            output={
                "verdict": verdict,
                "validation": response
            },
            artifacts={"validation_report": response}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general PM tasks."""
        description = task.get("description", "")

        response = await self.generate_response_async(
            f"PM task: {description}"
        )

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  PM-SPECIFIC METHODS
    # ============================================================

    def evaluate_opportunity(self, problem_description: str) -> Dict[str, Any]:
        """Quick evaluation of a problem opportunity."""
        prompt = f"""Evaluate this opportunity. Score 1-10 on severity, market size, solution clarity, and competition.

{problem_description}

End with: PURSUE / CONSIDER / PASS
"""
        response = self.generate_response(prompt)

        if "PURSUE" in response.upper():
            recommendation = "pursue"
        elif "CONSIDER" in response.upper():
            recommendation = "consider"
        else:
            recommendation = "pass"

        return {
            "recommendation": recommendation,
            "evaluation": response
        }

    def create_mvp_scope(self, full_requirements: str) -> str:
        """Define MVP scope from full requirements."""
        prompt = f"""Define MVP scope. Cut everything that isn't essential for one user to get value.

{full_requirements}

List: what stays (minimum viable), what gets cut, and the fastest path to ship.
"""
        return self.generate_response(prompt, use_first_principles=True)

    def write_release_notes(self, changes: List[str]) -> str:
        """Write user-facing release notes."""
        changes_text = "\n".join(f"- {c}" for c in changes)

        prompt = f"""Write release notes for users. Non-technical, benefit-focused, organized by category.

{changes_text}
"""
        return self.generate_response(prompt, use_first_principles=False)
