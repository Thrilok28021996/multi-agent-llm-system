"""Product Manager Agent - Product strategy and prioritization."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from tools import UnifiedTools


PM_SYSTEM_PROMPT = """You are the Product Manager. You translate problems into clear, buildable requirements.

Your job:
- Take a problem description and produce a concrete list of requirements the Developer can implement.
- Define what "done" looks like — clear acceptance criteria.
- Scope to MVP. Cut anything that is not essential to solving the core problem.
- Write requirements as specific, testable statements, not vague wishes.

Jobs-to-be-Done: What job is the user hiring this solution to do? State it as: When [situation], I want to [motivation], so I can [expected outcome].

User Persona: Define the primary user: Who are they? What is their technical level? What tools do they currently use? What frustrates them?

Scope Ruthlessness: If a feature does not directly serve the primary user's core job, cut it. Every feature has a maintenance cost. MVP means Minimum VIABLE — not minimum features.

Acceptance Criteria Rigor: Every requirement must have: (1) a specific input, (2) an expected output, (3) an exact command to verify. If you cannot write the test, you do not understand the requirement.

Competitive Awareness: How do existing solutions handle this? What is our specific advantage? If we cannot articulate the advantage in one sentence, we have not found it.

Good requirement: "The CLI tool accepts a filename argument and outputs word count to stdout."
Bad requirement: "The tool should be user-friendly and handle various inputs gracefully."

Keep requirements to 5-10 items for MVP. Each one should be something a developer can build and QA can verify.

Failure Definition: Define what "failure" looks like for this product. What would make a user abandon it after first use? Design requirements to prevent those failure modes.

Measurable Success: Every requirement must have a measurable success condition. "Works well" is not measurable. "Processes 100 items in under 5 seconds" is measurable.

Rescoping Protocol: When asked to rescope, cut features — never cut quality. A tool that does 1 thing perfectly beats a tool that does 3 things poorly.

Stakeholder Communication: When requirements change, explain the change in plain English: what was cut, why it was cut, and what impact users will notice. The team needs to understand the WHY before they can build the WHAT.
"""

PM_FIRST_PRINCIPLES = [
    "USER STORY: Write ONE user story that captures the core need. If it requires 'and' to describe, you are combining features — split them.",
    "EXISTING SOLUTIONS: Name 2-3 existing ways users solve this today. What is wrong with each? Our solution must fix those specific shortcomings.",
    "SCOPE KNIFE: List all features. For each, answer: 'Would the user pay for JUST this feature alone?' If no, it is not MVP.",
    "TESTABLE DONE: For each requirement, write the exact terminal command and expected output. Untestable = rewrite it.",
    "ANTI-BLOAT: Count requirements. If >7 for MVP, rank by user impact and cut the bottom 30%.",
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

        prompt = f"""
As Product Manager, I need to analyze this problem:

Problem Description:
{problem.get('description', 'No description')}

Source: {problem.get('source', 'Unknown')}
Severity: {problem.get('severity', 'Unknown')}

Research Data:
{research_data}
{self._get_problem_preamble("analyze_problem")}
Please analyze:
1. What is the core problem? (Not symptoms, but root cause)
2. Who experiences this problem? (Target users)
3. How severe is this problem? (Pain level)
4. How frequently does it occur?
5. What are users currently doing to solve it?
6. Is this a problem worth solving? Why/why not?
7. What would a good solution look like?

Provide a comprehensive problem analysis.
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

{self._get_principles_checklist()}
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

        prompt = f"""
As Product Manager, I need to prioritize these features:

Features:
{features_text}

Prioritization Criteria: {', '.join(criteria)}

Please:
1. Score each feature on each criterion (1-5)
2. Calculate overall priority
3. Rank features from highest to lowest priority
4. Provide rationale for top 3 priorities
5. Identify any quick wins
6. Identify any dependencies between features

Use a framework like RICE, MoSCoW, or similar.
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

        prompt = f"""
As Product Manager, I need to create user stories for this feature:

Feature: {feature.get('name', 'Unknown')}
Description: {feature.get('description', '')}

User Personas: {', '.join(personas) if personas else 'General users'}

Please create user stories in this format:
"As a [user type], I want [goal] so that [benefit]"

For each story, include:
1. User story statement
2. Acceptance criteria (Given/When/Then)
3. Priority (High/Medium/Low)
4. Estimated complexity (S/M/L/XL)

Create 3-7 user stories that cover the feature.
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
6. Overall verdict: PASS / PASS_WITH_NOTES / FAIL

Provide detailed validation feedback.
"""

        response = await self.generate_response_async(prompt)

        # Parse verdict
        if "PASS_WITH_NOTES" in response.upper():
            verdict = "pass_with_notes"
        elif "PASS" in response.upper():
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
            f"As Product Manager, please address: {description}"
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
        prompt = f"""
Quick opportunity evaluation:

{problem_description}

Score 1-10 on:
1. Problem severity (how much pain?)
2. Market size (how many people affected?)
3. Solution clarity (do we know how to solve it?)
4. Competitive landscape (are others solving it?)

Overall recommendation: PURSUE / CONSIDER / PASS
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
        prompt = f"""
Define the MVP scope from these requirements:

{full_requirements}

Identify:
1. The absolute minimum features for a usable product
2. What can be cut or simplified
3. What's the fastest path to user value

Be ruthless about cutting scope. What's the smallest thing we can ship?
"""
        return self.generate_response(prompt, use_first_principles=True)

    def write_release_notes(self, changes: List[str]) -> str:
        """Write user-facing release notes."""
        changes_text = "\n".join(f"- {c}" for c in changes)

        prompt = f"""
Write user-facing release notes for these changes:

{changes_text}

Make it:
- Clear and non-technical
- Focused on user benefits
- Well-organized by category
- Exciting where appropriate
"""
        return self.generate_response(prompt, use_first_principles=False)
