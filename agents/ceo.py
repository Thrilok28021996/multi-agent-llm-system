"""CEO Agent - Strategic decision maker and company vision leader."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from tools import UnifiedTools


CEO_SYSTEM_PROMPT = """You are the CEO of an autonomous AI company. You make the final call on what ships.

Strategic Evaluation Framework (MTRR):
- Market fit: Does this solve a validated problem that real users have?
- Technical soundness: Is the implementation correct and well-structured?
- Resource efficiency: Was this built with appropriate effort and complexity?
- Risk assessment: What could go wrong in production?

Decision Quality Checklist:
- Does this solve a validated problem? Is the evidence from execution, not assumption?
- Would a paying user choose this over alternatives?
- If you approve everything, you are not doing your job. Expect to reject 30-40% of first submissions.
- A good rejection is more valuable than a lazy approval. Your rejection should be a roadmap to success.
- Think like you are presenting this to a board of directors. What questions would they ask?

Decision framework:
- APPROVE only when evidence shows the solution works and addresses the stated problem.
- REJECT when evidence shows the solution is broken, wrong, or misses requirements.
- REQUEST_MORE_INFO when you lack sufficient evidence to decide either way.

Do NOT reject for style, missing nice-to-haves, or theoretical issues. Only reject for demonstrated failures.
When rejecting, give the developer a numbered fix list they can act on immediately.
When requesting more info, you MUST specify exactly what evidence you need.

Investment Thinking: Every approval is an investment of compute, time, and opportunity cost. Ask: "If I had 3 problems to choose from, would I still pick this one?"

Rejection as Mentorship: When you reject, your feedback should be so clear that the Developer could fix it without asking a single question. A vague rejection is worse than a bad approval.

Escalation Trigger: After round 4+ of rejection, do NOT repeat the same feedback. Instead, identify the systemic failure: is this a design problem (requires CTO redesign), a scope problem (requires PM rescoping), or a capability problem (requires different approach)? Drive the team forward — do not be a permanent blocker.

Approval Criteria (ALL must be true):
1. The solution demonstrably runs (execution evidence, not just code review)
2. It solves the stated problem (not a related problem, not a superset)
3. A user could install and use it within 5 minutes following the README
4. No critical security issues flagged by Security Engineer

Do NOT approve based on effort or number of iterations. Approve based on evidence.

Focus on clear acceptance criteria and success metrics. Keep responses concise and actionable. This is a first-draft scaffolding task — prioritize working structure over perfection.
"""

CEO_FIRST_PRINCIPLES = [
    "EVIDENCE AUDIT: List every piece of concrete evidence (test results, execution output, files created). If evidence is thin, REQUEST_MORE_INFO — do not guess.",
    "PROBLEM-SOLUTION FIT: Restate the original problem. Does this solution address the ROOT CAUSE or just symptoms? Surface-level fixes = REJECT.",
    "USER PERSPECTIVE: Would a real user prefer this over doing nothing or using an existing tool? If not clearly better, REJECT.",
    "RISK ASSESSMENT: What could go wrong in production? If unaddressed critical risks exist, REJECT with specific mitigations needed.",
    "DECISION INTEGRITY: Am I approving because the evidence is strong, or because I am tired of iterations? Fatigue-based approval = REJECT and escalate.",
    "OPPORTUNITY COST: What else could the team build instead? If this problem is not in the top 3 by impact, challenge whether we should continue.",
    "ITERATION FATIGUE GUARD: If this is round 5+ of approval, am I lowering my bar? Check: would I approve this if it were round 1?",
]


class CEOAgent(BaseAgent, AgentToolsMixin):
    """
    CEO Agent - Makes strategic decisions and provides company leadership.

    Enhanced with all 13 Claude Code tools and problem statement refinement.
    """

    def __init__(
        self,
        model: str = "qwen3-8b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="CEO",
            role=AgentRole.CEO,
            model=model,
            first_principles=CEO_FIRST_PRINCIPLES,
            system_prompt=CEO_SYSTEM_PROMPT,
            temperature=0.7,
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
            "strategic_decision_making",
            "opportunity_evaluation",
            "resource_allocation",
            "final_approval",
            "vision_setting",
            "company_announcements"
        ]

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a CEO task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "evaluate_opportunity":
                result = await self._evaluate_opportunity(task)
            elif task_type == "approve_solution":
                result = await self._approve_solution(task)
            elif task_type == "allocate_resources":
                result = await self._allocate_resources(task)
            elif task_type == "strategic_decision":
                result = await self._make_strategic_decision(task)
            elif task_type == "company_meeting":
                result = await self._lead_meeting(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    async def _evaluate_opportunity(self, task: Dict[str, Any]) -> TaskResult:
        """Evaluate a business opportunity."""
        problem = task.get("problem", {})
        market_analysis = task.get("market_analysis", "")

        bias_flags = task.get("bias_flags", "")
        counter_ev = task.get("counter_evidence", "")
        opposing = task.get("opposing_viewpoints", "")
        freshness = task.get("freshness_score", "")
        credibility = task.get("credibility", "")

        research_quality = ""
        if any([bias_flags, counter_ev, opposing]):
            research_quality = f"""
--- RESEARCH QUALITY FLAGS (read before deciding) ---
Bias in data: {bias_flags or 'None'}
Counter-evidence (reasons this is NOT a real problem): {counter_ev or 'None'}
Opposing viewpoints: {opposing or 'None'}
Data freshness score: {freshness or 'Unknown'} (1.0=very recent, 0.0=stale)
Credibility: {credibility or 'Not scored'}
"""

        prompt = f"""Evaluate this opportunity and decide whether to pursue it.

Problem: {problem.get('description', 'No description')}
Severity: {problem.get('severity', 'Unknown')}
Target Users: {problem.get('target_users', 'Unknown')}

Market Context:
{market_analysis}

Meeting Outcome: {task.get('meeting_outcome', 'No meeting data')}
{research_quality}
Answer these questions briefly:
1. Is this a real problem worth solving? What evidence supports this? (1-2 sentences)
2. Does the counter-evidence change your assessment? How? (1 sentence)
3. Can we build a useful solution with our capabilities? (1-2 sentences)
4. What's the main risk? (1 sentence)

Decide based on evidence, not optimism. If bias is high and counter-evidence is strong, lean toward NEED_MORE_INFO.
You MUST respond with a JSON block:
```json
{{"decision": "YES" or "NO" or "NEED_MORE_INFO", "confidence": 0.0-1.0, "reasoning": "..."}}
```"""

        compute = self._get_compute_config(prompt)
        if compute["consistency_samples"] > 1:
            response = await self.generate_with_consistency_async(
                prompt, n=compute["consistency_samples"]
            )
        else:
            response = await self.generate_response_async(prompt)

        # Use structured parser for decision extraction
        from utils.output_parser import StructuredOutputParser
        parser = StructuredOutputParser()
        parsed = parser.parse_decision(response)
        decision_str = parsed.get("decision", "uncertain").lower()

        if decision_str in ("yes", "approve", "approved"):
            decision = "approved"
        elif decision_str in ("need_more_info", "request_more_info", "uncertain"):
            decision = "needs_more_info"
        else:
            decision = "rejected"

        return TaskResult(
            success=True,
            output={
                "decision": decision,
                "reasoning": response,
                "problem": problem,
                "parsed": parsed
            },
            artifacts={"evaluation_report": response},
            confidence=parsed.get("confidence", 0.5)
        )

    async def _approve_solution(self, task: Dict[str, Any]) -> TaskResult:
        """Approve or reject a proposed solution."""
        solution = task.get("solution", {})
        qa_report = task.get("qa_report", "")
        security_report = task.get("security_report", "Not run")
        problem = task.get("original_problem", {})
        execution_summary = task.get("execution_summary", "Not run")

        root_cause = task.get("root_cause", "")

        # Structured pre-thinking: template-based reasoning (no extra LLM call)
        think_conclusion = self.structured_think(
            f"CEO final approval for: {problem.get('description', 'No description')[:150]}. "
            f"QA={str(qa_report)[:80]}. Execution={execution_summary[:80]}."
        )

        prompt = f"""[PRE-ANALYSIS]
{think_conclusion}

Review this solution for final approval.

PROBLEM: {problem.get('description', 'No description')}
{f"STATED ROOT CAUSE: {root_cause}" if root_cause else ""}
SOLUTION: {solution.get('description', 'No description')}
IMPLEMENTATION: {str(solution.get('implementation', 'No details'))}
QA REPORT: {str(qa_report)}
SECURITY REVIEW: {security_report}
CODE EXECUTION: {execution_summary}

{f"CRITICAL CHECK: Does this solution address the stated root cause: {root_cause}? If not, this should be rejected." if root_cause else ""}

Answer YES or NO for each:
1. Does code exist and were files written? [YES/NO]
2. Does the code run without crashing? [YES/NO/NOT_TESTED]
3. Does the solution solve the stated problem? [YES/NO/PARTIALLY]
4. Did QA find any critical blocking bugs? [YES/NO]

Decision table:
- Q1=YES, Q2=YES, Q3=YES, Q4=NO -> APPROVE
- Q1=YES, Q2=YES, Q3=PARTIALLY, Q4=NO -> APPROVE (MVP is acceptable)
- Q1=NO or Q3=NO -> REJECT with numbered fix list
- Q2=NOT_TESTED -> REQUEST_MORE_INFO (ask for execution evidence)
- Q4=YES (critical bugs exist) -> REJECT with the bug list
- Otherwise -> REJECT with numbered fix list

Example response:
```json
{{"decision": "APPROVE", "confidence": 0.85, "reasoning": "Code exists, runs, solves the problem, no critical bugs.", "issues": []}}
```

BOARD SUMMARY: Before deciding, write a 2-sentence board summary: What problem did we solve? What's the evidence it works?

EVIDENCE CITATION: List 3 specific pieces of EVIDENCE from the test results and execution output that support your decision. If you cannot cite 3 concrete pieces of evidence, respond with REQUEST_MORE_INFO.

RISK REGISTER: List the top 3 risks of shipping this solution. For each risk, state the mitigation.

{self._get_principles_checklist()}
You MUST respond with a JSON block:
```json
{{"decision": "APPROVE" or "REJECT" or "REQUEST_MORE_INFO", "confidence": 0.0-1.0, "reasoning": "...", "issues": ["issue1"]}}
```"""

        compute = self._get_compute_config(prompt)
        if compute["consistency_samples"] > 1:
            response = await self.generate_with_consistency_async(
                prompt, n=compute["consistency_samples"]
            )
        else:
            response = await self.generate_response_async(prompt)

        # Use structured parser — no bias
        from utils.output_parser import StructuredOutputParser
        parser = StructuredOutputParser()
        parsed = parser.parse_decision(response)
        decision_str = parsed.get("decision", "uncertain").lower()

        if decision_str in ("approve", "approved", "yes"):
            decision = "approved"
        elif decision_str in ("request_more_info", "need_more_info", "uncertain"):
            decision = "needs_more_info"
        else:
            decision = "rejected"

        return TaskResult(
            success=True,
            output={
                "decision": decision,
                "reasoning": response,
                "solution": solution,
                "issues": parsed.get("issues", []),
                "parsed": parsed
            },
            artifacts={"approval_report": response},
            confidence=parsed.get("confidence", 0.5)
        )

    async def _allocate_resources(self, task: Dict[str, Any]) -> TaskResult:
        """Allocate resources (agents) to a project."""
        project = task.get("project", {})
        available_agents = task.get("available_agents", [])

        prompt = f"""
As CEO, I need to allocate resources for this project:

Project: {project.get('name', 'Unknown')}
Description: {project.get('description', 'No description')}
Priority: {project.get('priority', 'Normal')}

Available Team Members:
{', '.join(available_agents)}

Please determine:
1. Which team members should work on this?
2. Who should lead the effort?
3. What's the recommended approach?
4. Any special considerations?

Provide a clear resource allocation plan.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "allocation": response,
                "project": project
            },
            artifacts={"allocation_plan": response}
        )

    async def _make_strategic_decision(self, task: Dict[str, Any]) -> TaskResult:
        """Make a strategic decision."""
        question = task.get("question", "")
        context = task.get("context", "")
        options = task.get("options", [])

        options_text = "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(options))

        prompt = f"""
As CEO, I need to make a strategic decision:

Question: {question}

Context:
{context}

Options:
{options_text if options else "No specific options provided - please recommend the best course of action"}

Please analyze and decide:
1. What are the key factors to consider?
2. What are the pros and cons of each option?
3. What is your decision and why?
4. What should be the next steps?
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "decision": response,
                "question": question
            },
            artifacts={"decision_report": response}
        )

    async def _lead_meeting(self, task: Dict[str, Any]) -> TaskResult:
        """Lead a company meeting."""
        agenda = task.get("agenda", "")
        attendees = task.get("attendees", [])
        context = task.get("context", "")

        prompt = f"""
As CEO, I'm leading a company meeting.

Attendees: {', '.join(attendees)}
Agenda: {agenda}
Context: {context}

Please:
1. Open the meeting with context
2. Address the agenda items
3. Facilitate discussion points
4. Summarize decisions and action items
5. Close with next steps

Provide the meeting notes and outcomes.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "meeting_notes": response,
                "agenda": agenda,
                "attendees": attendees
            },
            artifacts={"meeting_minutes": response}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general CEO tasks."""
        description = task.get("description", "")

        prompt = f"""
As CEO, I need to handle this task:

{description}

Please provide your response with clear reasoning and any decisions or directions.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  CEO-SPECIFIC METHODS
    # ============================================================

    def approve_project(self, project_summary: str) -> Dict[str, Any]:
        """Quick approval check for a project."""
        prompt = f"""
Quick evaluation needed for this project:

{project_summary}

Should we proceed? Answer with:
- APPROVED: [reason]
- REJECTED: [reason]
- NEEDS_MORE_INFO: [what information is needed]
"""

        response = self.generate_response(prompt, use_first_principles=True)

        if "APPROVED" in response.upper():
            return {"status": "approved", "response": response}
        elif "REJECTED" in response.upper():
            return {"status": "rejected", "response": response}
        else:
            return {"status": "needs_info", "response": response}

    def set_priority(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Prioritize a list of items/projects."""
        items_text = "\n".join(
            f"{i+1}. {item.get('name', 'Unknown')}: {item.get('description', '')}"
            for i, item in enumerate(items)
        )

        prompt = f"""
Please prioritize these items from most to least important:

{items_text}

Consider: business value, urgency, resource requirements, strategic alignment.

Return the prioritized list with brief justification for each ranking.
"""

        response = self.generate_response(prompt)

        # Return items with priority scores
        return [
            {**item, "priority_rank": i + 1, "ceo_notes": response}
            for i, item in enumerate(items)
        ]

    def make_announcement(self, topic: str, details: str) -> str:
        """Create a company-wide announcement."""
        prompt = f"""
Create a company-wide announcement about:

Topic: {topic}
Details: {details}

The announcement should be:
- Clear and professional
- Inspiring when appropriate
- Action-oriented if needed
- Concise but complete
"""

        return self.generate_response(prompt, use_first_principles=False)
