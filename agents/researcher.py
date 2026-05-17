"""Researcher Agent - Problem discovery and market analysis."""

from typing import Any, Dict, List, Optional

import aiohttp

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from tools import UnifiedTools
from ui.console import console


RESEARCHER_SYSTEM_PROMPT = """You are the Lead Researcher. You discover and validate problems worth solving.

Your job:
- Find real problems with multiple independent reports, failed workarounds, and measurable impact.
- Separate signal from noise: a problem affecting 100 users daily beats one affecting 10,000 users annually.
- For user-provided problems: validate scope and target users only — do not question whether the problem exists.
- For auto-discovered problems: require evidence from at least 2 independent platforms.

Signal criteria (a real problem has ALL three):
1. Multiple independent reports (different platforms, different users)
2. Failed workarounds — people tried to solve it and could not
3. Measurable impact — time wasted, money lost, or users churned

Source quality: Primary (user's own experience) > Secondary (reported by others) > Tertiary (aggregated).
For every claim: state who said it, when, on what platform, and how many agreed. No date = reject the source.
Trace every claim to its primary source — blog posts citing other blogs have low credibility.

Freshness: Discard problems with no reports in the last 60 days unless you verify the problem is still unsolved.
After finding evidence FOR a problem, check for evidence it is already solved. Report both sides.

Existing solutions: Name 2-3 existing solutions, what each does well, and where each fails. Our solution must address the specific failures.

Required output — end every research response with this synthesis paragraph:
SYNTHESIS: [problem] affects [who] with severity [low/medium/high]. Existing solutions fail at [specific gaps]. A better solution would [specific improvement]. Confidence: [0.0-1.0].
"""

RESEARCHER_FIRST_PRINCIPLES = [
    "SOURCE INDEPENDENCE: Are sources truly independent? Same platform or community = shared bias. Reddit + HN can reflect the same echo chamber — they count as ONE source.",
    "WORKAROUND CHECK: What do users currently do about this? Strong existing workarounds = low pain. No workarounds = high pain and unmet need.",
    "QUANTIFY IMPACT: Express pain as numbers — hours/week wasted, users affected, error frequency. Qualitative-only = insufficient.",
    "FRESHNESS CHECK: Is there a report from within the last 60 days? If not, verify the problem is still unsolved before proceeding.",
    "BIAS CHECK: Am I validating this problem because the evidence supports it, or because it is interesting to me? If the evidence is thin, say so explicitly.",
]


class ResearcherAgent(BaseAgent, AgentToolsMixin):
    """
    Researcher Agent - Discovers problems and conducts market research.

    Enhanced with all 13 Claude Code tools and problem statement refinement.
    Uses enhanced tools for web search and content fetching.
    """

    def __init__(
        self,
        model: str = "ministral-8b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            model=model,
            first_principles=RESEARCHER_FIRST_PRINCIPLES,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            temperature=0.6,
            max_tokens=4096
        )
        super().__init__(config, workspace_root, memory_persist_dir)

        # Initialize unified tools for web research
        self.tools = UnifiedTools(
            workspace_root=workspace_root,
            persist_dir=memory_persist_dir
        )

        # Initialize problem statement refiner
        self.problem_refiner = ProblemStatementRefiner()

        # Enable ReAct tool use loop
        self.enable_react_tools()

        # Research state
        self.discovered_problems: List[Dict[str, Any]] = []

    def get_capabilities(self) -> List[str]:
        return [
            "problem_discovery",
            "web_research",
            "pain_point_extraction",
            "market_analysis",
            "competitive_research",
            "trend_analysis"
        ]

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a Researcher task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "discover_problems":
                result = await self._discover_problems(task)
            elif task_type == "analyze_content":
                result = await self._analyze_content(task)
            elif task_type == "research_topic":
                result = await self._research_topic(task)
            elif task_type == "competitive_analysis":
                result = await self._competitive_analysis(task)
            elif task_type == "validate_problem":
                result = await self._validate_problem(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    async def _discover_problems(self, task: Dict[str, Any]) -> TaskResult:
        """Discover problems from web content."""
        sources = task.get("sources", [])
        domain = task.get("domain", "general")
        raw_content = task.get("content", "")

        prompt = f"""Identify user problems in this content.

Domain: {domain}
Sources: {', '.join(sources) if sources else 'Various'}

Content:
{raw_content}

For each problem found, output:
PROBLEM: [description]
SEVERITY: Low|Medium|High|Critical
FREQUENCY: Rare|Occasional|Common|Very Common
TARGET_USERS: [who experiences this]
EVIDENCE: [supporting quote or data]
"""

        response = await self.generate_response_async(prompt)

        # Extract problems (simplified - in production would use structured extraction)
        problems = self._extract_problems_from_response(response)

        # Store discovered problems
        self.discovered_problems.extend(problems)

        return TaskResult(
            success=True,
            output={
                "problems": problems,
                "analysis": response,
                "domain": domain
            },
            artifacts={"research_report": response}
        )

    async def _analyze_content(self, task: Dict[str, Any]) -> TaskResult:
        """Analyze specific content for pain points."""
        content = task.get("content", "")
        analysis_type = task.get("analysis_type", "pain_points")

        prompt = f"""Analyze for {analysis_type}. State key findings, main themes, and confidence level.

{content}
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={"analysis": response}
        )

    async def _research_topic(self, task: Dict[str, Any]) -> TaskResult:
        """Research a specific topic."""
        topic = task.get("topic", "")
        depth = task.get("depth", "standard")  # quick, standard, deep

        prompt = f"""Research: {topic} ({depth} analysis)

Cover: current state, key solutions, common problems, and unsolved gaps.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "topic": topic,
                "research": response
            },
            artifacts={"topic_research": response}
        )

    async def _competitive_analysis(self, task: Dict[str, Any]) -> TaskResult:
        """Analyze competitors in a space."""
        space = task.get("space", "")
        competitors = task.get("competitors", [])

        prompt = f"""Competitive analysis: {space}
Competitors: {', '.join(competitors) if competitors else 'identify key players'}

For each: positioning, strengths, weaknesses. Then: market gaps and our differentiation opportunity.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "space": space,
                "analysis": response
            },
            artifacts={"competitive_analysis": response}
        )

    async def _validate_problem(self, task: Dict[str, Any]) -> TaskResult:
        """Validate if a problem is worth solving."""
        problem = task.get("problem", {})
        evidence = task.get("evidence", "")

        # Check if this is a user-provided problem (high score = user input)
        is_user_provided = problem.get('score', 0) >= 100 or 'user_input' in str(problem.get('sources', []))

        if is_user_provided:
            prompt = f"""The user wants us to solve this problem:

Problem: {problem.get('description', '')}
Evidence: {evidence}

Since the user requested this, we will build it. But first, scope it:
1. Who is the primary user? (1 sentence)
2. What is the core deliverable? (1 sentence)
3. What is the biggest technical risk? (1 sentence)
4. What should we cut for MVP? (1 sentence)
{self._get_principles_checklist()}
Verdict: VALIDATED"""
        else:
            prompt = f"""Validate this discovered problem:

Problem: {problem.get('description', '')}
Evidence: {evidence}

Evaluate:
1. Is this a real, recurring problem?
2. How widespread is it?
3. Would people actively want a solution?

Verdict: VALIDATED / PARTIALLY_VALIDATED / NOT_VALIDATED
{self._get_principles_checklist()}
Be concise (3-5 sentences max)."""

        response = await self.generate_response_async(prompt)

        # Parse validation status — check NOT_VALIDATED before VALIDATED to avoid substring match
        import re as _re
        verdict_region = response[-300:].upper()
        if _re.search(r'\bNOT[_ ]?VALIDATED\b', verdict_region):
            status = "not_validated"
        elif _re.search(r'\bPARTIALLY', verdict_region):
            status = "partially_validated"
        elif _re.search(r'\bVALIDATED\b', verdict_region):
            status = "validated"
        else:
            status = "not_validated"

        return TaskResult(
            success=True,
            output={
                "status": status,
                "validation": response,
                "problem": problem
            },
            artifacts={"validation_report": response}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general research tasks."""
        description = task.get("description", "")

        response = await self.generate_response_async(
            f"Research task: {description}"
        )

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  RESEARCH METHODS
    # ============================================================

    def _extract_problems_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract structured problems from LLM response."""
        problems = []

        # Simple extraction - in production would use structured output
        lines = response.split("\n")
        current_problem = {}

        for line in lines:
            line = line.strip()
            if line.startswith("- Description:"):
                if current_problem:
                    problems.append(current_problem)
                current_problem = {
                    "description": line.replace("- Description:", "").strip()
                }
            elif line.startswith("- Severity:"):
                current_problem["severity"] = line.replace("- Severity:", "").strip()
            elif line.startswith("- Frequency:"):
                current_problem["frequency"] = line.replace("- Frequency:", "").strip()
            elif line.startswith("- Target Users:"):
                current_problem["target_users"] = line.replace("- Target Users:", "").strip()

        if current_problem:
            problems.append(current_problem)

        return problems

    async def fetch_web_content(self, url: str) -> Dict[str, Any]:
        """Fetch content from a URL."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            "success": True,
                            "content": content,
                            "url": url
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}",
                            "url": url
                        }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    async def fetch_reddit_posts(self, subreddit: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch posts from a Reddit subreddit (JSON API)."""
        url = f"https://www.reddit.com/r/{subreddit}/top/.json?t=week&limit={limit}"

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {"User-Agent": "CompanyAGI/1.0"}
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = []
                        for child in data.get("data", {}).get("children", []):
                            post = child.get("data", {})
                            posts.append({
                                "title": post.get("title", ""),
                                "text": post.get("selftext", ""),
                                "score": post.get("score", 0),
                                "comments": post.get("num_comments", 0),
                                "url": post.get("url", "")
                            })
                        return posts
        except Exception as e:
            console.warning(f"Error fetching Reddit: {e}")

        return []

    def synthesize_research(self, findings: List[Dict[str, Any]]) -> str:
        """Synthesize multiple research findings into a cohesive report."""
        findings_text = "\n\n".join(
            f"Finding {i+1}:\n{f.get('summary', str(f))}"
            for i, f in enumerate(findings)
        )

        prompt = f"""Synthesize into a research report.

{findings_text}

Output: executive summary, key themes, top problems found, recommended next steps.
"""
        return self.generate_response(prompt)

    def rank_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank discovered problems by potential value."""
        problems_text = "\n".join(
            f"{i+1}. {p.get('description', 'Unknown')} (Severity: {p.get('severity', '?')})"
            for i, p in enumerate(problems)
        )

        prompt = f"""Rank by business value (severity, audience size, willingness to pay, feasibility, competition):

{problems_text}

Return ranked list with score and one-sentence reasoning per problem.
"""
        response = self.generate_response(prompt)

        # Return problems with ranking metadata
        return [
            {**p, "rank": i + 1, "ranking_notes": response}
            for i, p in enumerate(problems)
        ]
