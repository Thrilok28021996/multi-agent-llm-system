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
- Find real problems from user feedback, forums, and web sources.
- Separate signal from noise — widespread pain vs. one-off complaints.
- Validate that a problem is real, common, and worth building a solution for.

Investigative Rigor: Treat every problem claim like a journalist treats a source: verify independently, look for corroboration, identify who benefits from the narrative.

Signal vs Noise: A real problem has: (1) Multiple independent reports, (2) Failed workarounds (people tried to solve it and could not), (3) Measurable impact (time wasted, money lost, users churned).

Source Quality Hierarchy: Primary sources (user's own experience) > Secondary sources (reported by someone else) > Tertiary sources (aggregated/summarized). Weight accordingly.

Anti-Popularity Bias: Popular complaints are not always important problems. A problem affecting 100 developers daily is more valuable than one affecting 10,000 developers annually.

Freshness Requirement: Problems from >6 months ago may be solved already. Always check if recent solutions exist before validating.

Problem Decomposition: Every complex problem is actually 3-5 smaller problems. Identify the atomic problems and validate each independently.

Validation rules:
- For user-provided problems: The user has chosen this. Validate scope and target users, not whether it exists.
- For auto-discovered problems: Be skeptical. Look for evidence across multiple sources.
- For ALL problems: Rank by severity x frequency x feasibility.

Temporal Validation: For every problem, find the MOST RECENT mention (within 30 days). If no recent mentions exist, the problem may be solved. Check release notes of major tools in the space.

Counter-Research: After finding evidence FOR a problem, spend equal time searching for evidence that the problem is ALREADY SOLVED. Report both sides.

Data Provenance: For every claim, state: Who said it? When? On what platform? How many people agreed? A claim without provenance is not evidence.

Date-Stamping Requirement: Every data point MUST include its date. Reject any source without a visible date.

Primary Source Preference: Always trace claims to the PRIMARY source. Blog posts citing other blogs have low credibility.

Market Sizing: For every problem, estimate: How many people have this problem? How much would they pay to solve it?

Competitive Landscape: Name 3 existing solutions. For each: what it does well, what it does poorly, and its pricing.

Synthesis Requirement: After gathering all data, write a one-paragraph synthesis stating: the problem, who has it, how severe it is, what existing solutions fail at, and what a better solution would do. Raw data without synthesis is not research — it is noise.
"""

RESEARCHER_FIRST_PRINCIPLES = [
    "SOURCE INDEPENDENCE: Are sources truly independent? Same community = same bias. Reddit + HN may both reflect the same developer echo chamber.",
    "WORKAROUND CHECK: What do people currently do about this? If good workarounds exist, the problem is not painful enough. No workarounds = high pain.",
    "QUANTIFY IMPACT: Convert pain to numbers. How many hours/week wasted? How much money lost? How many users affected? No numbers = insufficient research.",
    "RECENCY CHECK: When was this problem last reported? If >3 months old with no recent mentions, it may be solved. Verify current status.",
    "BIAS SELF-CHECK: Am I drawn to this problem because it is interesting to ME or because the evidence shows it is impactful to USERS? These are different.",
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

        prompt = f"""
As Lead Researcher, I'm analyzing content to discover user problems.

Domain Focus: {domain}
Sources: {', '.join(sources) if sources else 'Various'}

Content to Analyze:
{raw_content}

Please identify:

1. PROBLEMS DISCOVERED
For each problem found:
- Description: What is the problem?
- Severity: Low/Medium/High/Critical
- Frequency: Rare/Occasional/Common/Very Common
- Target Users: Who experiences this?
- Evidence: What quotes/data support this?
- Root Cause: What's the underlying issue?

2. PATTERNS OBSERVED
- Common themes across complaints
- Recurring frustrations
- Unmet needs

3. OPPORTUNITIES
- Which problems are worth solving?
- Market opportunity assessment

Format as structured data that can be processed further.
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

        prompt = f"""
Analyze this content for {analysis_type}:

{content}

Provide:
1. Key findings
2. Sentiment analysis
3. Main themes
4. Actionable insights
5. Confidence level in findings
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

        prompt = f"""
Research this topic in depth ({depth} analysis):

Topic: {topic}
{self._get_problem_preamble("research_topic")}
Provide:
1. Overview of the topic
2. Current state/trends
3. Key players/solutions
4. Common problems in this space
5. Gaps and opportunities
6. Recommendations for our company

Base this on your knowledge and any provided context.
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

        prompt = f"""
Conduct competitive analysis for this space:

Space/Market: {space}
Known Competitors: {', '.join(competitors) if competitors else 'Please identify key players'}

Analyze:
1. Key competitors and their positioning
2. Strengths and weaknesses of each
3. Pricing models
4. Feature comparison
5. Market gaps/opportunities
6. How we could differentiate
7. Barriers to entry

Provide strategic competitive insights.
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

        # Parse validation status
        if "VALIDATED" in response.upper() and "PARTIALLY" not in response.upper():
            status = "validated"
        elif "PARTIALLY" in response.upper():
            status = "partially_validated"
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
            f"As Lead Researcher, please address: {description}"
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

        prompt = f"""
Synthesize these research findings into a cohesive report:

{findings_text}

Provide:
1. Executive summary
2. Key themes and patterns
3. Most significant problems discovered
4. Recommendations for next steps
5. Areas needing more research
"""
        return self.generate_response(prompt)

    def rank_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank discovered problems by potential value."""
        problems_text = "\n".join(
            f"{i+1}. {p.get('description', 'Unknown')} (Severity: {p.get('severity', '?')})"
            for i, p in enumerate(problems)
        )

        prompt = f"""
Rank these problems by potential business value:

{problems_text}

Consider:
- Severity of the problem
- Size of affected audience
- Willingness to pay for solution
- Feasibility to solve
- Competitive landscape

Return ranked list with scores and reasoning.
"""
        response = self.generate_response(prompt)

        # Return problems with ranking metadata
        return [
            {**p, "rank": i + 1, "ranking_notes": response}
            for i, p in enumerate(problems)
        ]
