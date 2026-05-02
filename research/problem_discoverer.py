"""Problem Discoverer - Identifies problems and pain points from web content."""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

MAX_PROBLEM_AGE_DAYS = 60  # Only consider problems from the last 60 days

from .credibility import CredibilityScorer
from .web_scraper import WebScraper
from config.domains import DomainConfig

logger = logging.getLogger(__name__)


class ProblemSeverity(Enum):
    """Severity levels for discovered problems."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProblemFrequency(Enum):
    """Frequency levels for discovered problems."""

    VERY_COMMON = "very_common"
    COMMON = "common"
    OCCASIONAL = "occasional"
    RARE = "rare"


@dataclass
class DiscoveredProblem:
    """A problem discovered through research."""

    id: str
    description: str
    severity: ProblemSeverity
    frequency: ProblemFrequency = ProblemFrequency.COMMON
    target_users: str = ""
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    domain: str = "software"
    keywords: List[str] = field(default_factory=list)
    potential_solution_ideas: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    validation_status: str = "unvalidated"
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)  # For custom data like language

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "severity": self.severity.value,
            "frequency": self.frequency.value,
            "target_users": self.target_users,
            "evidence": self.evidence,
            "sources": self.sources,
            "domain": self.domain,
            "keywords": self.keywords,
            "potential_solution_ideas": self.potential_solution_ideas,
            "discovered_at": self.discovered_at.isoformat(),
            "validation_status": self.validation_status,
            "score": self.score,
            "metadata": self.metadata,
        }


class ProblemDiscoverer:
    """
    Discovers problems by analyzing content from various sources.
    Uses LLM to extract and analyze pain points.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        domain_config: DomainConfig = None
    ):
        self.model = model
        self.domain_config = domain_config or DomainConfig()
        self.scraper = WebScraper()

        # Storage
        self.discovered_problems: List[DiscoveredProblem] = []
        self.raw_content_cache: List[Dict[str, Any]] = []

        # Counter for IDs
        self._problem_counter = 0

    def _generate_problem_id(self) -> str:
        """Generate unique problem ID."""
        self._problem_counter += 1
        return f"PROB-{datetime.now().strftime('%Y%m%d')}-{self._problem_counter:04d}"

    async def discover_from_reddit(
        self,
        subreddits: List[str],
        limit_per_sub: int = 25,
        time_period: str = "month"
    ) -> List[DiscoveredProblem]:
        """
        Discover problems from Reddit subreddits.

        Args:
            subreddits: List of subreddit names
            limit_per_sub: Posts to fetch per subreddit
        """
        all_problems = []

        for subreddit in subreddits:
            # Fetch posts
            posts = await self.scraper.fetch_reddit_subreddit(
                subreddit,
                sort="top",
                time_period=time_period,
                limit=limit_per_sub
            )

            if not posts:
                continue

            # Combine post content
            content = self._format_reddit_posts(posts, subreddit)

            # Analyze for problems
            problems = await self._analyze_content_for_problems(
                content,
                source=f"reddit/r/{subreddit}",
                domain=self._infer_domain(subreddit)
            )

            all_problems.extend(problems)

        self.discovered_problems.extend(all_problems)
        return all_problems

    async def discover_from_hacker_news(
        self,
        limit: int = 30
    ) -> List[DiscoveredProblem]:
        """Discover problems from Hacker News stories and comments."""
        stories = await self.scraper.fetch_hacker_news_top(limit)

        if not stories:
            return []

        # Format content
        content = self._format_hn_stories(stories)

        # Analyze
        problems = await self._analyze_content_for_problems(
            content,
            source="hackernews",
            domain="tech_dev"
        )

        self.discovered_problems.extend(problems)
        return problems

    async def discover_from_search(
        self,
        queries: List[str],
        subreddit: str = None
    ) -> List[DiscoveredProblem]:
        """
        Discover problems by searching for specific pain point queries.

        Args:
            queries: Search queries (e.g., ["frustrated with", "hate when"])
            subreddit: Optional subreddit to limit search
        """
        all_problems = []

        for query in queries:
            results = await self.scraper.search_reddit(
                query,
                subreddit=subreddit,
                sort="relevance",
                limit=25
            )

            if not results:
                continue

            content = self._format_reddit_posts(results, subreddit or "all")

            problems = await self._analyze_content_for_problems(
                content,
                source=f"reddit_search:{query}",
                domain=self._infer_domain(subreddit) if subreddit else "general"
            )

            all_problems.extend(problems)

        self.discovered_problems.extend(all_problems)
        return all_problems

    async def discover_from_url(
        self,
        url: str,
        domain: str = "general"
    ) -> List[DiscoveredProblem]:
        """Discover problems from a specific URL."""
        content = await self.scraper.fetch_url(url)

        if not content.success:
            return []

        problems = await self._analyze_content_for_problems(
            f"Title: {content.title}\n\n{content.content}",
            source=url,
            domain=domain
        )

        self.discovered_problems.extend(problems)
        return problems

    async def discover_from_stackoverflow(
        self,
        tags: List[str] = None,
        limit: int = 20
    ) -> List[DiscoveredProblem]:
        """Discover problems from Stack Overflow top questions.

        Fetches top questions from Stack Overflow (filtered by tags if
        provided) and analyzes them for recurring problems and pain points.

        Args:
            tags: Optional list of SO tags to filter by (e.g. ['python', 'api']).
                  If None, uses general programming tags.
            limit: Maximum number of questions to analyze.

        Returns:
            List of discovered problems with source='stack_overflow'.
        """
        default_tags = tags or ["python", "javascript", "api", "web"]
        tag_query = ";".join(default_tags)

        # Stack Overflow API endpoint for recent questions by tag (last 30 days)
        since_epoch = int(time.time() - 30 * 86400)
        url = (
            f"https://api.stackexchange.com/2.3/questions"
            f"?order=desc&sort=votes&tagged={tag_query}"
            f"&site=stackoverflow&pagesize={limit}"
            f"&filter=withbody"
            f"&fromdate={since_epoch}"
        )

        content_result = await self.scraper.fetch_url(url)

        if not content_result.success:
            return []

        # Format the SO content for analysis
        formatted_lines = [
            f"Stack Overflow - Top Questions (tags: {', '.join(default_tags)})\n"
        ]

        try:
            data = json.loads(content_result.content)
            items = data.get("items", [])

            for item in items[:limit]:
                title = item.get("title", "")
                score = item.get("score", 0)
                answer_count = item.get("answer_count", 0)
                body = item.get("body", "")
                formatted_lines.append(
                    f"[Score: {score}, Answers: {answer_count}] {title}"
                )
                if body:
                    formatted_lines.append(f"  {body}")
                formatted_lines.append("")
        except (json.JSONDecodeError, KeyError):
            logger.debug("JSON parsing failed for Stack Overflow, using raw content")
            formatted_lines.append(content_result.content)

        content = "\n".join(formatted_lines)

        problems = await self._analyze_content_for_problems(
            content,
            source="stack_overflow",
            domain="tech_dev"
        )

        self.discovered_problems.extend(problems)
        return problems

    async def discover_from_github_issues(
        self,
        repos: List[str] = None,
        limit: int = 20
    ) -> List[DiscoveredProblem]:
        """Discover problems from GitHub issue trends.

        Fetches recent issues from specified GitHub repositories (or
        trending repos) and analyzes them for patterns indicating common
        developer problems.

        Args:
            repos: Optional list of 'owner/repo' strings to analyze.
                   If None, fetches from trending repositories.
            limit: Maximum number of issues to analyze per repo.

        Returns:
            List of discovered problems with source='github_issues'.
        """
        default_repos = repos or ["microsoft/vscode", "vercel/next.js", "facebook/react"]
        all_problems = []

        for repo in default_repos:
            # GitHub API endpoint for recent issues (last 30 days)
            since_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            url = (
                f"https://api.github.com/repos/{repo}/issues"
                f"?state=open&sort=comments&direction=desc&per_page={limit}"
                f"&since={since_date}"
            )

            content_result = await self.scraper.fetch_url(url)

            if not content_result.success:
                continue

            # Format the GitHub issues for analysis
            formatted_lines = [f"GitHub Issues - {repo}\n"]

            try:
                issues = json.loads(content_result.content)

                if not isinstance(issues, list):
                    continue

                for issue in issues[:limit]:
                    if issue.get("pull_request"):
                        continue

                    title = issue.get("title", "")
                    comments = issue.get("comments", 0)
                    labels = [
                        lbl.get("name", "") for lbl in issue.get("labels", [])
                    ]
                    body = (issue.get("body") or "")

                    formatted_lines.append(
                        f"[Comments: {comments}, Labels: {', '.join(labels)}] {title}"
                    )
                    if body:
                        formatted_lines.append(f"  {body}")
                    formatted_lines.append("")
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.debug("JSON parsing failed for GitHub issues, using raw content")
                formatted_lines.append(content_result.content)

            content = "\n".join(formatted_lines)

            problems = await self._analyze_content_for_problems(
                content,
                source="github_issues",
                domain="tech_dev"
            )

            all_problems.extend(problems)

        self.discovered_problems.extend(all_problems)
        return all_problems

    async def _analyze_content_for_problems(
        self,
        content: str,
        source: str,
        domain: str
    ) -> List[DiscoveredProblem]:
        """
        Use LLM to analyze content and extract problems.

        Args:
            content: Raw content to analyze
            source: Source identifier
            domain: Domain context
        """
        # Get domain-specific keywords
        pain_keywords = self.domain_config.get_pain_point_keywords()
        opportunity_keywords = self.domain_config.get_opportunity_keywords()

        prompt = f"""Analyze this content and identify specific user problems, pain points, and frustrations.

Content from {source}:
---
{content}
---

Pain point indicators: {', '.join(pain_keywords[:10])}
Opportunity indicators: {', '.join(opportunity_keywords[:10])}

For EACH distinct problem you find, provide:
1. PROBLEM: Clear description of the problem (what is the user struggling with?)
2. SEVERITY: critical/high/medium/low (how painful is this?)
3. FREQUENCY: very_common/common/occasional/rare (how often do people mention this?)
4. TARGET_USERS: Who experiences this problem?
5. EVIDENCE: Direct quotes or paraphrases from the content
6. KEYWORDS: Key terms related to this problem
7. SOLUTION_IDEAS: Brief ideas for potential solutions
8. LANGUAGE: What language is the source content in? (e.g., en, es, zh, ja, de, fr)

Format each problem as:
---PROBLEM---
PROBLEM: [description]
SEVERITY: [level]
FREQUENCY: [level]
TARGET_USERS: [description]
EVIDENCE: [quote1] | [quote2]
KEYWORDS: [keyword1], [keyword2]
SOLUTION_IDEAS: [idea1] | [idea2]
LANGUAGE: [language code]
---END---

Find 1-5 distinct problems. Focus on real, actionable problems that could be solved with software or automation. Skip vague complaints or problems without clear solutions."""

        try:
            from config.llm_client import get_llm_client
            from config.models import ModelConfig
            from config.roles import AgentRole
            model_spec = ModelConfig().get_model(AgentRole.RESEARCHER)
            response_text, _, _ = await get_llm_client().chat_async(
                model_spec,
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2048,
            )

            problems = self._parse_problems_response(response_text, source, domain)

            return problems

        except Exception as e:
            logger.warning(f"LLM analysis failed for {source}: {e}")
            return []

    def _parse_problems_response(
        self,
        response: str,
        source: str,
        domain: str
    ) -> List[DiscoveredProblem]:
        """Parse LLM response into structured problems."""
        problems = []

        # Split by problem markers - try exact markers first, then fallbacks
        if "---PROBLEM---" in response:
            problem_blocks = response.split("---PROBLEM---")[1:]
        else:
            # Fallback: split by numbered PROBLEM headings (### **PROBLEM 1:**, PROBLEM 2:, etc.)
            # Requires a digit to distinguish section headers from field labels like **PROBLEM:**
            problem_blocks = re.split(
                r'\n(?=#{0,3}\s*\*{0,2}\s*PROBLEM\s+\d+\s*[:.\)]\s*)',
                response,
                flags=re.IGNORECASE
            )
            if len(problem_blocks) <= 1:
                # Fallback: split by numbered headings (### 1., ## **1.**, etc.)
                problem_blocks = re.split(r'\n(?=#{1,3}\s*\*{0,2}\d+[\.\)]\s)', response)
            if len(problem_blocks) <= 1:
                # Fallback: split by numbered lines (1. Something, 2) Something)
                problem_blocks = re.split(r'\n(?=\d+[\.\)]\s+\*{0,2}[A-Z])', response)
            if len(problem_blocks) <= 1:
                # Last resort: treat entire response as one problem block
                problem_blocks = [response]

        for block in problem_blocks:
            if "---END---" in block:
                block = block.split("---END---")[0]

            try:
                problem_data = self._extract_problem_fields(block)

                if not problem_data.get("description"):
                    continue

                # Parse severity (strip markdown: **High** -> high)
                severity_str = self._clean_markdown(
                    problem_data.get("severity", "medium")
                ).lower().strip()
                try:
                    severity = ProblemSeverity(severity_str)
                except ValueError:
                    severity = ProblemSeverity.MEDIUM

                # Parse frequency (strip markdown: **very_common** -> very_common)
                freq_str = self._clean_markdown(
                    problem_data.get("frequency", "occasional")
                ).lower().strip()
                try:
                    frequency = ProblemFrequency(freq_str)
                except ValueError:
                    frequency = ProblemFrequency.OCCASIONAL

                # Build metadata with language if detected
                problem_metadata = {}
                if problem_data.get("language"):
                    problem_metadata["language"] = problem_data["language"]

                problem = DiscoveredProblem(
                    id=self._generate_problem_id(),
                    description=problem_data.get("description", ""),
                    severity=severity,
                    frequency=frequency,
                    target_users=problem_data.get("target_users", "Unknown"),
                    evidence=problem_data.get("evidence", []),
                    sources=[source],
                    domain=domain,
                    keywords=problem_data.get("keywords", []),
                    potential_solution_ideas=problem_data.get("solution_ideas", []),
                    metadata=problem_metadata,
                )

                # Calculate initial score
                problem.score = self._calculate_problem_score(problem)
                problems.append(problem)

            except Exception as e:
                logger.warning(f"Error parsing problem block: {e}")
                continue

        return problems

    def _clean_markdown(self, text: str) -> str:
        """Strip markdown formatting from text."""
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)        # *italic*
        text = re.sub(r'#{1,6}\s*', '', text)              # ### headings
        text = re.sub(r'`([^`]+)`', r'\1', text)           # `code`
        return text.strip()

    # Field labels the parser recognizes (after markdown stripping)
    _FIELD_PREFIXES = [
        "PROBLEM:", "SEVERITY:", "FREQUENCY:", "TARGET_USERS:",
        "EVIDENCE:", "KEYWORDS:", "SOLUTION_IDEAS:", "LANGUAGE:",
    ]

    def _extract_problem_fields(self, block: str) -> Dict[str, Any]:
        """Extract fields from a problem block.

        Handles multi-line values: if a field label line has no value,
        subsequent non-field lines are accumulated as the value.
        Also handles common misspellings (FREQUENITY, FREQUENY).
        """
        fields: Dict[str, Any] = {}
        lines = block.strip().split("\n")

        current_field: Optional[str] = None
        accumulator: List[str] = []

        def _flush():
            """Save accumulated lines to the current field."""
            nonlocal current_field, accumulator
            if current_field and accumulator:
                value = " ".join(accumulator).strip()
                if current_field == "description":
                    fields["description"] = value
                elif current_field == "severity":
                    fields["severity"] = value
                elif current_field == "frequency":
                    fields["frequency"] = value
                elif current_field == "target_users":
                    fields["target_users"] = value
                elif current_field == "evidence":
                    fields["evidence"] = [e.strip() for e in value.split("|") if e.strip()]
                elif current_field == "keywords":
                    fields["keywords"] = [k.strip() for k in value.split(",") if k.strip()]
                elif current_field == "solution_ideas":
                    fields["solution_ideas"] = [i.strip() for i in value.split("|") if i.strip()]
                elif current_field == "language":
                    fields["language"] = value.lower()
            current_field = None
            accumulator = []

        for line in lines:
            clean = self._clean_markdown(line.strip())
            if not clean or clean.startswith("---") or clean.startswith("|"):
                continue

            # Normalize common misspellings
            normalized = re.sub(r'^FREQUEN\w*:', 'FREQUENCY:', clean)

            # Check for PROBLEM heading with number: "PROBLEM 1: description..."
            m = re.match(r'^PROBLEM\s*\d*\s*[:.\)]\s*(.*)', normalized, re.IGNORECASE)
            if m:
                _flush()
                current_field = "description"
                val = m.group(1).strip()
                if val:
                    accumulator.append(val)
                continue

            # Check standard field prefixes
            matched = False
            for prefix in self._FIELD_PREFIXES:
                if prefix == "PROBLEM:":
                    continue  # Handled above
                if normalized.upper().startswith(prefix):
                    _flush()
                    field_key = {
                        "SEVERITY:": "severity",
                        "FREQUENCY:": "frequency",
                        "TARGET_USERS:": "target_users",
                        "EVIDENCE:": "evidence",
                        "KEYWORDS:": "keywords",
                        "SOLUTION_IDEAS:": "solution_ideas",
                        "LANGUAGE:": "language",
                    }[prefix]
                    current_field = field_key
                    val = normalized[len(prefix):].strip()
                    if val:
                        accumulator.append(val)
                    matched = True
                    break

            if not matched and current_field:
                # Continuation line for the current field
                accumulator.append(clean)

        _flush()
        return fields

    def source_diversity_score(self, problem_dict: Dict[str, Any]) -> float:
        """Calculate how many independent platforms contributed evidence.

        Measures source diversity to reward problems confirmed across
        multiple independent platforms, which indicates stronger signal.

        Args:
            problem_dict: A problem dict (or DiscoveredProblem.to_dict()
                output) containing a 'sources' list.

        Returns:
            A float score: 0.3 for 1 platform, 0.6 for 2, 0.9 for 3+.
        """
        sources = problem_dict.get("sources", [])
        if not sources:
            return 0.3

        # Extract unique platform identifiers from source strings
        platforms = set()
        for source in sources:
            normalized = source.lower().strip()
            if normalized.startswith("reddit") or normalized.startswith("reddit_search"):
                platforms.add("reddit")
            elif normalized in ("hackernews", "hn", "hacker_news"):
                platforms.add("hacker_news")
            elif normalized.startswith("github"):
                platforms.add("github")
            elif normalized.startswith("stack_overflow"):
                platforms.add("stack_overflow")
            elif normalized == "user_input":
                platforms.add("user_input")
            elif normalized == "product_hunt":
                platforms.add("product_hunt")
            elif normalized == "academic":
                platforms.add("academic")
            else:
                # Each unique unknown source counts as its own platform
                platforms.add(normalized)

        count = len(platforms)
        if count >= 3:
            return 0.9
        elif count == 2:
            return 0.6
        else:
            return 0.3

    def _calculate_freshness_weight(self, age_days: float) -> float:
        """Calculate a freshness weight multiplier based on content age.

        More recent content receives a higher multiplier, reflecting
        the assumption that newer data is more relevant for problem
        discovery.

        Args:
            age_days: Age of the content in days (fractional allowed).

        Returns:
            A weight multiplier between 0.2 and 1.0.
        """
        if age_days < 1.0:
            return 1.0
        elif age_days <= 7.0:
            return 0.95
        elif age_days <= 30.0:
            return 0.8
        elif age_days <= 60.0:
            return 0.5
        else:
            return 0.1  # Nearly filter out — hard filter catches these anyway

    def _calculate_problem_score(self, problem: DiscoveredProblem) -> float:
        """Calculate a score for problem prioritization.

        Combines severity, frequency, evidence count, and solution ideas
        into a base score, then adjusts by credibility, source diversity,
        and data freshness. This means problems confirmed across multiple
        platforms with recent evidence score highest.
        """
        score = 0.0

        # Severity weight
        severity_scores = {
            ProblemSeverity.CRITICAL: 4.0,
            ProblemSeverity.HIGH: 3.0,
            ProblemSeverity.MEDIUM: 2.0,
            ProblemSeverity.LOW: 1.0,
        }
        score += severity_scores.get(problem.severity, 2.0)

        # Frequency weight
        frequency_scores = {
            ProblemFrequency.VERY_COMMON: 4.0,
            ProblemFrequency.COMMON: 3.0,
            ProblemFrequency.OCCASIONAL: 2.0,
            ProblemFrequency.RARE: 1.0,
        }
        score += frequency_scores.get(problem.frequency, 2.0)

        # Evidence bonus
        score += min(len(problem.evidence), 3) * 0.5

        # Solution ideas bonus (problems with clear solution paths)
        score += min(len(problem.potential_solution_ideas), 3) * 0.3

        # Apply credibility adjustment based on source trustworthiness
        if problem.sources:
            credibility_scores = [
                CredibilityScorer.get_base_score(source)
                for source in problem.sources
            ]
            avg_credibility = sum(credibility_scores) / len(credibility_scores)
        else:
            avg_credibility = 0.5  # Neutral for problems with no source info

        score *= avg_credibility

        # Apply source diversity factor
        problem_dict = {"sources": problem.sources}
        diversity = self.source_diversity_score(problem_dict)
        score *= diversity

        # Apply freshness weight based on problem age
        age_days = (datetime.now() - problem.discovered_at).total_seconds() / 86400.0
        freshness = self._calculate_freshness_weight(age_days)
        score *= freshness

        return round(score, 2)

    def _format_reddit_posts(self, posts: List[Dict], subreddit: str) -> str:
        """Format Reddit posts for analysis."""
        formatted = [f"Reddit r/{subreddit} - Top Posts\n"]

        for post in posts:
            formatted.append(f"[Score: {post.get('score', 0)}] {post.get('title', '')}")
            if post.get("text"):
                text = post["text"]
                formatted.append(f"  {text}")
            formatted.append("")

        return "\n".join(formatted)

    def _format_hn_stories(self, stories: List[Dict]) -> str:
        """Format HackerNews stories for analysis."""
        formatted = ["Hacker News - Top Stories\n"]

        for story in stories:
            formatted.append(
                f"[Score: {story.get('score', 0)}, Comments: {story.get('comments', 0)}] "
                f"{story.get('title', '')}"
            )
            if story.get("text"):
                formatted.append(f"  {story['text']}")
            formatted.append("")

        return "\n".join(formatted)

    def _infer_domain(self, subreddit: str) -> str:
        """Infer domain from subreddit name."""
        subreddit_lower = subreddit.lower()

        domain_mapping = {
            "programming": "tech_dev",
            "webdev": "tech_dev",
            "learnprogramming": "tech_dev",
            "coding": "tech_dev",
            "entrepreneur": "business",
            "smallbusiness": "business",
            "startups": "business",
            "productivity": "productivity",
            "ecommerce": "ecommerce",
            "shopify": "ecommerce",
        }

        return domain_mapping.get(subreddit_lower, "general")

    # ============================================================
    #  SEARCH TERM EXPANSION
    # ============================================================

    async def expand_search_terms(self, query: str, num_alternatives: int = 4) -> List[str]:
        """Use LLM to generate alternative search queries for broader discovery.

        Given a problem query, generates alternative phrasings, synonyms,
        and adjacent problem spaces to search.

        Args:
            query: Original search query.
            num_alternatives: Number of alternative queries to generate.

        Returns:
            List of alternative search queries (includes original).
        """
        prompt = f"""Generate {num_alternatives} alternative search queries for discovering problems related to:
"{query}"

Each query should approach the topic from a DIFFERENT angle:
1. Use synonyms or different phrasing
2. Search an adjacent problem space
3. Use different terminology (e.g., technical vs. user-facing language)
4. Search for the consequence/symptom instead of the cause

Output ONLY the queries, one per line, no numbering or explanation."""

        try:
            from config.llm_client import get_llm_client
            from config.models import ModelConfig
            from config.roles import AgentRole
            model_spec = ModelConfig().get_model(AgentRole.RESEARCHER)
            text, _, _ = await get_llm_client().chat_async(
                model_spec,
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )
            alternatives = [
                line.strip().strip('"').strip("'")
                for line in text.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            # Always include the original
            return [query] + alternatives[:num_alternatives]
        except Exception as e:
            logger.warning(f"Search term expansion failed: {e}")
            return [query]

    async def discover_counter_evidence(
        self,
        problems: List[DiscoveredProblem],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search for counter-evidence for top problems.

        For each problem, searches for evidence that it's already solved
        or has good alternatives, producing balanced input for DataAnalyst.

        Args:
            problems: Problems to find counter-evidence for.
            top_n: Number of top problems to check.

        Returns:
            List of dicts with problem_id, counter_evidence_queries, and results.
        """
        counter_results = []
        sorted_problems = sorted(problems, key=lambda p: p.score, reverse=True)

        for problem in sorted_problems[:top_n]:
            desc_short = problem.description[:80]
            queries = [
                f'"{desc_short}" solved',
                f'"{desc_short}" alternative solution',
            ]
            # Add keyword-based counter queries
            if problem.keywords:
                kw = " ".join(problem.keywords[:3])
                queries.append(f"{kw} already solved 2024 2025")

            counter_data = {
                "problem_id": problem.id,
                "description": problem.description,
                "counter_queries": queries,
                "counter_evidence": [],
            }

            for q in queries:
                try:
                    results = await self.scraper.search_reddit(q, sort="relevance", limit=5)
                    if results:
                        counter_data["counter_evidence"].extend(
                            [r.get("title", "") for r in results[:2] if r.get("title")]
                        )
                except Exception as e:
                    logger.debug(f"Counter-evidence search failed for query '{q}': {e}")

                # Rate limiting
                import asyncio
                await asyncio.sleep(0.5)

            counter_results.append(counter_data)

        return counter_results

    # ============================================================
    #  EXISTING SOLUTION DETECTION
    # ============================================================

    def check_existing_solutions(self, problem: DiscoveredProblem) -> Dict[str, Any]:
        """Check if a problem already has well-known open-source solutions.

        Searches GitHub repos and PyPI for existing packages/projects that
        solve the problem, using keyword overlap to gauge confidence.

        Args:
            problem: The discovered problem to check.

        Returns:
            Dict with 'has_solution', 'solutions', and 'confidence'.
        """
        from .web_search import WebSearch

        keywords = problem.keywords[:5]
        if not keywords:
            return {"has_solution": False, "solutions": [], "confidence": 0.0}

        query = " ".join(keywords)
        searcher = WebSearch()

        # Run searches synchronously via the event loop
        loop = asyncio.get_event_loop()

        try:
            github_results = loop.run_until_complete(
                searcher.search_github_repos(query, min_stars=100)
            )
        except RuntimeError:
            # Already in an async context — use a helper
            github_results = []
        try:
            pypi_results = loop.run_until_complete(
                searcher.search_pypi(query, limit=5)
            )
        except RuntimeError:
            pypi_results = []

        return self._score_solutions(keywords, github_results, pypi_results)

    async def check_existing_solutions_async(self, problem: DiscoveredProblem) -> Dict[str, Any]:
        """Async version of check_existing_solutions."""
        from .web_search import WebSearch

        keywords = problem.keywords[:5]
        if not keywords:
            return {"has_solution": False, "solutions": [], "confidence": 0.0}

        query = " ".join(keywords)
        searcher = WebSearch()

        github_results = await searcher.search_github_repos(query, min_stars=100)
        pypi_results = await searcher.search_pypi(query, limit=5)

        return self._score_solutions(keywords, github_results, pypi_results)

    def _score_solutions(
        self,
        keywords: List[str],
        github_results: List[Dict[str, Any]],
        pypi_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Score how well search results match problem keywords."""
        keyword_set = {k.lower() for k in keywords}
        solutions = []
        best_confidence = 0.0

        for repo in github_results:
            text = f"{repo.get('name', '')} {repo.get('description', '')}".lower()
            text_words = set(text.split())
            if not keyword_set:
                continue
            overlap = len(keyword_set & text_words) / len(keyword_set)
            confidence = overlap
            # Boost for high-star repos
            if repo.get("stars", 0) >= 1000:
                confidence = min(confidence * 1.2, 1.0)
            if overlap >= 0.4:
                solutions.append({
                    "source": "github",
                    "name": repo.get("name", ""),
                    "url": repo.get("url", ""),
                    "confidence": round(confidence, 2),
                })
            best_confidence = max(best_confidence, confidence)

        for pkg in pypi_results:
            text = f"{pkg.get('name', '')} {pkg.get('description', '')}".lower()
            text_words = set(text.split())
            if not keyword_set:
                continue
            overlap = len(keyword_set & text_words) / len(keyword_set)
            if overlap >= 0.4:
                solutions.append({
                    "source": "pypi",
                    "name": pkg.get("name", ""),
                    "url": pkg.get("url", ""),
                    "confidence": round(overlap, 2),
                })
            best_confidence = max(best_confidence, overlap)

        return {
            "has_solution": best_confidence >= 0.3,
            "solutions": solutions,
            "confidence": round(best_confidence, 2),
        }

    async def filter_problems_with_solutions(
        self,
        problems: List[DiscoveredProblem]
    ) -> List[DiscoveredProblem]:
        """Filter out problems that already have well-known open-source solutions.

        Args:
            problems: List of discovered problems to check.

        Returns:
            Filtered list with strong-solution problems removed and
            medium-confidence problems score-penalized.
        """
        filtered = []

        for problem in problems:
            try:
                result = await self.check_existing_solutions_async(problem)
                confidence = result.get("confidence", 0.0)

                if confidence >= 0.7:
                    logger.info(
                        "Filtering problem '%s' — existing solutions (confidence=%.2f): %s",
                        problem.description[:60],
                        confidence,
                        [s["name"] for s in result.get("solutions", [])],
                    )
                    continue
                elif confidence >= 0.3:
                    problem.score *= 0.5
                    logger.info(
                        "Penalizing problem '%s' score by 50%% (solution confidence=%.2f)",
                        problem.description[:60],
                        confidence,
                    )

                filtered.append(problem)
            except Exception as e:
                logger.warning("Solution check failed for '%s': %s", problem.description[:60], e)
                filtered.append(problem)

            # Rate limiting
            await asyncio.sleep(1)

        return filtered

    # ============================================================
    #  RETRIEVAL METHODS
    # ============================================================

    def detect_trending(self) -> List[DiscoveredProblem]:
        """Detect trending problems mentioned in 3+ sources within last 7 days.

        Cross-references problems across sources. Problems mentioned in
        3+ sources within the last 7 days get a 'trending' flag and
        priority boost in opportunity evaluation.

        Returns:
            List of trending problems (subset of discovered_problems).
        """
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_problems = [
            p for p in self.discovered_problems
            if p.discovered_at >= seven_days_ago
        ]

        # Group by keyword overlap to find trending topics
        trending = []
        for problem in recent_problems:
            if len(problem.sources) >= 3:
                problem.metadata["trending"] = True
                problem.score *= 1.3  # 30% priority boost
                trending.append(problem)
            else:
                # Check if similar problems exist across sources
                similar_count = 0
                problem_keywords = set(k.lower() for k in problem.keywords)
                if len(problem_keywords) < 2:
                    continue
                for other in recent_problems:
                    if other.id == problem.id:
                        continue
                    other_keywords = set(k.lower() for k in other.keywords)
                    if problem_keywords and other_keywords:
                        overlap = len(problem_keywords & other_keywords)
                        min_size = min(len(problem_keywords), len(other_keywords))
                        if min_size > 0 and overlap / min_size > 0.5:
                            similar_count += 1

                if similar_count >= 2:  # 3+ similar problems = trending
                    problem.metadata["trending"] = True
                    problem.score *= 1.3
                    trending.append(problem)

        return trending

    def filter_by_freshness(self, max_age_days: int = None) -> List[DiscoveredProblem]:
        """Filter discovered problems to only include those within max_age_days."""
        max_age = max_age_days or MAX_PROBLEM_AGE_DAYS
        cutoff = datetime.now() - timedelta(days=max_age)
        return [p for p in self.discovered_problems if p.discovered_at >= cutoff]

    def get_top_problems(self, limit: int = 10) -> List[DiscoveredProblem]:
        """Get top problems by score, with trending problems boosted."""
        # Detect trending before sorting
        self.detect_trending()

        cutoff = datetime.now() - timedelta(days=MAX_PROBLEM_AGE_DAYS)
        fresh_problems = [p for p in self.discovered_problems if p.discovered_at >= cutoff]
        sorted_problems = sorted(
            fresh_problems,
            key=lambda p: p.score,
            reverse=True
        )
        return sorted_problems[:limit]

    def get_problems_by_domain(self, domain: str) -> List[DiscoveredProblem]:
        """Get problems for a specific domain."""
        return [p for p in self.discovered_problems if p.domain == domain]

    def get_problems_by_severity(
        self,
        severity: ProblemSeverity
    ) -> List[DiscoveredProblem]:
        """Get problems by severity level."""
        return [p for p in self.discovered_problems if p.severity == severity]

    def get_problem_by_id(self, problem_id: str) -> Optional[DiscoveredProblem]:
        """Get a specific problem by ID."""
        for p in self.discovered_problems:
            if p.id == problem_id:
                return p
        return None

    def export_problems(self) -> List[Dict[str, Any]]:
        """Export all problems as dictionaries."""
        return [p.to_dict() for p in self.discovered_problems]

    def clear_problems(self) -> None:
        """Clear all discovered problems."""
        self.discovered_problems.clear()
        self._problem_counter = 0

    # ============================================================
    #  VALIDATION FEEDBACK
    # ============================================================

    def apply_validation_adjustments(
        self,
        problems: List[DiscoveredProblem],
        validation_results: Dict[str, Dict[str, Any]]
    ) -> List[DiscoveredProblem]:
        """Feed DataAnalyst cross-validation results back into problem scores.

        Args:
            problems: List of discovered problems to adjust.
            validation_results: Mapping of problem ID to validation data
                containing a 'verdict' key (CONFIRMED, PARTIALLY_CONFIRMED,
                UNCONFIRMED, or CONTRADICTED).

        Returns:
            Adjusted list with contradicted problems removed and scores updated.
        """
        adjusted = []
        for problem in problems:
            verdict_data = validation_results.get(problem.id, {})
            verdict = verdict_data.get("verdict", "").upper()

            if verdict == "CONTRADICTED":
                logger.info(
                    "Removing contradicted problem: %s", problem.description[:60]
                )
                continue
            elif verdict == "UNCONFIRMED":
                problem.score *= 0.6  # 40% penalty
            elif verdict == "PARTIALLY_CONFIRMED":
                problem.score *= 0.8  # 20% penalty
            elif verdict == "CONFIRMED":
                problem.score *= 1.2  # 20% boost

            adjusted.append(problem)

        return adjusted

    async def revalidate_freshness(self, problem: DiscoveredProblem) -> DiscoveredProblem:
        """Search for recent mentions (last 7 days) of the problem keywords.

        If no recent mentions are found, penalizes score by 30%.
        If found, updates evidence with fresh data.

        Args:
            problem: The problem to revalidate.

        Returns:
            The problem with updated score and evidence.
        """
        keywords = problem.keywords[:5]
        if not keywords:
            return problem

        query = " ".join(keywords)
        try:
            results = await self.scraper.search_reddit(
                query, sort="new", limit=10
            )
            if results:
                # Found recent mentions — add as fresh evidence
                fresh_titles = [r.get("title", "") for r in results[:3] if r.get("title")]
                if fresh_titles:
                    problem.evidence.extend(
                        [f"[Fresh] {t}" for t in fresh_titles]
                    )
                    logger.info(
                        "Fresh evidence found for '%s': %d mentions",
                        problem.description[:40], len(fresh_titles)
                    )
            else:
                # No recent mentions — penalize
                problem.score *= 0.7
                logger.info(
                    "No fresh mentions for '%s' — score penalized 30%%",
                    problem.description[:40]
                )
        except Exception as e:
            logger.warning("Freshness revalidation failed for '%s': %s",
                           problem.description[:40], e)

        return problem
