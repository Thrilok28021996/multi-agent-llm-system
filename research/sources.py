"""Research sources configuration."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class SourceType(Enum):
    """Types of research sources."""

    REDDIT = "reddit"
    HACKER_NEWS = "hacker_news"
    STACK_OVERFLOW = "stack_overflow"
    GITHUB_ISSUES = "github_issues"
    PRODUCT_HUNT = "product_hunt"
    FORUM = "forum"
    REVIEW_SITE = "review_site"
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    API = "api"
    COUNTER_EVIDENCE = "counter_evidence"


@dataclass
class ResearchSource:
    """Configuration for a research source."""

    name: str
    source_type: SourceType
    url: str
    enabled: bool = True
    priority: int = 1  # Higher = more important
    rate_limit: float = 1.0  # Seconds between requests
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchSources:
    """
    Manages research sources for problem discovery.
    """

    def __init__(self):
        self.sources: Dict[str, ResearchSource] = {}
        self._init_default_sources()

    def _init_default_sources(self) -> None:
        """Initialize default research sources."""
        default_sources = [
            # Reddit - Technology/Programming
            ResearchSource(
                name="reddit_programming",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/programming",
                priority=3,
                keywords=["bug", "issue", "frustrating", "hate", "wish"],
                metadata={"subreddit": "programming"}
            ),
            ResearchSource(
                name="reddit_webdev",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/webdev",
                priority=3,
                keywords=["struggle", "problem", "help", "stuck"],
                metadata={"subreddit": "webdev"}
            ),
            ResearchSource(
                name="reddit_learnprogramming",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/learnprogramming",
                priority=2,
                keywords=["confused", "don't understand", "hard"],
                metadata={"subreddit": "learnprogramming"}
            ),

            # Reddit - Business/Entrepreneurship
            ResearchSource(
                name="reddit_entrepreneur",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/Entrepreneur",
                priority=3,
                keywords=["challenge", "struggle", "need help", "advice"],
                metadata={"subreddit": "Entrepreneur"}
            ),
            ResearchSource(
                name="reddit_smallbusiness",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/smallbusiness",
                priority=3,
                keywords=["problem", "issue", "frustrated"],
                metadata={"subreddit": "smallbusiness"}
            ),
            ResearchSource(
                name="reddit_startups",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/startups",
                priority=2,
                keywords=["validate", "idea", "problem"],
                metadata={"subreddit": "startups"}
            ),

            # Reddit - Productivity
            ResearchSource(
                name="reddit_productivity",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/productivity",
                priority=2,
                keywords=["distracted", "overwhelmed", "can't focus"],
                metadata={"subreddit": "productivity"}
            ),

            # Reddit - E-commerce
            ResearchSource(
                name="reddit_ecommerce",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/ecommerce",
                priority=2,
                keywords=["issue", "problem", "help"],
                metadata={"subreddit": "ecommerce"}
            ),

            # Hacker News
            ResearchSource(
                name="hacker_news",
                source_type=SourceType.HACKER_NEWS,
                url="https://news.ycombinator.com",
                priority=4,
                keywords=["problem", "solution", "struggling", "announce"],
                metadata={"api_base": "https://hacker-news.firebaseio.com/v0"}
            ),

            # Stack Overflow
            ResearchSource(
                name="stack_overflow",
                source_type=SourceType.STACK_OVERFLOW,
                url="https://stackoverflow.com",
                priority=3,
                keywords=["bug", "error", "help", "exception", "crash"],
                metadata={"tags": ["python", "javascript", "api", "web"]}
            ),
            # GitHub Issues
            ResearchSource(
                name="github_issues",
                source_type=SourceType.GITHUB_ISSUES,
                url="https://github.com",
                priority=3,
                keywords=["bug", "feature request", "enhancement", "breaking"],
                metadata={"repos": ["popular"]}
            ),
            # Product Hunt
            ResearchSource(
                name="product_hunt",
                source_type=SourceType.PRODUCT_HUNT,
                url="https://www.producthunt.com",
                priority=2,
                keywords=["launch", "alternative", "tool", "app"],
                metadata={"category": "developer_tools"}
            ),

            # Reddit - DevOps/SRE
            ResearchSource(
                name="reddit_devops",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/devops",
                priority=2,
                keywords=["incident", "outage", "pain point", "tooling"],
                metadata={"subreddit": "devops"}
            ),

            # Reddit - SaaS
            ResearchSource(
                name="reddit_saas",
                source_type=SourceType.REDDIT,
                url="https://reddit.com/r/SaaS",
                priority=3,
                keywords=["churn", "feature request", "integration"],
                metadata={"subreddit": "SaaS"}
            ),
        ]

        for source in default_sources:
            self.sources[source.name] = source

    def add_source(self, source: ResearchSource) -> None:
        """Add a research source."""
        self.sources[source.name] = source

    def remove_source(self, name: str) -> bool:
        """Remove a research source."""
        if name in self.sources:
            del self.sources[name]
            return True
        return False

    def get_source(self, name: str) -> ResearchSource:
        """Get a specific source by name."""
        return self.sources.get(name)

    def get_enabled_sources(self) -> List[ResearchSource]:
        """Get all enabled sources."""
        return [s for s in self.sources.values() if s.enabled]

    def get_sources_by_type(self, source_type: SourceType) -> List[ResearchSource]:
        """Get sources of a specific type."""
        return [s for s in self.sources.values() if s.source_type == source_type]

    def get_sources_by_priority(self, min_priority: int = 1) -> List[ResearchSource]:
        """Get sources meeting minimum priority."""
        sources = [s for s in self.sources.values() if s.priority >= min_priority]
        return sorted(sources, key=lambda x: x.priority, reverse=True)

    def get_reddit_subreddits(self) -> List[str]:
        """Get list of configured Reddit subreddits."""
        return [
            s.metadata.get("subreddit")
            for s in self.sources.values()
            if s.source_type == SourceType.REDDIT and s.enabled
        ]

    def enable_source(self, name: str) -> bool:
        """Enable a source."""
        if name in self.sources:
            self.sources[name].enabled = True
            return True
        return False

    def disable_source(self, name: str) -> bool:
        """Disable a source."""
        if name in self.sources:
            self.sources[name].enabled = False
            return True
        return False

    def get_all_keywords(self) -> List[str]:
        """Get all keywords from enabled sources."""
        keywords = set()
        for source in self.get_enabled_sources():
            keywords.update(source.keywords)
        return list(keywords)

    def export_config(self) -> Dict[str, Any]:
        """Export sources configuration."""
        return {
            name: {
                "name": s.name,
                "type": s.source_type.value,
                "url": s.url,
                "enabled": s.enabled,
                "priority": s.priority,
                "keywords": s.keywords,
            }
            for name, s in self.sources.items()
        }
