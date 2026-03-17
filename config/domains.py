"""Domain configuration for problem research."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ResearchDomain(Enum):
    """Research domains for problem discovery."""

    TECH_DEV = "tech_dev"
    BUSINESS = "business"
    ECOMMERCE = "ecommerce"
    PRODUCTIVITY = "productivity"
    HEALTH_WELLNESS = "health_wellness"
    EDUCATION = "education"
    FINANCE = "finance"
    CREATIVE = "creative"


@dataclass
class DomainSource:
    """A source for research within a domain."""

    name: str
    url: str
    source_type: str  # reddit, forum, review_site, news
    selectors: Dict[str, str] = field(default_factory=dict)  # CSS selectors for scraping
    keywords: List[str] = field(default_factory=list)
    priority: int = 1  # Higher = more important


@dataclass
class DomainSpec:
    """Specification for a research domain."""

    name: str
    description: str
    sources: List[DomainSource]
    pain_point_keywords: List[str]
    opportunity_keywords: List[str]


# Domain configurations
DOMAIN_CONFIGS: Dict[ResearchDomain, DomainSpec] = {
    ResearchDomain.TECH_DEV: DomainSpec(
        name="Technology & Development",
        description="Software development, tools, and programming challenges",
        sources=[
            DomainSource(
                name="Reddit Programming",
                url="https://www.reddit.com/r/programming/top/.json?t=week",
                source_type="reddit",
                keywords=["frustrating", "hate", "wish", "annoying", "problem"],
                priority=2
            ),
            DomainSource(
                name="Reddit WebDev",
                url="https://www.reddit.com/r/webdev/top/.json?t=week",
                source_type="reddit",
                keywords=["struggle", "issue", "help", "bug"],
                priority=2
            ),
            DomainSource(
                name="HackerNews",
                url="https://hacker-news.firebaseio.com/v0/topstories.json",
                source_type="api",
                keywords=["problem", "solution", "struggling"],
                priority=3
            ),
        ],
        pain_point_keywords=[
            "frustrating", "annoying", "broken", "hate", "wish there was",
            "can't believe", "why doesn't", "should be easier", "waste of time",
            "nightmare", "headache", "struggle", "pain point"
        ],
        opportunity_keywords=[
            "would pay for", "need a tool", "looking for", "anyone know",
            "recommend", "alternative to", "better way to", "automate"
        ]
    ),

    ResearchDomain.BUSINESS: DomainSpec(
        name="Business & Entrepreneurship",
        description="Small business, startups, and entrepreneurship challenges",
        sources=[
            DomainSource(
                name="Reddit Entrepreneur",
                url="https://www.reddit.com/r/Entrepreneur/top/.json?t=week",
                source_type="reddit",
                keywords=["challenge", "problem", "struggle", "help"],
                priority=2
            ),
            DomainSource(
                name="Reddit SmallBusiness",
                url="https://www.reddit.com/r/smallbusiness/top/.json?t=week",
                source_type="reddit",
                keywords=["issue", "frustrated", "advice"],
                priority=2
            ),
            DomainSource(
                name="Indie Hackers",
                url="https://www.indiehackers.com/feed",
                source_type="forum",
                keywords=["problem", "validate", "idea"],
                priority=3
            ),
        ],
        pain_point_keywords=[
            "losing money", "wasting time", "manual process", "inefficient",
            "expensive", "complicated", "no good solution", "stuck"
        ],
        opportunity_keywords=[
            "would pay", "need help with", "looking for tool", "automate",
            "save time", "streamline", "simplify"
        ]
    ),

    ResearchDomain.PRODUCTIVITY: DomainSpec(
        name="Productivity & Workflow",
        description="Personal and team productivity challenges",
        sources=[
            DomainSource(
                name="Reddit Productivity",
                url="https://www.reddit.com/r/productivity/top/.json?t=week",
                source_type="reddit",
                keywords=["help", "struggling", "advice", "tips"],
                priority=2
            ),
        ],
        pain_point_keywords=[
            "distracted", "overwhelmed", "can't focus", "too many",
            "losing track", "forget", "disorganized", "chaotic"
        ],
        opportunity_keywords=[
            "better system", "workflow", "organize", "track", "manage"
        ]
    ),

    ResearchDomain.ECOMMERCE: DomainSpec(
        name="E-commerce & Retail",
        description="Online selling and retail challenges",
        sources=[
            DomainSource(
                name="Reddit Ecommerce",
                url="https://www.reddit.com/r/ecommerce/top/.json?t=week",
                source_type="reddit",
                keywords=["problem", "issue", "help"],
                priority=2
            ),
        ],
        pain_point_keywords=[
            "abandoned cart", "conversion", "shipping", "inventory",
            "customer service", "returns", "fraud"
        ],
        opportunity_keywords=[
            "increase sales", "reduce", "automate", "optimize"
        ]
    ),
}


class DomainConfig:
    """Domain configuration manager."""

    def __init__(self, enabled_domains: List[ResearchDomain] = None):
        self.configs = DOMAIN_CONFIGS.copy()
        self.enabled_domains = enabled_domains or list(ResearchDomain)

    def get_domain(self, domain: ResearchDomain) -> DomainSpec:
        """Get domain specification."""
        return self.configs.get(domain)

    def get_all_sources(self) -> List[DomainSource]:
        """Get all sources from enabled domains."""
        sources = []
        for domain in self.enabled_domains:
            if domain in self.configs:
                sources.extend(self.configs[domain].sources)
        return sorted(sources, key=lambda s: s.priority, reverse=True)

    def get_pain_point_keywords(self) -> List[str]:
        """Get all pain point keywords from enabled domains."""
        keywords = set()
        for domain in self.enabled_domains:
            if domain in self.configs:
                keywords.update(self.configs[domain].pain_point_keywords)
        return list(keywords)

    def get_opportunity_keywords(self) -> List[str]:
        """Get all opportunity keywords from enabled domains."""
        keywords = set()
        for domain in self.enabled_domains:
            if domain in self.configs:
                keywords.update(self.configs[domain].opportunity_keywords)
        return list(keywords)
