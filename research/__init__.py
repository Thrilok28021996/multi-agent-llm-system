"""Research module for Company AGI."""

from .web_scraper import WebScraper
from .web_search import WebSearch, quick_search, SearchResult
from .problem_discoverer import (
    ProblemDiscoverer,
    ProblemSeverity,
    ProblemFrequency,
    DiscoveredProblem,
)
from .problem_statement_refiner import ProblemStatementRefiner
from .cross_validator import CrossValidator
from .credibility import CredibilityScorer
from .sources import ResearchSources

__all__ = [
    "WebScraper",
    "WebSearch",
    "quick_search",
    "SearchResult",
    "ProblemDiscoverer",
    "ProblemSeverity",
    "ProblemFrequency",
    "DiscoveredProblem",
    "ProblemStatementRefiner",
    "CrossValidator",
    "CredibilityScorer",
    "ResearchSources",
]
