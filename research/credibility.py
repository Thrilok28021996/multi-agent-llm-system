"""Source credibility scoring for research data.

Provides a static scoring system that evaluates the trustworthiness of
discovered problems based on their source, recency, and corroboration
from independent sources.
"""

from typing import Dict, Optional
from datetime import datetime


# Base credibility scores per source type (0.0-1.0)
SOURCE_CREDIBILITY: Dict[str, float] = {
    "github_issues": 0.85,
    "stack_overflow": 0.8,
    "academic": 0.9,
    "hacker_news": 0.6,
    "reddit_programming": 0.5,
    "reddit_webdev": 0.5,
    "reddit_learnprogramming": 0.35,
    "reddit_entrepreneur": 0.45,
    "reddit_smallbusiness": 0.45,
    "reddit_startups": 0.4,
    "reddit_productivity": 0.35,
    "reddit_ecommerce": 0.4,
    "reddit_devops": 0.55,
    "reddit_saas": 0.45,
    "product_hunt": 0.5,
    "web_search": 0.55,
    "user_input": 1.0,
}

RECENCY_DECAY_DAYS = 45  # After 45 days, score starts decaying (tech moves fast)
CORROBORATION_BONUS = 0.1  # Per independent confirming source

# Enhanced freshness settings
FRESHNESS_REQUIRED = True           # Enforce freshness scoring
FRESHNESS_PENALTY_30_DAYS = 0.5     # 50% penalty for data older than 30 days
FRESHNESS_PENALTY_90_DAYS = 0.8     # 80% penalty for data older than 90 days


def score_freshness(
    published_date: Optional[datetime] = None,
    discovered_at: Optional[datetime] = None
) -> float:
    """Score data freshness.

    Returns 1.0 for <7 days, 0.8 for <30 days, 0.5 for <90 days,
    0.2 for <180 days, 0.0 for >180 days.

    Args:
        published_date: When the content was originally published.
        discovered_at: When the content was discovered/scraped.

    Returns:
        A float between 0.0 and 1.0 representing freshness.
    """
    ref_date = published_date or discovered_at or datetime.now()
    age_days = (datetime.now() - ref_date).total_seconds() / 86400.0
    if age_days < 7:
        return 1.0
    if age_days < 30:
        return 0.8
    if age_days < 90:
        return 0.5
    if age_days < 180:
        return 0.2
    return 0.0


class CredibilityScorer:
    """Scores the credibility of discovered problems based on source,
    recency, and cross-source corroboration."""

    @staticmethod
    def get_base_score(source_name: str) -> float:
        """Get the base credibility score for a given source.

        Args:
            source_name: The identifier of the source (e.g. 'github_issues',
                'reddit_programming'). For Reddit search sources formatted as
                'reddit_search:<query>', the prefix 'reddit_search' is
                normalized to 'reddit' and then matched.

        Returns:
            A float between 0.0 and 1.0 representing base credibility.
            Returns 0.5 (neutral) for unknown sources.
        """
        # Normalize source names that come in as paths or prefixes
        # e.g. "reddit/r/programming" -> try "reddit_programming"
        normalized = source_name.lower().strip()

        # Direct lookup first
        if normalized in SOURCE_CREDIBILITY:
            return SOURCE_CREDIBILITY[normalized]

        # Handle "reddit/r/<subreddit>" format from ProblemDiscoverer
        if normalized.startswith("reddit/r/"):
            subreddit = normalized.replace("reddit/r/", "")
            key = f"reddit_{subreddit}"
            if key in SOURCE_CREDIBILITY:
                return SOURCE_CREDIBILITY[key]

        # Handle "reddit_search:<query>" format
        if normalized.startswith("reddit_search:"):
            # Reddit search results have unknown subreddit context;
            # use a conservative average reddit credibility
            return 0.45

        # Handle "hackernews" alias
        if normalized in ("hackernews", "hn"):
            return SOURCE_CREDIBILITY["hacker_news"]

        # Unknown source gets a neutral score
        return 0.5

    @staticmethod
    def apply_recency_decay(score: float, discovered_at: datetime) -> float:
        """Apply time-based decay to a credibility score.

        Problems discovered more than RECENCY_DECAY_DAYS ago have their
        score gradually reduced. The decay is linear: at 2x the decay
        window the score is halved, at 3x it is one-third, etc.

        Args:
            score: The current credibility score (0.0-1.0).
            discovered_at: When the problem was first discovered.

        Returns:
            The decayed score, never lower than 0.1 * original score.
        """
        now = datetime.now()
        age = now - discovered_at
        age_days = age.total_seconds() / 86400.0

        if age_days <= RECENCY_DECAY_DAYS:
            return score

        # Linear decay: score reduces proportionally beyond the decay window
        # At 180 days (2x window), multiplier = 0.5
        # At 270 days (3x window), multiplier = 0.33
        decay_factor = RECENCY_DECAY_DAYS / age_days
        decayed = score * decay_factor

        # Floor: never drop below 10% of original score
        return max(decayed, score * 0.1)

    @staticmethod
    def apply_corroboration_bonus(
        score: float,
        num_confirming_sources: int
    ) -> float:
        """Apply a bonus for independent source corroboration.

        Each additional independent source that confirms the same problem
        adds CORROBORATION_BONUS to the score, up to a maximum of 1.0.

        Args:
            score: The current credibility score (0.0-1.0).
            num_confirming_sources: Number of independent sources that
                confirm this problem (not counting the original source).

        Returns:
            The boosted score, capped at 1.0.
        """
        if num_confirming_sources <= 0:
            return score

        bonus = num_confirming_sources * CORROBORATION_BONUS
        return min(score + bonus, 1.0)

    @staticmethod
    def apply_freshness_penalty(
        score: float,
        discovered_at: datetime,
        has_timestamp: bool = True
    ) -> float:
        """Apply freshness penalty with linear decay (no cliff).

        - 0-7 days: no penalty (fresh)
        - 7-120 days: linear decay from 100% to 20% of score
        - 120+ days: floor at 20% of original score
        - No timestamp: max 70% of original score

        Args:
            score: The current credibility score.
            discovered_at: When the problem was first discovered.
            has_timestamp: Whether the source included a date/timestamp.

        Returns:
            The freshness-adjusted score.
        """
        if not FRESHNESS_REQUIRED:
            return score

        # No timestamp = can't verify freshness
        if not has_timestamp:
            return score * 0.7

        now = datetime.now()
        age_days = (now - discovered_at).total_seconds() / 86400.0

        if age_days <= 7:
            return score  # Fresh data, no penalty

        # Linear decay: 100% at day 7 → 20% at day 120
        # multiplier = 1.0 - (age_days - 7) * (0.8 / 113)
        decay_range = 113.0  # 120 - 7
        decay_amount = min((age_days - 7) / decay_range, 1.0) * 0.8
        multiplier = max(1.0 - decay_amount, 0.2)

        return score * multiplier

    @staticmethod
    def score_problem(
        source_name: str,
        discovered_at: datetime,
        num_sources: int
    ) -> float:
        """Calculate the full credibility score for a problem.

        Combines base source credibility, recency decay, and
        corroboration bonus into a single score.

        Args:
            source_name: Primary source identifier.
            discovered_at: When the problem was first discovered.
            num_sources: Total number of independent sources (including
                the primary source). Corroboration bonus is applied for
                each source beyond the first.

        Returns:
            A float between 0.0 and 1.0 representing overall credibility.
        """
        score = CredibilityScorer.get_base_score(source_name)

        # Apply freshness penalty (single age-based adjustment, not compound)
        # Note: apply_recency_decay() is still available for standalone use
        # but is NOT applied here to avoid double-penalizing old data.
        score = CredibilityScorer.apply_freshness_penalty(score, discovered_at)

        # num_sources includes the primary source; corroboration is
        # from additional sources only
        confirming = max(num_sources - 1, 0)
        score = CredibilityScorer.apply_corroboration_bonus(score, confirming)

        return round(score, 3)
