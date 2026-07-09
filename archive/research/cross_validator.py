"""Cross-validation and deduplication for research findings.

Provides utilities to deduplicate similar problems discovered from
multiple sources, cross-validate findings for confidence scoring,
and detect systematic biases in the research data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Set


class BiasType(Enum):
    """Types of bias that can affect research findings."""

    COMMUNITY_BIAS = "community_bias"       # Only from one community
    POPULARITY_BIAS = "popularity_bias"     # Only popular/upvoted opinions
    RECENCY_BIAS = "recency_bias"           # Only recent problems
    ENGLISH_ONLY = "english_only"           # Missing non-English sources


@dataclass
class BiasFlag:
    """A detected bias in the research data."""

    bias_type: BiasType
    description: str
    severity: str  # "low", "medium", "high"
    affected_problems: List[str] = field(default_factory=list)  # problem IDs


class CrossValidator:
    """Cross-validates and deduplicates discovered problems.

    This class operates on lists of problem dictionaries (as returned
    by DiscoveredProblem.to_dict()) and enriches them with validation
    metadata.
    """

    # Threshold for keyword overlap to consider two problems duplicates
    DUPLICATE_KEYWORD_OVERLAP = 0.6

    # Minimum number of sources for a problem to be considered validated
    MIN_SOURCES_FOR_VALIDATION = 2

    @staticmethod
    def _get_keyword_set(problem: Dict[str, Any]) -> Set[str]:
        """Extract a normalized keyword set from a problem dict."""
        keywords = problem.get("keywords", [])
        return {k.strip().lower() for k in keywords if k.strip()}

    @staticmethod
    def _keyword_overlap_ratio(set_a: Set[str], set_b: Set[str]) -> float:
        """Calculate the Jaccard-like overlap ratio between two keyword sets.

        Returns the size of the intersection divided by the size of the
        smaller set. This avoids penalizing problems with many keywords.
        Returns 0.0 if either set is empty.
        """
        if not set_a or not set_b:
            return 0.0

        intersection = set_a & set_b
        smaller_size = min(len(set_a), len(set_b))
        return len(intersection) / smaller_size

    @staticmethod
    def _merge_problems(
        primary: Dict[str, Any],
        duplicate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge a duplicate problem into the primary problem.

        Combines sources, evidence, keywords, and solution ideas from
        both problems, keeping the primary's core fields (id,
        description, severity, etc.).
        """
        merged = dict(primary)

        # Merge sources (deduplicated)
        all_sources = list(primary.get("sources", []))
        for src in duplicate.get("sources", []):
            if src not in all_sources:
                all_sources.append(src)
        merged["sources"] = all_sources

        # Merge evidence (deduplicated)
        all_evidence = list(primary.get("evidence", []))
        for ev in duplicate.get("evidence", []):
            if ev not in all_evidence:
                all_evidence.append(ev)
        merged["evidence"] = all_evidence

        # Merge keywords (deduplicated)
        all_keywords = list(primary.get("keywords", []))
        for kw in duplicate.get("keywords", []):
            if kw not in all_keywords:
                all_keywords.append(kw)
        merged["keywords"] = all_keywords

        # Merge solution ideas (deduplicated)
        all_ideas = list(primary.get("potential_solution_ideas", []))
        for idea in duplicate.get("potential_solution_ideas", []):
            if idea not in all_ideas:
                all_ideas.append(idea)
        merged["potential_solution_ideas"] = all_ideas

        # Keep the higher score
        merged["score"] = max(
            primary.get("score", 0.0),
            duplicate.get("score", 0.0)
        )

        # Keep the higher severity
        severity_order = ["critical", "high", "medium", "low"]
        primary_sev = primary.get("severity", "medium")
        dup_sev = duplicate.get("severity", "medium")
        primary_idx = severity_order.index(primary_sev) if primary_sev in severity_order else 2
        dup_idx = severity_order.index(dup_sev) if dup_sev in severity_order else 2
        merged["severity"] = severity_order[min(primary_idx, dup_idx)]

        return merged

    @classmethod
    def deduplicate(cls, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate problems by keyword overlap.

        If two problems share more than 60% of their keywords (measured
        by the overlap ratio against the smaller keyword set), they are
        considered duplicates. The problem with the higher score is kept
        as the primary and the other is merged into it.

        Args:
            problems: List of problem dicts (from DiscoveredProblem.to_dict()).

        Returns:
            Deduplicated list with merged sources and evidence.
        """
        if not problems:
            return []

        # Track which indices have been merged into another
        merged_into: Dict[int, int] = {}  # duplicate_idx -> primary_idx
        result_problems = list(problems)  # shallow copy

        for i in range(len(problems)):
            if i in merged_into:
                continue

            keywords_i = cls._get_keyword_set(problems[i])

            for j in range(i + 1, len(problems)):
                if j in merged_into:
                    continue

                keywords_j = cls._get_keyword_set(problems[j])
                overlap = cls._keyword_overlap_ratio(keywords_i, keywords_j)

                if overlap > cls.DUPLICATE_KEYWORD_OVERLAP:
                    # Determine primary (higher score)
                    score_i = problems[i].get("score", 0.0)
                    score_j = problems[j].get("score", 0.0)

                    if score_i >= score_j:
                        result_problems[i] = cls._merge_problems(
                            result_problems[i], problems[j]
                        )
                        merged_into[j] = i
                    else:
                        result_problems[j] = cls._merge_problems(
                            result_problems[j], problems[i]
                        )
                        merged_into[i] = j
                        break  # i is now merged, stop comparing it

        # Return only non-merged problems
        deduplicated = [
            result_problems[idx]
            for idx in range(len(result_problems))
            if idx not in merged_into
        ]

        return deduplicated

    @classmethod
    def validate_source_diversity(
        cls, problems: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Flag problems confirmed by only 1 source type.

        Require at least 2 different source types for high-confidence problems.
        Single-source-type problems get max 0.5 credibility.

        Args:
            problems: List of problem dicts.

        Returns:
            The same list with 'source_diversity_score' and
            'source_types' fields added.
        """
        enriched = []
        for problem in problems:
            updated = dict(problem)
            sources = problem.get("sources", [])

            # Categorize each source
            source_types = set()
            for source in sources:
                source_types.add(cls._categorize_source(source))

            num_types = len(source_types)
            updated["source_types"] = sorted(source_types)

            if num_types >= 3:
                updated["source_diversity_score"] = 1.0
            elif num_types == 2:
                updated["source_diversity_score"] = 0.8
            elif num_types == 1:
                # Single source type — cap credibility
                updated["source_diversity_score"] = 0.5
                # Cap the problem's overall score
                current_score = updated.get("score", 0.0)
                updated["score"] = min(current_score, current_score * 0.5)
            else:
                updated["source_diversity_score"] = 0.0

            enriched.append(updated)

        return enriched

    @classmethod
    def cross_validate(cls, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cross-validate problems by checking independent source confirmation.

        Each problem receives a 'cross_validation_score' field:
        - Base score of 0.0
        - +0.1 per independent source that confirms the problem
        - Problems with only 1 source get a 'single_source_warning' flag

        Args:
            problems: List of problem dicts.

        Returns:
            The same list with 'cross_validation_score' and
            'single_source_warning' fields added to each problem.
        """
        validated = []

        for problem in problems:
            enriched = dict(problem)
            sources = problem.get("sources", [])
            num_sources = len(sources)

            # Each independent source beyond the first adds 0.1
            cross_score = max(num_sources - 1, 0) * 0.1
            enriched["cross_validation_score"] = round(min(cross_score, 1.0), 2)

            # Flag single-source problems
            enriched["single_source_warning"] = num_sources <= 1

            validated.append(enriched)

        return validated

    @classmethod
    def detect_bias(cls, problems: List[Dict[str, Any]]) -> List[BiasFlag]:
        """Detect systematic biases in the research data.

        Checks for:
        - community_bias: >80% of problems from the same source type
        - popularity_bias: all evidence from top-voted content only
        - recency_bias: all problems discovered in the last 7 days
        - english_only: no problems with non-English metadata

        Args:
            problems: List of problem dicts.

        Returns:
            List of BiasFlag objects describing detected biases.
        """
        if not problems:
            return []

        flags: List[BiasFlag] = []
        all_ids = [p.get("id", "unknown") for p in problems]

        # --- Community bias ---
        # Count source type occurrences across all problems
        source_type_counts: Dict[str, int] = {}
        total_source_refs = 0

        for problem in problems:
            for source in problem.get("sources", []):
                # Normalize source to a type category
                source_type = cls._categorize_source(source)
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
                total_source_refs += 1

        if total_source_refs > 0:
            for source_type, count in source_type_counts.items():
                ratio = count / total_source_refs
                if ratio > 0.8:
                    severity = "high" if ratio > 0.95 else "medium"
                    affected = [
                        p.get("id", "unknown")
                        for p in problems
                        if any(
                            cls._categorize_source(s) == source_type
                            for s in p.get("sources", [])
                        )
                    ]
                    flags.append(BiasFlag(
                        bias_type=BiasType.COMMUNITY_BIAS,
                        description=(
                            f"{ratio:.0%} of source references come from "
                            f"'{source_type}'. Consider diversifying research "
                            f"sources for more balanced findings."
                        ),
                        severity=severity,
                        affected_problems=affected,
                    ))

        # --- Popularity bias ---
        # Check if all evidence appears to come from high-score content
        # Heuristic: if every problem has "Score:" in evidence suggesting
        # only top-voted content was analyzed
        has_low_score_evidence = False
        for problem in problems:
            metadata = problem.get("metadata", {})
            if metadata.get("includes_low_score_content", False):
                has_low_score_evidence = True
                break
            # Also check if any source is from search (more diverse)
            for source in problem.get("sources", []):
                if "search" in source.lower():
                    has_low_score_evidence = True
                    break
            if has_low_score_evidence:
                break

        if not has_low_score_evidence and len(problems) > 2:
            flags.append(BiasFlag(
                bias_type=BiasType.POPULARITY_BIAS,
                description=(
                    "All problems appear to come from top-voted/popular "
                    "content. Less popular but valid problems may be missed. "
                    "Consider including 'new' or 'controversial' sort orders."
                ),
                severity="medium",
                affected_problems=all_ids,
            ))

        # --- Recency bias ---
        # Check if all problems were discovered in the last 7 days
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        all_recent = True

        for problem in problems:
            discovered_str = problem.get("discovered_at")
            if discovered_str:
                try:
                    discovered = datetime.fromisoformat(discovered_str)
                    if discovered < seven_days_ago:
                        all_recent = False
                        break
                except (ValueError, TypeError):
                    # Cannot parse; assume not recent
                    all_recent = False
                    break
            else:
                all_recent = False
                break

        if all_recent and len(problems) > 2:
            flags.append(BiasFlag(
                bias_type=BiasType.RECENCY_BIAS,
                description=(
                    "All problems were discovered within the last 7 days. "
                    "Older, persistent problems may be underrepresented. "
                    "Consider expanding the time window for research."
                ),
                severity="low",
                affected_problems=all_ids,
            ))

        # --- English-only bias ---
        # Only flag if language data exists and is all English.
        # If no problems have language metadata, we can't determine bias.
        has_non_english = False
        has_language_data = False
        for problem in problems:
            metadata = problem.get("metadata", {})
            language = metadata.get("language", "")
            if language:
                has_language_data = True
                if language.lower() not in ("en", "english"):
                    has_non_english = True
                    break

        if has_language_data and not has_non_english and len(problems) > 0:
            flags.append(BiasFlag(
                bias_type=BiasType.ENGLISH_ONLY,
                description=(
                    "All research findings are from English-language sources. "
                    "Problems specific to non-English-speaking markets may be "
                    "missed. Consider adding multilingual research sources."
                ),
                severity="low",
                affected_problems=all_ids,
            ))

        return flags

    @staticmethod
    def _categorize_source(source: str) -> str:
        """Categorize a source string into a broad type.

        Args:
            source: Source identifier (e.g. 'reddit/r/programming',
                'hackernews', 'stack_overflow').

        Returns:
            A category string like 'reddit', 'hackernews', 'stackoverflow',
            'github', 'producthunt', or 'other'.
        """
        source_lower = source.lower()

        if "reddit" in source_lower:
            return "reddit"
        elif "hackernews" in source_lower or "hacker_news" in source_lower or "hn" == source_lower:
            return "hackernews"
        elif "stackoverflow" in source_lower or "stack_overflow" in source_lower:
            return "stackoverflow"
        elif "github" in source_lower:
            return "github"
        elif "producthunt" in source_lower or "product_hunt" in source_lower:
            return "producthunt"
        else:
            return "other"
