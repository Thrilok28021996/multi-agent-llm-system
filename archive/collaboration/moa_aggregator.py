"""Mixture-of-Agents review aggregator for synthesizing multiple agent reviews."""
from typing import Dict


class MoAReviewAggregator:
    """
    Collects independent reviews from multiple agents and synthesizes into one.
    All inference is sequential on the single loaded model.
    """

    def __init__(self, synthesizer_agent):
        self.synthesizer = synthesizer_agent

    async def aggregate_reviews(
        self,
        artifact: str,
        reviews: Dict[str, str],
        context: str = "",
    ) -> str:
        reviews_block = "\n\n".join(
            f"--- {reviewer} Review ---\n{text}"
            for reviewer, text in reviews.items()
        )
        prompt = (
            f"Synthesize independent code/artifact reviews from multiple senior reviewers.\n\n"
            f"ARTIFACT (first 1500 chars):\n{artifact[:1500]}\n\n"
            f"INDEPENDENT REVIEWS:\n{reviews_block}\n\n"
            f"CONTEXT: {context}\n\n"
            "Produce:\n"
            "CONSENSUS_ISSUES: [issues all reviewers agree on — must fix]\n"
            "SINGLE_REVIEWER_FLAGS: [flagged by one reviewer only — include your verdict]\n"
            "ACTIONABLE_FIXES:\n"
            "1. [CRITICAL/HIGH/MEDIUM] Description\n"
            "...\n"
            "VERDICT: PASS / CONDITIONAL_PASS / FAIL"
        )
        return await self.synthesizer.generate_response_async(prompt)
