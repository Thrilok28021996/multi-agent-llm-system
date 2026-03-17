"""Lightweight Tree-of-Thoughts for sequential single-model inference."""
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ThoughtBranch:
    branch_id: int
    approach: str
    reasoning: str
    score: float = 0.0
    evaluation: str = ""


class TreeOfThoughts:
    """
    Generates N diverse thought branches, scores each, returns the best.
    Total LLM calls: 1 (branch gen) + N (evaluations) + 1 (final) = N+2
    With N=3: 5 sequential calls. All on the same loaded model.
    """

    def __init__(self, n_branches: int = 3):
        self.n_branches = n_branches

    async def generate_best(
        self,
        agent,
        task_prompt: str,
        evaluation_criteria: str = "",
    ) -> Tuple[str, ThoughtBranch]:
        # Step 1: Generate N diverse approaches in one call
        branch_prompt = (
            f"Task: {task_prompt}\n\n"
            f"Generate {self.n_branches} DISTINCT approaches. Each must use a fundamentally different strategy.\n\n"
            + "\n".join(
                f"APPROACH {i}:\n[approach name and full description]\n---"
                for i in range(1, self.n_branches + 1)
            )
        )
        branches_raw = await agent.generate_response_async(branch_prompt, use_first_principles=False)
        branches = self._parse_branches(branches_raw)

        if len(branches) < 2:
            # Not enough branches parsed — fall back to standard generation
            fallback = await agent.generate_response_async(task_prompt)
            return fallback, ThoughtBranch(branch_id=0, approach="standard", reasoning=fallback)

        criteria = evaluation_criteria or "correctness, completeness, feasibility, and alignment with requirements"

        # Step 2: Score each branch
        for branch in branches:
            eval_prompt = (
                f"Evaluate this approach for the task below.\n\n"
                f"TASK: {task_prompt[:400]}\n\n"
                f"APPROACH:\n{branch.reasoning[:600]}\n\n"
                f"Criteria: {criteria}\n\n"
                "Respond with:\nSCORE: X.X\nREASONING: [brief explanation]"
            )
            eval_response = await agent.generate_response_async(eval_prompt, use_first_principles=False)
            score_match = re.search(r"SCORE:\s*([\d.]+)", eval_response)
            branch.score = float(score_match.group(1)) if score_match else 0.5
            branch.evaluation = eval_response

        # Step 3: Pick best
        best = max(branches, key=lambda b: b.score)

        # Step 4: Generate final detailed response using best approach
        final_prompt = (
            f"Task: {task_prompt}\n\n"
            f"Use this approach (score {best.score:.2f}):\n{best.reasoning}\n\n"
            "Now produce the complete, detailed final response."
        )
        final_response = await agent.generate_response_async(final_prompt)
        return final_response, best

    def _parse_branches(self, raw: str) -> List[ThoughtBranch]:
        parts = re.split(r"APPROACH\s+\d+:\s*", raw, flags=re.IGNORECASE)
        branches = []
        for i, part in enumerate(parts[1:], start=1):
            text = part.split("---")[0].strip()
            if not text:
                continue
            name = text.split("\n")[0][:80]
            branches.append(ThoughtBranch(branch_id=i, approach=name, reasoning=text))
        return branches
