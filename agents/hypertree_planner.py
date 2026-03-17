"""
HyperTree Planning: Decomposes complex tasks into subtask trees with self-reflection.
Improves multi-step task execution by planning before executing.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlanNode:
    id: str
    task: str
    parent_id: Optional[str] = None
    children: List['PlanNode'] = field(default_factory=list)
    result: str = ""
    status: str = "pending"  # pending, in_progress, done, failed
    depth: int = 0


class HyperTreePlanner:
    """
    Decomposes a complex task into a tree of subtasks, then executes each.
    Uses self-reflection to revise the plan if subtasks fail.
    Max depth: 2 (to control inference cost on 16GB).
    """

    def __init__(self, max_depth: int = 2, max_children: int = 3):
        self.max_depth = max_depth
        self.max_children = max_children

    async def plan_and_execute(self, agent, task: str, context: str = "") -> str:
        """
        Decompose task into subtasks, execute each, synthesize result.
        Total LLM calls: 1 (decompose) + N subtasks (execute) + 1 (reflect) + 1 (synthesize)
        For max 3 subtasks: ~6 calls total.
        """
        # Step 1: Decompose into subtasks
        decompose_prompt = (
            f"Break down this complex task into {self.max_children} concrete, sequential subtasks.\n\n"
            f"TASK: {task}\n\n"
            f"CONTEXT: {context[:400]}\n\n"
            "Each subtask should be independently executable and build on previous ones.\n"
            f"Format:\n"
            + "\n".join(f"SUBTASK {i}: [specific actionable task]" for i in range(1, self.max_children + 1))
        )
        plan_response = await agent.generate_response_async(decompose_prompt)
        subtasks = self._parse_subtasks(plan_response)

        if not subtasks:
            # Decomposition failed — fall back to direct execution
            return await agent.generate_response_async(task)

        # Step 2: Execute each subtask sequentially
        results = []
        accumulated_context = context
        for i, subtask in enumerate(subtasks[:self.max_children]):
            exec_prompt = (
                f"Execute this subtask as part of the larger goal.\n\n"
                f"LARGER GOAL: {task[:300]}\n\n"
                f"CURRENT SUBTASK {i+1}/{len(subtasks)}: {subtask}\n\n"
                f"COMPLETED SO FAR:\n{accumulated_context[-800:] if accumulated_context else 'Nothing yet'}\n\n"
                "Complete ONLY this subtask. Be specific and concrete."
            )
            result = await agent.generate_response_async(exec_prompt)
            results.append(f"Subtask {i+1} ({subtask[:50]}): {result}")
            accumulated_context += f"\n\nSubtask {i+1} result: {result[:400]}"

        # Step 3: Self-reflection — check if subtask results are consistent
        reflect_prompt = (
            f"Review these subtask results for consistency and completeness.\n\n"
            f"ORIGINAL TASK: {task[:300]}\n\n"
            f"SUBTASK RESULTS:\n" + "\n".join(results)
            + "\n\nAre the results consistent? Any gaps? Answer YES/NO and explain briefly."
        )
        reflection = await agent.generate_response_async(reflect_prompt)

        # Step 4: Synthesize final result
        synth_prompt = (
            f"Synthesize these subtask results into a complete final response.\n\n"
            f"ORIGINAL TASK: {task[:300]}\n\n"
            f"SUBTASK RESULTS:\n" + "\n".join(results)
            + f"\n\nREFLECTION: {reflection[:300]}\n\n"
            "Produce the final complete response that fully addresses the original task."
        )
        return await agent.generate_response_async(synth_prompt)

    def _parse_subtasks(self, text: str) -> List[str]:
        matches = re.findall(r'SUBTASK\s+\d+:\s*(.+)', text, re.IGNORECASE)
        return [m.strip() for m in matches if m.strip()]
