"""Multi-agent debate orchestrator for improving architecture decisions."""
from dataclasses import dataclass
from typing import List


@dataclass
class DebateRound:
    round_number: int
    agent_a_position: str
    agent_b_position: str


@dataclass
class DebateOutcome:
    topic: str
    rounds: List[DebateRound]
    final_position: str
    consensus_reached: bool = False


class DebateOrchestrator:
    """
    Structured debate between two agents. Sequential inference only.
    2 rounds = 4 inference calls + 1 synthesis = 5 total calls.
    """

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds

    async def run_debate(
        self,
        agent_a,
        agent_b,
        topic: str,
        initial_a: str = "",
        initial_b: str = "",
        synthesizer=None,
    ) -> DebateOutcome:
        # Initial positions
        pos_a = initial_a or await agent_a.generate_response_async(
            f"State your position on: {topic}\nProvide 3 key arguments. Be specific and concise."
        )
        pos_b = initial_b or await agent_b.generate_response_async(
            f"Topic: {topic}\n\nOther party's position:\n{pos_a[:500]}\n\n"
            "State YOUR position. Where do you agree? Where do you disagree? Give 3 key arguments."
        )

        rounds = []
        for r in range(1, self.max_rounds + 1):
            new_a = await agent_a.generate_response_async(
                f"DEBATE ROUND {r} - Topic: {topic}\n\n"
                f"Your position: {pos_a[:400]}\n\nOpponent: {pos_b[:400]}\n\n"
                "1. Acknowledge valid opponent points.\n"
                "2. Rebut their weakest argument.\n"
                "3. State your refined position."
            )
            new_b = await agent_b.generate_response_async(
                f"DEBATE ROUND {r} - Topic: {topic}\n\n"
                f"Your position: {pos_b[:400]}\n\nOpponent: {new_a[:400]}\n\n"
                "1. Acknowledge valid opponent points.\n"
                "2. Rebut their weakest argument.\n"
                "3. State your refined position."
            )
            rounds.append(DebateRound(r, new_a, new_b))
            pos_a, pos_b = new_a, new_b

        synth = synthesizer or agent_b
        final = await synth.generate_response_async(
            f"Synthesize this debate outcome on: {topic}\n\n"
            f"Final position A ({agent_a.name}):\n{pos_a[:600]}\n\n"
            f"Final position B ({agent_b.name}):\n{pos_b[:600]}\n\n"
            "Output:\n"
            "CONSENSUS: [points both agree on]\n"
            "UNRESOLVED: [genuine remaining disagreements]\n"
            "FINAL_RECOMMENDATION: [best unified recommendation to move forward]"
        )

        return DebateOutcome(topic=topic, rounds=rounds, final_position=final)
