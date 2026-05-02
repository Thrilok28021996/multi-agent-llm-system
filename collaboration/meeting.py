"""Agent collaboration through meetings, debates, and consensus building."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from config.llm_client import get_llm_client
from config.models import ModelConfig
from config.roles import AgentRole

from ui.console import console


class MeetingType(Enum):
    """Types of meetings."""
    BRAINSTORM = "brainstorm"        # Generate ideas
    DECISION = "decision"            # Make a decision
    REVIEW = "review"                # Review work
    PLANNING = "planning"            # Plan next steps
    DEBATE = "debate"                # Argue different perspectives
    STANDUP = "standup"              # Brief status check: what did/will you do, blockers?
    RETROSPECTIVE = "retrospective"  # What worked, what didn't, what to improve?
    DEVILS_ADVOCATE = "devils_advocate"  # Adversarial review: advocate vs critic
    ONE_ON_ONE = "one_on_one"        # Supervisor reviews report's performance


@dataclass
class MeetingParticipant:
    """A participant in a meeting."""
    name: str
    role: str
    model: str
    perspective: str = ""  # Their viewpoint/bias for debates


@dataclass
class MeetingMessage:
    """A message in a meeting."""
    speaker: str
    content: str
    message_type: str = "statement"  # statement, question, proposal, vote
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MeetingOutcome:
    """Outcome of a meeting."""
    decision: Optional[str] = None
    action_items: List[str] = field(default_factory=list)
    consensus_reached: bool = False
    votes: Dict[str, str] = field(default_factory=dict)
    key_insights: List[str] = field(default_factory=list)


class AgentMeeting:
    """
    Facilitates meetings between agents with real collaboration.
    Agents can debate, brainstorm, and reach consensus.
    """

    def __init__(self, meeting_type: MeetingType, topic: str):
        self.meeting_type = meeting_type
        self.topic = topic
        self.participants: List[MeetingParticipant] = []
        self.transcript: List[MeetingMessage] = []
        self.outcome = MeetingOutcome()
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        self.max_rounds = 5

    def add_participant(
        self,
        name: str,
        role: str,
        model: str,
        perspective: str = ""
    ) -> None:
        """Add a participant to the meeting."""
        self.participants.append(MeetingParticipant(
            name=name,
            role=role,
            model=model,
            perspective=perspective
        ))

    async def run(self, context: str = "") -> MeetingOutcome:
        """
        Run the meeting and return the outcome.

        Args:
            context: Background context for the meeting
        """
        self.started_at = datetime.now()

        # Show meeting start
        console.show_meeting(
            self.topic,
            [p.name for p in self.participants],
            []
        )

        if self.meeting_type == MeetingType.BRAINSTORM:
            await self._run_brainstorm(context)
        elif self.meeting_type == MeetingType.DECISION:
            await self._run_decision(context)
        elif self.meeting_type == MeetingType.DEBATE:
            await self._run_debate(context)
        elif self.meeting_type == MeetingType.REVIEW:
            await self._run_review(context)
        elif self.meeting_type == MeetingType.STANDUP:
            await self._run_standup(context)
        elif self.meeting_type == MeetingType.RETROSPECTIVE:
            await self._run_retrospective(context)
        elif self.meeting_type == MeetingType.DEVILS_ADVOCATE:
            await self._run_devils_advocate(context)
        elif self.meeting_type == MeetingType.ONE_ON_ONE:
            await self._run_one_on_one(context)
        else:
            await self._run_planning(context)

        self.ended_at = datetime.now()
        return self.outcome

    async def _run_brainstorm(self, context: str) -> None:
        """Run a brainstorming session."""
        console.info(f"Starting brainstorm on: {self.topic}")

        ideas = []

        # Each participant contributes ideas
        for participant in self.participants:
            prompt = f"""You are {participant.name}, the {participant.role} in a company meeting.

Topic: {self.topic}
Context: {context}

Previous ideas shared:
{chr(10).join(f'- {idea}' for idea in ideas) if ideas else 'None yet'}

Share 2-3 creative ideas from your perspective as {participant.role}. Be concise.
Build on previous ideas if relevant. Format as bullet points."""

            response = await self._get_response(participant, prompt)

            # Record message
            self.transcript.append(MeetingMessage(
                speaker=participant.name,
                content=response,
                message_type="proposal"
            ))

            console.agent_thinking(participant.name, response)

            # Extract ideas
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    ideas.append(f"[{participant.name}] {line[1:].strip()}")

        # Summarize and vote on best ideas
        self.outcome.key_insights = ideas[:10]
        self.outcome.consensus_reached = True

    async def _run_decision(self, context: str) -> None:
        """Run a decision-making meeting."""
        console.info(f"Decision meeting on: {self.topic}")

        # Round 1: Each participant shares their view
        views = []
        for participant in self.participants:
            prompt = f"""You are {participant.name}, the {participant.role}.

DECISION NEEDED: {self.topic}
CONTEXT: {context}

From your area of expertise ({participant.role}), give your recommendation.

Evaluate honestly based on evidence. Do not default to either YES or NO.
Cite specific evidence from the context to support your position.

Respond in this exact format:
RECOMMENDATION: YES or NO or NEED_MORE_INFO (one word)
EVIDENCE: [specific fact or data point supporting your recommendation]
REASONING: [1-2 sentences explaining why]"""

            response = await self._get_response(participant, prompt)
            views.append({"speaker": participant.name, "view": response})

            self.transcript.append(MeetingMessage(
                speaker=participant.name,
                content=response,
                message_type="proposal"
            ))

            console.agent_message(participant.name, "all", response)

        # Round 2: Seek consensus
        consensus_prompt = f"""Based on the discussion about "{self.topic}":

{chr(10).join(f'{v["speaker"]}: {v["view"]}' for v in views)}

Synthesize these views into a final decision. Weigh the evidence presented by each participant.
If participants disagree, explain why you favor one side over the other.
If evidence is insufficient, decide REQUEST_MORE_INFO rather than guessing.

State the decision in one sentence, then list any noted concerns as bullet points."""

        # Use CEO's model for final decision
        ceo = next((p for p in self.participants if "CEO" in p.name), self.participants[0])
        final_decision = await self._get_response(ceo, consensus_prompt)

        self.outcome.decision = final_decision
        self.outcome.consensus_reached = True

        console.agent_decision(ceo.name, final_decision)

    async def _run_debate(self, context: str) -> None:
        """Run a debate with different perspectives."""
        console.info(f"Debate on: {self.topic}")

        # Assign perspectives if not set
        perspectives = ["advocate", "skeptic", "pragmatist"]
        for i, participant in enumerate(self.participants):
            if not participant.perspective:
                participant.perspective = perspectives[i % len(perspectives)]

        # Multiple rounds of debate
        for round_num in range(min(3, self.max_rounds)):
            console.info(f"Round {round_num + 1}")

            for participant in self.participants:
                # Get previous arguments
                prev_args = [
                    f"{m.speaker} ({next((p.perspective for p in self.participants if p.name == m.speaker), 'unknown')}): {m.content}"
                    for m in self.transcript[-6:]  # Last 6 messages
                ]

                prompt = f"""You are {participant.name}, the {participant.role}.
Your perspective in this debate: {participant.perspective.upper()}

Topic: {self.topic}
Context: {context}

Previous arguments:
{chr(10).join(prev_args) if prev_args else 'Opening statements'}

As a {participant.perspective}, make your argument (2-3 sentences).
{'Challenge the other perspectives.' if round_num > 0 else 'State your opening position.'}"""

                response = await self._get_response(participant, prompt)

                self.transcript.append(MeetingMessage(
                    speaker=participant.name,
                    content=response,
                    message_type="statement"
                ))

                console.agent_message(
                    participant.name,
                    "all",
                    f"[{participant.perspective}] {response}"
                )

        # Synthesize conclusion
        all_args = "\n".join(f"{m.speaker}: {m.content}" for m in self.transcript)
        synthesis_prompt = f"""Synthesize this debate on "{self.topic}":

{all_args}

Provide:
1. Key points from each perspective
2. A balanced conclusion
3. Recommended action"""

        facilitator = self.participants[0]
        conclusion = await self._get_response(facilitator, synthesis_prompt)

        self.outcome.decision = conclusion
        self.outcome.key_insights = [m.content for m in self.transcript[:5]]

    async def _run_review(self, context: str) -> None:
        """Run a review meeting."""
        console.info(f"Review meeting: {self.topic}")

        feedback = []

        for participant in self.participants:
            prompt = f"""You are {participant.name}, the {participant.role}, reviewing a completed solution.

WORK UNDER REVIEW:
{context}

Topic: {self.topic}

VERDICT RULES (evaluate honestly based on evidence):
- APPROVE: Evidence shows the solution addresses the stated problem and is functional.
- NEEDS_CHANGES: Evidence shows a specific, fixable gap. You MUST name the exact issue.
- REJECT: Evidence shows the solution is fundamentally wrong or does not address the problem.

Review whether the solution WORKS based on evidence presented. Cite specific strengths or problems you observe.

Respond in this exact format:
STRENGTHS: [1-2 brief points]
ISSUES: [1-2 specific issues, or "None" if the solution works]
VERDICT: [exactly one of: APPROVE / NEEDS_CHANGES / REJECT]"""

            response = await self._get_response(participant, prompt)
            feedback.append({"reviewer": participant.name, "feedback": response})

            self.transcript.append(MeetingMessage(
                speaker=participant.name,
                content=response,
                message_type="review"
            ))

            # Parse verdict — no default bias, require explicit verdict
            from utils.output_parser import StructuredOutputParser
            parser = StructuredOutputParser()
            verdict = parser.parse_verdict(response)
            # Map parser output to meeting verdict
            if verdict in ("pass", "approve"):
                verdict = "approve"
            elif verdict in ("fail", "reject"):
                verdict = "reject"
            elif verdict in ("pass_with_issues", "needs_changes"):
                verdict = "needs_changes"
            else:
                verdict = "approve"  # uncertain with no explicit rejection = approve

            self.outcome.votes[participant.name] = verdict
            console.agent_message(participant.name, "all", f"Verdict: {verdict.upper()}")

        # Determine overall outcome — require majority approval, no ties
        approvals = sum(1 for v in self.outcome.votes.values() if v == "approve")
        rejections = sum(1 for v in self.outcome.votes.values() if v == "reject")
        total = len(self.outcome.votes)

        if total > 0 and approvals > total / 2:
            self.outcome.decision = "approved"
            self.outcome.consensus_reached = True
        elif rejections > 0:
            self.outcome.decision = "needs_revision"
            self.outcome.action_items = [
                f.get("feedback", "") for f in feedback
            ]
        else:
            # No clear majority — default to needs_revision, not approved
            self.outcome.decision = "needs_revision"
            self.outcome.consensus_reached = False
            self.outcome.action_items = [
                f.get("feedback", "") for f in feedback
            ]

    async def _run_planning(self, context: str) -> None:
        """Run a planning meeting."""
        console.info(f"Planning meeting: {self.topic}")

        # Gather input from all participants
        all_tasks = []

        for participant in self.participants:
            prompt = f"""You are {participant.name}, the {participant.role}.

Planning topic: {self.topic}
Context: {context}

From your perspective, what tasks need to be done?
List 2-3 specific, actionable tasks with owners.
Format: - [Owner] Task description"""

            response = await self._get_response(participant, prompt)

            self.transcript.append(MeetingMessage(
                speaker=participant.name,
                content=response,
                message_type="proposal"
            ))

            # Extract tasks
            for line in response.split("\n"):
                if line.strip().startswith("-"):
                    all_tasks.append(line.strip()[1:].strip())

            console.agent_thinking(participant.name, response)

        # Consolidate action items
        self.outcome.action_items = all_tasks[:10]
        self.outcome.consensus_reached = True
        self.outcome.decision = f"Planned {len(self.outcome.action_items)} action items"

    async def _run_standup(self, context: str) -> None:
        """Run a brief standup meeting: what did/will you do, blockers?"""
        console.info(f"Standup: {self.topic}")

        for participant in self.participants:
            prompt = f"""You are {participant.name}, the {participant.role}, in a brief standup meeting.

Topic: {self.topic}
Context: {context}

Answer briefly (1-2 sentences each):
1. What did you do since last check-in?
2. What will you do next?
3. Any blockers?"""

            response = await self._get_response(participant, prompt)

            self.transcript.append(MeetingMessage(
                speaker=participant.name,
                content=response,
                message_type="statement"
            ))

            console.agent_thinking(participant.name, response)

        self.outcome.consensus_reached = True
        self.outcome.decision = "Standup complete"
        self.outcome.action_items = [
            m.content for m in self.transcript
        ]

    async def _run_retrospective(self, context: str) -> None:
        """Run a retrospective: what worked, what didn't, what to improve?"""
        console.info(f"Retrospective: {self.topic}")

        learnings = []

        for participant in self.participants:
            prompt = f"""You are {participant.name}, the {participant.role}, in a retrospective meeting.

Topic: {self.topic}
Context: {context}

Reflect on this workflow run. Answer briefly:
1. What worked well?
2. What didn't work well?
3. What should we improve for next time?

Be specific and actionable (2-3 sentences total)."""

            response = await self._get_response(participant, prompt)

            self.transcript.append(MeetingMessage(
                speaker=participant.name,
                content=response,
                message_type="statement"
            ))

            learnings.append(f"[{participant.name}] {response}")
            console.agent_thinking(participant.name, response)

        self.outcome.consensus_reached = True
        self.outcome.decision = "Retrospective complete"
        self.outcome.key_insights = learnings[:10]
        self.outcome.action_items = [
            f"Improvement from {m.speaker}: {m.content}"
            for m in self.transcript
        ]

    async def _run_devils_advocate(self, context: str) -> None:
        """Run a devil's advocate meeting: advocate vs critic, judge decides.

        Requires at least 3 participants:
        - First participant: Advocate (argues FOR)
        - Second participant: Critic (argues AGAINST with specific evidence)
        - Third participant: Judge (weighs both sides objectively)
        """
        console.info(f"Devil's advocate review: {self.topic}")

        if len(self.participants) < 3:
            console.warning("Devil's advocate requires 3+ participants. Falling back to review.")
            await self._run_review(context)
            return

        advocate = self.participants[0]
        critic = self.participants[1]
        judge = self.participants[2]

        # Round 1: Advocate makes the case FOR
        advocate_prompt = f"""You are {advocate.name}, the {advocate.role}, acting as the ADVOCATE.

TOPIC: {self.topic}
CONTEXT: {context}

Your job: Make the strongest possible case FOR this solution/proposal.
Cite specific evidence from the context. Be persuasive but honest.
If there are genuine weaknesses, acknowledge them briefly but argue they are manageable.

Present your case (3-5 sentences):"""

        advocate_response = await self._get_response(advocate, advocate_prompt)
        self.transcript.append(MeetingMessage(
            speaker=advocate.name, content=f"[ADVOCATE] {advocate_response}",
            message_type="proposal"
        ))
        console.agent_message(advocate.name, "all", f"[ADVOCATE] {advocate_response}")

        # Round 2: Critic argues AGAINST
        critic_prompt = f"""You are {critic.name}, the {critic.role}, acting as the CRITIC.

TOPIC: {self.topic}
CONTEXT: {context}

ADVOCATE'S CASE: {advocate_response}

Your job: Find real problems and argue AGAINST this solution/proposal.
You MUST cite specific evidence for each concern. No vague objections.
Focus on: missing requirements, technical risks, quality gaps, unaddressed edge cases.

Present your counterarguments (3-5 sentences, each with specific evidence):"""

        critic_response = await self._get_response(critic, critic_prompt)
        self.transcript.append(MeetingMessage(
            speaker=critic.name, content=f"[CRITIC] {critic_response}",
            message_type="statement"
        ))
        console.agent_message(critic.name, "all", f"[CRITIC] {critic_response}")

        # Round 3: Judge weighs both sides
        judge_prompt = f"""You are {judge.name}, the {judge.role}, acting as the JUDGE.

TOPIC: {self.topic}
CONTEXT: {context}

ADVOCATE ({advocate.name}): {advocate_response}
CRITIC ({critic.name}): {critic_response}

Your job: Weigh both sides objectively and reach a verdict.
For each of the critic's concerns, state whether it is valid or not and why.
Do not default to either side — decide based on the strength of evidence.

VERDICT (exactly one): APPROVE / NEEDS_CHANGES / REJECT
REASONING: [2-3 sentences explaining your decision]"""

        judge_response = await self._get_response(judge, judge_prompt)
        self.transcript.append(MeetingMessage(
            speaker=judge.name, content=f"[JUDGE] {judge_response}",
            message_type="vote"
        ))
        console.agent_decision(judge.name, f"[JUDGE] {judge_response}")

        # Parse judge's verdict
        from utils.output_parser import StructuredOutputParser
        parser = StructuredOutputParser()
        verdict = parser.parse_verdict(judge_response)

        if verdict in ("approve", "pass"):
            self.outcome.decision = "approved"
            self.outcome.consensus_reached = True
        elif verdict in ("reject", "fail"):
            self.outcome.decision = "rejected"
            self.outcome.consensus_reached = True
        else:
            self.outcome.decision = "needs_revision"
            self.outcome.consensus_reached = False

        self.outcome.votes = {
            advocate.name: "approve",
            critic.name: "reject",
            judge.name: verdict
        }
        self.outcome.key_insights = [advocate_response, critic_response, judge_response]

    async def _run_one_on_one(self, context: str) -> None:
        """Run a 1:1 meeting between supervisor and report.

        First participant = supervisor, second = report.
        Supervisor reviews report's recent performance and provides feedback.
        """
        console.info(f"1:1 meeting: {self.topic}")

        if len(self.participants) < 2:
            console.warning("1:1 requires 2 participants.")
            self.outcome.decision = "Insufficient participants"
            return

        supervisor = self.participants[0]
        report = self.participants[1]

        # Supervisor gives feedback
        supervisor_prompt = f"""You are {supervisor.name}, the {supervisor.role}, in a 1:1 meeting with {report.name} ({report.role}).

CONTEXT (recent performance data):
{context}

Provide constructive feedback:
1. STRENGTHS: What {report.name} did well (be specific)
2. IMPROVEMENTS: What {report.name} should improve (be specific and actionable)
3. GOALS: 1-2 specific goals for the next sprint
4. SUPPORT: What you will do to help them succeed

Be direct and specific. Avoid vague praise or criticism."""

        supervisor_response = await self._get_response(supervisor, supervisor_prompt)
        self.transcript.append(MeetingMessage(
            speaker=supervisor.name, content=supervisor_response,
            message_type="review"
        ))
        console.agent_message(supervisor.name, report.name, supervisor_response)

        # Report responds
        report_prompt = f"""You are {report.name}, the {report.role}, in a 1:1 with your supervisor {supervisor.name}.

Your supervisor's feedback:
{supervisor_response}

Respond briefly:
1. What you agree with
2. What support you need
3. Your commitment for next sprint

Keep it to 2-3 sentences."""

        report_response = await self._get_response(report, report_prompt)
        self.transcript.append(MeetingMessage(
            speaker=report.name, content=report_response,
            message_type="statement"
        ))
        console.agent_message(report.name, supervisor.name, report_response)

        self.outcome.decision = "1:1 complete"
        self.outcome.consensus_reached = True
        self.outcome.key_insights = [supervisor_response, report_response]
        self.outcome.action_items = [
            f"[{report.name}] {line.strip()}"
            for line in supervisor_response.split("\n")
            if "goal" in line.lower() or "improve" in line.lower()
        ][:5]

    def _clean_response(self, text: str) -> str:
        """Clean LLM response by removing thinking tags and artifacts."""
        import re

        if not text:
            return ""

        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove other artifacts
        text = re.sub(r'<\/?begin.*?>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\/?end.*?>', '', text, flags=re.IGNORECASE)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    async def _get_response(self, participant: MeetingParticipant, prompt: str) -> str:
        """Get a response from a participant."""
        try:
            # Resolve the model spec from the participant's role
            role_enum = next((r for r in AgentRole if r.value == participant.role), None)
            _mc = ModelConfig()
            model_spec = _mc.get_model(role_enum) if role_enum else next(iter(_mc.configs.values()))
            text, _, _ = get_llm_client().chat(
                model_spec,
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            return self._clean_response(text)
        except Exception as e:
            return f"[Error: {str(e)}]"

    def get_transcript(self) -> str:
        """Get the full meeting transcript."""
        lines = [
            f"=== Meeting: {self.topic} ===",
            f"Type: {self.meeting_type.value}",
            f"Participants: {', '.join(p.name for p in self.participants)}",
            f"Started: {self.started_at}",
            "",
            "=== Transcript ===",
        ]

        for msg in self.transcript:
            lines.append(f"[{msg.speaker}] ({msg.message_type}): {msg.content}")

        lines.extend([
            "",
            "=== Outcome ===",
            f"Decision: {self.outcome.decision}",
            f"Consensus: {self.outcome.consensus_reached}",
            f"Action Items: {len(self.outcome.action_items)}",
        ])

        return "\n".join(lines)


async def quick_meeting(
    topic: str,
    participants: List[Dict[str, str]],
    meeting_type: MeetingType = MeetingType.DECISION,
    context: str = ""
) -> MeetingOutcome:
    """
    Run a quick meeting with minimal setup.

    Args:
        topic: Meeting topic
        participants: List of {"name": ..., "role": ..., "model": ...}
        meeting_type: Type of meeting
        context: Additional context

    Returns:
        MeetingOutcome with decision and action items
    """
    meeting = AgentMeeting(meeting_type, topic)

    for p in participants:
        meeting.add_participant(
            name=p["name"],
            role=p["role"],
            model=p.get("model", "mistral:latest")
        )

    return await meeting.run(context)
