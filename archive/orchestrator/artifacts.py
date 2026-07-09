"""Typed artifact envelopes for the spec-driven workflow.

Artifact chain (Minimal BMAD + Typed Envelopes):
  Researcher  → (existing research artifacts)
  PM          → RequirementsDoc  — GIVEN/WHEN/THEN acceptance criteria
  CTO         → ArchitectureNote — ADRs + sharded Story list
  Developer   → consumes one Story at a time (keeps prompt <600 tokens)
  QA          → QAReport         — maps each AC to PASS/FAIL/UNTESTED
  CEO         → reads digest of all artifacts

All dataclasses expose to_dict() for storage in WorkflowState.artifacts
and format_*() helpers that produce compact prompt-ready strings.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# PM artifact
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceCriterion:
    """A single testable GIVEN/WHEN/THEN criterion."""
    id: str                      # "AC-1", "AC-2", ...
    given: str                   # precondition / context
    when: str                    # action taken
    then: str                    # expected observable outcome
    priority: str = "P0"         # P0=must, P1=should, P2=nice-to-have
    verification_cmd: str = ""   # exact CLI command to verify (optional)


@dataclass
class RequirementsDoc:
    """PM artifact — structured product requirements with GIVEN/WHEN/THEN criteria."""
    artifact_type: str = "requirements_doc"
    schema_version: str = "1.0"
    produced_by: str = "pm"
    problem_summary: str = ""
    jobs_to_be_done: str = ""    # When [situation], I want [motivation], so I can [outcome]
    acceptance_criteria: List[AcceptanceCriterion] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    raw_text: str = ""           # full LLM response preserved for reference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "produced_by": self.produced_by,
            "problem_summary": self.problem_summary,
            "jobs_to_be_done": self.jobs_to_be_done,
            "acceptance_criteria": [
                {
                    "id": ac.id,
                    "given": ac.given,
                    "when": ac.when,
                    "then": ac.then,
                    "priority": ac.priority,
                    "verification_cmd": ac.verification_cmd,
                }
                for ac in self.acceptance_criteria
            ],
            "out_of_scope": self.out_of_scope,
            "raw_text": self.raw_text,
        }

    def format_for_agent(self, max_criteria: int = 10) -> str:
        """Compact prompt-ready representation (target ~600 tokens)."""
        lines = [f"PROBLEM: {self.problem_summary}"]
        if self.jobs_to_be_done:
            lines.append(f"JOB-TO-BE-DONE: {self.jobs_to_be_done}")
        lines.append("\nACCEPTANCE CRITERIA:")
        for ac in self.acceptance_criteria[:max_criteria]:
            lines.append(f"\n{ac.id} [{ac.priority}]:")
            lines.append(f"  GIVEN: {ac.given}")
            lines.append(f"  WHEN:  {ac.when}")
            lines.append(f"  THEN:  {ac.then}")
            if ac.verification_cmd:
                lines.append(f"  VERIFY: {ac.verification_cmd}")
        if self.out_of_scope:
            lines.append("\nOUT OF SCOPE:")
            for item in self.out_of_scope[:5]:
                lines.append(f"  - {item}")
        return "\n".join(lines)

    def p0_criteria_ids(self) -> List[str]:
        """Return IDs of all P0 (must-have) acceptance criteria."""
        return [ac.id for ac in self.acceptance_criteria if ac.priority == "P0"]


# ---------------------------------------------------------------------------
# CTO artifacts
# ---------------------------------------------------------------------------

@dataclass
class ArchitecturalDecision:
    """A single Architecture Decision Record (ADR)."""
    id: str            # "ADR-1"
    title: str
    decision: str
    rationale: str
    consequences: str
    alternatives_rejected: str = ""


@dataclass
class Story:
    """
    Atomic, self-contained work item for the Developer (BMAD story-sharding pattern).
    Kept small (<600 tokens when formatted) so a 7-8B model can implement it reliably
    without seeing the full architecture document.
    """
    artifact_type: str = "story"
    schema_version: str = "1.0"
    produced_by: str = "cto"
    story_id: str = ""                               # "story-1", "story-2", ...
    title: str = ""
    description: str = ""                            # what to build
    files: List[str] = field(default_factory=list)  # files this story touches
    ac_ids: List[str] = field(default_factory=list) # AC-N IDs this story covers
    tech_context: str = ""                           # language, framework, data flow snippet
    dependencies: List[str] = field(default_factory=list)  # other story_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "produced_by": self.produced_by,
            "story_id": self.story_id,
            "title": self.title,
            "description": self.description,
            "files": self.files,
            "ac_ids": self.ac_ids,
            "tech_context": self.tech_context,
            "dependencies": self.dependencies,
        }

    def format_for_developer(self, requirements_doc: Optional[RequirementsDoc] = None) -> str:
        """Self-contained developer prompt (target <600 tokens)."""
        lines = [
            f"STORY: {self.story_id} — {self.title}",
            f"\n{self.description}",
        ]
        if self.tech_context:
            lines.append(f"\nTECH CONTEXT:\n{self.tech_context}")
        if self.files:
            lines.append("\nFILES TO CREATE:")
            for f in self.files:
                lines.append(f"  - {f}")
        if self.ac_ids and requirements_doc:
            ac_map = {ac.id: ac for ac in requirements_doc.acceptance_criteria}
            relevant = [ac_map[aid] for aid in self.ac_ids if aid in ac_map]
            if relevant:
                lines.append("\nACCEPTANCE CRITERIA TO SATISFY:")
                for ac in relevant:
                    lines.append(f"  {ac.id}: GIVEN {ac.given} / WHEN {ac.when} / THEN {ac.then}")
                    if ac.verification_cmd:
                        lines.append(f"         VERIFY: {ac.verification_cmd}")
        return "\n".join(lines)


@dataclass
class ArchitectureNote:
    """CTO artifact — ADRs + sharded story list."""
    artifact_type: str = "architecture_note"
    schema_version: str = "1.0"
    produced_by: str = "cto"
    language: str = ""
    framework: str = ""
    key_libraries: List[str] = field(default_factory=list)
    entry_point: str = ""
    architecture_decisions: List[ArchitecturalDecision] = field(default_factory=list)
    stories: List[Story] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "produced_by": self.produced_by,
            "language": self.language,
            "framework": self.framework,
            "key_libraries": self.key_libraries,
            "entry_point": self.entry_point,
            "architecture_decisions": [
                {
                    "id": adr.id,
                    "title": adr.title,
                    "decision": adr.decision,
                    "rationale": adr.rationale,
                    "consequences": adr.consequences,
                    "alternatives_rejected": adr.alternatives_rejected,
                }
                for adr in self.architecture_decisions
            ],
            "stories": [s.to_dict() for s in self.stories],
            "raw_text": self.raw_text,
        }


# ---------------------------------------------------------------------------
# QA artifact
# ---------------------------------------------------------------------------

@dataclass
class CriterionResult:
    """QA result for a single acceptance criterion."""
    criterion_id: str
    status: str      # "PASS", "FAIL", "UNTESTED"
    evidence: str    # what was observed
    notes: str = ""


@dataclass
class QAReport:
    """QA artifact — maps each acceptance criterion to a test result."""
    artifact_type: str = "qa_report"
    schema_version: str = "1.0"
    produced_by: str = "qa"
    verdict: str = "FAIL"          # "PASS", "PASS_WITH_ISSUES", "FAIL"
    criterion_results: List[CriterionResult] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    minor_issues: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "produced_by": self.produced_by,
            "verdict": self.verdict,
            "criterion_results": [
                {
                    "criterion_id": r.criterion_id,
                    "status": r.status,
                    "evidence": r.evidence,
                    "notes": r.notes,
                }
                for r in self.criterion_results
            ],
            "critical_issues": self.critical_issues,
            "minor_issues": self.minor_issues,
            "raw_text": self.raw_text,
        }

    def passed_count(self) -> int:
        return sum(1 for r in self.criterion_results if r.status == "PASS")

    def failed_count(self) -> int:
        return sum(1 for r in self.criterion_results if r.status == "FAIL")

    def coverage_summary(self) -> str:
        total = len(self.criterion_results)
        passed = self.passed_count()
        failed = self.failed_count()
        untested = total - passed - failed
        return f"{passed}/{total} passed, {failed} failed, {untested} untested"
