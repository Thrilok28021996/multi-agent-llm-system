"""Parsers that extract typed artifacts from LLM text responses.

Each parser takes raw LLM output and produces the corresponding typed artifact.
Parsers are intentionally lenient — they always return a valid artifact even if
the LLM output is malformed, falling back to sensible defaults.
"""
import re
from typing import List

from orchestrator.artifacts import (
    AcceptanceCriterion, RequirementsDoc,
    ArchitecturalDecision, Story, ArchitectureNote,
    QAReport, CriterionResult,
)


class RequirementsParser:
    """Parse PM LLM output into a RequirementsDoc."""

    _JTBD_RE = re.compile(
        r'(?:JOB[- ]TO[- ]BE[- ]DONE|JTBD)[:\s]*(.+?)(?=\n\n|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    _REQ_SPLIT_RE = re.compile(
        r'(?:REQUIREMENT\s*\[?\d+\]?|AC-\d+)[:\s.]',
        re.IGNORECASE,
    )
    _GIVEN_RE = re.compile(r'GIVEN[:\s]+(.+?)(?=WHEN[:\s]|\Z)', re.IGNORECASE | re.DOTALL)
    _WHEN_RE  = re.compile(r'WHEN[:\s]+(.+?)(?=THEN[:\s]|\Z)', re.IGNORECASE | re.DOTALL)
    _THEN_RE  = re.compile(r'THEN[:\s]+(.+?)(?=GIVEN[:\s]|REQUIREMENT|AC-\d|---|\n\n|\Z)', re.IGNORECASE | re.DOTALL)
    _VERIFY_RE = re.compile(
        r'(?:MEASURABLE_CRITERIA|VERIFY(?:ICATION)?|VERIFICATION)[:\s]+(.+?)(?=GIVEN|REQUIREMENT|AC-\d|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    _PRIORITY_RE = re.compile(r'PRIORITY[:\s]+(P[012])', re.IGNORECASE)
    _OUT_OF_SCOPE_RE = re.compile(
        r'OUT OF SCOPE[:\s]*\n((?:[-*]\s*.+\n?)+)',
        re.IGNORECASE,
    )

    def parse(self, text: str, problem_summary: str = "") -> RequirementsDoc:
        doc = RequirementsDoc(problem_summary=problem_summary, raw_text=text)

        m = self._JTBD_RE.search(text)
        if m:
            doc.jobs_to_be_done = m.group(1).strip()[:250]

        doc.acceptance_criteria = self._extract_criteria(text)

        m = self._OUT_OF_SCOPE_RE.search(text)
        if m:
            items = re.findall(r'[-*]\s*(.+)', m.group(1))
            doc.out_of_scope = [i.strip() for i in items][:10]

        return doc

    def _extract_criteria(self, text: str) -> List[AcceptanceCriterion]:
        criteria = []
        blocks = self._REQ_SPLIT_RE.split(text)

        for i, block in enumerate(blocks[1:], 1):
            given_m   = self._GIVEN_RE.search(block)
            when_m    = self._WHEN_RE.search(block)
            then_m    = self._THEN_RE.search(block)
            priority_m = self._PRIORITY_RE.search(block)
            verify_m  = self._VERIFY_RE.search(block)

            priority = priority_m.group(1) if priority_m else "P0"
            verify   = verify_m.group(1).strip()[:200] if verify_m else ""

            if given_m and when_m and then_m:
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{i}",
                    given=given_m.group(1).strip()[:200],
                    when=when_m.group(1).strip()[:200],
                    then=then_m.group(1).strip()[:200],
                    priority=priority,
                    verification_cmd=verify,
                ))
            else:
                # Fallback: use first non-empty sentence as description
                desc = block.strip()[:200]
                if not desc:
                    continue
                criteria.append(AcceptanceCriterion(
                    id=f"AC-{i}",
                    given="the system is running",
                    when=f"the user performs: {desc[:100]}",
                    then="the expected behaviour occurs as described",
                    priority=priority,
                    verification_cmd=verify,
                ))

            if len(criteria) >= 10:
                break

        return criteria


class ArchitectureParser:
    """Parse CTO LLM output into an ArchitectureNote with ADRs and sharded Stories."""

    _LANGUAGE_RE   = re.compile(r'LANGUAGE[:\s]+(\w[\w+#]*)', re.IGNORECASE)
    _FRAMEWORK_RE  = re.compile(r'FRAMEWORK[:\s]+(.+?)(?:\n|$)', re.IGNORECASE)
    _ENTRY_RE      = re.compile(r'ENTRY POINT[:\s]+`?(.+?)`?(?:\s+-\s+.+)?(?:\n|$)', re.IGNORECASE)
    _LIBRARIES_RE  = re.compile(r'KEY LIBRARIES[:\s]+(.+?)(?:\n|$)', re.IGNORECASE)
    _FILES_BLOCK_RE = re.compile(r'FILES?[:\s]*\n((?:\d+\.\s+`[^`]+`[^\n]*\n?)+)', re.IGNORECASE)
    _FILE_LINE_RE  = re.compile(r'`([^`]+)`\s*[-–]?\s*(.+?)(?:\n|$)')
    _ADR_SPLIT_RE  = re.compile(r'ADR-(\d+)[:\s]', re.IGNORECASE)
    _STORY_SPLIT_RE = re.compile(r'STORY-?(\d+)[:\s]', re.IGNORECASE)

    def parse(self, text: str) -> ArchitectureNote:
        note = ArchitectureNote(raw_text=text)

        m = self._LANGUAGE_RE.search(text)
        if m:
            note.language = m.group(1).strip()

        m = self._FRAMEWORK_RE.search(text)
        if m:
            fw = m.group(1).strip()
            note.framework = "" if fw.lower() in ("none", "n/a", "-") else fw[:60]

        m = self._ENTRY_RE.search(text)
        if m:
            note.entry_point = m.group(1).strip()

        m = self._LIBRARIES_RE.search(text)
        if m:
            note.key_libraries = [lib.strip() for lib in m.group(1).split(",")][:8]

        note.architecture_decisions = self._extract_adrs(text)

        # Check if LLM produced explicit STORY-N blocks; otherwise auto-shard
        if self._STORY_SPLIT_RE.search(text):
            note.stories = self._parse_story_blocks(text, note)
        else:
            all_files = self._extract_files(text)
            note.stories = self._auto_shard(all_files, note)

        return note

    def _extract_adrs(self, text: str) -> List[ArchitecturalDecision]:
        adrs = []
        parts = self._ADR_SPLIT_RE.split(text)

        for i in range(1, len(parts), 2):
            adr_num = parts[i].strip()
            block   = parts[i + 1] if i + 1 < len(parts) else ""
            lines   = block.strip().split("\n")
            title   = lines[0].strip()[:80] if lines else f"ADR-{adr_num}"
            rest    = "\n".join(lines[1:]) if len(lines) > 1 else block

            def _extract(pattern: str) -> str:
                m = re.search(pattern, rest, re.IGNORECASE | re.DOTALL)
                return m.group(1).strip()[:300] if m else ""

            adrs.append(ArchitecturalDecision(
                id=f"ADR-{adr_num}",
                title=title,
                decision=_extract(r'Decision[:\s]+(.+?)(?=Rationale|Consequences|Alternatives|ADR-|\Z)'),
                rationale=_extract(r'Rationale[:\s]+(.+?)(?=Decision|Consequences|Alternatives|ADR-|\Z)'),
                consequences=_extract(r'Consequences[:\s]+(.+?)(?=Decision|Rationale|Alternatives|ADR-|\Z)'),
                alternatives_rejected=_extract(r'Alternatives? Rejected[:\s]+(.+?)(?=Decision|Rationale|Consequences|ADR-|\Z)'),
            ))

            if len(adrs) >= 5:
                break

        return adrs

    def _extract_files(self, text: str) -> List[str]:
        files: List[str] = []

        m = self._FILES_BLOCK_RE.search(text)
        if m:
            for fm in self._FILE_LINE_RE.finditer(m.group(1)):
                files.append(fm.group(1).strip())

        if not files:
            # Fallback: backtick-quoted paths that look like filenames
            for fm in re.finditer(
                r'`([\w./\-]+\.(?:py|js|ts|go|rs|rb|md|txt|json|yaml|toml|cfg|sh|css|html))`',
                text,
            ):
                p = fm.group(1)
                if p not in files:
                    files.append(p)

        return files[:20]

    def _parse_story_blocks(self, text: str, note: "ArchitectureNote") -> List[Story]:
        """Parse explicit STORY-N: blocks that the CTO generated."""
        stories = []
        parts = self._STORY_SPLIT_RE.split(text)

        for i in range(1, len(parts), 2):
            story_num = parts[i].strip()
            block     = parts[i + 1] if i + 1 < len(parts) else ""
            lines     = block.strip().split("\n")
            title     = lines[0].strip()[:80] if lines else f"Story {story_num}"
            rest      = "\n".join(lines[1:]) if len(lines) > 1 else block

            files = [f.strip() for f in re.findall(r'`([^`]+\.\w+)`', rest)][:10]

            ac_ids: List[str] = []
            covers_m = re.search(r'Covers?[:\s]+(.+?)(?:\n|\Z)', rest, re.IGNORECASE)
            if covers_m:
                ac_ids = re.findall(r'AC-\d+', covers_m.group(1), re.IGNORECASE)

            deps: List[str] = []
            dep_m = re.search(r'Depends? on[:\s]+(.+?)(?:\n|\Z)', rest, re.IGNORECASE)
            if dep_m:
                deps = [f"story-{d}" for d in re.findall(r'\d+', dep_m.group(1))]

            tech_m = re.search(r'Context[:\s]+(.+?)(?:\n|\Z)', rest, re.IGNORECASE)
            tech = tech_m.group(1).strip()[:200] if tech_m else (
                f"Language: {note.language}, Framework: {note.framework}"
            )

            stories.append(Story(
                story_id=f"story-{story_num}",
                title=title,
                description=rest[:400].strip(),
                files=files,
                ac_ids=ac_ids,
                tech_context=tech,
                dependencies=deps,
            ))

            if len(stories) >= 5:
                break

        return stories

    def _auto_shard(self, all_files: List[str], note: "ArchitectureNote") -> List[Story]:
        """Auto-shard files into logical story clusters when no STORY blocks exist."""
        if not all_files:
            return [Story(
                story_id="story-1",
                title="Full Implementation",
                description=note.raw_text[:500].strip(),
                tech_context=f"Language: {note.language}, Framework: {note.framework}",
            )]

        core, config, tests = [], [], []
        for f in all_files:
            fl = f.lower()
            if any(x in fl for x in ("test", "spec", "_test.")):
                tests.append(f)
            elif any(x in fl for x in (
                "config", "settings", "requirements", ".toml", ".yaml",
                ".cfg", ".env", ".json", "dockerfile", "makefile",
            )):
                config.append(f)
            else:
                core.append(f)

        stories: List[Story] = []
        tech_base = f"Language: {note.language}"
        if note.framework:
            tech_base += f", Framework: {note.framework}"
        if note.key_libraries:
            tech_base += f", Libraries: {', '.join(note.key_libraries[:4])}"

        if core:
            stories.append(Story(
                story_id="story-1",
                title="Core Implementation",
                description=(
                    f"Implement the core functionality.\n"
                    f"Entry point: {note.entry_point or 'see architecture above'}"
                ),
                files=core,
                tech_context=tech_base,
            ))

        if config:
            stories.append(Story(
                story_id=f"story-{len(stories)+1}",
                title="Configuration & Dependencies",
                description="Create all configuration files, dependency manifests, and environment setup.",
                files=config,
                tech_context=f"Language: {note.language}",
                dependencies=[stories[0].story_id] if stories else [],
            ))

        if tests:
            stories.append(Story(
                story_id=f"story-{len(stories)+1}",
                title="Tests",
                description="Write tests for the core implementation.",
                files=tests,
                tech_context=tech_base,
                dependencies=[s.story_id for s in stories],
            ))

        return stories or [Story(
            story_id="story-1",
            title="Full Implementation",
            description=note.raw_text[:500].strip(),
            files=all_files,
            tech_context=tech_base,
        )]


class QAReportParser:
    """Parse QA LLM output into a QAReport."""

    _VERDICT_RE   = re.compile(r'\b(PASS_WITH_ISSUES|PASS_WITH_NOTES|PASS|FAIL)\b', re.IGNORECASE)
    _AC_STATUS_RE = re.compile(r'(AC-\d+)[:\s]+(?:status[:\s]+)?(PASS|FAIL|UNTESTED)', re.IGNORECASE)
    _CRITICAL_RE  = re.compile(
        r'CRITICAL[:\s]+(.+?)(?=MAJOR|MINOR|TRIVIAL|PASS|FAIL|\n\n|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    _MINOR_RE = re.compile(
        r'(?:MINOR|TRIVIAL)[:\s]+(.+?)(?=CRITICAL|MAJOR|PASS|FAIL|\n\n|\Z)',
        re.IGNORECASE | re.DOTALL,
    )

    def parse(self, text: str, ac_ids: List[str] = None) -> QAReport:
        report = QAReport(raw_text=text)

        m = self._VERDICT_RE.search(text)
        if m:
            v = m.group(1).upper()
            report.verdict = "PASS_WITH_ISSUES" if v in ("PASS_WITH_ISSUES", "PASS_WITH_NOTES") else v

        results: dict = {}
        for m in self._AC_STATUS_RE.finditer(text):
            ac_id  = m.group(1).upper()
            status = m.group(2).upper()
            start  = max(0, m.start() - 50)
            end    = min(len(text), m.end() + 150)
            results[ac_id] = CriterionResult(
                criterion_id=ac_id,
                status=status,
                evidence=text[start:end].strip()[:300],
            )

        if ac_ids:
            for ac_id in ac_ids:
                if ac_id not in results:
                    results[ac_id] = CriterionResult(
                        criterion_id=ac_id,
                        status="UNTESTED",
                        evidence="Not explicitly tested in QA report.",
                    )

        report.criterion_results = list(results.values())

        for m in self._CRITICAL_RE.finditer(text):
            report.critical_issues.append(m.group(1).strip()[:200])
        for m in self._MINOR_RE.finditer(text):
            report.minor_issues.append(m.group(1).strip()[:200])

        return report
