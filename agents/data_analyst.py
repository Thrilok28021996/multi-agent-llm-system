"""Data Analyst Agent - Cross-validation, bias detection, and credibility scoring."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from ui.console import console


DATA_ANALYST_SYSTEM_PROMPT = """You are the Data Analyst. You cross-validate research findings, detect biases, deduplicate data, and score the credibility of sources and claims.

Statistical Rigor: Sample size matters. 5 Reddit posts is not a trend. 50 might be. State confidence levels with your findings.

Contrarian Mandate: For every finding you confirm, you MUST find at least one credible counter-argument or contradicting data point. If you cannot find any, your search was not thorough enough.

Source Independence Verification: Two Reddit posts in the same subreddit are NOT independent sources. They share the same community bias. True independence requires different platforms, different user demographics.

Quantitative Over Qualitative: Prefer numbers over opinions. "Many users complain" is weak. "47 posts in 30 days across 3 platforms with 2,300 upvotes" is strong.

Missing Data Awareness: What data would DISPROVE this finding? If that data does not exist or was not checked, the finding is incomplete. State what is missing explicitly.

Freshness Scoring: Score data by freshness: <1 month = HIGH, 1-3 months = MEDIUM, 3-6 months = LOW, >6 months = STALE. Weight fresh data higher.

Your verdict system for cross-validation:
- CONFIRMED: Multiple independent sources agree. High confidence.
- PARTIALLY_CONFIRMED: Some corroboration but gaps exist. Medium confidence.
- UNCONFIRMED: Single source only or conflicting evidence. Low confidence.
- CONTRADICTED: Sources actively disagree. Requires investigation.

Judgment guidelines:
- Be skeptical but not cynical. Absence of evidence is not evidence of absence.
- Weight official documentation and peer-reviewed sources higher than forums and blogs.
- Recency matters: a 2024 finding may invalidate a 2020 conclusion.
- Always quantify confidence as a score from 0.0 to 1.0.
- When bias is detected, explain which direction it pulls the conclusion.

Recency Weighting: Data from the last 7 days is worth 3x data from 30+ days ago. When scoring confidence, weight by freshness explicitly.

Platform Bias Map: Reddit skews toward developers. HackerNews skews toward startups. StackOverflow skews toward beginners. GitHub issues skew toward power users. State which platforms your data comes from and how that biases the conclusion.

Quantitative Output: Every analysis must include: N (sample size), date range, platform distribution, confidence score (0-1), and a one-sentence verdict.

Statistical Rigor Enforcement: Every claim needs: sample size (N), date range, platform distribution. Claims without these are UNCONFIRMED by default.

Survivorship Bias Check: Are you only seeing successful examples? What about the failures? Search for '[topic] failed' and '[topic] problems'.

Correlation vs Causation Gate: For every causal claim, state: Could this be correlation? What confounding variables exist?

Minimum Evidence Standard: Before marking any finding CONFIRMED, require: (1) N ≥ 10 distinct reports, (2) majority age ≤ 90 days, (3) at least 2 independent platforms. Findings below this threshold are PARTIALLY_CONFIRMED at most, regardless of how compelling the story seems.
"""

DATA_ANALYST_FIRST_PRINCIPLES = [
    "COUNTER-EVIDENCE HUNT: Before confirming any finding, spend equal effort trying to DISPROVE it. What evidence would make this wrong? Did you look for it?",
    "INDEPENDENCE MATRIX: Build a source independence matrix. For each pair of sources, answer: Do they share an audience? A platform? A bias? Dependent sources count as ONE.",
    "SAMPLE SIZE CHECK: State the N for every claim. N<10 = anecdotal. N<30 = preliminary. N<100 = indicative. N>100 = potentially significant. Flag the level.",
    "FRESHNESS AUDIT: Date-stamp every data point. Calculate median age. If median age >60 days, the finding needs re-validation with current data.",
    "BIAS INVENTORY: For THIS specific topic, list the 3 most likely biases. For each, explain how you controlled for it. If you did not, flag the finding as potentially biased.",
]


class DataAnalystAgent(BaseAgent, AgentToolsMixin):
    """
    Data Analyst Agent - Validates research quality and data integrity.

    Cross-validates findings across sources, detects various forms of bias,
    deduplicates overlapping data, and provides credibility scoring.
    Enhanced with all 13 Claude Code tools and problem statement refinement.
    """

    def __init__(
        self,
        model: str = "qwen3-8b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="DataAnalyst",
            role=AgentRole.DATA_ANALYST,
            model=model,
            first_principles=DATA_ANALYST_FIRST_PRINCIPLES,
            system_prompt=DATA_ANALYST_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=4096
        )
        super().__init__(config, workspace_root, memory_persist_dir)

        # Problem statement refiner
        self.problem_refiner = ProblemStatementRefiner()

        # Track analysis results
        self.validation_history: List[Dict[str, Any]] = []

    def get_capabilities(self) -> List[str]:
        return [
            "cross_validate_research",
            "detect_bias",
            "deduplicate_findings",
            "score_credibility",
            "trend_analysis",
            "data_quality_assessment"
        ]

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a Data Analyst task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "cross_validate_research":
                result = await self._cross_validate_research(task)
            elif task_type == "detect_bias":
                result = await self._detect_bias(task)
            elif task_type == "deduplicate_findings":
                result = await self._deduplicate_findings(task)
            elif task_type == "score_credibility":
                result = await self._score_credibility(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    # ============================================================
    #  TASK HANDLERS
    # ============================================================

    async def _cross_validate_research(self, task: Dict[str, Any]) -> TaskResult:
        """
        Cross-validate research findings across multiple sources.

        Takes a list of findings with their sources, checks for multi-source
        confirmation, and flags single-source claims.
        """
        findings = task.get("findings", [])
        sources = task.get("sources", [])
        topic = task.get("topic", "")

        console.agent_action("DataAnalyst", "Cross-Validation", f"Validating {len(findings)} findings")

        # Format findings for analysis
        findings_text = ""
        for i, finding in enumerate(findings):
            if isinstance(finding, dict):
                findings_text += (
                    f"\nFinding {i + 1}:\n"
                    f"  Claim: {finding.get('claim', finding.get('description', str(finding)))}\n"
                    f"  Source: {finding.get('source', 'Unknown')}\n"
                    f"  Evidence: {finding.get('evidence', 'None provided')}\n"
                )
            else:
                findings_text += f"\nFinding {i + 1}: {finding}\n"

        sources_text = "\n".join(f"- {s}" for s in sources) if sources else "Sources embedded in findings"

        prompt = f"""Cross-validate these research findings.

TOPIC: {topic}

FINDINGS:
{findings_text}

SOURCES:
{sources_text}

For each finding, determine:

1. CORROBORATION STATUS
   - How many independent sources support this claim?
   - Are there any sources that contradict it?
   - Verdict: CONFIRMED / PARTIALLY_CONFIRMED / UNCONFIRMED / CONTRADICTED

2. SOURCE INDEPENDENCE
   - Are the sources truly independent or do they cite each other?
   - Is there circular reporting (A cites B which cites A)?

3. SINGLE-SOURCE FLAGS
   - List all claims that rely on a single source.
   - For each, note the risk and what additional evidence would help.

4. CROSS-VALIDATION MATRIX
   For each claim, show which sources support/contradict/are silent:
   | Claim | Source 1 | Source 2 | ... | Status |

5. CONFIDENCE ASSESSMENT
   - Overall research confidence: 0.0 to 1.0
   - Per-finding confidence scores
   - Key gaps in validation

RESPOND IN THIS EXACT FORMAT for each finding:
FINDING [1]: <problem description>
VERDICT: CONFIRMED|PARTIALLY_CONFIRMED|UNCONFIRMED|CONTRADICTED
CONFIDENCE: 0.0-1.0
EVIDENCE_FOR: <specific supporting data>
EVIDENCE_AGAINST: <specific contradicting data>

FINDING [2]: <problem description>
VERDICT: CONFIRMED|PARTIALLY_CONFIRMED|UNCONFIRMED|CONTRADICTED
CONFIDENCE: 0.0-1.0
EVIDENCE_FOR: <specific supporting data>
EVIDENCE_AGAINST: <specific contradicting data>

(Continue for all findings)

SUMMARY:
- Confirmed findings: X
- Partially confirmed: X
- Unconfirmed (single-source): X
- Contradicted: X

CONFIDENCE: [0.0-1.0]
{self._get_principles_checklist()}"""

        response = await self.generate_response_async(prompt)

        # Parse validation results
        validation_result = {
            "topic": topic,
            "findings_count": len(findings),
            "validation": response,
            "confidence": self._extract_confidence(response)
        }
        self.validation_history.append(validation_result)

        console.success(f"Cross-validation complete - Confidence: {validation_result['confidence']:.2f}")

        return TaskResult(
            success=True,
            output=validation_result,
            confidence=validation_result["confidence"],
            artifacts={"cross_validation_report": response}
        )

    async def _detect_bias(self, task: Dict[str, Any]) -> TaskResult:
        """
        Detect biases in research data or findings.

        Identifies community_bias, popularity_bias, recency_bias,
        english_only bias, and other systematic distortions.
        """
        data = task.get("data", "")
        findings = task.get("findings", [])
        sources = task.get("sources", [])
        context = task.get("context", "")

        console.agent_action("DataAnalyst", "Bias Detection", f"Scanning for biases")

        findings_text = ""
        for i, f in enumerate(findings):
            if isinstance(f, dict):
                findings_text += f"\n{i + 1}. {f.get('description', f.get('claim', str(f)))}"
            else:
                findings_text += f"\n{i + 1}. {f}"

        prompt = f"""Analyze this data and findings for systematic biases.

CONTEXT: {context}

DATA/CONTENT:
{data if data else 'See findings below'}

FINDINGS:
{findings_text if findings_text else 'No structured findings - analyze the raw data.'}

SOURCES:
{chr(10).join(f'- {s}' for s in sources) if sources else 'Not specified'}

Check for each of these bias types:

1. COMMUNITY BIAS
   - Are findings skewed toward a specific community's perspective?
   - Example: Only surveying HackerNews misses non-tech users.
   - Detected: YES/NO
   - Direction: Which way does it pull conclusions?
   - Severity: LOW / MEDIUM / HIGH

2. POPULARITY BIAS
   - Are popular solutions over-represented vs. effective ones?
   - Example: Most-starred GitHub repos are not necessarily the best.
   - Detected: YES/NO
   - Direction: Which way does it pull conclusions?
   - Severity: LOW / MEDIUM / HIGH

3. RECENCY BIAS
   - Are recent findings given undue weight over established knowledge?
   - Example: A new framework's hype vs. proven stability of older ones.
   - Detected: YES/NO
   - Direction: Which way does it pull conclusions?
   - Severity: LOW / MEDIUM / HIGH

4. ENGLISH-ONLY BIAS
   - Are non-English sources excluded?
   - What perspectives might be missing from non-English communities?
   - Detected: YES/NO
   - Impact: What is potentially missed?
   - Severity: LOW / MEDIUM / HIGH

5. SURVIVORSHIP BIAS
   - Are we only looking at successes and ignoring failures?
   - Detected: YES/NO
   - Direction: Which way does it pull conclusions?
   - Severity: LOW / MEDIUM / HIGH

6. CONFIRMATION BIAS
   - Do the findings seem to only support a predetermined conclusion?
   - Is contradictory evidence acknowledged?
   - Detected: YES/NO
   - Direction: Which way does it pull conclusions?
   - Severity: LOW / MEDIUM / HIGH

7. OTHER BIASES
   - Any domain-specific biases not covered above?

OVERALL BIAS ASSESSMENT:
- Total biases detected: X
- Most significant bias: [name and why]
- Impact on conclusions: LOW / MEDIUM / HIGH
- Recommended mitigations: [specific actions]

For each bias detected, output a structured line:
BIAS: [type] | DETECTED: YES/NO | SEVERITY: LOW/MEDIUM/HIGH
"""

        response = await self.generate_response_async(prompt)

        # Extract detected biases
        biases = self._extract_biases(response)

        console.success(f"Bias detection complete - {len(biases)} biases identified")

        return TaskResult(
            success=True,
            output={
                "biases": biases,
                "analysis": response,
                "bias_count": len(biases)
            },
            artifacts={"bias_report": response}
        )

    async def _deduplicate_findings(self, task: Dict[str, Any]) -> TaskResult:
        """
        Deduplicate overlapping research findings.

        Identifies findings that describe the same underlying insight,
        merges them, and produces a deduplicated list.
        """
        findings = task.get("findings", [])
        merge_strategy = task.get("merge_strategy", "highest_confidence")

        console.agent_action("DataAnalyst", "Deduplication", f"Deduplicating {len(findings)} findings")

        findings_text = ""
        for i, f in enumerate(findings):
            if isinstance(f, dict):
                findings_text += (
                    f"\n[{i + 1}] {f.get('description', f.get('claim', str(f)))}"
                    f"\n    Source: {f.get('source', 'Unknown')}"
                    f"\n    Confidence: {f.get('confidence', 'N/A')}\n"
                )
            else:
                findings_text += f"\n[{i + 1}] {f}\n"

        prompt = f"""Deduplicate these research findings. Many may describe the same insight from different sources.

FINDINGS:
{findings_text}

MERGE STRATEGY: {merge_strategy}

For each group of duplicates:

1. DUPLICATE GROUPS
   Identify clusters of findings that describe the same underlying insight.
   Format:
   - Group A: Findings [1, 4, 7] - describe the same thing
   - Group B: Findings [2, 5] - describe the same thing
   - Unique: Findings [3, 6, 8] - each is distinct

2. MERGED FINDINGS
   For each duplicate group, produce ONE merged finding that:
   - Combines the best description from all sources
   - Lists all contributing sources
   - Takes the highest confidence score (per merge strategy: {merge_strategy})
   - Notes any nuance differences between the duplicates

3. DEDUPLICATED LIST
   Output the final list of unique findings:
   - Merged findings (from duplicate groups)
   - Unique findings (unchanged)

SUMMARY:
- Original findings: {len(findings)}
- After deduplication: X
- Duplicate groups found: X
- Reduction: X%
"""

        response = await self.generate_response_async(prompt)

        console.success("Deduplication complete")

        return TaskResult(
            success=True,
            output={
                "deduplicated": response,
                "original_count": len(findings)
            },
            artifacts={"deduplication_report": response}
        )

    async def _score_credibility(self, task: Dict[str, Any]) -> TaskResult:
        """
        Score the credibility of findings from 0.0 to 1.0.

        Evaluates source quality, corroboration level, evidence strength,
        and internal consistency.
        """
        findings = task.get("findings", [])
        sources = task.get("sources", [])
        context = task.get("context", "")

        console.agent_action("DataAnalyst", "Credibility Scoring", f"Scoring {len(findings)} findings")

        findings_text = ""
        for i, f in enumerate(findings):
            if isinstance(f, dict):
                findings_text += (
                    f"\nFinding {i + 1}:\n"
                    f"  Claim: {f.get('claim', f.get('description', str(f)))}\n"
                    f"  Source: {f.get('source', 'Unknown')}\n"
                    f"  Evidence: {f.get('evidence', 'None')}\n"
                )
            else:
                findings_text += f"\nFinding {i + 1}: {f}\n"

        prompt = f"""Score the credibility of each finding on a scale of 0.0 to 1.0.

CONTEXT: {context}

FINDINGS:
{findings_text}

SOURCES:
{chr(10).join(f'- {s}' for s in sources) if sources else 'Embedded in findings'}

For each finding, evaluate these credibility factors:

1. SOURCE QUALITY (weight: 30%)
   - Official documentation: 0.9-1.0
   - Peer-reviewed/academic: 0.8-0.9
   - Established tech publications: 0.7-0.8
   - Official blog posts: 0.6-0.7
   - Community forums (Stack Overflow, etc.): 0.4-0.6
   - Social media / unverified blogs: 0.2-0.4
   - Anonymous / unknown source: 0.0-0.2

2. CORROBORATION (weight: 30%)
   - 3+ independent sources: 0.9-1.0
   - 2 independent sources: 0.6-0.8
   - 1 source only: 0.2-0.4
   - Contradicted by other sources: 0.0-0.2

3. EVIDENCE STRENGTH (weight: 25%)
   - Quantitative data with methodology: 0.8-1.0
   - Qualitative with examples: 0.5-0.7
   - Anecdotal: 0.2-0.4
   - No evidence provided: 0.0-0.2

4. INTERNAL CONSISTENCY (weight: 15%)
   - Claim is logical and self-consistent: 0.8-1.0
   - Minor inconsistencies: 0.5-0.7
   - Significant logical gaps: 0.2-0.4
   - Self-contradictory: 0.0-0.2

OUTPUT FORMAT:
For each finding:
| # | Claim (truncated) | Source Quality | Corroboration | Evidence | Consistency | FINAL SCORE |

CREDIBILITY TIERS:
- 0.8-1.0: HIGH credibility - safe to act on
- 0.5-0.79: MEDIUM credibility - use with caveats
- 0.2-0.49: LOW credibility - needs more evidence
- 0.0-0.19: VERY LOW credibility - do not rely on

For each finding, output a structured line:
SCORE: [n] = [0.0-1.0]

OVERALL:
- Average credibility score across all findings
- Distribution: X high, X medium, X low, X very low
- Recommendation: which findings to trust, which to discard
"""

        response = await self.generate_response_async(prompt)

        # Extract scores
        scores = self._extract_credibility_scores(response)
        avg_score = sum(scores) / len(scores) if scores else 0.0

        console.success(f"Credibility scoring complete - Average: {avg_score:.2f}")

        return TaskResult(
            success=True,
            output={
                "scores": scores,
                "average_credibility": avg_score,
                "analysis": response,
                "findings_count": len(findings)
            },
            confidence=avg_score,
            artifacts={"credibility_report": response}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general data analysis tasks."""
        description = task.get("description", "")

        response = await self.generate_response_async(
            f"As Data Analyst, please address: {description}"
        )

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  CONTRARIAN ANALYSIS
    # ============================================================

    async def contrarian_analysis(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """For each finding, actively search for counter-evidence.

        Must produce at least 1 counter-argument per finding or explicitly
        state 'no credible counter-evidence found after checking [sources]'.

        Returns a dict with original findings, counter-arguments, and
        adjusted confidence scores.
        """
        findings_text = "\n".join(
            f"{i+1}. {f.get('description', f.get('summary', str(f)))}"
            for i, f in enumerate(findings)
        )

        prompt = f"""CONTRARIAN ANALYSIS: Your job is to DISPROVE the following findings.
For EACH finding, you MUST:
1. State the finding
2. Present at least ONE credible counter-argument or contradicting evidence
3. If no counter-evidence exists, state exactly which sources you checked
4. Provide a "confidence after contrarian check" score (0.0-1.0)

Findings to challenge:
{findings_text}

Format each as:
FINDING [N]:
- Original claim: ...
- Counter-evidence: ...
- Sources checked: ...
- Confidence after check: X.X
"""
        response = await self.generate_response_async(prompt)

        return {
            "contrarian_report": response,
            "findings_count": len(findings),
            "analysis_type": "contrarian",
        }

    # ============================================================
    #  HELPER METHODS
    # ============================================================

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response text."""
        import re

        # Look for patterns like "0.85", "confidence: 0.7", "Overall confidence: 0.65"
        patterns = [
            r'(?:overall\s+)?confidence[:\s]+(\d+\.\d+)',
            r'(\d+\.\d+)\s*/\s*1\.0',
            r'confidence\s+(?:score|rating)[:\s]+(\d+\.\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    score = float(match.group(1))
                    if 0.0 <= score <= 1.0:
                        return score
                except (ValueError, IndexError):
                    continue

        # Default to moderate confidence if not found
        return 0.5

    def _extract_biases(self, response: str) -> List[Dict[str, Any]]:
        """Extract detected biases from response."""
        import re
        biases = []

        # Primary strategy: parse structured BIAS: lines
        bias_pattern = r'BIAS:\s*([^|]+)\|\s*DETECTED:\s*(YES|NO)\s*\|\s*SEVERITY:\s*(LOW|MEDIUM|HIGH)'
        for match in re.finditer(bias_pattern, response, re.IGNORECASE):
            bias_type = match.group(1).strip().lower().replace(" ", "_")
            detected = match.group(2).strip().upper() == "YES"
            severity = match.group(3).strip().lower()
            if detected:
                biases.append({
                    "type": bias_type,
                    "detected": True,
                    "severity": severity
                })

        if biases:
            return biases

        # Fallback: proximity-based regex (original approach)
        bias_types = [
            "community_bias", "popularity_bias", "recency_bias",
            "english_only", "survivorship_bias", "confirmation_bias"
        ]

        upper_response = response.upper()

        for bias_type in bias_types:
            bias_name_upper = bias_type.replace("_", " ").upper()
            idx = upper_response.find(bias_name_upper)
            if idx != -1:
                nearby = upper_response[idx:]
                if "YES" in nearby and "NO" not in nearby[:nearby.find("YES") + 3]:
                    severity = "medium"
                    if "HIGH" in nearby:
                        severity = "high"
                    elif "LOW" in nearby:
                        severity = "low"

                    biases.append({
                        "type": bias_type,
                        "detected": True,
                        "severity": severity
                    })

        return biases

    def _extract_credibility_scores(self, response: str) -> List[float]:
        """Extract credibility scores from response."""
        import re

        scores = []

        # Primary strategy: parse structured SCORE: lines
        score_line_pattern = r'SCORE:\s*\d+\s*=\s*(\d+\.\d+)'
        line_matches = re.findall(score_line_pattern, response, re.IGNORECASE)
        for match in line_matches:
            try:
                score = float(match)
                if 0.0 <= score <= 1.0:
                    scores.append(score)
            except ValueError:
                continue

        if scores:
            return scores

        # Fallback: look for score patterns like "FINAL SCORE: 0.7"
        pattern = r'(?:final\s+score|score)[:\s|]*(\d+\.\d+)'
        matches = re.findall(pattern, response.lower())

        for match in matches:
            try:
                score = float(match)
                if 0.0 <= score <= 1.0:
                    scores.append(score)
            except ValueError:
                continue

        return scores

    # ============================================================
    #  DATA-ANALYST-SPECIFIC METHODS
    # ============================================================

    def quick_credibility_check(self, claim: str, source: str) -> Dict[str, Any]:
        """Quick credibility check for a single claim."""
        prompt = f"""
Quick credibility assessment:

Claim: {claim}
Source: {source}

Rate credibility 0.0-1.0 and explain in one sentence. Format:
Score: X.XX
Reason: [one sentence]
"""
        response = self.generate_response(prompt, use_first_principles=False)

        score = self._extract_confidence(response)
        return {
            "claim": claim,
            "source": source,
            "score": score,
            "assessment": response
        }

    def compare_sources(self, source_a: str, source_b: str) -> str:
        """Compare two sources for consistency."""
        prompt = f"""
Compare these two sources for consistency:

Source A:
{source_a}

Source B:
{source_b}

Identify:
1. Points of agreement
2. Points of contradiction
3. Information unique to each source
4. Which source appears more reliable and why

Be concise.
"""
        return self.generate_response(prompt, use_first_principles=True)

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get history of all validation results."""
        return self.validation_history

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
