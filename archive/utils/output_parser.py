"""Structured output parser for LLM responses.

Replaces brittle string matching with multi-strategy parsing.
Pure logic — no LLM calls, no I/O.
"""

import json
import re
from typing import Dict, Any, Optional


class StructuredOutputParser:
    """Parse LLM responses into structured decisions."""

    # ----------------------------------------------------------------
    # Decision parsing
    # ----------------------------------------------------------------

    def parse_decision(self, response: str) -> Dict[str, Any]:
        """Extract a decision from an LLM response.

        Tries multiple strategies in order:
        1. Find JSON block in response
        2. Look for structured markers (DECISION:, VERDICT:, etc.)
        3. Return "uncertain" with the raw response

        Returns:
            Dict with keys: decision, confidence, reasoning, issues
        """
        if not response or not response.strip():
            return {"decision": "uncertain", "confidence": 0.0,
                    "reasoning": "", "issues": []}

        # Strategy 1: Try to find JSON block
        parsed = self._try_json_extraction(response)
        if parsed and "decision" in parsed:
            return self._normalize_decision(parsed)

        # Strategy 2: Look for structured markers
        parsed = self._try_marker_extraction(response)
        if parsed and parsed.get("decision") != "uncertain":
            return parsed

        # Strategy 3: Return uncertain
        return {
            "decision": "uncertain",
            "confidence": 0.3,
            "reasoning": response,
            "issues": []
        }

    def parse_verdict(self, response: str) -> str:
        """Extract QA/review verdict from response.

        Returns one of: pass, pass_with_issues, fail, approve, reject,
                        needs_changes, uncertain
        """
        if not response:
            return "uncertain"

        # Strategy 1: JSON extraction
        parsed = self._try_json_extraction(response)
        if parsed:
            verdict = parsed.get("verdict", parsed.get("decision", ""))
            mapped = self._map_verdict(str(verdict))
            if mapped != "uncertain":
                return mapped

        # Strategy 2: Marker extraction
        upper = response.upper()

        # Check for explicit markers (order matters: more specific first)
        if "PASS_WITH_ISSUES" in upper:
            return "pass_with_issues"
        if "NEEDS_CHANGES" in upper:
            return "needs_changes"

        # Look for standalone APPROVE/REJECT/PASS/FAIL with word boundaries
        # to avoid matching "APPROVAL" or "PASSING"
        approve_match = re.search(r'\bAPPROVE\b', upper)
        reject_match = re.search(r'\bREJECT\b', upper)
        pass_match = re.search(r'\bPASS\b', upper)
        fail_match = re.search(r'\bFAIL\b', upper)

        # If both approve and reject found, use position — last word wins
        if approve_match and reject_match:
            if reject_match.start() > approve_match.start():
                return "reject"
            return "approve"

        if reject_match:
            return "reject"
        if approve_match:
            return "approve"

        # FAIL takes precedence over PASS (conservative)
        if fail_match and pass_match:
            if fail_match.start() > pass_match.start():
                return "fail"
            return "pass"
        if fail_match:
            return "fail"
        if pass_match:
            return "pass"

        return "uncertain"

    # ----------------------------------------------------------------
    # Internal strategies
    # ----------------------------------------------------------------

    def _try_json_extraction(self, response: str) -> Optional[Dict[str, Any]]:
        """Try to extract a JSON object from the response.

        Handles:
        - ```json ... ``` blocks
        - Bare {...} objects
        - Multiple JSON blocks (takes the first valid one)
        """
        # Try ```json blocks first
        json_blocks = re.findall(
            r'```(?:json)?\s*\n?(.*?)\n?\s*```',
            response, re.DOTALL
        )
        for block in json_blocks:
            parsed = self._safe_json_parse(block.strip())
            if parsed:
                return parsed

        # Try bare JSON objects
        brace_matches = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
        for match in brace_matches:
            parsed = self._safe_json_parse(match)
            if parsed:
                return parsed

        return None

    def _try_marker_extraction(self, response: str) -> Dict[str, Any]:
        """Extract decision from structured markers like DECISION: YES."""
        result: Dict[str, Any] = {
            "decision": "uncertain",
            "confidence": 0.5,
            "reasoning": "",
            "issues": []
        }

        lines = response.split("\n")
        for line in lines:
            line_stripped = line.strip()
            upper = line_stripped.upper()

            # Decision markers
            for marker in ("DECISION:", "VERDICT:", "RECOMMENDATION:"):
                if upper.startswith(marker):
                    value = line_stripped[len(marker):].strip()
                    mapped = self._map_verdict(value.upper())
                    if mapped != "uncertain":
                        result["decision"] = mapped
                    break

            # Confidence markers
            if upper.startswith("CONFIDENCE:"):
                conf_str = line_stripped[len("CONFIDENCE:"):].strip()
                try:
                    conf = float(conf_str.replace("%", "").strip())
                    if conf > 1.0:
                        conf = conf / 100.0
                    result["confidence"] = max(0.0, min(1.0, conf))
                except ValueError:
                    pass

            # Reasoning markers
            if upper.startswith("REASONING:") or upper.startswith("REASON:"):
                marker_len = len("REASONING:") if upper.startswith("REASONING:") else len("REASON:")
                result["reasoning"] = line_stripped[marker_len:].strip()

            # Issues markers
            if upper.startswith("ISSUES:") or upper.startswith("PROBLEMS:"):
                marker_len = len("ISSUES:") if upper.startswith("ISSUES:") else len("PROBLEMS:")
                issues_str = line_stripped[marker_len:].strip()
                if issues_str and issues_str.lower() not in ("none", "n/a", "[]"):
                    result["issues"] = [i.strip() for i in issues_str.split(",") if i.strip()]

        return result

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
        """Safely parse a JSON string, returning None on failure."""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return None

    @staticmethod
    def _map_verdict(value: str) -> str:
        """Map various verdict strings to canonical values."""
        value = value.strip().upper()

        approve_words = {"YES", "APPROVE", "APPROVED", "PASS", "ACCEPT",
                         "PROCEED", "GO", "SHIP"}
        reject_words = {"NO", "REJECT", "REJECTED", "FAIL", "DENY",
                        "DENIED", "STOP", "BLOCK"}
        more_info_words = {"NEED_MORE_INFO", "REQUEST_MORE_INFO",
                           "NEED MORE INFO", "UNCERTAIN", "MAYBE",
                           "NEEDS_CHANGES", "NEEDS CHANGES"}
        partial_words = {"PASS_WITH_ISSUES", "PARTIAL", "CONDITIONAL",
                         "PASS WITH ISSUES"}

        if value in approve_words:
            return "approve"
        if value in reject_words:
            return "reject"
        if value in partial_words:
            return "pass_with_issues"
        if value in more_info_words:
            return "needs_changes"
        return "uncertain"

    # ----------------------------------------------------------------
    # Agent-specific parsers
    # ----------------------------------------------------------------

    def parse_data_analyst_verdict(self, text: str) -> Dict[str, Any]:
        """Extract per-finding VERDICT/CONFIDENCE/EVIDENCE from DataAnalyst cross-validation.

        Returns:
            Dict with 'findings' list (each has verdict, confidence, evidence_for, evidence_against)
            and 'summary' dict.
        """
        if not text:
            return {"findings": [], "summary": {}}

        # Strategy 1: JSON block
        parsed = self._try_json_extraction(text)
        if parsed and "findings" in parsed:
            return parsed

        # Strategy 2: Marker extraction for FINDING [N]: blocks
        findings = []
        current: Dict[str, Any] = {}
        for line in text.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("FINDING") and ("]" in upper or ":" in upper):
                if current:
                    findings.append(current)
                # Extract finding description after the marker
                desc = re.sub(r'^FINDING\s*\[?\d*\]?\s*:?\s*', '', stripped, flags=re.IGNORECASE)
                current = {"description": desc, "verdict": "UNCONFIRMED",
                           "confidence": 0.5, "evidence_for": "", "evidence_against": ""}
            elif upper.startswith("VERDICT:"):
                val = stripped[len("VERDICT:"):].strip().upper()
                if val in ("CONFIRMED", "PARTIALLY_CONFIRMED", "UNCONFIRMED", "CONTRADICTED"):
                    current["verdict"] = val
            elif upper.startswith("CONFIDENCE:"):
                try:
                    conf = float(stripped[len("CONFIDENCE:"):].strip())
                    if conf > 1.0:
                        conf /= 100.0
                    current["confidence"] = max(0.0, min(1.0, conf))
                except ValueError:
                    pass
            elif upper.startswith("EVIDENCE_FOR:"):
                current["evidence_for"] = stripped[len("EVIDENCE_FOR:"):].strip()
            elif upper.startswith("EVIDENCE_AGAINST:"):
                current["evidence_against"] = stripped[len("EVIDENCE_AGAINST:"):].strip()

        if current:
            findings.append(current)

        # Build summary
        verdicts = [f.get("verdict", "") for f in findings]
        summary = {
            "confirmed": verdicts.count("CONFIRMED"),
            "partially_confirmed": verdicts.count("PARTIALLY_CONFIRMED"),
            "unconfirmed": verdicts.count("UNCONFIRMED"),
            "contradicted": verdicts.count("CONTRADICTED"),
            "total": len(findings),
        }

        return {"findings": findings, "summary": summary}

    def parse_qa_result(self, text: str) -> Dict[str, Any]:
        """Extract TEST_RESULT, ROOT_CAUSE_ADDRESSED, ISSUES from QA response.

        Returns:
            Dict with test_result, root_cause_addressed, root_cause_explanation, issues.
        """
        if not text:
            return {"test_result": "uncertain", "root_cause_addressed": None,
                    "root_cause_explanation": "", "issues": []}

        parsed = self._try_json_extraction(text)
        if parsed and "test_result" in parsed:
            return parsed

        result: Dict[str, Any] = {
            "test_result": "uncertain",
            "root_cause_addressed": None,
            "root_cause_explanation": "",
            "issues": [],
        }

        for line in text.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("TEST_RESULT:"):
                val = stripped[len("TEST_RESULT:"):].strip()
                mapped = self._map_verdict(val.upper())
                if mapped != "uncertain":
                    result["test_result"] = mapped
            elif upper.startswith("ROOT_CAUSE_ADDRESSED:"):
                val = stripped[len("ROOT_CAUSE_ADDRESSED:"):].strip().upper()
                result["root_cause_addressed"] = val in ("YES", "TRUE", "Y")
            elif upper.startswith("ROOT_CAUSE_EXPLANATION:"):
                result["root_cause_explanation"] = stripped[len("ROOT_CAUSE_EXPLANATION:"):].strip()
            elif upper.startswith("ISSUES:") or upper.startswith("BLOCKING_ISSUES:"):
                marker_len = len("BLOCKING_ISSUES:") if upper.startswith("BLOCKING_ISSUES:") else len("ISSUES:")
                issues_str = stripped[marker_len:].strip()
                if issues_str and issues_str.lower() not in ("none", "n/a", "[]"):
                    result["issues"] = [i.strip() for i in issues_str.split(",") if i.strip()]

        # Also try to extract verdict from parse_verdict as fallback for test_result
        if result["test_result"] == "uncertain":
            verdict = self.parse_verdict(text)
            if verdict != "uncertain":
                result["test_result"] = verdict

        return result

    def parse_ceo_decision(self, text: str) -> Dict[str, Any]:
        """Extract DECISION, EVIDENCE_CITED, FEEDBACK from CEO response.

        Returns:
            Dict with decision, confidence, evidence_cited, feedback, issues.
        """
        if not text:
            return {"decision": "uncertain", "confidence": 0.5,
                    "evidence_cited": "", "feedback": "", "issues": []}

        # Try JSON first (CEO is already prompted to output JSON)
        parsed = self._try_json_extraction(text)
        if parsed and "decision" in parsed:
            normalized = self._normalize_decision(parsed)
            normalized.setdefault("evidence_cited", parsed.get("evidence_cited", ""))
            normalized.setdefault("feedback", parsed.get("feedback", parsed.get("reasoning", "")))
            return normalized

        # Marker extraction
        result = self._try_marker_extraction(text)
        result.setdefault("evidence_cited", "")
        result.setdefault("feedback", result.get("reasoning", ""))

        for line in text.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("EVIDENCE_CITED:") or upper.startswith("EVIDENCE:"):
                marker_len = len("EVIDENCE_CITED:") if upper.startswith("EVIDENCE_CITED:") else len("EVIDENCE:")
                result["evidence_cited"] = stripped[marker_len:].strip()
            elif upper.startswith("FEEDBACK:"):
                result["feedback"] = stripped[len("FEEDBACK:"):].strip()

        return result

    def parse_cto_design(self, text: str) -> Dict[str, Any]:
        """Extract ROOT_CAUSE, ARCHITECTURE_APPROACH, COMPONENTS from CTO design output.

        Returns:
            Dict with root_cause, architecture_approach, components, risk_assessment.
        """
        if not text:
            return {"root_cause": "", "architecture_approach": "",
                    "components": [], "risk_assessment": ""}

        parsed = self._try_json_extraction(text)
        if parsed and "root_cause" in parsed:
            return parsed

        result: Dict[str, Any] = {
            "root_cause": "",
            "architecture_approach": "",
            "components": [],
            "risk_assessment": "",
        }

        for line in text.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("ROOT_CAUSE:") or upper.startswith("ROOT CAUSE:"):
                marker_len = len("ROOT_CAUSE:") if upper.startswith("ROOT_CAUSE:") else len("ROOT CAUSE:")
                result["root_cause"] = stripped[marker_len:].strip()
            elif upper.startswith("ARCHITECTURE_APPROACH:") or upper.startswith("ARCHITECTURE APPROACH:"):
                marker_len = len("ARCHITECTURE_APPROACH:") if upper.startswith("ARCHITECTURE_APPROACH:") else len("ARCHITECTURE APPROACH:")
                result["architecture_approach"] = stripped[marker_len:].strip()
            elif upper.startswith("COMPONENTS:"):
                comp_str = stripped[len("COMPONENTS:"):].strip()
                if comp_str:
                    result["components"] = [c.strip() for c in comp_str.split(",") if c.strip()]
            elif upper.startswith("RISK_ASSESSMENT:") or upper.startswith("RISK ASSESSMENT:"):
                marker_len = len("RISK_ASSESSMENT:") if upper.startswith("RISK_ASSESSMENT:") else len("RISK ASSESSMENT:")
                result["risk_assessment"] = stripped[marker_len:].strip()

        # If components is empty, try to find bullet-point components after COMPONENTS marker
        if not result["components"]:
            in_components = False
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped.upper().startswith("COMPONENTS:"):
                    in_components = True
                    continue
                if in_components:
                    if stripped.upper().startswith(("RISK", "ROOT", "ARCHITECTURE")):
                        break
                    if stripped.startswith(("-", "*", "•")) or re.match(r'^\d+\.', stripped):
                        comp = re.sub(r'^[-*•\d.]+\s*', '', stripped).strip()
                        if comp:
                            result["components"].append(comp)

        return result

    # ----------------------------------------------------------------
    # Root cause depth scoring
    # ----------------------------------------------------------------

    def score_root_cause_depth(self, text: str) -> int:
        """Score the depth of root cause analysis in text.

        Returns:
            1 = symptom level (surface description)
            2 = proximate cause (immediate trigger)
            3 = root cause (underlying issue)
            4 = systemic cause (structural/design flaw)
        """
        if not text:
            return 1

        text_lower = text.lower()

        # Level 4: Systemic cause indicators
        systemic_indicators = [
            "systemic", "structural", "architectural flaw", "design pattern",
            "fundamental assumption", "paradigm", "organizational",
            "process failure", "systematic"
        ]
        if sum(1 for ind in systemic_indicators if ind in text_lower) >= 2:
            return 4

        # Level 3: Root cause indicators
        root_cause_indicators = [
            "root cause", "underlying", "because.*because", "fundamental",
            "core issue", "the real problem", "actually caused by",
            "stems from", "originates from", "the reason is"
        ]
        if sum(1 for ind in root_cause_indicators if re.search(ind, text_lower)) >= 2:
            return 3

        # Level 2: Proximate cause indicators
        proximate_indicators = [
            "caused by", "triggered by", "results from", "due to",
            "leads to", "because", "reason:", "the error occurs when"
        ]
        if sum(1 for ind in proximate_indicators if ind in text_lower) >= 1:
            return 2

        # Level 1: Just symptoms
        return 1

    # ----------------------------------------------------------------
    # Internal helpers (continued)
    # ----------------------------------------------------------------

    @staticmethod
    def _normalize_decision(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a parsed JSON decision to standard format."""
        decision = str(parsed.get("decision", "uncertain")).strip()
        confidence = parsed.get("confidence", 0.5)
        reasoning = parsed.get("reasoning", "")
        issues = parsed.get("issues", [])

        # Normalize confidence
        if isinstance(confidence, str):
            try:
                confidence = float(confidence.replace("%", ""))
                if confidence > 1.0:
                    confidence = confidence / 100.0
            except ValueError:
                confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))

        # Normalize issues to list
        if isinstance(issues, str):
            issues = [i.strip() for i in issues.split(",") if i.strip()]

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": str(reasoning),
            "issues": list(issues) if issues else []
        }
