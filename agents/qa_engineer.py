"""QA Engineer Agent - Testing and quality assurance."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from ui.console import console
from utils.enhanced_review_system import EnhancedReviewSystem


QA_SYSTEM_PROMPT = """You are the QA Engineer. You validate that solutions work and meet requirements.

Testing Methodology: Apply systematic testing: (1) Happy path — does the basic case work? (2) Boundary values — what about edge cases? (3) Error cases — what about invalid input? (4) Integration — do components work together?

Bug Classification Rigor: Classify with precision — CRITICAL: data loss, security hole, crash on happy path. MAJOR: feature does not work as specified. MINOR: cosmetic, non-blocking UX issue. TRIVIAL: style preference.

Requirements Traceability: For EACH requirement from the PM, verify: (1) Is it implemented? (2) Does it work correctly? (3) What is the evidence? Map every requirement to a test result.

Regression Awareness: When code is changed to fix one issue, verify that previously working features still work. Fixing A should not break B.

Anti-Perfectionism Balance: Your job is to ensure quality, not to achieve perfection. A working MVP with known minor issues is shippable. A perfect solution that takes forever is not.

Evidence-Based Verdicts: Every FAIL must include: (1) What was expected, (2) What actually happened, (3) Steps to reproduce, (4) Suggested fix. A FAIL without reproduction steps is not useful.

Your verdict system:
- PASS: Solution works and meets the core requirements. Ship it.
- PASS_WITH_ISSUES: Solution works but has minor issues (cosmetic, non-blocking). Still shippable.
- FAIL: Solution does not work or fundamentally misses the requirements. Must be fixed.

Focus on: does it work? Can a user run it? Does it solve the problem? Everything else is secondary.

Execution-First Validation: ALWAYS try to run the code before reviewing it. A code review without execution is opinion. Execution results are evidence.

Regression Tracking: Keep a running list of previously-passed tests. When new code is submitted, verify ALL previous passes still hold. New fixes that break old features = FAIL.

User Journey Testing: Test the complete user journey: install → configure → first use → common use → error recovery. Each step must work.

Regression Suite: Maintain a running list of test cases. When code is modified, re-run ALL previous test cases. New fixes that break old features = FAIL.

User Acceptance Testing: Test as the end user would: install from scratch, follow README, try the happy path. Document every step.

Performance Baseline: Measure: startup time, memory usage, response time for main operation. Flag if any exceeds reasonable limits for the project type.

Test Repeatability: Every test described must be deterministic — run the same command, get the same result. Flaky tests (sometimes pass, sometimes fail) must be flagged and quarantined. A test that randomly passes is not evidence.

Focus on structure validation and basic test scaffolding. Generate test stubs with clear descriptions of what each test should verify. Mark tests that need real implementation with # TODO: implement test logic.
"""

QA_FIRST_PRINCIPLES = [
    "REQUIREMENTS COVERAGE: List every PM requirement. For each, state: TESTED/UNTESTED and PASS/FAIL. 100% coverage before any verdict.",
    "HAPPY PATH FIRST: Run the main use case mentally. Does it work start to finish? If the happy path fails, everything else is irrelevant — immediate FAIL.",
    "EDGE CASE SWEEP: For every input, test: empty, null, maximum size, special characters, unicode, negative numbers (where applicable). List which ones fail.",
    "ERROR MESSAGE QUALITY: When the code fails, does it tell the user what went wrong and how to fix it? Cryptic errors = MAJOR issue.",
    "VERDICT INTEGRITY: Am I passing this because it is genuinely good, or because I want the loop to end? Quality fatigue = call it out and escalate, do not lower standards.",
]


class QAEngineerAgent(BaseAgent, AgentToolsMixin):
    """
    QA Engineer Agent - Ensures quality through testing and validation.

    Enhanced with all 13 Claude Code tools, problem statement refinement,
    and context-aware code review system.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder-7b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="QAEngineer",
            role=AgentRole.QA_ENGINEER,
            model=model,
            first_principles=QA_FIRST_PRINCIPLES,
            system_prompt=QA_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=4096
        )
        super().__init__(config, workspace_root, memory_persist_dir)

        # Track found issues
        self.found_issues: List[Dict[str, Any]] = []

        # Enhanced review system
        self.review_system = EnhancedReviewSystem()

        # Problem statement refiner
        self.problem_refiner = ProblemStatementRefiner()

    def get_capabilities(self) -> List[str]:
        return [
            "test_design",
            "test_execution",
            "bug_finding",
            "code_review_qa",
            "validation",
            "quality_reporting"
        ]

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a QA task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "create_test_plan":
                result = await self._create_test_plan(task)
            elif task_type == "run_tests":
                result = await self._run_tests(task)
            elif task_type == "review_code":
                result = await self._review_code(task)
            elif task_type == "validate_solution":
                result = await self._validate_solution(task)
            elif task_type == "find_bugs":
                result = await self._find_bugs(task)
            elif task_type == "generate_qa_report":
                result = await self._generate_qa_report(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    async def _create_test_plan(self, task: Dict[str, Any]) -> TaskResult:
        """Create a comprehensive test plan."""
        feature = task.get("feature", {})
        requirements = task.get("requirements", "")
        scope = task.get("scope", "comprehensive")

        prompt = f"""
Create a test plan for this feature:

Feature: {feature.get('name', 'Unknown')}
Description: {feature.get('description', '')}

Requirements:
{requirements}

Scope: {scope}

Provide a comprehensive test plan including:

1. TEST STRATEGY
   - Testing approach
   - Types of testing needed
   - Testing priorities

2. TEST CASES
   For each test case:
   - Test ID
   - Description
   - Preconditions
   - Steps
   - Expected result
   - Priority (P0/P1/P2/P3)

3. EDGE CASES
   - Boundary conditions
   - Error conditions
   - Unusual inputs

4. TEST DATA
   - Required test data
   - Test environment needs

5. ACCEPTANCE CRITERIA
   - Pass/fail criteria
   - Coverage requirements
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "test_plan": response,
                "feature": feature
            },
            artifacts={"test_plan": response}
        )

    async def _run_tests(self, task: Dict[str, Any]) -> TaskResult:
        """Run tests and report results."""
        test_path = task.get("test_path", ".")
        test_framework = task.get("framework", "pytest")

        # Run tests using command executor
        result = self.use_tool(
            "run_tests",
            test_path=test_path,
            framework=test_framework
        )

        # Analyze test output
        output = result.get("stdout", "") + result.get("stderr", "")

        prompt = f"""
Analyze these test results:

```
{output}
```

Provide:
1. Summary (total tests, passed, failed, skipped)
2. Failed tests analysis (if any)
3. Recommendations
4. Overall verdict: PASS / FAIL
"""

        analysis = await self.generate_response_async(prompt)

        return TaskResult(
            success=result.get("success", False),
            output={
                "raw_output": output,
                "analysis": analysis,
                "test_path": test_path
            },
            artifacts={"test_results": analysis}
        )

    async def _review_code(self, task: Dict[str, Any]) -> TaskResult:
        """Review code with proper context analysis (Claude Code-style)."""
        code = task.get("code", "")
        file_path = task.get("file_path", "unknown")
        focus = task.get("focus", ["bugs", "security", "quality"])
        user_query = task.get("user_query", None)
        purpose = task.get("purpose", None)
        requirements = task.get("requirements", [])

        console.agent_action("QAEngineer", "Code Review", f"Reviewing {file_path}")

        # Step 1: Create review context with proper understanding
        console.info("Analyzing query and context...")
        context = self.review_system.create_context(
            file_path=file_path,
            code=code,
            user_query=user_query,
            purpose=purpose,
            requirements=requirements
        )

        console.info(f"Context: {context.language}, {len(context.functions)} functions, {len(context.classes)} classes")

        # Step 2: Review with context-aware analysis
        console.info("Performing context-aware review...")
        result = self.review_system.review_with_context(context, focus_areas=focus)

        console.info(f"Review complete - Found {len(result.suggestions)} issues")

        # Step 3: Generate Claude Code-style report
        console.info("Generating detailed report...")
        review_report = result.to_claude_code_format()

        # Extract issues for compatibility
        issues = []
        for suggestion in result.suggestions:
            issues.append({
                "severity": suggestion.severity.value,
                "category": suggestion.category.value,
                "location": f"{suggestion.file_path}:{suggestion.line_start}",
                "description": suggestion.explanation,
                "suggestion": suggestion.suggested_code
            })

        self.found_issues.extend(issues)

        console.success(f"Review complete - {result.critical_issues_count} critical, "
                        f"{result.high_issues_count} high, {result.medium_issues_count} medium issues")

        return TaskResult(
            success=True,
            output={
                "review": review_report,
                "issues": issues,
                "file_path": file_path,
                "summary": result.get_summary(),
                "meets_requirements": result.meets_requirements,
                "critical_count": result.critical_issues_count,
                "high_count": result.high_issues_count
            },
            artifacts={"code_review": review_report}
        )

    async def _validate_solution(self, task: Dict[str, Any]) -> TaskResult:
        """Validate a solution against requirements with proper context review."""
        solution = task.get("solution", {})
        requirements = task.get("requirements", "")
        original_problem = task.get("problem", {})

        console.agent_action("QAEngineer", "Validating Solution", "Starting context review")

        # Step 1: Review and understand the original query/problem
        problem_description = original_problem.get('description', 'Not specified')
        console.info(f"Problem: {problem_description}")

        # Step 2: Parse requirements into structured format
        console.info("Analyzing requirements...")
        req_list = []
        if isinstance(requirements, str):
            # Split requirements by newlines or bullet points
            req_list = [r.strip() for r in requirements.split('\n') if r.strip()]
        elif isinstance(requirements, list):
            req_list = requirements

        console.info(f"Found {len(req_list)} requirements to validate")

        # Step 3: Review solution code with full context
        console.info("Reviewing solution with full context...")
        implementation = solution.get('implementation', '')

        if implementation:
            # Use enhanced review system for code validation
            files = solution.get('files', {})

            all_issues = []
            meets_all_requirements = True

            for file_path, code_content in files.items():
                console.info(f"  Reviewing {file_path}...")

                context = self.review_system.create_context(
                    file_path=file_path,
                    code=code_content,
                    user_query=problem_description,
                    purpose=solution.get('description', ''),
                    requirements=req_list
                )

                review_result = self.review_system.review_with_context(
                    context,
                    focus_areas=['bugs', 'security', 'quality']
                )

                all_issues.extend(review_result.suggestions)

                if not review_result.meets_requirements:
                    meets_all_requirements = False

            console.info(f"Code review complete - {len(all_issues)} issues found")

        # Step 4: Determine review depth based on trust context
        review_depth = task.get("review_depth", "standard")
        if review_depth == "thorough":
            depth_instruction = "This solution has a HISTORY OF ISSUES. Be EXTRA thorough. Check EVERY file, EVERY function, EVERY import. Do NOT give benefit of the doubt."
        elif review_depth == "light":
            depth_instruction = "Quick review — developer has a strong track record. Focus on critical functionality only."
        else:
            depth_instruction = "Standard review — check functionality and code quality."
        # Override with workflow-provided instruction if available
        depth_instruction = task.get("review_depth_instruction", depth_instruction)

        # Step 5: LLM-based validation for completeness
        console.info("LLM validation for completeness...")

        review_feedback = task.get("review_feedback", "")

        execution_results = task.get("execution_results", {})
        execution_ran = bool(execution_results)
        execution_passed = execution_results.get("success", False) if execution_ran else False
        execution_summary = execution_results.get("summary", "")

        if execution_ran:
            exec_label = "PASSED - code runs without errors" if execution_passed else f"FAILED - {execution_summary}"
        else:
            exec_label = "NOT RUN - code execution was not performed"

        dev_conf = task.get("developer_confidence")
        dev_conf_note = (
            f"\nDEVELOPER SELF-CONFIDENCE: {dev_conf:.0%} — Developer flagged uncertainty. "
            "Test the uncertain areas FIRST and with extra scrutiny."
            if dev_conf is not None and dev_conf < 0.75 else ""
        )

        cto_peer = task.get("cto_peer_review", "")
        cto_peer_note = (
            f"\nCTO PEER REVIEW ISSUES (fix these too): {cto_peer[:400]}"
            if cto_peer else ""
        )

        # Build acceptance criteria section from RequirementsDoc if available
        req_doc_dict = task.get("requirements_doc", {})
        ac_list = req_doc_dict.get("acceptance_criteria", []) if req_doc_dict else []
        ac_ids = [ac["id"] for ac in ac_list]

        if ac_list:
            ac_section = "\nACCEPTANCE CRITERIA (validate each explicitly):\n"
            for ac in ac_list:
                ac_section += (
                    f"\n{ac['id']} [{ac.get('priority','P0')}]:\n"
                    f"  GIVEN: {ac.get('given','')}\n"
                    f"  WHEN:  {ac.get('when','')}\n"
                    f"  THEN:  {ac.get('then','')}\n"
                )
                if ac.get("verification_cmd"):
                    ac_section += f"  VERIFY: {ac['verification_cmd']}\n"
            ac_verdict_instructions = (
                "\nFor each AC, output: <AC-ID>: PASS|FAIL|UNTESTED — <one-line evidence>\n"
                "Example: AC-1: PASS — verified by running `python main.py sample.txt`; output matched expected\n"
            )
        else:
            ac_section = ""
            ac_verdict_instructions = ""

        prompt = f"""Validate whether this solution solves the problem.

REVIEW DEPTH: {depth_instruction}{dev_conf_note}{cto_peer_note}

PROBLEM: {problem_description}

REQUIREMENTS:
{chr(10).join(f'- {r}' for r in req_list) if req_list else '- Solve the stated problem'}
{ac_section}
SOLUTION: {solution.get('description', '')}

IMPLEMENTATION: {implementation}

{f"CODE REVIEW FEEDBACK: {review_feedback}" if review_feedback else ""}

CODE EXECUTION RESULTS: {exec_label}
{f"Runtime Errors: {chr(10).join(execution_results.get('runtime_errors', []))}" if execution_results.get('runtime_errors') else ""}
{f"Test Output: {execution_results.get('test_output', '')}" if execution_results.get('test_output') else ""}
{ac_verdict_instructions}
Answer each question with YES, NO, or N/A:
1. Does the solution address the core problem?
2. Were files/code actually generated?
3. Could a user run this and get a working result?
4. Does the code execute without runtime errors? (N/A if execution was not run)

VERDICT RULES (follow EXACTLY):
- Q1=YES, Q2=YES, Q3=YES, Q4=YES or N/A → PASS
- Q1=YES, Q2=YES, Q3=YES, Q4=NO → FAIL (code crashes)
- Q1=YES, Q2=YES, Q3=NO → PASS_WITH_ISSUES (works but not perfectly runnable)
- Q1=NO or Q2=NO → FAIL (core problem unsolved or no code exists)
- 3 of 4 YES and the NO is Q3 → PASS_WITH_ISSUES
- Any other combination → FAIL with specific blocking issues listed

ROOT_CAUSE_ADDRESSED: YES|NO
ROOT_CAUSE_EXPLANATION: [if NO, what symptom does the solution address instead of the root cause?]

Your verdict (exactly one of): PASS / PASS_WITH_ISSUES / FAIL

If FAIL, list the specific blocking issues (max 3).
If PASS_WITH_ISSUES, list the non-blocking issues (max 3).
{self._get_principles_checklist()}
Verdict:"""

        response = await self.generate_response_async(prompt)

        # Use structured parser — no bias toward pass
        from utils.output_parser import StructuredOutputParser
        parser = StructuredOutputParser()
        llm_verdict = parser.parse_verdict(response)
        # Map to QA-specific verdicts
        if llm_verdict in ("approve", "pass"):
            llm_verdict = "pass"
        elif llm_verdict in ("pass_with_issues", "needs_changes"):
            llm_verdict = "pass_with_issues"
        elif llm_verdict in ("reject", "fail"):
            llm_verdict = "fail"
        else:
            llm_verdict = "pass_with_issues"  # uncertain = flag for review, not auto-fail

        # FIX 4: Apply deterministic rules when execution results are available
        syntax_errors = 0
        tests_passed_count = 0
        tests_failed_count = 0
        critical_issues_list = []

        if execution_ran:
            # Count syntax errors from compilation results
            syntax_errors = len(execution_results.get("syntax_errors", []))
            # Count test results if provided
            tests_passed_count = execution_results.get("tests_passed", 0)
            tests_failed_count = execution_results.get("tests_failed", 0)
            # Collect critical issues (runtime errors are treated as critical)
            runtime_errors = execution_results.get("runtime_errors", [])
            if runtime_errors:
                critical_issues_list.extend(runtime_errors)
            # If execution failed outright, treat as a critical issue
            if not execution_passed:
                critical_issues_list.append(execution_summary or "Execution failed")

        # Also collect critical issues from code review
        if implementation:
            for issue in all_issues:
                sev = getattr(getattr(issue, "severity", None), "value", "")
                if sev in ("critical", "error"):
                    desc = getattr(issue, "explanation", str(issue))
                    critical_issues_list.append(desc)

        verdict = self._compute_deterministic_verdict(
            syntax_errors=syntax_errors,
            tests_passed=tests_passed_count,
            tests_failed=tests_failed_count,
            critical_issues=critical_issues_list,
            llm_verdict=llm_verdict,
        )

        if verdict != llm_verdict:
            console.warning(
                f"[DeterministicQA] Verdict overridden: LLM said '{llm_verdict}' "
                f"-> deterministic rules say '{verdict}' "
                f"(syntax_errors={syntax_errors}, tests_failed={tests_failed_count}, "
                f"critical_issues={len(critical_issues_list)})"
            )

        # Parse QA response into typed QAReport
        from orchestrator.artifact_parser import QAReportParser
        qa_parser = QAReportParser()
        qa_report = qa_parser.parse(response, ac_ids=ac_ids)
        # Use deterministic verdict rather than LLM-extracted one
        qa_report.verdict = verdict.upper()
        if critical_issues_list:
            qa_report.critical_issues = [str(i)[:200] for i in critical_issues_list[:5]]

        return TaskResult(
            success=True,
            output={
                "verdict": verdict,
                "llm_verdict": llm_verdict,
                "validation": response,
                "qa_report": qa_report.to_dict(),
                "deterministic_inputs": {
                    "syntax_errors": syntax_errors,
                    "tests_passed": tests_passed_count,
                    "tests_failed": tests_failed_count,
                    "critical_issues_count": len(critical_issues_list),
                },
            },
            artifacts={
                "validation_report": response,
                "qa_report": qa_report.to_dict(),
            },
        )

    async def _find_bugs(self, task: Dict[str, Any]) -> TaskResult:
        """Actively hunt for bugs in code or system."""
        target = task.get("target", "")
        code = task.get("code", "")
        context = task.get("context", "")

        prompt = f"""
Hunt for bugs in this:

Target: {target}
Context: {context}

```
{code}
```

Think like a malicious user or an edge case expert. Find:
1. Ways to break it
2. Invalid inputs that aren't handled
3. Race conditions or timing issues
4. Resource leaks
5. Error paths that fail badly
6. Security vulnerabilities

For each bug found:
- Severity: Critical/High/Medium/Low
- Reproduction steps
- Expected behavior
- Actual/potential behavior
- Suggested fix
"""

        response = await self.generate_response_async(prompt)

        # Extract bugs
        bugs = self._extract_issues(response)
        self.found_issues.extend(bugs)

        return TaskResult(
            success=True,
            output={
                "bugs_found": len(bugs),
                "bugs": bugs,
                "report": response
            },
            artifacts={"bug_report": response}
        )

    async def _generate_qa_report(self, task: Dict[str, Any]) -> TaskResult:
        """Generate a comprehensive QA report."""
        project = task.get("project", "")
        test_results = task.get("test_results", {})
        issues = task.get("issues", self.found_issues)
        execution_results = task.get("execution_results", {})

        issues_text = "\n".join(
            f"- [{i.get('severity', 'Unknown')}] {i.get('description', 'No description')}"
            if isinstance(i, dict) else f"- {i}"
            for i in issues
        )

        execution_section = ""
        if execution_results:
            exec_passed = execution_results.get("success", True)
            exec_summary = execution_results.get("summary", "")
            execution_section = f"""
Code Execution Results:
- Status: {"PASSED" if exec_passed else "FAILED"}
- Summary: {exec_summary}
{f"- Runtime Errors: {chr(10).join(execution_results.get('runtime_errors', []))}" if execution_results.get('runtime_errors') else ""}
{f"- Test Output: {execution_results.get('test_output', '')}" if execution_results.get('test_output') else ""}
"""

        prompt = f"""
Generate a QA report for: {project}

Test Results:
{test_results}

Issues Found:
{issues_text if issues else 'No issues found'}
{execution_section}
Generate a comprehensive QA report including:
1. Executive Summary
2. Testing Coverage
3. Code Execution Results
4. Issues Summary (by severity)
5. Detailed Issue List
6. Risk Assessment
7. Recommendations
8. Release Readiness: READY / NOT_READY / CONDITIONAL

Format as a professional QA report.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "report": response,
                "total_issues": len(issues)
            },
            artifacts={"qa_report": response}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general QA tasks."""
        description = task.get("description", "")

        response = await self.generate_response_async(
            f"As QA Engineer, please address: {description}"
        )

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  HELPER METHODS
    # ============================================================

    def _compute_deterministic_verdict(
        self,
        syntax_errors: int,
        tests_passed: int,
        tests_failed: int,
        critical_issues: list,
        llm_verdict: str,
    ) -> str:
        """Apply deterministic rules to override or confirm the LLM verdict.

        Hard rules (in priority order):
        1. Any syntax errors -> fail
        2. Any critical issues (security vulns, broken imports) -> fail
        3. All tests failed and none passed -> fail
        4. Some tests failed -> pass_with_issues
        5. Fall back to LLM verdict when no hard signal

        Args:
            syntax_errors: Count of Python syntax errors found
            tests_passed: Count of passing tests
            tests_failed: Count of failing tests
            critical_issues: List of critical issue descriptions
            llm_verdict: The verdict string produced by the LLM ("pass", "pass_with_issues", "fail")

        Returns:
            Final verdict string: "pass", "pass_with_issues", or "fail"
        """
        if syntax_errors > 0:
            return "fail"
        if critical_issues:  # security vulns, broken imports
            return "fail"
        if tests_failed > 0 and tests_passed == 0:
            return "fail"
        if tests_failed > 0:
            return "pass_with_issues"
        # Fall back to LLM verdict only if no hard signals
        return llm_verdict

    def _extract_issues(self, response: str) -> List[Dict[str, Any]]:
        """Extract issues from review response."""
        issues = []

        # Simple extraction based on severity markers
        lines = response.split("\n")
        current_issue = {}

        severity_keywords = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        for line in lines:
            line_upper = line.upper()
            for severity in severity_keywords:
                if severity in line_upper and (":" in line or "-" in line):
                    if current_issue:
                        issues.append(current_issue)
                    current_issue = {
                        "severity": severity.lower(),
                        "description": line.split(":", 1)[-1].strip() if ":" in line else line
                    }
                    break

        if current_issue:
            issues.append(current_issue)

        return issues

    # ============================================================
    #  QA-SPECIFIC METHODS
    # ============================================================

    def quick_review(self, code: str) -> str:
        """Quick code review for obvious issues."""
        prompt = f"""
Quick review - find obvious issues:

```
{code}
```

List only clear bugs, security issues, or quality problems. Be concise.
"""
        return self.generate_response(prompt, use_first_principles=False)

    def suggest_test_cases(self, function_description: str) -> str:
        """Suggest test cases for a function."""
        prompt = f"""
Suggest test cases for this function:

{function_description}

Provide:
- Happy path tests
- Edge cases
- Error cases
- Boundary value tests

Format as a simple list.
"""
        return self.generate_response(prompt)

    def assess_risk(self, change_description: str) -> Dict[str, Any]:
        """Assess risk of a change."""
        prompt = f"""
Assess the risk of this change:

{change_description}

Provide:
1. Risk Level: Low/Medium/High/Critical
2. Main risks
3. Testing recommendations
4. Rollback considerations
"""
        response = self.generate_response(prompt, use_first_principles=True)

        risk_level = "medium"
        if "CRITICAL" in response.upper():
            risk_level = "critical"
        elif "HIGH" in response.upper():
            risk_level = "high"
        elif "LOW" in response.upper():
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "assessment": response
        }

    def get_all_issues(self) -> List[Dict[str, Any]]:
        """Get all found issues."""
        return self.found_issues

    def clear_issues(self) -> None:
        """Clear found issues list."""
        self.found_issues.clear()
