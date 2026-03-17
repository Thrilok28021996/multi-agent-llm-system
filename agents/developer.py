"""Developer Agent - Solution implementation and coding."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from templates import generate_project
from tools import UnifiedTools
from ui.console import console
from utils.enhanced_code_parser import EnhancedCodeParser, extract_single_code_block
from utils.error_recovery import ErrorRecoverySystem, PartialSuccessHandler


DEVELOPER_SYSTEM_PROMPT = """You are the Lead Developer. You write working code that solves the stated problem.

Your priorities (in order):
1. Make it work — code must be functional and solve the problem
2. Make it clear — readable, well-structured, obvious naming
3. Make it complete — include all files needed to run (entry point, dependencies, README)

Engineering Discipline: Write code you would be proud to show in a code review. Every function should have a clear single purpose. Every error should be handled, not silenced.

First-Principles Coding: Before writing code, understand the problem completely. Trace the data flow mentally from input to output. Only then write the code.

Error Handling Philosophy: Handle errors at the boundary where you can do something useful about them. Do not catch exceptions just to re-raise them. Do not silently swallow errors.

Testing Mindset: Write code that is easy to test. Pure functions over side effects. Dependency injection over hard-coded dependencies. If a function is hard to test, it is doing too much.

Fix Discipline: When fixing bugs: (1) Reproduce the bug first, (2) Understand WHY it happened, (3) Fix the root cause not the symptom, (4) Verify the fix does not break anything else.

Anti-Pattern Awareness: Avoid: God classes, deep nesting (>3 levels), magic numbers, copy-paste code, premature optimization, stringly-typed interfaces.

Rules:
- Always output complete, runnable files. Never output partial snippets or pseudocode.
- Output each file as: FILE: path/to/file.ext followed by a code block. Both **File Path:** and FILE: formats work.
- When fixing bugs or addressing feedback, re-output the complete corrected file, not just the diff.
- Keep solutions simple. A working 50-line script beats an over-engineered 500-line framework.

When fixing issues from QA or CEO feedback:
- Read the feedback carefully — fix exactly what was flagged
- Re-generate the affected files with corrections applied
- If the feedback is vague, improve the most likely problem areas (error handling, missing functionality)

Root Cause Implementation: Before writing code, state the root cause of the problem in one sentence. Every function you write must trace back to addressing that root cause. Code that doesn't serve the root cause gets deleted.

Working Incrementally: Build the smallest thing that proves the core works first. Then add features one at a time. Never write all files at once and hope they work together.

Self-Testing: After writing each file, mentally execute it line by line. What happens with empty input? What happens with the largest reasonable input? Fix issues before outputting.

Test-Driven Development: Write the test FIRST, then the implementation. If you cannot write the test, you do not understand the requirement.

Code Review Self-Check: Before submitting: (1) Would I approve this in a code review? (2) Is every function under 30 lines? (3) Are there any hardcoded values that should be config?

Dependency Justification: For each import or external dependency, state why the standard library is not sufficient.

Done Checklist: Before submitting: (1) Every file runs without import errors, (2) Entry point works with the exact command in README, (3) No hardcoded paths, secrets, or debug print statements remain, (4) Error messages are human-readable, not raw tracebacks.

Confidence Declaration: At the END of every implementation response, add a line: "CONFIDENCE: X.X" where X.X is 0.0–1.0. Be honest. 0.9+ means you are certain it works. Below 0.6 means you have serious doubts — flag the uncertain part explicitly so QA knows where to focus.

Generate clean, well-structured scaffolding code. Use standard patterns and libraries. Add TODO comments where business logic needs human implementation. Mark incomplete sections clearly with # TODO: <what needs to be done>.
"""

DEVELOPER_FIRST_PRINCIPLES = [
    "INPUT-OUTPUT TRACE: Start at the entry point. For EVERY possible input, trace to the output. Does it produce the correct result? Does it handle invalid input? Does it handle edge cases (empty, null, huge)?",
    "DEPENDENCY MINIMALISM: List every import. For each: is it in stdlib? If not, is it essential? Could you write the 10 lines yourself instead of importing a library?",
    "ERROR PATH AUDIT: For every external call (file I/O, network, user input), what happens on failure? If the answer is 'crash' or 'undefined', fix it.",
    "READABILITY TEST: Could a developer who has never seen this code understand what it does in 60 seconds? If not, rename variables, extract functions, add a one-line comment.",
    "COMPLETENESS CHECK: Can someone clone this repo and run it with ONLY what is in the output? requirements.txt, entry point, README with run command — all present?",
    "ROOT CAUSE TRACE: State the root cause. For every function, answer: 'How does this address the root cause?' If it does not, delete it.",
]


class DeveloperAgent(BaseAgent, AgentToolsMixin):
    """
    Developer Agent - Implements solutions and writes code.

    Uses Claude Code-style granular tools for file operations.
    Now includes all 13 Claude Code tools + problem statement refinement.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder-7b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="Developer",
            role=AgentRole.DEVELOPER,
            model=model,
            first_principles=DEVELOPER_FIRST_PRINCIPLES,
            system_prompt=DEVELOPER_SYSTEM_PROMPT,
            temperature=0.3,  # Low temperature for precise code
            max_tokens=8192  # Larger for code generation
        )
        super().__init__(config, workspace_root, memory_persist_dir)

        # Initialize unified tools (Claude Code-style)
        self.tools = UnifiedTools(
            workspace_root=workspace_root,
            persist_dir=memory_persist_dir
        )

        # Initialize error recovery system
        self.error_recovery = ErrorRecoverySystem(
            max_retries=3,
            enable_logging=True
        )

        # Initialize problem statement refiner
        self.problem_refiner = ProblemStatementRefiner()

        # Enable ReAct tool use loop
        self.enable_react_tools()

    def _add_draft_header(self, filepath: str, content: str) -> str:
        """Add a draft notice to generated files."""
        ext = Path(filepath).suffix
        if ext == ".py":
            header = "# AUTO-GENERATED SCAFFOLD — Review and complete TODOs before production use\n"
        elif ext in (".js", ".ts"):
            header = "// AUTO-GENERATED SCAFFOLD — Review and complete TODOs before production use\n"
        elif ext in (".md",):
            header = "> **AUTO-GENERATED SCAFFOLD** — Review and complete before use\n\n"
        else:
            return content
        if not content.startswith(header):
            return header + content
        return content

    def _safe_output_path(self, file_path: str, output_dir: str) -> str:
        """
        Safely resolve a file path within the output directory.

        Prevents path traversal attacks (e.g., ../../../etc/passwd) by
        resolving the final path and verifying it stays within output_dir.
        """
        from pathlib import Path

        if not output_dir:
            return file_path

        # Resolve the combined path to eliminate .. traversals
        base = Path(output_dir).resolve()
        combined = (base / file_path).resolve()

        # Verify the resolved path is within the output directory
        try:
            combined.relative_to(base)
        except ValueError:
            # Path escapes output_dir - force it inside
            safe_name = Path(file_path).name  # Use just the filename
            combined = base / safe_name

        return str(combined)

    def refine_problem_statement(self, problem: str) -> str:
        """
        Refine a problem statement before building solution.

        This ensures we have a clear, concise, actionable problem
        statement before starting development.

        Args:
            problem: Original problem description

        Returns:
            Refined problem statement (formatted)
        """
        console.agent_action("Developer", "Refining Problem", problem)
        console.info(f"Original: {problem}")

        refined = self.problem_refiner.refine(problem)

        console.info(f"Refined: {refined.refined_statement}")
        console.info(f"Type: {refined.problem_type.value}")
        console.info(f"Actionable: {'Yes' if refined.is_actionable else 'No'}")
        console.info(f"Confidence: {refined.confidence:.0%}")

        if not refined.is_actionable:
            console.warning("Problem statement may not be fully actionable")
            for note in refined.refinement_notes:
                console.warning(f"  - {note}")

        return self.problem_refiner.format_refined_statement(refined)

    def get_capabilities(self) -> List[str]:
        return [
            "code_implementation",
            "bug_fixing",
            "code_refactoring",
            "test_writing",
            "documentation",
            "file_operations"
        ]

    async def write_file_with_recovery(
        self,
        file_path: str,
        content: str
    ) -> Dict[str, Any]:
        """Write file with automatic error recovery."""
        async def write_operation():
            return self.use_tool("write_file", path=file_path, content=content)

        result = await self.error_recovery.retry_async(
            write_operation,
            f"write {file_path}"
        )

        if result.success:
            return result.result
        else:
            return {
                "success": False,
                "error": str(result.error),
                "suggested_fix": self.error_recovery.suggest_fix(result.error, file_path)
            }

    async def write_multiple_files_with_recovery(
        self,
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Write multiple files with partial success handling."""
        handler = PartialSuccessHandler()

        for file_path, content in files.items():
            try:
                result = await self.write_file_with_recovery(file_path, content)

                if result.get("success"):
                    handler.add_success(file_path)
                else:
                    error_msg = result.get("error", "Unknown error")
                    handler.add_failure(file_path, Exception(error_msg))

            except Exception as e:
                handler.add_failure(file_path, e)

        summary = handler.get_summary()

        return {
            "success": summary["success_rate"] > 0,
            "files_written": handler.successes,
            "files_failed": summary["failed_items"],
            "errors": summary["errors"],
            "is_partial": summary["is_partial"],
            "summary": summary
        }

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a Developer task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "implement_feature":
                result = await self._implement_feature(task)
            elif task_type == "fix_bug":
                result = await self._fix_bug(task)
            elif task_type == "write_code":
                result = await self._write_code(task)
            elif task_type == "write_tests":
                result = await self._write_tests(task)
            elif task_type == "refactor":
                result = await self._refactor_code(task)
            elif task_type == "create_project":
                result = await self._create_project(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    async def _implement_feature(self, task: Dict[str, Any]) -> TaskResult:
        """Implement a feature based on specifications.

        If the task includes structured 'stories' (from CTO ArchitectureNote),
        each story is implemented separately to keep each prompt small enough
        for a 7-8B model to handle reliably. Results are merged.
        Otherwise falls back to the original single-prompt approach.
        """
        spec = task.get("specification", "")
        architecture = task.get("architecture", "")
        language = task.get("language", "python")
        output_dir = task.get("output_dir", "")
        file_structure = task.get("file_structure", {})

        # --- Story-driven path (BMAD-style sharding) ---
        stories_raw = task.get("stories", [])
        req_doc_dict = task.get("requirements_doc", {})
        if stories_raw:
            return await self._implement_stories(
                stories_raw, req_doc_dict, language, output_dir, spec
            )

        # --- Legacy single-prompt path ---
        compute = self._get_compute_config(str(task))

        prompt = f"""MANDATORY PREAMBLE (include at the top of your response):
ROOT CAUSE: [one sentence — the fundamental reason this problem exists]
SOLUTION APPROACH: [one sentence — how your code addresses this root cause]
VERIFICATION: [one sentence — how to verify the root cause is addressed]

Implement the following feature. Output complete, working {language} code.

SPECIFICATION:
{spec}

ARCHITECTURE:
{architecture}

{f"FILE STRUCTURE: {file_structure}" if file_structure else ""}

Requirements:
- Every file must be complete and runnable — no stubs, no TODOs, no placeholders
- Include a clear entry point (main function or script)
- Include any config files needed (requirements.txt, package.json, etc.)
- Keep it simple and focused on solving the specification

Output each file using this EXACT format (required for parsing):
**File Path:** `path/to/file.ext`
```{language}
complete code here
```

{self._get_principles_checklist()}
Before outputting, mentally trace through the code:
- Does the entry point exist and call the right functions?
- Are all imports present and correct?
- Would `python main.py` (or equivalent) actually run?

BEFORE YOU FINISH, self-check:
- [ ] Does every file have complete, runnable code (no TODOs, no stubs)?
- [ ] Is there a clear entry point a user can run?
- [ ] Does the code actually implement the specification, not just scaffold it?
- [ ] Are dependency/config files included (requirements.txt, package.json, etc.)?
If any check fails, fix it before outputting.
"""

        response = await self.generate_response_async(prompt)

        # Parse files from response with Python syntax validation (FIX 2)
        output_dir = task.get("output_dir", "")
        files = self._parse_code_files_with_syntax_validation(
            response, max_retries=2, task_context=spec
        )

        # Write files using file operations, scoped to output_dir
        written_files = []
        for file_path, content in files.items():
            write_path = self._safe_output_path(file_path, output_dir)
            content = self._add_draft_header(write_path, content)
            result = self.use_tool("write_file", path=write_path, content=content)
            if result.get("success"):
                written_files.append(write_path)

        # FIX 3: Run compilation check after writing
        compilation = self._run_compilation_check(files, output_dir)
        if not compilation["success"] and compilation["failures"]:
            error_summary = "\n".join(
                f"  {fp}: {err}" for fp, err in compilation["failures"].items()
            )
            console.warning(f"[CompilationCheck] Failures detected:\n{error_summary}")
            # One LLM fix attempt for compilation failures
            fix_prompt = (
                f"The following files failed compilation. Fix the errors and re-output the complete corrected files.\n\n"
                f"COMPILATION ERRORS:\n{error_summary}\n\n"
                f"Re-output ONLY the failing files using:\n"
                f"**File Path:** `path/to/file.py`\n```python\ncomplete corrected code\n```"
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": fix_prompt},
            ]
            try:
                fix_response = self._call_llm_with_retry(messages, max_retries=1)
                fixed_files = self._parse_code_files_with_syntax_validation(fix_response, max_retries=1)
                for fpath, content in fixed_files.items():
                    write_path = self._safe_output_path(fpath, output_dir)
                    result = self.use_tool("write_file", path=write_path, content=content)
                    if result.get("success") and write_path not in written_files:
                        written_files.append(write_path)
                    files[fpath] = content
                # Re-run compilation check to update summary
                compilation = self._run_compilation_check(files, output_dir)
            except Exception as exc:
                console.warning(f"[CompilationCheck] Fix attempt failed: {exc}")

        # AgentCoder test-execution feedback loop
        loop_result = await self._test_and_fix_loop(
            files=files,
            written_files=written_files,
            output_dir=output_dir,
            language=language,
        )
        files = loop_result["files"]
        written_files = loop_result["written_files"]
        test_result = loop_result["test_result"]

        return TaskResult(
            success=len(written_files) > 0,
            output={
                "files_written": written_files,
                "implementation": response,
                "compilation": compilation,
                "test_result": {
                    "passed": test_result.passed if test_result else 0,
                    "failed": test_result.failed if test_result else 0,
                    "errors": test_result.errors if test_result else 0,
                    "success": test_result.success if test_result else False,
                } if test_result else None,
            },
            artifacts={"code": files}
        )

    async def _implement_stories(
        self,
        stories_raw: List[Dict[str, Any]],
        req_doc_dict: Dict[str, Any],
        language: str,
        output_dir: str,
        fallback_spec: str,
    ) -> TaskResult:
        """Implement each CTO story atomically (BMAD story-sharding pattern).

        Each story is a self-contained prompt kept under ~600 tokens so a
        7-8B model can handle it reliably. Results are merged into a single
        TaskResult.
        """
        from orchestrator.artifacts import RequirementsDoc, Story, AcceptanceCriterion

        # Reconstruct RequirementsDoc from dict for formatting
        req_doc: Optional[RequirementsDoc] = None
        if req_doc_dict:
            req_doc = RequirementsDoc(
                problem_summary=req_doc_dict.get("problem_summary", ""),
                jobs_to_be_done=req_doc_dict.get("jobs_to_be_done", ""),
                raw_text=req_doc_dict.get("raw_text", ""),
            )
            for ac in req_doc_dict.get("acceptance_criteria", []):
                req_doc.acceptance_criteria.append(AcceptanceCriterion(
                    id=ac.get("id", ""),
                    given=ac.get("given", ""),
                    when=ac.get("when", ""),
                    then=ac.get("then", ""),
                    priority=ac.get("priority", "P0"),
                    verification_cmd=ac.get("verification_cmd", ""),
                ))

        all_written: List[str] = []
        all_files: Dict[str, str] = {}
        combined_response: List[str] = []
        overall_success = False

        for story_dict in stories_raw:
            story = Story(
                story_id=story_dict.get("story_id", "story-1"),
                title=story_dict.get("title", "Implementation"),
                description=story_dict.get("description", fallback_spec[:400]),
                files=story_dict.get("files", []),
                ac_ids=story_dict.get("ac_ids", []),
                tech_context=story_dict.get("tech_context", ""),
            )

            story_prompt = story.format_for_developer(req_doc)

            prompt = f"""MANDATORY PREAMBLE:
ROOT CAUSE: [one sentence — the fundamental reason this problem exists]
SOLUTION APPROACH: [one sentence — how this story addresses the root cause]

Implement this story. Output complete, working {language} code.

{story_prompt}

Rules:
- Every file must be complete and runnable — no stubs, no TODOs, no placeholders
- Include all imports and config files needed
- Keep it focused: only implement what this story describes

Output each file using:
**File Path:** `path/to/file.ext`
```{language}
complete code here
```

SELF-CHECK:
- [ ] All files listed under FILES TO CREATE are present in output
- [ ] Every file runs without import errors
- [ ] Acceptance criteria above are satisfied by the code
"""
            console.info(f"[Developer] Implementing {story.story_id}: {story.title}")
            response = await self.generate_response_async(prompt)
            combined_response.append(f"### {story.story_id}: {story.title}\n{response}")

            # Parse and write files for this story
            files = self._parse_code_files_with_syntax_validation(
                response, max_retries=2, task_context=story.description
            )

            for file_path, content in files.items():
                write_path = self._safe_output_path(file_path, output_dir)
                content = self._add_draft_header(write_path, content)
                result = self.use_tool("write_file", path=write_path, content=content)
                if result.get("success"):
                    all_written.append(write_path)
                    overall_success = True
                all_files[file_path] = content

        # Run compilation check on all collected files
        compilation = self._run_compilation_check(all_files, output_dir)

        # AgentCoder test-execution feedback loop (runs after all stories written)
        if output_dir and all_files:
            loop_result = await self._test_and_fix_loop(
                files=all_files,
                written_files=all_written,
                output_dir=output_dir,
                language=language,
            )
            all_files = loop_result["files"]
            all_written = loop_result["written_files"]
            test_result = loop_result["test_result"]
        else:
            test_result = None

        full_response = "\n\n".join(combined_response)
        return TaskResult(
            success=overall_success,
            output={
                "files_written": all_written,
                "implementation": full_response,
                "compilation": compilation,
                "stories_implemented": len(stories_raw),
                "test_result": {
                    "passed": test_result.passed if test_result else 0,
                    "failed": test_result.failed if test_result else 0,
                    "errors": test_result.errors if test_result else 0,
                    "success": test_result.success if test_result else False,
                } if test_result else None,
            },
            artifacts={"code": all_files},
        )

    async def _test_and_fix_loop(
        self,
        files: Dict[str, str],
        written_files: List[str],
        output_dir: str,
        language: str,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """AgentCoder test-execution feedback loop.

        Runs actual tests against written files. On failure, feeds test output
        back to the LLM for a targeted fix. Repeats up to max_iterations times.
        Returns a dict with 'files', 'written_files', and 'test_result'.
        """
        from tools.test_runner import TestRunner

        runner = TestRunner()
        test_result = None
        iteration = 0

        while iteration < max_iterations:
            # Run tests in the output directory
            test_result = runner.run_tests(output_dir)
            iteration += 1

            if test_result.success or (test_result.failed == 0 and test_result.errors == 0):
                console.info(
                    f"[TestLoop] Iteration {iteration}: All tests passed "
                    f"({test_result.passed} passed)"
                )
                break

            failure_summary = (
                f"Tests failed (iteration {iteration}/{max_iterations}):\n"
                f"  Passed: {test_result.passed}  Failed: {test_result.failed}  Errors: {test_result.errors}\n\n"
            )
            if test_result.failures:
                failure_summary += "FAILURES:\n"
                for f in test_result.failures[:10]:
                    name = f.get("name", f.get("test", "unknown"))
                    msg = f.get("message", f.get("error", ""))
                    failure_summary += f"  [{name}]\n{msg.strip()}\n\n"
            if test_result.error_output:
                failure_summary += f"STDERR:\n{test_result.error_output[-2000:]}\n"

            console.warning(f"[TestLoop] {failure_summary[:300]}...")

            if iteration >= max_iterations:
                console.warning(f"[TestLoop] Max iterations reached, stopping.")
                break

            # Build fix prompt with current file contents
            current_code_snippets = ""
            for fpath, content in files.items():
                ext = Path(fpath).suffix.lstrip(".") or language
                current_code_snippets += f"**File Path:** `{fpath}`\n```{ext}\n{content}\n```\n\n"

            fix_prompt = (
                f"The following test failures occurred. Fix the code so all tests pass.\n\n"
                f"{failure_summary}\n"
                f"CURRENT CODE:\n{current_code_snippets}\n"
                f"Output ONLY the corrected files using:\n"
                f"**File Path:** `path/to/file.ext`\n```{language}\ncomplete corrected code\n```\n\n"
                f"Fix ONLY what the test failures indicate. Do not change unrelated code."
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": fix_prompt},
            ]
            try:
                fix_response = self._call_llm_with_retry(messages, max_retries=1)
                fixed_files = self._parse_code_files_with_syntax_validation(
                    fix_response, max_retries=1
                )
                for fpath, content in fixed_files.items():
                    write_path = self._safe_output_path(fpath, output_dir)
                    result = self.use_tool("write_file", path=write_path, content=content)
                    if result.get("success"):
                        if write_path not in written_files:
                            written_files.append(write_path)
                    files[fpath] = content
            except Exception as exc:
                console.warning(f"[TestLoop] Fix attempt {iteration} failed: {exc}")
                break

        return {
            "files": files,
            "written_files": written_files,
            "test_result": test_result,
        }

    async def _write_code(self, task: Dict[str, Any]) -> TaskResult:
        """Write code for a specific requirement."""
        requirement = task.get("requirement", "")
        language = task.get("language", "python")
        context = task.get("context", "")

        prompt = f"""
Write {language} code for this requirement:

{requirement}

Context:
{context}

Provide:
1. Complete, working code
2. Comments explaining key parts
3. Example usage
4. Any assumptions made

Write clean, production-ready code.
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "code": response,
                "language": language
            },
            artifacts={"code_snippet": response}
        )

    async def _fix_bug(self, task: Dict[str, Any]) -> TaskResult:
        """Fix a bug or address QA/CEO rejection feedback."""
        bug_description = task.get("bug_description", "")
        code = task.get("code", "")
        file_path = task.get("file_path", "")
        error_message = task.get("error_message", "")
        qa_report = task.get("qa_report", "")
        original_requirements = task.get("original_requirements", "")
        output_dir = task.get("output_dir", "")

        strategy_context = task.get("strategy_context", "")

        prompt = f"""Fix issues in an existing solution.

FEEDBACK / BUG DESCRIPTION:
{bug_description}

{f"QA REPORT: {qa_report}" if qa_report else ""}
{f"ORIGINAL REQUIREMENTS: {original_requirements}" if original_requirements else ""}
{f"ERROR MESSAGE: {error_message}" if error_message else ""}
{f"FILE: {file_path}" if file_path else ""}
{f"CURRENT CODE:{chr(10)}```{chr(10)}{code}{chr(10)}```" if code else ""}
{f"PREVIOUS ATTEMPTS (do NOT repeat):{chr(10)}{strategy_context}" if strategy_context else ""}

Steps:
1. Read the feedback. List the specific issues (max 5).
2. For each issue, identify the exact file and line.
3. Apply the fix. Output the COMPLETE corrected file.

Output each fixed file using this EXACT format:
**File Path:** `path/to/file.ext`
```language
complete corrected code here
```

Output COMPLETE files, not partial diffs.

SELF-CHECK:
- Did I fix every issue in the feedback?
- Is each file complete and runnable?
- Did I avoid changing unrelated code?

Before outputting, mentally trace through the code:
- Does the entry point exist and call the right functions?
- Are all imports present and correct?
- Would `python main.py` (or equivalent) actually run?
"""

        response = await self.generate_response_async(prompt)

        # Parse and write fixed files, with syntax validation (FIX 2)
        files = self._parse_code_files_with_syntax_validation(
            response, max_retries=2, task_context=bug_description
        )
        written_files = []

        if files:
            for fpath, content in files.items():
                write_path = self._safe_output_path(fpath, output_dir)
                content = self._add_draft_header(write_path, content)
                result = self.use_tool("write_file", path=write_path, content=content)
                if result.get("success"):
                    written_files.append(write_path)
        elif file_path:
            # Fallback: extract single code block for the specified file
            fixed_code = self._extract_code_block(response)
            if fixed_code:
                write_path = self._safe_output_path(file_path, output_dir)
                self.use_tool("write_file", path=write_path, content=fixed_code)
                written_files.append(write_path)
                files[file_path] = fixed_code

        # FIX 3: Run compilation check after writing
        if files:
            compilation = self._run_compilation_check(files, output_dir)
            if not compilation["success"] and compilation["failures"]:
                error_summary = "\n".join(
                    f"  {fp}: {err}" for fp, err in compilation["failures"].items()
                )
                console.warning(f"[CompilationCheck] Fix: failures:\n{error_summary}")
                fix_prompt = (
                    f"The following files failed compilation after your fix. Correct the errors.\n\n"
                    f"COMPILATION ERRORS:\n{error_summary}\n\n"
                    f"Re-output ONLY the failing files using:\n"
                    f"**File Path:** `path/to/file.py`\n```python\ncomplete corrected code\n```"
                )
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": fix_prompt},
                ]
                try:
                    fix_response = self._call_llm_with_retry(messages, max_retries=1)
                    fixed_files = self._parse_code_files_with_syntax_validation(fix_response, max_retries=1)
                    for fpath, content in fixed_files.items():
                        write_path = self._safe_output_path(fpath, output_dir)
                        result = self.use_tool("write_file", path=write_path, content=content)
                        if result.get("success") and write_path not in written_files:
                            written_files.append(write_path)
                    compilation = self._run_compilation_check({**files, **fixed_files}, output_dir)
                except Exception as exc:
                    console.warning(f"[CompilationCheck] Fix attempt failed: {exc}")
        else:
            compilation = {"passed": 0, "failed": 0, "summary": "No files to check", "success": True}

        return TaskResult(
            success=True,
            output={
                "fix": response,
                "files_fixed": written_files,
                "file_path": file_path,
                "compilation": compilation,
            },
            artifacts={"bug_fix": response}
        )

    async def _write_tests(self, task: Dict[str, Any]) -> TaskResult:
        """Write tests for code."""
        code = task.get("code", "")
        file_path = task.get("file_path", "")
        test_framework = task.get("framework", "pytest")

        prompt = f"""
Write comprehensive tests for this code using {test_framework}:

File: {file_path}
```
{code}
```

Include:
1. Unit tests for each function/method
2. Edge cases
3. Error handling tests
4. Integration tests if applicable

Use proper test naming conventions and include docstrings.
"""

        response = await self.generate_response_async(prompt)

        # Extract test code
        test_code = self._extract_code_block(response)
        test_file = file_path.replace(".py", "_test.py") if file_path else "tests.py"
        output_dir = task.get("output_dir", "")

        if test_code:
            write_path = self._safe_output_path(test_file, output_dir)
            self.use_tool("write_file", path=write_path, content=test_code)

        return TaskResult(
            success=True,
            output={
                "tests": response,
                "test_file": test_file
            },
            artifacts={"test_code": test_code}
        )

    async def _refactor_code(self, task: Dict[str, Any]) -> TaskResult:
        """Refactor existing code."""
        code = task.get("code", "")
        file_path = task.get("file_path", "")
        goals = task.get("goals", ["improve readability", "reduce complexity"])

        prompt = f"""
Refactor this code:

File: {file_path}
```
{code}
```

Refactoring Goals:
{chr(10).join(f'- {g}' for g in goals)}

Please:
1. Provide refactored code
2. Explain each change made
3. Note any breaking changes
4. Suggest further improvements

Maintain existing functionality while improving code quality.

Before outputting, mentally trace through the code:
- Does the entry point exist and call the right functions?
- Are all imports present and correct?
- Would `python main.py` (or equivalent) actually run?
"""

        response = await self.generate_response_async(prompt)

        # Extract refactored code
        refactored_code = self._extract_code_block(response)

        output_dir = task.get("output_dir", "")
        if file_path and refactored_code:
            write_path = self._safe_output_path(file_path, output_dir)
            self.use_tool("write_file", path=write_path, content=refactored_code)

        return TaskResult(
            success=True,
            output={
                "refactored": response,
                "file_path": file_path
            },
            artifacts={"refactored_code": refactored_code}
        )

    async def _create_project(self, task: Dict[str, Any]) -> TaskResult:
        """Create a new project structure with template support."""
        project_name = task.get("name", "new_project")
        project_type = task.get("project_type", task.get("language", "python"))
        framework = task.get("framework", None)  # Specific framework (react, fastapi, etc.)
        description = task.get("description", "")
        features = task.get("features", [])
        output_dir = task.get("output_dir", ".")

        console.agent_action("Developer", "Creating Project", f"{project_name} (type: {project_type}, framework: {framework})")

        # Try to use template system if framework is specified
        if framework:
            try:
                console.info(f"Using template system for {framework}...")
                files = generate_project(
                    framework=framework,
                    output_dir=output_dir,
                    project_name=project_name
                )

                created_files = []
                for file_path, content in files.items():
                    console.info(f"  Creating {file_path}")
                    result = self.use_tool("write_file", path=file_path, content=content)
                    if result.get("success"):
                        created_files.append(file_path)

                if created_files:
                    console.success(f"Project scaffolded using {framework} template - {len(created_files)} files created")
                    return TaskResult(
                        success=True,
                        output={
                            "project_name": project_name,
                            "framework": framework,
                            "files_created": created_files,
                            "method": "template"
                        },
                        artifacts={"project_structure": files}
                    )
            except Exception as e:
                console.warning(f"Template generation failed: {e}, falling back to LLM generation")
                # Fall through to LLM-based generation

        # Fallback to LLM-based project generation
        console.info("Using LLM-based project generation...")

        # Language-specific configuration hints
        config_hints = {
            "python": "pyproject.toml or requirements.txt, setup.py if needed",
            "javascript": "package.json, .eslintrc, .prettierrc",
            "typescript": "package.json, tsconfig.json, .eslintrc",
            "go": "go.mod, go.sum",
            "rust": "Cargo.toml",
            "java": "pom.xml or build.gradle",
            "csharp": ".csproj file",
        }

        config_hint = config_hints.get(project_type, "appropriate configuration files")

        prompt = f"""
Create a project structure for:

Name: {project_name}
Language/Type: {project_type}
Description: {description}
Features: {', '.join(features) if features else 'Core functionality'}

Provide:
1. Complete directory structure following {project_type} best practices
2. Initial files with boilerplate code
3. Configuration files ({config_hint})
4. README with setup and run instructions
5. Basic test setup for {project_type}

IMPORTANT: Format each file with this EXACT format (required for parsing):
**File Path:** `path/to/file.ext`
```{project_type}
code here
```

Example for Python:
**File Path:** `src/main.py`
```python
def main():
    print("Hello World")

if __name__ == "__main__":
    main()
```
"""

        response = await self.generate_response_async(prompt)

        # Parse and create files, with syntax validation for .py files (FIX 2)
        files = self._parse_code_files_with_syntax_validation(
            response, max_retries=2, task_context=description
        )
        created_files = []

        for file_path, content in files.items():
            write_path = self._safe_output_path(file_path, output_dir)
            content = self._add_draft_header(write_path, content)
            console.info(f"  Creating {write_path}")
            result = self.use_tool("write_file", path=write_path, content=content)
            if result.get("success"):
                created_files.append(write_path)

        # FIX 3: Run compilation check after writing
        if files:
            compilation = self._run_compilation_check(files, output_dir)
            if not compilation["success"] and compilation["failures"]:
                error_summary = "\n".join(
                    f"  {fp}: {err}" for fp, err in compilation["failures"].items()
                )
                console.warning(f"[CompilationCheck] Project: failures:\n{error_summary}")
                fix_prompt = (
                    f"The following project files failed compilation. Fix the errors.\n\n"
                    f"COMPILATION ERRORS:\n{error_summary}\n\n"
                    f"Re-output ONLY the failing files using:\n"
                    f"**File Path:** `path/to/file.py`\n```python\ncomplete corrected code\n```"
                )
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": fix_prompt},
                ]
                try:
                    fix_response = self._call_llm_with_retry(messages, max_retries=1)
                    fixed_files = self._parse_code_files_with_syntax_validation(fix_response, max_retries=1)
                    for fpath, content in fixed_files.items():
                        write_path = self._safe_output_path(fpath, output_dir)
                        result = self.use_tool("write_file", path=write_path, content=content)
                        if result.get("success") and write_path not in created_files:
                            created_files.append(write_path)
                    files.update(fixed_files)
                except Exception as exc:
                    console.warning(f"[CompilationCheck] Fix attempt failed: {exc}")

        if created_files:
            console.success(f"Project created using LLM generation - {len(created_files)} files")

        return TaskResult(
            success=len(created_files) > 0,
            output={
                "project_name": project_name,
                "files_created": created_files,
                "method": "llm"
            },
            artifacts={"project_structure": files}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general development tasks."""
        description = task.get("description", "")

        response = await self.generate_response_async(
            f"As a Senior Developer, please address: {description}"
        )

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  HELPER METHODS
    # ============================================================

    def _validate_python_syntax(self, filepath: str, content: str) -> tuple:
        """Validate Python syntax using ast.parse.

        Args:
            filepath: File path (used only for error messages)
            content: Python source code to validate

        Returns:
            (valid: bool, error_message: str)
        """
        import ast
        try:
            ast.parse(content)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def _run_compilation_check(self, files: Dict[str, str], output_dir: str = "") -> Dict[str, Any]:
        """Run lightweight compilation checks on written .py files.

        For each .py file: validates syntax via ast + py_compile.
        Checks requirements.txt exists if referenced.
        Returns a summary dict and sends failures back to LLM for one fix attempt.

        Args:
            files: Dict mapping file path -> content
            output_dir: Base output directory (for resolving paths)

        Returns:
            {
                "passed": int,
                "failed": int,
                "failures": {filepath: error_msg},
                "requirements_noted": bool,
                "summary": str,
                "success": bool,
            }
        """
        import ast
        import os

        passed = 0
        failed = 0
        failures: Dict[str, str] = {}

        py_files = {fp: content for fp, content in files.items() if fp.endswith(".py")}

        for fp, content in py_files.items():
            # First do ast validation (no disk required)
            valid, err = self._validate_python_syntax(fp, content)
            if not valid:
                failed += 1
                failures[fp] = err
                continue

            # Then try py_compile on the actual file on disk
            write_path = self._safe_output_path(fp, output_dir) if output_dir else fp
            if os.path.isfile(write_path):
                try:
                    result = self.use_tool(
                        "execute_command",
                        command=(
                            f'python -c "'
                            f'import ast, py_compile; '
                            f'ast.parse(open(\\"{write_path}\\").read()); '
                            f'py_compile.compile(\\"{write_path}\\", doraise=True)"'
                        ),
                        timeout=10,
                    )
                    if result.get("success") or result.get("returncode", 1) == 0:
                        passed += 1
                    else:
                        failed += 1
                        stderr = result.get("stderr", "") or result.get("output", "")
                        failures[fp] = stderr[:300] if stderr else "Compilation failed"
                except Exception as exc:
                    # execute_command not available or timed out — fall back to ast-only pass
                    passed += 1
            else:
                # File not on disk yet — ast pass is enough
                passed += 1

        # Note requirements.txt
        has_requirements = "requirements.txt" in files
        requirements_noted = False
        if has_requirements:
            requirements_noted = True
            console.info("[CompilationCheck] requirements.txt present — dependencies NOT installed (noted only)")

        summary = f"{passed}/{len(py_files)} Python files passed compilation"
        if failures:
            summary += f"; {failed} failed: {', '.join(failures.keys())}"

        console.info(f"[CompilationCheck] {summary}")

        return {
            "passed": passed,
            "failed": failed,
            "failures": failures,
            "requirements_noted": requirements_noted,
            "summary": summary,
            "success": failed == 0,
        }

    def _parse_code_files_with_syntax_validation(
        self,
        response: str,
        max_retries: int = 2,
        task_context: str = "",
    ) -> Dict[str, str]:
        """Parse code files from LLM response, validating Python syntax.

        For .py files: validates syntax before accepting.
        If any file has syntax errors: re-prompts the LLM with errors (max 2 retries).
        Only returns files that pass validation.

        Args:
            response: Raw LLM response text
            max_retries: Max syntax-fix retry attempts
            task_context: Original task description (used in retry prompts)

        Returns:
            Dict of {filepath: content} where all .py files have valid syntax
        """
        files = self._parse_code_files(response)

        for attempt in range(max_retries + 1):
            syntax_errors: Dict[str, str] = {}
            for fp, content in files.items():
                if fp.endswith(".py"):
                    valid, err = self._validate_python_syntax(fp, content)
                    if not valid:
                        syntax_errors[fp] = err

            if not syntax_errors:
                break  # All .py files pass

            console.warning(
                f"[SyntaxValidation] Attempt {attempt + 1}: "
                f"{len(syntax_errors)} file(s) have syntax errors: "
                + ", ".join(syntax_errors.keys())
            )

            if attempt == max_retries:
                # Final attempt exhausted — return only valid files
                console.warning("[SyntaxValidation] Max retries exhausted. Excluding files with syntax errors.")
                files = {fp: c for fp, c in files.items() if fp not in syntax_errors}
                break

            # Build error feedback for LLM retry
            error_lines = "\n".join(
                f"  {fp}: {err}" for fp, err in syntax_errors.items()
            )
            retry_prompt = (
                f"The following Python files in your previous response have syntax errors. "
                f"Fix ONLY the syntax errors and re-output the complete corrected files.\n\n"
                f"SYNTAX ERRORS:\n{error_lines}\n\n"
                f"Original task context: {task_context}\n\n"
                f"Re-output ONLY the files with errors, using the same file format:\n"
                f"**File Path:** `path/to/file.py`\n```python\ncomplete corrected code\n```"
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": retry_prompt},
            ]
            try:
                fix_response = self._call_llm_with_retry(messages, max_retries=1)
                fixed_files = self._parse_code_files(fix_response)
                # Merge: replace errored files with fixed versions
                for fp, content in fixed_files.items():
                    files[fp] = content
            except Exception as exc:
                console.warning(f"[SyntaxValidation] Retry failed: {exc}")
                files = {fp: c for fp, c in files.items() if fp not in syntax_errors}
                break

        return files

    def _parse_code_files(self, response: str) -> Dict[str, str]:
        """Parse code files from LLM response - supports multiple formats with enhanced error handling."""
        # Use enhanced parser with error collection
        parser = EnhancedCodeParser(strict_mode=False)

        try:
            parsed_files = parser.parse(response)

            # Log any warnings
            if parser.warnings:
                warning_msg = parser.get_parse_report()
                console.warning(f"Code parsing warnings:\n{warning_msg}")

            # Convert ParsedFile objects to simple dict
            files = {path: pf.content for path, pf in parsed_files.items()}

            # Log confidence scores
            low_confidence = [
                f"{path} (confidence: {pf.confidence:.2f})"
                for path, pf in parsed_files.items()
                if pf.confidence < 0.7
            ]
            if low_confidence:
                console.warning(f"Low confidence files: {', '.join(low_confidence)}")

            return files

        except Exception as e:
            # Fallback to basic parsing on critical failure
            console.warning(f"Enhanced parser failed: {e}, falling back to simple extraction")
            return self._parse_code_files_fallback(response)

    def _parse_code_files_fallback(self, response: str) -> Dict[str, str]:
        """Fallback parser using simple regex (when enhanced parser fails)."""
        import re
        files = {}

        # Try to find file markers with basic patterns
        file_pattern = r'(?:===|###|####)?\s*(?:FILE|File):\s*`?([^\n`]+)`?'
        matches = list(re.finditer(file_pattern, response))

        if not matches:
            # No files found, try to extract as single code block
            code = extract_single_code_block(response)
            if code:
                files['output.py'] = code

        for i, match in enumerate(matches):
            filepath = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(response)

            content_section = response[start:end]
            code = extract_single_code_block(content_section)

            if code:
                files[filepath] = code

        return files

    def _extract_code_block(self, response: str) -> str:
        """Extract code from markdown code block (uses enhanced parser)."""
        return extract_single_code_block(response)

    # ============================================================
    #  DEVELOPER-SPECIFIC METHODS
    # ============================================================

    def quick_implement(self, description: str, language: str = "python") -> str:
        """Quick implementation for simple tasks."""
        prompt = f"""
Implement this in {language}:

{description}

Provide clean, working code only (no explanation needed).
"""
        return self.generate_response(prompt, use_first_principles=False)

    def explain_code(self, code: str) -> str:
        """Explain what code does."""
        prompt = f"""
Explain this code:

```
{code}
```

Provide:
1. High-level overview
2. Step-by-step walkthrough
3. Key concepts used
4. Potential issues or improvements
"""
        return self.generate_response(prompt)

    def suggest_improvements(self, code: str) -> str:
        """Suggest improvements for code."""
        prompt = f"""
Review this code and suggest improvements:

```
{code}
```

Consider:
- Performance optimizations
- Code clarity
- Error handling
- Security
- Best practices
"""
        return self.generate_response(prompt, use_first_principles=True)
