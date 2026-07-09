#!/usr/bin/env python3
"""
Interactive Coding Mode - Similar to Claude Code, Codex, Gemini CLI

Works directly in your current directory, edits existing files, and provides
interactive code generation and review.

Usage:
    python interactive_mode.py "Add user authentication to my Flask app"
    python interactive_mode.py --review auth.py
    python interactive_mode.py --fix "Bug in login function"
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import DeveloperAgent, QAEngineerAgent, CTOAgent
from config.models import ModelConfig
from tools import UnifiedTools
from ui.console import console
from utils.enhanced_interactive_session import (
    EnhancedInteractiveSession,
    ProgressIndicator
)
from utils.cost_tracker import CostTracker
from utils.structured_logging import get_structured_logger, LogLevel
from utils.error_recovery import ErrorRecoverySystem, RetryStrategy


class InteractiveCodingSession:
    """
    Interactive coding session - works like Claude Code.

    - Generates files in current directory
    - Edits existing files
    - Reviews code
    - Provides explanations
    """

    def __init__(
        self,
        workspace_root: str = ".",
        model_config: ModelConfig = None,
        language: str = "python",
        data_dir: str = None
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.model_config = model_config or ModelConfig()
        self.language = language

        # Data directory for persistence (memory, logs, costs)
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path.home() / ".multi-agent-llm-company-system"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        memory_dir = str(self.data_dir / "memory")

        # Initialize agents with persistent memory
        self.developer = DeveloperAgent(
            model=self.model_config.get_model_name("developer"),
            workspace_root=str(self.workspace_root),
            memory_persist_dir=memory_dir
        )

        self.qa = QAEngineerAgent(
            model=self.model_config.get_model_name("qa_engineer"),
            workspace_root=str(self.workspace_root),
            memory_persist_dir=memory_dir
        )

        self.cto = CTOAgent(
            model=self.model_config.get_model_name("cto"),
            workspace_root=str(self.workspace_root),
            memory_persist_dir=memory_dir
        )

        # Enable streaming on all agents for real-time output
        for agent in [self.developer, self.qa, self.cto]:
            agent._streaming_enabled = True
            agent._stream_callback = console.create_stream_callback(agent.name)

        # Initialize tools with persistent memory
        self.tools = UnifiedTools(
            workspace_root=str(self.workspace_root),
            persist_dir=memory_dir
        )

        # Infrastructure: cost tracking
        import utils.cost_tracker as _ct_mod
        self.cost_tracker = CostTracker(
            history_file=self.data_dir / "costs" / "cost_history.json"
        )
        _ct_mod._cost_tracker = self.cost_tracker

        # Infrastructure: structured logging
        self.logger = get_structured_logger(
            name="interactive",
            level=LogLevel.INFO,
            json_output=self.data_dir / "logs" / "interactive.jsonl"
        )

        # Infrastructure: error recovery
        self.error_recovery = ErrorRecoverySystem(
            max_retries=2,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            enable_logging=True
        )

        # Enhanced session for multi-turn conversations
        self.enhanced_session = None

    async def conversational_mode(self, initial_request: Optional[str] = None):
        """
        Start a conversational coding session with context retention.

        This mode allows multiple back-and-forth interactions like Claude Code.
        """
        console.section("CONVERSATIONAL CODING MODE")
        console.info("Multi-turn conversation with context retention enabled")
        console.info(f"Working directory: {self.workspace_root}")
        console.info("Type 'exit' to quit, 'files' to see modified files, 'summary' for session info")
        console.info("Type 'cost' to see token usage\n")

        # Start cost tracking session
        self.cost_tracker.start_session("interactive")
        self.logger.info("Interactive session started")

        # Initialize enhanced session
        self.enhanced_session = EnhancedInteractiveSession(
            workspace_root=str(self.workspace_root)
        )

        # Setup progress indicator
        progress = ProgressIndicator()
        self.enhanced_session.set_progress_callback(progress.update)

        # Handle initial request if provided
        if initial_request:
            console.info(f"User: {initial_request}\n")
            result = await self.enhanced_session.chat(
                user_message=initial_request,
                agent=self.developer,
                task_type="implement_feature"
            )

            console.success("Assistant:")
            print(result["response"])
            print()

        # Start conversation loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() == 'exit':
                    console.info("Ending conversation session...")
                    break

                elif user_input.lower() == 'files':
                    files = self.enhanced_session.list_modified_files()
                    console.info(f"\nModified files in this session ({len(files)}):")
                    for f in files:
                        console.info(f"  - {f}")
                    print()
                    continue

                elif user_input.lower() == 'summary':
                    summary = self.enhanced_session.get_conversation_summary()
                    console.info(f"\n{summary}\n")
                    continue

                elif user_input.lower() == 'cost':
                    session = self.cost_tracker.get_current_session()
                    if session:
                        console.info(f"\n  Tokens used: {session.total_tokens:,}")
                        console.info(f"  Requests: {session.request_count}")
                    else:
                        console.info("\n  No token usage recorded yet.")
                    print()
                    continue

                elif user_input.lower().startswith('review '):
                    # Quick review command
                    file_path = user_input[7:].strip()
                    await self.review_code([file_path])
                    continue

                # Determine task type based on keywords
                task_type = "implement_feature"
                agent = self.developer

                if any(word in user_input.lower() for word in ['review', 'check', 'analyze', 'issues']):
                    task_type = "review_code"
                    agent = self.qa
                elif any(word in user_input.lower() for word in ['fix', 'bug', 'error', 'broken']):
                    task_type = "fix_bug"
                    agent = self.developer
                elif any(word in user_input.lower() for word in ['refactor', 'improve', 'optimize']):
                    task_type = "refactor"
                    agent = self.developer
                elif any(word in user_input.lower() for word in ['explain', 'what does', 'how does']):
                    task_type = "general_task"
                    agent = self.developer

                # Execute with context
                result = await self.enhanced_session.chat(
                    user_message=user_input,
                    agent=agent,
                    task_type=task_type
                )

                # Display response
                console.success("\nAssistant:")
                print(result["response"])
                print()

            except KeyboardInterrupt:
                console.info("\nEnding conversation session...")
                break
            except EOFError:
                break
            except Exception as e:
                console.error(f"\nError: {e}\n")
                continue

        # End cost tracking and show final summary
        cost_session = self.cost_tracker.end_session()
        self.logger.info("Interactive session ended")

        summary = self.enhanced_session.get_conversation_summary()
        console.info(f"\nSession Summary:")
        print(summary)
        if cost_session:
            console.info(f"  Tokens used: {cost_session.total_tokens:,}")
            console.info(f"  Requests: {cost_session.request_count}")

    async def code_request(self, request: str, context_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Handle a coding request (like Claude Code).

        Args:
            request: What to build/change (e.g., "Add user authentication")
            context_files: Files to read for context

        Returns:
            Result with files created/modified
        """
        console.section("INTERACTIVE CODING SESSION")
        console.info(f"Request: {request}")
        console.info(f"Working directory: {self.workspace_root}")

        # Read context files
        context = ""
        if context_files:
            console.info(f"Reading context from {len(context_files)} files...")
            for file_path in context_files:
                try:
                    content = (self.workspace_root / file_path).read_text()
                    context += f"\n\n=== {file_path} ===\n{content}"
                except Exception as e:
                    console.warning(f"Could not read {file_path}: {e}")

        # Analyze existing codebase
        console.agent_action("Developer", "Analyzing codebase", "Understanding current structure...")
        codebase_files = list(self.workspace_root.glob("*.py"))[:10]  # First 10 Python files
        for file in codebase_files:
            try:
                content = file.read_text()
                context += f"\n\n=== {file.name} ===\n{content}"
            except Exception:
                pass

        # Generate implementation
        console.agent_action("Developer", "Implementing", f"{request}...")

        task = {
            "type": "implement_feature",
            "specification": request,
            "architecture": f"Work directly in current directory. Language: {self.language}. Follow existing code patterns.",
            "file_structure": {"current_dir": str(self.workspace_root)},
            "language": self.language,
            "context": context
        }

        result = await self.developer.execute_task(task)

        if result.success:
            impl = result.output
            files_written = impl.get("files_written", [])

            console.success(f"Implementation complete!")
            console.info(f"Files modified: {len(files_written)}")

            for file_path in files_written:
                console.info(f"  ✓ {file_path}")

            return {
                "success": True,
                "files": files_written,
                "implementation": impl.get("implementation", "")
            }
        else:
            console.error("Implementation failed")
            return {"success": False, "error": result.output}

    async def review_code(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Review code files (like Claude Code review).

        Args:
            file_paths: Files to review

        Returns:
            Review results with suggestions
        """
        console.section("CODE REVIEW")
        console.info(f"Reviewing {len(file_paths)} files...")

        reviews = {}

        for file_path in file_paths:
            full_path = self.workspace_root / file_path
            if not full_path.exists():
                console.warning(f"File not found: {file_path}")
                continue

            console.agent_action("QA Engineer", "Reviewing", file_path)

            try:
                code = full_path.read_text()

                task = {
                    "type": "review_code",
                    "code": code,
                    "file_path": file_path,
                    "language": self.language
                }

                result = await self.qa.execute_task(task)

                reviews[file_path] = {
                    "issues": result.output.get("issues", []) if isinstance(result.output, dict) else [],
                    "suggestions": result.output.get("suggestions", []) if isinstance(result.output, dict) else [],
                    "rating": result.output.get("rating", "unknown") if isinstance(result.output, dict) else "unknown"
                }

                # Display review
                review_data = reviews[file_path]
                console.info(f"\n{file_path}:")
                if isinstance(result.output, dict):
                    console.info(f"  Rating: {review_data['rating']}")
                    if review_data['issues']:
                        console.warning(f"  Issues: {len(review_data['issues'])}")
                        for issue in review_data['issues'][:3]:
                            console.warning(f"    - {issue}")
                    if review_data['suggestions']:
                        console.info(f"  Suggestions: {len(review_data['suggestions'])}")
                        for suggestion in review_data['suggestions'][:3]:
                            console.info(f"    - {suggestion}")

            except Exception as e:
                console.error(f"Review failed for {file_path}: {e}")
                reviews[file_path] = {"error": str(e)}

        return {"reviews": reviews}

    async def fix_issue(self, issue_description: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Fix a specific issue (like Claude Code fix).

        Args:
            issue_description: What to fix
            file_path: Optional specific file to fix

        Returns:
            Fix results
        """
        console.section("FIXING ISSUE")
        console.info(f"Issue: {issue_description}")

        if file_path:
            console.info(f"Target file: {file_path}")
            full_path = self.workspace_root / file_path
            if not full_path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            code = full_path.read_text()

            console.agent_action("Developer", "Fixing", issue_description)

            task = {
                "type": "fix_bug",
                "bug_description": issue_description,
                "code": code,
                "file_path": file_path,
                "error_message": ""
            }

            result = await self.developer.execute_task(task)

            if result.success:
                console.success(f"Fixed in {file_path}")
                return {
                    "success": True,
                    "file": file_path,
                    "fix": result.output.get("fix", "")
                }
            else:
                return {"success": False, "error": "Fix failed"}

        else:
            # General fix - analyze codebase and fix
            console.agent_action("Developer", "Analyzing", "Finding files related to issue...")

            # Find relevant files
            relevant_files = []
            for py_file in self.workspace_root.glob("*.py"):
                try:
                    content = py_file.read_text()
                    # Simple heuristic: check if issue keywords are in file
                    if any(keyword.lower() in content.lower() for keyword in issue_description.split()[:5]):
                        relevant_files.append(py_file.name)
                except Exception:
                    pass

            if relevant_files:
                console.info(f"Found {len(relevant_files)} relevant files")
                return await self.fix_issue(issue_description, relevant_files[0])
            else:
                return {"success": False, "error": "Could not find relevant files"}

    async def explain_code(self, file_path: str) -> Dict[str, Any]:
        """
        Explain code in a file (like Claude Code explain).

        Args:
            file_path: File to explain

        Returns:
            Explanation
        """
        console.section("CODE EXPLANATION")
        console.info(f"File: {file_path}")

        full_path = self.workspace_root / file_path
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        code = full_path.read_text()

        console.agent_action("CTO", "Analyzing", "Understanding code structure...")

        task = {
            "type": "explain_code",
            "code": code,
            "file_path": file_path
        }

        result = await self.cto.execute_task(task)

        explanation = result.output.get("explanation", str(result.output)) if isinstance(result.output, dict) else str(result.output)

        console.info("\nExplanation:")
        console.info(explanation)

        return {
            "success": True,
            "explanation": explanation
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Interactive Coding Mode - Like Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Code generation in current directory
  python interactive_mode.py "Add user authentication with JWT"
  python interactive_mode.py "Create a REST API for todo items"

  # Review existing code
  python interactive_mode.py --review app.py models.py

  # Fix issues
  python interactive_mode.py --fix "Memory leak in user session" --file auth.py
  python interactive_mode.py --fix "SQL injection vulnerability"

  # Explain code
  python interactive_mode.py --explain utils.py

  # Specify language
  python interactive_mode.py "Add TypeScript types" --language typescript
  python interactive_mode.py "Add Go tests" --language go

This mode works directly in your current directory, just like Claude Code!
        """
    )

    parser.add_argument(
        "request",
        nargs="?",
        help="What to build/implement (e.g., 'Add user login')"
    )

    parser.add_argument(
        "--review",
        nargs="+",
        metavar="FILE",
        help="Review code files"
    )

    parser.add_argument(
        "--fix",
        type=str,
        metavar="ISSUE",
        help="Fix an issue"
    )

    parser.add_argument(
        "--file",
        type=str,
        metavar="FILE",
        help="Specific file for fix operation"
    )

    parser.add_argument(
        "--explain",
        type=str,
        metavar="FILE",
        help="Explain code in file"
    )

    parser.add_argument(
        "--context",
        nargs="+",
        metavar="FILE",
        help="Context files to read"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="python",
        metavar="LANG",
        help="Programming language (python, javascript, typescript, go, rust)"
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Working directory (default: current directory)"
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start conversational mode with multi-turn interactions (like Claude Code)"
    )

    args = parser.parse_args()

    if not any([args.request, args.review, args.fix, args.explain, args.chat]):
        parser.print_help()
        sys.exit(1)

    # Setup
    console.print_header()
    console.info(f"Working directory: {Path(args.dir).resolve()}")
    console.info(f"Language: {args.language}")

    config = ModelConfig()

    session = InteractiveCodingSession(
        workspace_root=args.dir,
        model_config=config,
        language=args.language
    )

    # Execute command
    result = None  # Initialize to avoid unbound variable

    if args.chat:
        # Conversational mode
        await session.conversational_mode(initial_request=args.request)
        return  # No summary needed for conversational mode

    elif args.review:
        result = await session.review_code(args.review)

    elif args.fix:
        result = await session.fix_issue(args.fix, args.file)

    elif args.explain:
        result = await session.explain_code(args.explain)

    elif args.request:
        result = await session.code_request(args.request, args.context)

    # Print summary
    console.section("SESSION COMPLETE")
    if result and result.get("success"):
        console.success("Operation completed successfully")
    elif result:
        console.error(f"Operation failed: {result.get('error', 'Unknown error')}")
    else:
        console.error("No operation was executed")


if __name__ == "__main__":
    asyncio.run(main())
