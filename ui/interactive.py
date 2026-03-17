"""
Interactive User Interface Tools for Company AGI.

Provides Claude Code-style interactive features:
- AskUserQuestion tool for clarification
- Multi-select and single-select options
- Timeout handling
- Input validation
"""

import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TextIO


class QuestionType(Enum):
    """Types of questions."""
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    FREE_TEXT = "free_text"
    YES_NO = "yes_no"
    CONFIRMATION = "confirmation"


@dataclass
class QuestionOption:
    """An option for a question."""
    label: str
    description: str = ""
    value: Any = None

    def __post_init__(self) -> None:
        if self.value is None:
            self.value = self.label


@dataclass
class Question:
    """A question to ask the user."""
    question: str
    header: str = ""
    question_type: QuestionType = QuestionType.SINGLE_SELECT
    options: List[QuestionOption] = field(default_factory=list)
    multi_select: bool = False
    default: Optional[str] = None
    required: bool = True
    timeout_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "header": self.header,
            "type": self.question_type.value,
            "options": [{"label": o.label, "description": o.description} for o in self.options],
            "multi_select": self.multi_select,
            "default": self.default,
            "required": self.required,
        }


@dataclass
class QuestionResult:
    """Result of asking a question."""
    question: str
    answer: Any
    answered_at: str = ""
    timed_out: bool = False
    skipped: bool = False

    def __post_init__(self) -> None:
        if not self.answered_at:
            self.answered_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "answered_at": self.answered_at,
            "timed_out": self.timed_out,
            "skipped": self.skipped,
        }


class AskUserQuestion:
    """
    Interactive question tool for user clarification.

    Features:
    - Single and multi-select options
    - Free text input
    - Yes/No confirmations
    - Timeout handling
    - Input validation
    """

    def __init__(
        self,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
        color_enabled: bool = True,
        default_timeout: Optional[float] = None,
    ):
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.color_enabled = color_enabled
        self.default_timeout = default_timeout

        self._colors = {
            "header": "\033[1;36m",    # Bold Cyan
            "question": "\033[1;37m",  # Bold White
            "option": "\033[33m",      # Yellow
            "selected": "\033[32m",    # Green
            "error": "\033[31m",       # Red
            "hint": "\033[90m",        # Gray
            "reset": "\033[0m",
        }

        if not color_enabled:
            self._colors = {k: "" for k in self._colors}

    def _print(self, text: str) -> None:
        """Print to output stream."""
        self.output_stream.write(text)
        self.output_stream.flush()

    def _read_input(self, timeout: Optional[float] = None) -> Optional[str]:
        """Read input with optional timeout."""
        if timeout is None:
            try:
                return self.input_stream.readline().strip()
            except (EOFError, KeyboardInterrupt):
                return None

        # Timeout implementation
        result: List[Optional[str]] = [None]
        event = threading.Event()

        def read_thread() -> None:
            try:
                result[0] = self.input_stream.readline().strip()
            except (EOFError, KeyboardInterrupt):
                result[0] = None
            finally:
                event.set()

        thread = threading.Thread(target=read_thread, daemon=True)
        thread.start()

        if event.wait(timeout):
            return result[0]
        return None  # Timed out

    def ask(
        self,
        question: str,
        options: Optional[List[QuestionOption]] = None,
        question_type: QuestionType = QuestionType.SINGLE_SELECT,
        header: str = "",
        default: Optional[str] = None,
        timeout: Optional[float] = None,
        multi_select: bool = False,
    ) -> QuestionResult:
        """
        Ask a question and get user response.

        Args:
            question: The question to ask
            options: Available options (for select types)
            question_type: Type of question
            header: Short header/label
            default: Default value if no input
            timeout: Timeout in seconds
            multi_select: Allow multiple selections

        Returns:
            QuestionResult with the answer
        """
        timeout = timeout or self.default_timeout
        c = self._colors

        # Print header if provided
        if header:
            self._print(f"\n{c['header']}[{header}]{c['reset']}\n")

        # Print question
        self._print(f"{c['question']}{question}{c['reset']}\n")

        # Handle different question types
        if question_type == QuestionType.YES_NO:
            return self._ask_yes_no(question, default, timeout)
        elif question_type == QuestionType.CONFIRMATION:
            return self._ask_confirmation(question, timeout)
        elif question_type == QuestionType.FREE_TEXT:
            return self._ask_free_text(question, default, timeout)
        elif question_type == QuestionType.MULTI_SELECT or multi_select:
            return self._ask_multi_select(question, options or [], timeout)
        else:
            return self._ask_single_select(question, options or [], default, timeout)

    def _ask_yes_no(
        self,
        question: str,
        default: Optional[str],
        timeout: Optional[float],
    ) -> QuestionResult:
        """Ask a yes/no question."""
        c = self._colors

        default_hint = ""
        if default:
            default_hint = f" {c['hint']}(default: {default}){c['reset']}"

        self._print(f"  {c['option']}[y/n]{c['reset']}{default_hint}: ")

        response = self._read_input(timeout)

        if response is None:
            if timeout:
                return QuestionResult(question=question, answer=default, timed_out=True)
            return QuestionResult(question=question, answer=default, skipped=True)

        if not response and default:
            response = default

        answer = response.lower() in ["y", "yes", "true", "1"]
        return QuestionResult(question=question, answer=answer)

    def _ask_confirmation(
        self,
        question: str,
        timeout: Optional[float],
    ) -> QuestionResult:
        """Ask for confirmation (press Enter to continue)."""
        c = self._colors

        self._print(f"  {c['hint']}Press Enter to continue, Ctrl+C to cancel...{c['reset']}")

        try:
            response = self._read_input(timeout)
            if response is None and timeout:
                return QuestionResult(question=question, answer=False, timed_out=True)
            return QuestionResult(question=question, answer=True)
        except KeyboardInterrupt:
            self._print("\n")
            return QuestionResult(question=question, answer=False, skipped=True)

    def _ask_free_text(
        self,
        question: str,
        default: Optional[str],
        timeout: Optional[float],
    ) -> QuestionResult:
        """Ask for free text input."""
        c = self._colors

        default_hint = ""
        if default:
            default_hint = f" {c['hint']}(default: {default}){c['reset']}"

        self._print(f"  {c['option']}>{c['reset']}{default_hint} ")

        response = self._read_input(timeout)

        if response is None:
            if timeout:
                return QuestionResult(question=question, answer=default, timed_out=True)
            return QuestionResult(question=question, answer=default, skipped=True)

        return QuestionResult(question=question, answer=response or default)

    def _ask_single_select(
        self,
        question: str,
        options: List[QuestionOption],
        default: Optional[str],
        timeout: Optional[float],
    ) -> QuestionResult:
        """Ask single-select question."""
        c = self._colors

        if not options:
            return self._ask_free_text(question, default, timeout)

        # Display options
        for i, opt in enumerate(options, 1):
            desc = f" - {c['hint']}{opt.description}{c['reset']}" if opt.description else ""
            self._print(f"  {c['option']}{i}.{c['reset']} {opt.label}{desc}\n")

        # Add "Other" option
        self._print(f"  {c['option']}{len(options) + 1}.{c['reset']} Other (custom input)\n")

        default_hint = ""
        if default:
            # Find default option number
            for i, opt in enumerate(options, 1):
                if opt.label == default or opt.value == default:
                    default_hint = f" {c['hint']}(default: {i}){c['reset']}"
                    break

        self._print(f"\n  Enter choice (1-{len(options) + 1}){default_hint}: ")

        response = self._read_input(timeout)

        if response is None:
            if timeout:
                return QuestionResult(question=question, answer=default, timed_out=True)
            return QuestionResult(question=question, answer=default, skipped=True)

        if not response and default:
            # Find and return default
            for opt in options:
                if opt.label == default or opt.value == default:
                    return QuestionResult(question=question, answer=opt.value)
            return QuestionResult(question=question, answer=default)

        try:
            choice = int(response)
            if 1 <= choice <= len(options):
                return QuestionResult(question=question, answer=options[choice - 1].value)
            elif choice == len(options) + 1:
                # "Other" selected - ask for custom input
                self._print(f"  {c['option']}Enter custom value:{c['reset']} ")
                custom = self._read_input(timeout)
                return QuestionResult(question=question, answer=custom or default)
        except ValueError:
            pass

        # Invalid input - treat as custom value
        return QuestionResult(question=question, answer=response)

    def _ask_multi_select(
        self,
        question: str,
        options: List[QuestionOption],
        timeout: Optional[float],
    ) -> QuestionResult:
        """Ask multi-select question."""
        c = self._colors

        if not options:
            return QuestionResult(question=question, answer=[])

        # Display options
        for i, opt in enumerate(options, 1):
            desc = f" - {c['hint']}{opt.description}{c['reset']}" if opt.description else ""
            self._print(f"  {c['option']}{i}.{c['reset']} {opt.label}{desc}\n")

        self._print(f"\n  {c['hint']}Enter choices separated by commas (e.g., 1,3,4):{c['reset']} ")

        response = self._read_input(timeout)

        if response is None:
            if timeout:
                return QuestionResult(question=question, answer=[], timed_out=True)
            return QuestionResult(question=question, answer=[], skipped=True)

        if not response:
            return QuestionResult(question=question, answer=[])

        # Parse selections
        selected = []
        for part in response.split(","):
            part = part.strip()
            try:
                choice = int(part)
                if 1 <= choice <= len(options):
                    selected.append(options[choice - 1].value)
            except ValueError:
                # Non-numeric - might be a label
                for opt in options:
                    if opt.label.lower() == part.lower():
                        selected.append(opt.value)
                        break

        return QuestionResult(question=question, answer=selected)

    def ask_questions(
        self,
        questions: List[Question],
    ) -> Dict[str, QuestionResult]:
        """
        Ask multiple questions.

        Args:
            questions: List of Question objects

        Returns:
            Dict mapping question text to QuestionResult
        """
        results = {}

        for q in questions:
            result = self.ask(
                question=q.question,
                options=q.options,
                question_type=q.question_type,
                header=q.header,
                default=q.default,
                timeout=q.timeout_seconds,
                multi_select=q.multi_select,
            )
            results[q.question] = result

            # If required and skipped/timed out with no answer, retry
            if q.required and (result.skipped or result.timed_out) and not result.answer:
                c = self._colors
                self._print(f"{c['error']}This question is required.{c['reset']}\n")
                # Could implement retry logic here

        return results


# Singleton instance
_ask_user: Optional[AskUserQuestion] = None


def get_ask_user_question(
    color_enabled: bool = True,
) -> AskUserQuestion:
    """Get or create the global AskUserQuestion instance."""
    global _ask_user
    if _ask_user is None:
        _ask_user = AskUserQuestion(color_enabled=color_enabled)
    return _ask_user


def reset_ask_user_question() -> None:
    """Reset the global AskUserQuestion instance."""
    global _ask_user
    _ask_user = None


def ask_user(
    question: str,
    options: Optional[List[Dict[str, str]]] = None,
    multi_select: bool = False,
    default: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Any:
    """
    Convenience function to ask a single question.

    Args:
        question: Question text
        options: List of {"label": str, "description": str} dicts
        multi_select: Allow multiple selections
        default: Default value
        timeout: Timeout in seconds

    Returns:
        The user's answer
    """
    asker = get_ask_user_question()

    question_options = None
    if options:
        question_options = [
            QuestionOption(label=o["label"], description=o.get("description", ""))
            for o in options
        ]

    q_type = QuestionType.MULTI_SELECT if multi_select else QuestionType.SINGLE_SELECT
    if not options:
        q_type = QuestionType.FREE_TEXT

    result = asker.ask(
        question=question,
        options=question_options,
        question_type=q_type,
        default=default,
        timeout=timeout,
        multi_select=multi_select,
    )

    return result.answer
