"""Rich console UI for real-time streaming output."""

import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .logger import logger


class AgentStatus(Enum):
    """Status of an agent."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    DONE = "done"


@dataclass
class AgentState:
    """Current state of an agent for display."""
    name: str
    role: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: str = ""
    thinking: str = ""
    last_action: str = ""
    messages_sent: int = 0
    messages_received: int = 0


class CompanyConsole:
    """
    Rich console UI for Company AGI.
    Provides real-time streaming output of agent activities.
    """

    # Agent role colors
    AGENT_COLORS = {
        "CEO": "bold red",
        "CTO": "bold blue",
        "ProductManager": "bold green",
        "Researcher": "bold yellow",
        "Developer": "bold cyan",
        "QAEngineer": "bold magenta",
    }

    # Status icons
    STATUS_ICONS = {
        AgentStatus.IDLE: "💤",
        AgentStatus.THINKING: "🤔",
        AgentStatus.ACTING: "⚡",
        AgentStatus.WAITING: "⏳",
        AgentStatus.DONE: "✅",
    }

    # Log levels: debug=0, info=1, warning=2, error=3
    LOG_LEVELS = {"debug": 0, "info": 1, "warning": 2, "error": 3}

    def __init__(self):
        self.console = Console()
        self.agents: Dict[str, AgentState] = {}
        self.messages: List[Dict[str, Any]] = []
        self.current_phase = "Initializing"
        self.start_time = datetime.now()
        self._log_level = 1  # default: info

    def set_log_level(self, level: str) -> None:
        """Set console log level: debug, info, warning, error."""
        self._log_level = self.LOG_LEVELS.get(level, 1)

    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM output by removing thinking tags and other artifacts."""
        if not text:
            return ""

        # Strip <think>/<​/think> tags but keep the reasoning content visible
        text = re.sub(r'</?think>', '', text)

        # Remove other common artifacts
        text = re.sub(r'<\/?begin.*?>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\/?end.*?>', '', text, flags=re.IGNORECASE)

        # Remove leading/trailing whitespace and extra newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def register_agent(self, name: str, role: str) -> None:
        """Register an agent for display."""
        self.agents[name] = AgentState(name=name, role=role)

    def set_phase(self, phase: str) -> None:
        """Set the current workflow phase."""
        self.current_phase = phase
        self.print_phase_header(phase)
        logger.log_phase(phase)

    def print_phase_header(self, phase: str) -> None:
        """Print a phase header."""
        self.console.print()
        self.console.rule(f"[bold cyan]Phase: {phase}[/]", style="cyan")
        self.console.print()

    def agent_thinking(self, agent_name: str, thought: str) -> None:
        """Show agent thinking in real-time."""
        # Log full thought before cleaning
        logger.log_agent_thinking(agent_name, thought)

        # Clean the thought output for display
        thought = self._clean_llm_output(thought)

        if agent_name in self.agents:
            self.agents[agent_name].status = AgentStatus.THINKING
            self.agents[agent_name].thinking = thought

        color = self.AGENT_COLORS.get(agent_name, "white")
        icon = self.STATUS_ICONS[AgentStatus.THINKING]

        if thought.strip():  # Only show if there's content
            self.console.print(
                Panel(
                    Text(thought, style="italic"),
                    title=f"{icon} [{color}]{agent_name}[/] is thinking...",
                    border_style="dim",
                    padding=(0, 1)
                )
            )

    def agent_action(self, agent_name: str, action: str, details: str = "") -> None:
        """Show agent taking an action."""
        # Log full action
        logger.log_agent_action(agent_name, action, details)

        if agent_name in self.agents:
            self.agents[agent_name].status = AgentStatus.ACTING
            self.agents[agent_name].last_action = action

        color = self.AGENT_COLORS.get(agent_name, "white")
        icon = self.STATUS_ICONS[AgentStatus.ACTING]

        content = f"[bold]{action}[/bold]"
        if details:
            content += f"\n{details}"

        self.console.print(
            Panel(
                content,
                title=f"{icon} [{color}]{agent_name}[/] action",
                border_style=color.replace("bold ", ""),
                padding=(0, 1)
            )
        )

    def agent_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        message_type: str = "general"
    ) -> None:
        """Show inter-agent communication."""
        # Log full message before cleaning
        logger.log_agent_message(from_agent, to_agent, message, message_type)

        from_color = self.AGENT_COLORS.get(from_agent, "white")
        to_color = self.AGENT_COLORS.get(to_agent, "white")

        # Clean the message for display
        clean_message = self._clean_llm_output(message)

        # Update message counts
        if from_agent in self.agents:
            self.agents[from_agent].messages_sent += 1
        if to_agent in self.agents:
            self.agents[to_agent].messages_received += 1

        # Store message
        self.messages.append({
            "from": from_agent,
            "to": to_agent,
            "message": clean_message,
            "type": message_type,
            "time": datetime.now()
        })

        if clean_message.strip():  # Only show if there's content
            self.console.print(
                f"  📨 [{from_color}]{from_agent}[/] → [{to_color}]{to_agent}[/]: {clean_message}",
                style="dim"
            )

    def agent_decision(self, agent_name: str, decision: str, reasoning: str = "") -> None:
        """Show an agent making a decision."""
        # Log full decision before cleaning
        logger.log_decision(agent_name, decision, reasoning)

        color = self.AGENT_COLORS.get(agent_name, "white")

        # Clean the decision and reasoning for display
        clean_decision = self._clean_llm_output(decision)
        clean_reasoning = self._clean_llm_output(reasoning)

        content = f"[bold green]Decision:[/] {clean_decision}"
        if clean_reasoning:
            content += f"\n[dim]Reasoning: {clean_reasoning}[/]"

        self.console.print(
            Panel(
                content,
                title=f"🎯 [{color}]{agent_name}[/] Decision",
                border_style="green",
                padding=(0, 1)
            )
        )

    def show_problem(self, problem: Dict[str, Any]) -> None:
        """Display a discovered problem."""
        # Log full problem
        logger.log_problem(problem)

        severity_colors = {
            "critical": "red",
            "high": "yellow",
            "medium": "blue",
            "low": "green"
        }

        severity = problem.get("severity", "medium")
        color = severity_colors.get(severity, "white")

        tree = Tree(f"[bold]🔍 Problem Discovered[/]")
        tree.add(f"[{color}]Severity: {severity.upper()}[/]")
        tree.add(f"Description: {problem.get('description', 'N/A')}")
        tree.add(f"Target Users: {problem.get('target_users', 'N/A')}")
        tree.add(f"Domain: {problem.get('domain', 'N/A')}")

        self.console.print(Panel(tree, border_style=color))

    def show_solution(self, solution: Dict[str, Any]) -> None:
        """Display a solution."""
        tree = Tree(f"[bold green]💡 Solution[/]")
        tree.add(f"Description: {solution.get('description', 'N/A')}")

        if solution.get("files"):
            files_branch = tree.add("Files Created:")
            for f in solution.get("files", [])[:5]:
                files_branch.add(f"📄 {f}")

        self.console.print(Panel(tree, border_style="green"))

    def show_meeting(
        self,
        topic: str,
        participants: List[str],
        discussion: List[Dict[str, str]],
        outcome: Dict[str, Any] = None
    ) -> None:
        """Display a meeting between agents."""
        # Log the full meeting
        logger.log_meeting(topic, participants, discussion, outcome or {})

        self.console.print()
        self.console.rule(f"[bold]🤝 Meeting: {topic}[/]", style="yellow")

        # Show participants
        participant_text = " ".join(
            f"[{self.AGENT_COLORS.get(p, 'white')}]{p}[/]"
            for p in participants
        )
        self.console.print(f"Participants: {participant_text}")
        self.console.print()

        # Show discussion (cleaned and truncated for display)
        for entry in discussion:
            speaker = entry.get("speaker", "Unknown")
            message = self._clean_llm_output(entry.get("message", entry.get("content", "")))
            color = self.AGENT_COLORS.get(speaker, "white")

            if message.strip():
                self.console.print(f"  [{color}]{speaker}[/]: {message}")

        self.console.print()
        self.console.rule(style="yellow")

    def show_progress(self, tasks: List[Dict[str, Any]]) -> None:
        """Show progress of tasks."""
        table = Table(title="Task Progress")
        table.add_column("Task", style="cyan")
        table.add_column("Assigned To", style="magenta")
        table.add_column("Status", style="green")

        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌"
        }

        for task in tasks:
            icon = status_icons.get(task.get("status", "pending"), "❓")
            table.add_row(
                task.get("description", "Unknown"),
                task.get("assigned_to", "Unassigned"),
                f"{icon} {task.get('status', 'pending')}"
            )

        self.console.print(table)

    def show_agent_status(self) -> None:
        """Show current status of all agents."""
        table = Table(title="Agent Status")
        table.add_column("Agent", style="cyan")
        table.add_column("Status")
        table.add_column("Current Task")
        table.add_column("Messages")

        for name, state in self.agents.items():
            color = self.AGENT_COLORS.get(name, "white")
            icon = self.STATUS_ICONS.get(state.status, "❓")

            table.add_row(
                f"[{color}]{name}[/]",
                f"{icon} {state.status.value}",
                state.current_task if state.current_task else "-",
                f"↑{state.messages_sent} ↓{state.messages_received}"
            )

        self.console.print(table)

    def show_code(self, code: str, language: str = "python", title: str = "") -> None:
        """Display code with syntax highlighting."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title=title or f"Code ({language})"))

    def start_streaming(self, agent_name: str, action: str = "generating") -> None:
        """Start a streaming output section."""
        color = self.AGENT_COLORS.get(agent_name, "white")
        self.console.print(f"  [dim]🔄 [{color}]{agent_name}[/] {action}...[/]", end="")
        self._streaming_buffer = ""

    def stream_token(self, token: str) -> None:
        """Display a single streaming token."""
        # Print token directly for real-time display
        sys.stdout.write(token)
        sys.stdout.flush()
        self._streaming_buffer = getattr(self, '_streaming_buffer', '') + token

    def end_streaming(self) -> None:
        """End a streaming output section."""
        self.console.print()  # New line after streaming completes
        self._streaming_buffer = ""

    def create_stream_callback(self, agent_name: str):
        """Create a callback function for streaming output from an agent."""
        def callback(token: str):
            # Only show non-thinking tokens
            if not token.startswith('<think>') and '</think>' not in token:
                sys.stdout.write(token)
                sys.stdout.flush()
        return callback

    def success(self, message: str) -> None:
        """Show success message."""
        if self._log_level <= 1:
            self.console.print(f"[bold green]✅ {message}[/]")

    def error(self, message: str) -> None:
        """Show error message."""
        self.console.print(f"[bold red]❌ {message}[/]")

    def warning(self, message: str) -> None:
        """Show warning message."""
        if self._log_level <= 2:
            self.console.print(f"[bold yellow]⚠️ {message}[/]")

    def info(self, message: str) -> None:
        """Show info message."""
        if self._log_level <= 1:
            self.console.print(f"[bold blue]ℹ️ {message}[/]")

    def debug(self, message: str) -> None:
        """Show debug message (only in verbose mode)."""
        if self._log_level <= 0:
            self.console.print(f"[dim]🔍 {message}[/]")

    def section(self, title: str) -> None:
        """Print a section header."""
        logger.log_phase(title)
        if self._log_level <= 1:
            self.console.print()
            self.console.rule(f"[bold cyan]{title}[/]", style="cyan")
            self.console.print()

    def print_header(self) -> None:
        """Print the main application header."""
        self.console.print()
        self.console.print(Panel(
            "[bold cyan]COMPANY AGI[/]\n"
            "[dim]Autonomous Multi-Agent System[/]",
            border_style="cyan",
            padding=(1, 2)
        ))
        self.console.print()
        self.start_time = datetime.now()

    def print_summary(self, result: Dict[str, Any]) -> None:
        """Print workflow summary."""
        duration = (datetime.now() - self.start_time).total_seconds()

        # Log solution/result
        logger.log_solution(result)

        # Save logs
        log_file = logger.save()
        transcript_file = logger.get_transcript_path()

        self.console.print()
        self.console.rule("[bold]Workflow Complete[/]", style="green")

        # Status
        status = result.get("status", "unknown")
        status_color = "green" if status == "completed" else "red"
        self.console.print(f"Status: [{status_color}]{status.upper()}[/]")

        # Duration
        self.console.print(f"Duration: {duration:.1f} seconds")

        # Decisions
        decisions = result.get("decisions", [])
        if decisions:
            self.console.print(f"\nDecisions Made: {len(decisions)}")
            for d in decisions:
                icon = "✅" if d.get("decision") == "approved" else "❌"
                self.console.print(f"  {icon} {d.get('phase', 'unknown')}: {d.get('decision', 'unknown')}")

        # Problem
        if result.get("problem"):
            self.console.print(f"\nProblem: {result['problem'].get('description', 'N/A')}")

        # Error
        if result.get("error"):
            self.console.print(f"\n[red]Error: {result['error']}[/]")

        # Log file locations
        self.console.print(f"\n[dim]Full conversation logs:[/]")
        self.console.print(f"  [cyan]Transcript:[/] {transcript_file}")
        self.console.print(f"  [cyan]JSON Log:[/] {log_file}")

        self.console.print()


# Global console instance
console = CompanyConsole()


def stream_text(text: str, delay: float = 0.02) -> None:
    """Stream text character by character."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()
