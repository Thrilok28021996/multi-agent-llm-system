#!/usr/bin/env python3
"""
Company AGI - Autonomous Multi-Agent System

A fully autonomous AGI that simulates a company workflow using local LLMs via Ollama.
Install Ollama: https://ollama.ai  |  Start server: ollama serve
Pull models: ollama pull qwen3:8b && ollama pull mistral:8b-instruct-2410-q4_K_M && ollama pull qwen2.5-coder:7b
Each agent role uses first principles thinking to discover and solve real-world problems.

Usage:
    python main.py --help                    Show help
    python main.py --check-models            Check which models are available
    python main.py --run                     Run the full workflow
    python main.py --discover                Only run problem discovery
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.models import ModelConfig
from config.config_loader import load_config, ConfigLoader
from ui.console import console
from research.problem_discoverer import DiscoveredProblem, ProblemSeverity


def get_data_dir() -> Path:
    """Get ~/.multi-agent-llm-company-system/ for internal data (logs, memory, reports)."""
    custom = os.environ.get("MULTI_AGENT_LLM_DATA_DIR")
    data_dir = Path(custom) if custom else Path.home() / ".multi-agent-llm-company-system"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def create_custom_problem(
    description: str,
    domain: str = "software",
    language: str = "python"
) -> DiscoveredProblem:
    """Create a custom problem from user input."""
    import hashlib
    problem_id = hashlib.sha256(description.encode()).hexdigest()[:12]

    return DiscoveredProblem(
        id=f"custom_{problem_id}",
        description=description,
        severity=ProblemSeverity.HIGH,  # User-provided problems are high priority
        domain=domain,
        evidence=[f"User requirement: {description}", f"Target language: {language}"],
        sources=["user_input"],
        target_users="As specified by user",
        potential_solution_ideas=[f"Implement in {language}"],
        keywords=[language, domain],
        score=100.0,  # High score for user-provided problems
        metadata={"language": language}  # Store language preference
    )


def check_models(config: ModelConfig) -> dict:
    """Check which required models are available on the configured backend."""
    from config.llm_client import _get_backend, OllamaBackend

    # Use env-overridden configs, not bare MODEL_CONFIGS
    unique_specs = {spec.model_id(_get_backend()): spec for spec in config.configs.values()}
    backend = _get_backend()
    status = {}

    if backend == "ollama":
        try:
            import ollama as _ollama
            host = OllamaBackend()._host()
            client = _ollama.Client(host=host)
            pulled = {m["name"] for m in client.list().get("models", [])}
            for model_id, spec in unique_specs.items():
                found = any(model_id in p for p in pulled)
                status[model_id] = f"{'found' if found else 'missing'} ({model_id})"
        except Exception as e:
            for model_id in unique_specs:
                status[model_id] = f"server unreachable ({e})"
    else:
        # LM Studio exposes /v1/models — probe it to see which models are loaded.
        host = os.environ.get("LMSTUDIO_HOST", "http://localhost:1234/v1").rstrip("/")
        url = f"{host}/models"
        try:
            import urllib.request
            import urllib.error
            with urllib.request.urlopen(url, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            available_ids = {m.get("id") for m in payload.get("data", []) if m.get("id")}
            for model_id in unique_specs:
                if model_id in available_ids:
                    status[model_id] = f"loaded ({model_id})"
                elif available_ids:
                    # LM Studio is reachable but this model isn't currently loaded.
                    status[model_id] = f"not loaded — LM Studio has: {', '.join(sorted(available_ids))[:120]}"
                else:
                    status[model_id] = f"not found ({model_id})"
        except Exception as e:
            # Connection error — warn but don't crash.
            print(f"Warning: could not reach LM Studio at {url}: {e}")
            for model_id in unique_specs:
                status[model_id] = f"server unreachable ({e})"

    return status


def print_model_status(status: dict) -> None:
    """Print model availability status."""
    from config.llm_client import _get_backend, OllamaBackend
    backend = _get_backend()
    server = OllamaBackend()._host() if backend == "ollama" else "LM Studio"
    print("\n=== Model Status ===")
    print(f"Backend: {backend}  |  Server: {server}\n")
    for model, state in status.items():
        if "loaded (" in state or state.startswith("found "):
            icon = "✓"
        elif "not loaded" in state or "assumed" in state:
            icon = "~"
        else:
            icon = "✗"
        print(f"  {icon} {model}: {state}")
    missing = [m for m, s in status.items() if "missing" in s or "unreachable" in s or "not found" in s]
    if missing:
        print(f"\n✗ {len(missing)} model(s) not available.")
        print("  Start Ollama: ollama serve")
        print("  Pull models: ollama pull qwen3:8b  &&  ollama pull mistral:8b-instruct-2410-q4_K_M  &&  ollama pull qwen2.5-coder:7b")
    else:
        print("\n✓ All required models found!")


async def run_discovery_only(config: ModelConfig) -> None:
    """Run only the problem discovery phase."""
    from research.problem_discoverer import ProblemDiscoverer
    from research.sources import ResearchSources

    data_dir = get_data_dir()

    console.print_header()
    console.section("Problem Discovery Mode")
    console.info("Searching for problems across configured sources...")

    discoverer = ProblemDiscoverer(
        model=config.get_model_name("researcher")
    )
    sources = ResearchSources()

    # Get subreddits to search
    subreddits = sources.get_reddit_subreddits()[:3]  # Limit for speed
    console.info(f"Searching subreddits: {', '.join(subreddits)}")

    # Discover problems
    console.agent_action("Researcher", "Scanning Reddit", f"Subreddits: {', '.join(subreddits)}")
    problems = await discoverer.discover_from_reddit(
        subreddits,
        limit_per_sub=10
    )

    # Also check HackerNews
    console.agent_action("Researcher", "Scanning Hacker News", "Looking for tech problems...")
    hn_problems = await discoverer.discover_from_hacker_news(limit=15)
    problems.extend(hn_problems)

    # Display results
    console.section(f"Discovered {len(problems)} Problems")

    top_problems = discoverer.get_top_problems(limit=5)
    for problem in top_problems:
        console.show_problem(problem.to_dict())

    # Save to file
    output_file = data_dir / "reports" / f"problems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(discoverer.export_problems(), indent=2))
    console.success(f"Full results saved to: {output_file}")


async def run_continuous_workflow(
    config: ModelConfig,
    output_dir: str = "output/solutions",
    delay: int = 60,
    max_iterations: Optional[int] = None,
    workspace_root: Optional[str] = None,
    app_config=None
) -> None:
    """Run the workflow in continuous mode."""
    from orchestrator.workflow import CompanyWorkflow

    data_dir = get_data_dir()
    ws_root = workspace_root or os.getcwd()

    console.print_header()
    console.info("Model Configuration:")
    config.print_config()
    console.info(f"Working directory: {ws_root}")
    console.info(f"Output directory: {output_dir}/")
    console.info("Initializing agents for continuous mode...")

    run_code_execution = True
    if app_config is not None:
        run_code_execution = getattr(getattr(app_config, 'output', None), 'run_tests', True)

    workflow = CompanyWorkflow(
        workspace_root=ws_root,
        model_config=config,
        memory_persist_dir=str(data_dir / "memory"),
        output_dir=output_dir,
        data_dir=str(data_dir),
        run_code_execution=run_code_execution
    )

    console.success("All agents initialized!")

    results = await workflow.run_continuous(
        delay_between_runs=delay,
        max_iterations=max_iterations
    )

    # Save all results
    output_file = data_dir / "reports" / f"continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2, default=str))
    console.success(f"All results saved to: {output_file}")


async def run_full_workflow(
    config: ModelConfig,
    custom_problem: Optional[DiscoveredProblem] = None,
    output_dir: str = "output/solutions",
    workspace_root: Optional[str] = None,
    max_approval_rounds: int = 3,
    app_config=None,
    resume: bool = False,
    interactive: bool = False,
    max_tokens: Optional[int] = None,
    session_id: Optional[str] = None,
    dry_run: bool = False,
    offline: bool = False,
    enable_escalation: bool = True,
    enable_security_review: bool = True,
    enable_retrospective: bool = True,
    max_workflow_minutes: int = 0,
    force_stop: bool = False,
    scaffold_mode: bool = False
) -> None:
    """Run the complete company workflow."""
    from orchestrator.workflow import CompanyWorkflow

    data_dir = get_data_dir()
    ws_root = workspace_root or os.getcwd()

    console.print_header()

    # Print model config with rich console
    console.info("Model Configuration:")
    config.print_config()

    if custom_problem:
        console.section("Custom Problem")
        console.show_problem(custom_problem.to_dict())

    console.info(f"Working directory: {ws_root}")
    console.info(f"Output directory: {output_dir}/")
    if max_approval_rounds:
        console.info(f"Max approval rounds: {max_approval_rounds}")
    else:
        console.info("Approval mode: automated (retries until CEO approves)")
    console.info("Initializing agents...")

    # Show agent initialization
    agents_to_init = ["CEO", "CTO", "ProductManager", "Researcher", "Developer", "QAEngineer",
                       "DevOpsEngineer", "DataAnalyst", "SecurityEngineer"]
    for agent in agents_to_init:
        console.agent_thinking(agent, "Initializing...")

    # Determine if code execution checks should run
    run_code_execution = True
    if app_config is not None:
        run_code_execution = getattr(getattr(app_config, 'output', None), 'run_tests', True)

    # Build LLM config overrides from config file (only if explicitly set)
    llm_overrides = {}
    if app_config is not None:
        llm = app_config.llm
        if llm.temperature != 0.7:  # Non-default = user explicitly set it
            llm_overrides["temperature"] = llm.temperature
        if llm.max_tokens != 4096:
            llm_overrides["max_tokens"] = llm.max_tokens
        if llm.streaming:
            llm_overrides["streaming"] = True

    # Read workflow behavior flags from config
    enable_meetings = True
    enable_learning = True
    if app_config is not None:
        enable_meetings = app_config.workflow.enable_meetings
        enable_learning = app_config.workflow.enable_learning

    # Dry-run disables code execution
    if dry_run:
        run_code_execution = False
        console.info("Dry-run mode: skipping code execution, testing, and delivery")

    workflow = CompanyWorkflow(
        workspace_root=ws_root,
        model_config=config,
        memory_persist_dir=str(data_dir / "memory"),
        output_dir=output_dir,
        data_dir=str(data_dir),
        run_code_execution=run_code_execution,
        llm_config=llm_overrides or None,
        enable_meetings=enable_meetings,
        enable_learning=enable_learning,
        enable_escalation=enable_escalation,
        enable_security_review=enable_security_review,
        enable_retrospective=enable_retrospective,
        max_workflow_minutes=max_workflow_minutes,
        force_stop=force_stop,
        scaffold_mode=scaffold_mode
    )

    console.success("All agents initialized!")

    if custom_problem:
        console.info(f"Working on: {custom_problem.description}")
    elif offline:
        console.warning("Offline mode: no web discovery available")
    else:
        console.info("Starting autonomous problem discovery...")

    console.info("Starting workflow...\n")

    # Run the workflow
    # Set token budget limit if specified
    if max_tokens is not None:
        workflow.cost_tracker.set_budget(max_tokens)

    # Offline forces auto_discover off; dry_run skips delivery
    auto_discover = (custom_problem is None) and not offline

    result = await workflow.run_full_workflow(
        auto_discover=auto_discover,
        problem=custom_problem,
        max_approval_rounds=max_approval_rounds,
        resume=resume,
        interactive=interactive,
        session_id=session_id,
        dry_run=dry_run,
        scaffold_mode=scaffold_mode
    )

    # Display results with rich console
    console.print_summary(result)

    # Save full result
    output_file = data_dir / "reports" / f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2, default=str))
    console.success(f"Full results saved to: {output_file}")


def list_solutions() -> None:
    """List past workflow solutions from history."""
    data_dir = get_data_dir()
    results_file = data_dir / "results" / "workflow_history.json"

    if not results_file.exists():
        print("No solutions found yet. Run a workflow first.")
        return

    try:
        history = json.loads(results_file.read_text())
    except Exception:
        print("Could not read solution history.")
        return

    if not history:
        print("No solutions found yet.")
        return

    print(f"\n{'='*70}")
    print(f"  Solution History ({len(history)} runs)")
    print(f"{'='*70}\n")

    for entry in reversed(history):
        status_icon = "+" if entry.get("status") == "completed" else "-"
        problem = entry.get("problem", "Unknown")
        started = entry.get("started_at", "")[:19] if entry.get("started_at") else "N/A"
        duration = entry.get("duration_seconds")
        duration_str = f"{duration:.0f}s" if duration else "N/A"
        delivery_dir = entry.get("delivery_dir", "")

        print(f"  [{status_icon}] #{entry.get('id', '?')} | {started} | {duration_str}")
        print(f"      {problem}")
        if delivery_dir:
            print(f"      Dir: {delivery_dir}")
        print()

    print(f"Data directory: {data_dir}")
    print()


def feedback_solution(solution_id: int, rating: str) -> None:
    """Record user feedback on a past solution to improve future runs."""
    data_dir = get_data_dir()
    results_file = data_dir / "results" / "workflow_history.json"

    if not results_file.exists():
        print("No solutions found. Run a workflow first.")
        return

    try:
        history = json.loads(results_file.read_text())
    except Exception:
        print("Could not read solution history.")
        return

    # Find the solution
    entry = None
    for e in history:
        if e.get("id") == solution_id:
            entry = e
            break

    if not entry:
        print(f"Solution #{solution_id} not found. Use --list-solutions to see available IDs.")
        return

    is_positive = rating.lower() in ("good", "great", "positive", "yes", "+", "1")

    # Record feedback into learning system for all agents
    from memory.learning import AgentLearning
    memory_dir = str(data_dir / "memory")
    for agent_name in ["Developer", "QAEngineer", "CTO", "CEO", "ProductManager", "Researcher"]:
        learning = AgentLearning(agent_name=agent_name, persist_dir=memory_dir)
        learning.add_lesson(
            task_type="workflow",
            lesson=f"{'Positive' if is_positive else 'Negative'} feedback on: {entry.get('problem', 'unknown')}",
            importance=0.8 if is_positive else 0.6
        )

    # Store feedback in history entry
    entry["user_feedback"] = {"rating": rating, "positive": is_positive}
    with open(results_file, "w") as f:
        json.dump(history, f, indent=2)

    icon = "+" if is_positive else "-"
    print(f"[{icon}] Feedback recorded for solution #{solution_id}: {rating}")
    print("  This will improve future workflow runs.")


def export_solution(solution_id: int) -> None:
    """Export a solution as a zip bundle with metadata."""
    import zipfile

    data_dir = get_data_dir()
    results_file = data_dir / "results" / "workflow_history.json"

    if not results_file.exists():
        print("No solutions found. Run a workflow first.")
        return

    try:
        history = json.loads(results_file.read_text())
    except Exception:
        print("Could not read solution history.")
        return

    entry = None
    for e in history:
        if e.get("id") == solution_id:
            entry = e
            break

    if not entry:
        print(f"Solution #{solution_id} not found. Use --list-solutions to see available IDs.")
        return

    delivery_dir = entry.get("delivery_dir", "")
    if not delivery_dir or not Path(delivery_dir).exists():
        print(f"Solution #{solution_id} directory not found: {delivery_dir}")
        return

    project_path = Path(delivery_dir)
    zip_name = f"solution_{solution_id}.zip"
    zip_path = Path.cwd() / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add metadata
        metadata = {
            "solution_id": solution_id,
            "problem": entry.get("problem", ""),
            "domain": entry.get("domain", ""),
            "status": entry.get("status", ""),
            "started_at": entry.get("started_at", ""),
            "completed_at": entry.get("completed_at", ""),
            "duration_seconds": entry.get("duration_seconds"),
            "decisions": entry.get("decisions", []),
            "user_feedback": entry.get("user_feedback"),
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        # Add all project files
        for file_path in project_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                arcname = str(file_path.relative_to(project_path))
                zf.write(file_path, arcname)

    print(f"Exported solution #{solution_id} to: {zip_path}")
    print(f"  Problem: {entry.get('problem', '')}")


def main():
    parser = argparse.ArgumentParser(
        description="Company AGI - Autonomous Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  company-agi --scaffold "FastAPI CRUD app for todos"   # Fast scaffold mode (skips research)
  company-agi "Build a CLI tool for managing todos"
  company-agi "Create a REST API" --language python
  company-agi "Build X" --approve             # Pause at each phase for approval
  company-agi --enhance ./myproject "Add auth" # Enhance existing project
  company-agi --list-solutions                # View past solutions
  company-agi --feedback 3 good              # Rate solution #3
  company-agi --export 3                     # Export solution #3 as zip
  company-agi --rerun 3                      # Re-run solution #3
  company-agi --list-sessions                # View past sessions
  company-agi --session-id abc123            # Resume a specific session
  company-agi "Build X" --dry-run            # Design only, no code execution
  company-agi "Build X" --offline            # No web research
  company-agi --run --verbose                # Debug-level output
  company-agi --run --quiet                  # Errors only
  company-agi "Build X" --max-tokens 50000   # Cap token usage
  company-agi --generate-config              # Create default config.yaml
  company-agi                                 # Interactive mode
  company-agi --check-models                  # Check Ollama models
  company-agi --run                           # Auto-discover problems
  company-agi --discover                      # Only discover problems
  company-agi --continuous --max-iterations 5 # Continuous mode
  company-agi --resume                        # Resume from checkpoint

Internal data: ~/.multi-agent-llm-company-system/
Generated code: current directory (override with --output-dir or --target)
        """
    )

    # Positional argument: problem description (optional)
    parser.add_argument(
        "problem",
        nargs="?",
        default=None,
        help="Problem to solve (e.g., 'Build a REST API'). Omit for interactive mode."
    )

    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Check which required models are installed"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full autonomous workflow (auto-discover problems)"
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Only run problem discovery phase"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "lmstudio"],
        default=None,
        help="LLM backend: ollama or lmstudio"
    )
    parser.add_argument(
        "--problem",
        type=str,
        dest="problem_flag",
        metavar="DESCRIPTION",
        help="Provide your own problem/requirement (backward-compat alias for positional arg)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="software",
        metavar="DOMAIN",
        help="Domain for the problem (e.g., software, business, productivity)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        metavar="LANG",
        help="Programming language for the solution (python, javascript, typescript, go, rust, java)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory where generated solutions will be saved (default: current directory)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (default when no arguments given)"
    )
    parser.add_argument(
        "--current-dir",
        action="store_true",
        help="Generate files in current directory (this is now the default)"
    )
    parser.add_argument(
        "--target",
        type=str,
        metavar="DIR",
        help="Target directory to generate files in (overrides current directory)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run in continuous mode - discover and solve problems in a loop"
    )
    parser.add_argument(
        "--continuous-delay",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Delay between runs in continuous mode (default: 60)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        metavar="N",
        help="Maximum iterations in continuous mode (default: unlimited)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        metavar="N",
        help="Max QA+CEO approval rounds (default: 0 = unlimited, keeps retrying until CEO approves)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume workflow from last checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="Path to config file (YAML or JSON)"
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate a default config.yaml file and exit"
    )
    parser.add_argument(
        "--list-solutions",
        action="store_true",
        help="List past workflow solutions and their status"
    )
    parser.add_argument(
        "--enhance",
        type=str,
        metavar="DIR",
        help="Enhance an existing project (e.g., --enhance ./myproject 'Add authentication')"
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Interactive approval mode: pause at key decision points for human approval"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        default=None,
        help="Stop workflow when total token usage exceeds N (e.g., --max-tokens 50000)"
    )
    parser.add_argument(
        "--feedback",
        nargs=2,
        metavar=("ID", "RATING"),
        help="Rate a past solution (e.g., --feedback 3 good)"
    )
    parser.add_argument(
        "--export",
        type=int,
        metavar="ID",
        help="Export a solution as a zip bundle (e.g., --export 3)"
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List past workflow sessions"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        metavar="ID",
        help="Resume a specific session by ID (e.g., --session-id abc123def456)"
    )
    parser.add_argument(
        "--rerun",
        type=int,
        metavar="N",
        help="Re-run a past solution by ID from --list-solutions (e.g., --rerun 3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run through design phases but skip code execution, testing, and delivery"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip web research (requires --problem or stdin input)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug-level output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors and final summary"
    )
    parser.add_argument(
        "--max-workflow-minutes",
        type=int,
        default=0,
        metavar="N",
        help="Time-based hard stop in minutes (default: 0 = no limit)"
    )
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Disable escalation system (revert to simple retry behavior)"
    )
    parser.add_argument(
        "--no-security",
        action="store_true",
        help="Skip security review phase"
    )
    parser.add_argument(
        "--no-retrospective",
        action="store_true",
        help="Skip retrospective phase after delivery"
    )
    parser.add_argument(
        "--force-stop",
        action="store_true",
        help="Allow partial delivery when round limits are hit (default: never stop until solved)"
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        metavar="PROBLEM",
        help="Multiple problems to solve sequentially (e.g., --problems 'Build X' 'Fix Y' 'Create Z')"
    )
    parser.add_argument(
        "--scaffold",
        type=str,
        metavar="DESCRIPTION",
        help="Generate a first-draft scaffold (e.g., --scaffold 'FastAPI CRUD app for todos'). Skips research/discovery and goes straight to PM -> CTO -> Developer pipeline."
    )

    args = parser.parse_args()

    # Handle generate-config first
    if args.generate_config:
        loader = ConfigLoader()
        loader.save_default_config("config.yaml")
        return

    # Handle list-solutions
    if args.list_solutions:
        list_solutions()
        return

    # Handle --feedback
    if args.feedback:
        try:
            sol_id = int(args.feedback[0])
        except ValueError:
            print("Error: Solution ID must be a number. E.g., --feedback 3 good")
            sys.exit(1)
        feedback_solution(sol_id, args.feedback[1])
        return

    # Handle --export
    if args.export:
        export_solution(args.export)
        return

    # Handle --list-sessions
    if args.list_sessions:
        from memory.session import SessionManager
        sm = SessionManager(sessions_dir="output/sessions")
        sessions = sm.list_sessions(limit=20)
        if not sessions:
            print("No sessions found.")
        else:
            print(f"\n{'ID':<14} {'State':<12} {'Name':<40} {'Messages':<10} {'Updated'}")
            print("-" * 100)
            for s in sessions:
                name = (s.get("name") or "")
                updated = s.get("updated_at", "")[:19]
                print(f"{s['id']:<14} {s['state']:<12} {name:<40} {s['message_count']:<10} {updated}")
        return

    # Handle --rerun
    if args.rerun:
        data_dir = get_data_dir()
        results_file = data_dir / "results" / "workflow_history.json"
        if not results_file.exists():
            print("No solutions found. Run a workflow first.")
            sys.exit(1)
        try:
            history = json.loads(results_file.read_text())
        except Exception:
            print("Could not read solution history.")
            sys.exit(1)
        match = next((e for e in history if e.get("id") == args.rerun), None)
        if not match:
            print(f"Solution #{args.rerun} not found. Use --list-solutions to see available IDs.")
            sys.exit(1)
        problem_desc = match.get("problem", "")
        if not problem_desc:
            print(f"Solution #{args.rerun} has no problem description to re-run.")
            sys.exit(1)
        domain = match.get("domain", "software")
        language = match.get("language", "python")
        print(f"Re-running solution #{args.rerun}: {problem_desc}...")
        # Re-run falls through to normal --run path with this problem
        args.run = True
        args.problem = problem_desc
        # Set domain/language so they flow into the workflow
        if not getattr(args, 'domain', None):
            args.domain = domain
        if not getattr(args, 'language', None) or args.language == "python":
            args.language = language

    # Validate conflicting flags
    if args.enhance and args.run:
        print("Error: Cannot use --enhance and --run together. Use --enhance to modify an existing project.")
        sys.exit(1)
    if getattr(args, 'approve', False) and args.continuous:
        print("Error: Cannot use --approve (interactive) with --continuous (loop) mode.")
        sys.exit(1)
    if args.max_tokens is not None and args.max_tokens <= 0:
        print("Error: --max-tokens must be a positive integer (e.g., --max-tokens 50000).")
        sys.exit(1)
    if args.verbose and args.quiet:
        print("Error: Cannot use --verbose and --quiet together.")
        sys.exit(1)
    if args.offline and not args.problem and not args.enhance and not sys.stdin.isatty():
        pass  # stdin piped — OK
    elif args.offline and not args.problem and not args.enhance and not args.rerun:
        print("Error: --offline requires --problem or stdin input (no web discovery available).")
        sys.exit(1)

    # Apply verbose/quiet to console
    if args.verbose:
        console.set_log_level("debug")
    elif args.quiet:
        console.set_log_level("error")

    # Run startup validation (directories, LLM backend, models, environment)
    try:
        from config.validation import validate_config_on_startup
        validate_config_on_startup(exit_on_error=False)
    except Exception:
        pass  # Validation is best-effort; don't block startup

    # Load config from file if available
    app_config = load_config(args.config)

    # Initialize model config
    config = ModelConfig()

    # Apply model settings from config file
    if app_config.models:
        config.set_model_for_role("ceo", app_config.models.ceo)
        config.set_model_for_role("cto", app_config.models.cto)
        config.set_model_for_role("product_manager", app_config.models.product_manager)
        config.set_model_for_role("researcher", app_config.models.researcher)
        config.set_model_for_role("developer", app_config.models.developer)
        config.set_model_for_role("qa_engineer", app_config.models.qa_engineer)

    # CLI backend override
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
        print(f"Using LLM backend: {args.backend}")

    # Workspace root defaults to CWD; --target overrides
    workspace_root = os.getcwd()
    if args.target:
        target_dir = Path(args.target).resolve()
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created target directory: {target_dir}")
        workspace_root = str(target_dir)
        print(f"Target directory: {workspace_root}")

    # Output dir defaults to workspace_root (CWD)
    output_dir = args.output_dir or workspace_root

    # Use output dir from config if not specified on CLI
    if args.output_dir is None and app_config.output.solutions_dir != "output/solutions":
        output_dir = app_config.output.solutions_dir

    # Validate output dir is writable
    output_path = Path(output_dir)
    if output_path.exists() and not os.access(str(output_path), os.W_OK):
        print(f"Error: Output directory is not writable: {output_dir}")
        sys.exit(1)

    # Resolve problem description: positional > --problem-flag > config > stdin
    problem_desc = args.problem or args.problem_flag or app_config.problem
    if not problem_desc and not sys.stdin.isatty():
        # Read problem from stdin pipe: echo "Build X" | python main.py
        problem_desc = sys.stdin.read().strip()
        if problem_desc:
            print(f"Read problem from stdin: {problem_desc}...")
    domain = args.domain if args.domain != "software" else app_config.domain
    language = args.language if args.language != "python" else app_config.language

    # Config file defaults for flags (CLI overrides config)
    if not args.approve and app_config.workflow.interactive_approval:
        args.approve = True
    if args.max_tokens is None and app_config.workflow.token_budget is not None:
        args.max_tokens = app_config.workflow.token_budget
    if not args.enhance and app_config.workflow.enhance_dir:
        args.enhance = app_config.workflow.enhance_dir

    # Execute requested action
    if args.check_models:
        status = check_models(config)
        print_model_status(status)

    elif args.discover:
        # Check models first
        status = check_models(config)
        missing = [n for n, s in status.items() if "missing" in s]
        if missing:
            print(f"Warning: Missing models: {', '.join(missing)}")
            print("Pull missing models: ollama pull <model>\n")

        asyncio.run(run_discovery_only(config))

    elif args.enhance:
        # Enhance an existing project
        enhance_dir = Path(args.enhance).resolve()
        if not enhance_dir.exists():
            print(f"Error: Project directory not found: {enhance_dir}")
            sys.exit(1)
        if not enhance_dir.is_dir():
            print(f"Error: --enhance expects a directory, not a file: {enhance_dir}")
            sys.exit(1)

        # The positional arg becomes the enhancement description
        if not problem_desc:
            print("Error: Provide what to enhance, e.g.:")
            print(f'  python main.py --enhance {args.enhance} "Add user authentication"')
            sys.exit(1)

        status = check_models(config)
        missing = [n for n, s in status.items() if "missing" in s]
        if missing:
            print(f"Warning: Missing models: {', '.join(missing)}")
            print("Pull missing models: ollama pull <model>\n")

        # Build enhancement problem that includes project context
        enhance_desc = (
            f"Enhance an existing {language} project at {enhance_dir}.\n"
            f"Enhancement: {problem_desc}\n"
            f"IMPORTANT: Read the existing codebase first, understand its structure, "
            f"then make targeted changes. Do NOT rewrite files from scratch - "
            f"edit existing files where possible."
        )
        custom_problem = create_custom_problem(enhance_desc, domain, language)
        print(f"Enhancing project: {enhance_dir}")
        print(f"Enhancement: {problem_desc}")

        # Point workspace and output at the existing project
        asyncio.run(run_full_workflow(
            config, custom_problem,
            output_dir=str(enhance_dir),
            workspace_root=str(enhance_dir),
            max_approval_rounds=args.max_rounds,
            app_config=app_config,
            resume=args.resume,
            interactive=args.approve,
            max_tokens=args.max_tokens,
            session_id=args.session_id,
            dry_run=args.dry_run,
            offline=args.offline
        ))

    elif args.problems:
        # Multiple problems: solve sequentially using backlog
        from company.backlog import ProjectBacklog
        backlog = ProjectBacklog()
        for idx, prob in enumerate(args.problems, 1):
            backlog.add_problem(
                problem_id=f"problem_{idx}",
                description=prob,
                domain=domain,
                severity=1.0 - (idx - 1) * 0.01,  # slight priority by order
            )
        print(f"Backlog: {backlog.size()} problems queued")

        # Check models
        status = check_models(config)
        missing = [n for n, s in status.items() if "missing" in s]
        if missing:
            print(f"Warning: Missing models: {', '.join(missing)}")
            print("Pull missing models: ollama pull <model>\n")

        while True:
            item = backlog.get_next()
            if item is None:
                break
            backlog.mark_in_progress(item.id)
            print(f"\n{'='*60}")
            print(f"Solving [{item.id}]: {item.description}")
            print(f"{'='*60}")
            custom_problem = create_custom_problem(item.description, domain, language)
            try:
                asyncio.run(run_full_workflow(
                    config, custom_problem, output_dir, workspace_root,
                    args.max_rounds, app_config, args.resume, args.approve,
                    args.max_tokens, args.session_id, args.dry_run, args.offline,
                    enable_escalation=not args.no_escalation,
                    enable_security_review=not args.no_security,
                    enable_retrospective=not args.no_retrospective,
                    max_workflow_minutes=args.max_workflow_minutes,
                    force_stop=args.force_stop
                ))
                backlog.mark_completed(item.id)
                print(f"Completed [{item.id}]: {item.description}")
            except Exception as e:
                backlog.mark_skipped(item.id, reason=str(e))
                print(f"Skipped [{item.id}]: {e}")

        # Summary
        summary = backlog.to_dict()
        print(f"\nBacklog summary: {summary['completed']} completed, "
              f"{summary['pending']} pending, "
              f"{summary['in_progress']} in-progress")

    elif args.scaffold:
        # Scaffold mode: skip research/discovery, go straight to PM -> CTO -> Developer
        status = check_models(config)
        missing = [n for n, s in status.items() if "missing" in s]
        if missing:
            print(f"Warning: Missing models: {', '.join(missing)}")
            print("Pull missing models: ollama pull <model>\n")

        scaffold_problem = create_custom_problem(args.scaffold, domain, language)
        print(f"Scaffold mode: {args.scaffold}")
        print(f"Target language: {language}")
        print(f"Working directory: {workspace_root}")
        print(f"Output directory: {output_dir}/")
        print("Skipping research/discovery — going straight to PM -> CTO -> Developer pipeline.")

        asyncio.run(run_full_workflow(
            config, scaffold_problem, output_dir, workspace_root, args.max_rounds,
            app_config, args.resume, args.approve, args.max_tokens, args.session_id,
            args.dry_run, offline=True,
            enable_escalation=not args.no_escalation,
            enable_security_review=False,
            enable_retrospective=not args.no_retrospective,
            max_workflow_minutes=args.max_workflow_minutes,
            force_stop=args.force_stop,
            scaffold_mode=True
        ))

    elif args.run or problem_desc:
        # Problem given or --run flag: run the full workflow
        # Check models first
        status = check_models(config)
        missing = [n for n, s in status.items() if "missing" in s]
        if missing:
            print(f"Warning: Missing models: {', '.join(missing)}")
            print("Pull missing models: ollama pull <model>\n")

        # Check if user provided a custom problem (CLI or config)
        custom_problem = None
        if problem_desc:
            custom_problem = create_custom_problem(problem_desc, domain, language)
            print(f"Using custom problem: {problem_desc}...")
            print(f"Target language: {language}")
            print(f"Working directory: {workspace_root}")
            print(f"Output directory: {output_dir}/")

        asyncio.run(run_full_workflow(
            config, custom_problem, output_dir, workspace_root, args.max_rounds,
            app_config, args.resume, args.approve, args.max_tokens, args.session_id,
            args.dry_run, args.offline,
            enable_escalation=not args.no_escalation,
            enable_security_review=not args.no_security,
            enable_retrospective=not args.no_retrospective,
            max_workflow_minutes=args.max_workflow_minutes,
            force_stop=args.force_stop
        ))

    elif args.continuous:
        # Check models first
        status = check_models(config)
        missing = [n for n, s in status.items() if "missing" in s]
        if missing:
            print(f"Warning: Missing models: {', '.join(missing)}")
            print("Pull missing models: ollama pull <model>\n")

        asyncio.run(run_continuous_workflow(
            config,
            output_dir=output_dir,
            delay=args.continuous_delay,
            max_iterations=args.max_iterations,
            workspace_root=workspace_root,
            app_config=app_config
        ))

    else:
        # No flags, no problem -> enter interactive chat mode
        from interactive_mode import InteractiveCodingSession
        data_dir = get_data_dir()
        session = InteractiveCodingSession(
            workspace_root=workspace_root,
            model_config=config,
            language=language,
            data_dir=str(data_dir)
        )
        asyncio.run(session.conversational_mode())


if __name__ == "__main__":
    main()
