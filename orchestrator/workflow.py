"""Company workflow orchestrator - Manages the end-to-end problem-solving workflow."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import (CEOAgent, CTOAgent, DataAnalystAgent, DeveloperAgent,
                    DevOpsEngineerAgent, ProductManagerAgent, QAEngineerAgent,
                    ResearcherAgent, SecurityEngineerAgent)
from collaboration.meeting import MeetingType, quick_meeting
from config.models import ModelConfig
from memory.session import SessionManager, SessionState
from memory.shared_memory import SharedMemory
from orchestrator.escalation import EscalationAction, EscalationManager
from orchestrator.message_bus import MessageBus
from orchestrator.plan_mode import PlanManager
from orchestrator.task_manager import TaskManager, TaskPriority
from research.cross_validator import CrossValidator
from research.problem_discoverer import (DiscoveredProblem, ProblemDiscoverer,
                                         ProblemFrequency, ProblemSeverity)
from research.sources import ResearchSources
from tools import UnifiedTools
from tools.git_tools import GitTools
from ui.console import console
from utils.cost_tracker import CostTracker
from utils.error_recovery import ErrorRecoverySystem, RetryStrategy
from utils.health_check import HealthChecker, HealthStatus
from utils.memory_monitor import MemoryMonitor
from utils.progress_tracker import ProgressTracker
from utils.structured_logging import LogLevel, get_structured_logger

# Confidence band thresholds for graduated response
CONFIDENCE_CRITICAL = 0.3   # Requires immediate intervention
CONFIDENCE_LOW = 0.5        # Requires supervisor notification
CONFIDENCE_MEDIUM = 0.7     # Standard review
CONFIDENCE_HIGH = 0.85      # Light review


class WorkflowPhase(Enum):
    """Phases of the company workflow."""

    IDLE = "idle"
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    ANALYSIS = "analysis"
    OPPORTUNITY_EVALUATION = "opportunity_evaluation"
    TECHNICAL_DESIGN = "technical_design"
    DESIGN_REVIEW = "design_review"
    IMPLEMENTATION = "implementation"
    CODE_EXECUTION = "code_execution"
    QA_VALIDATION = "qa_validation"
    SECURITY_REVIEW = "security_review"
    CEO_APPROVAL = "ceo_approval"
    RETROSPECTIVE = "retrospective"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowState:
    """Current state of the workflow."""

    phase: WorkflowPhase = WorkflowPhase.IDLE
    current_problem: Optional[DiscoveredProblem] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# ============================================================
#  INITIALIZATION
# ============================================================


class CompanyWorkflow:
    """
    Orchestrates the full company workflow from problem discovery to solution delivery.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        model_config: ModelConfig = None,
        memory_persist_dir: str = None,
        output_dir: str = "output/solutions",
        data_dir: Optional[str] = None,
        run_code_execution: bool = True,
        llm_config: Optional[Dict[str, Any]] = None,
        enable_meetings: bool = True,
        enable_learning: bool = True,
        enable_escalation: bool = True,
        enable_security_review: bool = True,
        enable_retrospective: bool = True,
        max_workflow_minutes: int = 0,
        force_stop: bool = False,
        scaffold_mode: bool = False
    ):
        self.workspace_root = workspace_root
        self.model_config = model_config or ModelConfig()
        self.memory_persist_dir = memory_persist_dir
        self.output_dir = output_dir
        self.run_code_execution = run_code_execution
        self.llm_config = llm_config or {}
        self.enable_meetings = enable_meetings
        self.enable_learning = enable_learning
        self.enable_escalation = enable_escalation
        self.enable_security_review = enable_security_review
        self.enable_retrospective = enable_retrospective
        self.max_workflow_minutes = max_workflow_minutes
        self.force_stop = force_stop
        self.scaffold_mode = scaffold_mode
        # data_dir is for internal Company-AGI data (logs, reports)
        # Defaults to workspace_root for backwards compatibility
        self.data_dir = data_dir or workspace_root

        # Initialize agents
        self._init_agents()

        # Apply LLM config overrides to agents (user config file settings)
        if self.llm_config:
            for agent in self.agents.values():
                if "temperature" in self.llm_config:
                    agent.config.temperature = self.llm_config["temperature"]
                if "max_tokens" in self.llm_config:
                    agent.config.max_tokens = self.llm_config["max_tokens"]
                if self.llm_config.get("streaming"):
                    agent._streaming_enabled = True
                    agent._stream_callback = console.create_stream_callback(agent.name)

        # Initialize infrastructure - logs go to data_dir, not workspace_root
        self.message_bus = MessageBus(log_dir=f"{self.data_dir}/logs")
        self.shared_memory = SharedMemory(persist_dir=memory_persist_dir or f"{self.data_dir}/memory")
        self.problem_discoverer = ProblemDiscoverer(
            model=self.model_config.get_model_name("researcher")
        )
        self.research_sources = ResearchSources()

        # Initialize unified tools (Claude Code style)
        self.tools = UnifiedTools(
            workspace_root=workspace_root,
            persist_dir=memory_persist_dir
        )

        # Register agents with message bus
        self._register_agents()

        # Initialize progress tracker
        self.progress = ProgressTracker(
            metrics_file=Path(self.data_dir) / "metrics" / "phase_metrics.json"
        )
        self.progress.add_phases([
            {"id": "research", "name": "Research"},
            {"id": "data_analysis", "name": "Data Analysis"},
            {"id": "analysis", "name": "Analysis"},
            {"id": "opportunity", "name": "Opportunity Eval"},
            {"id": "design", "name": "Technical Design"},
            {"id": "design_review", "name": "Design Review"},
            {"id": "implementation", "name": "Implementation"},
            {"id": "code_execution", "name": "Code Execution"},
            {"id": "qa_validation", "name": "QA Validation"},
            {"id": "security_review", "name": "Security Review"},
            {"id": "ceo_approval", "name": "CEO Approval"},
            {"id": "delivery", "name": "Delivery"},
            {"id": "retrospective", "name": "Retrospective"},
        ])

        # Initialize error recovery
        self.error_recovery = ErrorRecoverySystem(
            max_retries=2,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=2.0,
            enable_logging=True
        )

        # Initialize health checker
        self.health_checker = HealthChecker()

        # Initialize cost tracker and set as global singleton so agents can use it
        import utils.cost_tracker as _cost_tracker_mod
        self.cost_tracker = CostTracker(
            history_file=Path(self.data_dir) / "costs" / "cost_history.json"
        )
        _cost_tracker_mod._cost_tracker = self.cost_tracker

        # Task manager for tracking agent work
        self.task_manager = TaskManager()

        # Plan manager for structured execution plans
        self.plan_manager = PlanManager(plans_dir=f"{self.data_dir}/plans")

        # Structured logger with JSON output for tracing
        self.structured_logger = get_structured_logger(
            name="company_agi",
            level=LogLevel.INFO,
            json_output=Path(self.data_dir) / "logs" / "structured.jsonl"
        )

        # Memory monitor with warning callbacks and auto-compaction
        def _compact_agent_memory(agent_name: str) -> None:
            console.warning(f"[Memory] Agent {agent_name} memory high — compacting")
            agent = self.agents.get(agent_name)
            if agent:
                agent.memory.compact()

        # Memory monitor disabled — its background thread + psutil polling
        # waste RAM better reserved for Ollama LLM inference.
        self.memory_monitor = MemoryMonitor(
            on_warning=lambda w: console.warning(f"[Memory] {w.message}"),
            on_compact_needed=_compact_agent_memory,
        )
        self.memory_monitor.disabled = True

        # Git tools for version control in delivery
        self.git_tools = GitTools(repo_path=workspace_root)

        # Session manager for session persistence / resume
        self.session_manager = SessionManager(
            sessions_dir=f"{self.data_dir}/sessions"
        )
        # Clean up old sessions on startup
        try:
            removed = self.session_manager.cleanup_old_sessions()
            if removed > 0:
                console.debug(f"Cleaned up {removed} old session(s)")
        except Exception as e:
            console.warning(f"Session cleanup failed (non-critical): {e}")

        # Hooks manager for lifecycle events
        from utils.hooks import HooksManager
        self.hooks_manager = HooksManager()

        # Escalation manager for graduated recovery
        self.escalation_manager = EscalationManager()
        strat_path = Path(self.data_dir) / "escalation" / "strategy_memory.json"
        self.escalation_manager.load_strategy_memory(strat_path)

        # Cross-validator for research deduplication
        self.cross_validator = CrossValidator()

        # Performance tracker — records task outcomes and computes KPIs
        # Load from disk so KPIs persist across runs
        from company.performance import PerformanceTracker
        perf_path = Path(self.data_dir) / "performance" / "kpis.json"
        self.performance_tracker = PerformanceTracker.load(perf_path)

        # Org chart — defines reporting lines and escalation paths
        from company.organization import OrgChart
        self.org_chart = OrgChart.default()

        # Trust tracker — inter-agent trust based on review outcomes
        # Load from disk so trust persists across runs
        from company.trust import TrustTracker
        trust_path = Path(self.data_dir) / "trust" / "trust_scores.json"
        self.trust_tracker = TrustTracker.load(trust_path)

        # Sprint manager for team alignment and progress tracking
        from company.sprint import SprintManager
        self.sprint_manager = SprintManager()
        self._current_sprint_id: str = ""

        # Agent personality and experience


        from agents.personality import (DEFAULT_PERSONALITIES, AgentExperience,
                                        AgentPersonality)
        self._experience_dir = Path(self.data_dir) / "experience"
        for agent_name, agent in self.agents.items():
            role_key = agent.role.value
            # Apply personality
            personality = DEFAULT_PERSONALITIES.get(role_key, AgentPersonality())
            agent._personality = personality
            # Adjust temperature based on personality
            agent.config.temperature = personality.adjust_temperature(agent.config.temperature)
            # Load experience from disk
            exp_path = self._experience_dir / f"{agent_name}.json"
            agent._experience = AgentExperience.load(exp_path)

        # Company culture — inject values into all agent prompts
        from company.culture import COMPANY_CULTURE
        culture_text = COMPANY_CULTURE.get_prompt_injection()
        for agent in self.agents.values():
            agent.system_prompt += culture_text

        # Agent pool for headcount tracking
        from company.hiring import AgentPool
        self.agent_pool = AgentPool()
        for agent_name, agent in self.agents.items():
            self.agent_pool.register(agent_name, agent.role.value)

        # Meeting managers
        from company.meetings import StandupManager, IncidentResponseManager
        self.standup_manager = StandupManager()
        self.incident_manager = IncidentResponseManager()

        # Project backlog for multi-problem management
        from company.backlog import ProjectBacklog
        self.backlog = ProjectBacklog()

        # RAG store for injecting relevant past solutions into agent prompts
        from memory.rag_store import RAGStore
        self.rag_store = RAGStore(persist_dir=str(Path(self.output_dir) / "rag"))
        for agent in self.agents.values():
            agent.set_rag_store(self.rag_store)

        # Workflow state
        self.state = WorkflowState()
        self.workflow_history: List[WorkflowState] = []

    def _init_agents(self) -> None:
        """Initialize all company agents."""
        # Use get_model_name() which accepts string role names
        ceo_model = self.model_config.get_model_name("ceo")
        cto_model = self.model_config.get_model_name("cto")
        pm_model = self.model_config.get_model_name("product_manager")
        researcher_model = self.model_config.get_model_name("researcher")
        dev_model = self.model_config.get_model_name("developer")
        qa_model = self.model_config.get_model_name("qa_engineer")
        devops_model = self.model_config.get_model_name("devops_engineer")
        data_analyst_model = self.model_config.get_model_name("data_analyst")
        security_model = self.model_config.get_model_name("security_engineer")

        self.ceo = CEOAgent(
            model=ceo_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.cto = CTOAgent(
            model=cto_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.product_manager = ProductManagerAgent(
            model=pm_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.researcher = ResearcherAgent(
            model=researcher_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.developer = DeveloperAgent(
            model=dev_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.qa_engineer = QAEngineerAgent(
            model=qa_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.devops_engineer = DevOpsEngineerAgent(
            model=devops_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.data_analyst = DataAnalystAgent(
            model=data_analyst_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )
        self.security_engineer = SecurityEngineerAgent(
            model=security_model,
            workspace_root=self.workspace_root,
            memory_persist_dir=self.memory_persist_dir
        )

        self.agents = {
            "CEO": self.ceo,
            "CTO": self.cto,
            "ProductManager": self.product_manager,
            "Researcher": self.researcher,
            "Developer": self.developer,
            "QAEngineer": self.qa_engineer,
            "DevOpsEngineer": self.devops_engineer,
            "DataAnalyst": self.data_analyst,
            "SecurityEngineer": self.security_engineer
        }

        # Enable ReAct tools for all agents that need to call tools in their loops
        for agent in [
            self.developer,
            self.researcher,
            self.cto,
            self.qa_engineer,
            self.security_engineer,
            self.devops_engineer,
            self.data_analyst,
        ]:
            if hasattr(agent, 'enable_react_tools'):
                agent.enable_react_tools()

    def _register_agents(self) -> None:
        """Register all agents with the message bus."""
        for name, agent in self.agents.items():
            self.message_bus.register_agent(name, agent)

        # Wire shared memory into all agents for institutional knowledge
        for agent in self.agents.values():
            agent._shared_memory = self.shared_memory

        # Set up topic subscriptions
        self.message_bus.subscribe("CEO", "decisions")
        self.message_bus.subscribe("CEO", "approvals")
        self.message_bus.subscribe("CTO", "technical")
        self.message_bus.subscribe("ProductManager", "product")
        self.message_bus.subscribe("Developer", "implementation")
        self.message_bus.subscribe("QAEngineer", "testing")
        self.message_bus.subscribe("DevOpsEngineer", "deployment")
        self.message_bus.subscribe("DataAnalyst", "research")
        self.message_bus.subscribe("SecurityEngineer", "security")

    # ============================================================
    #  MAIN WORKFLOW ENTRY POINT
    # ============================================================

    async def run_full_workflow(
        self,
        auto_discover: bool = True,
        problem: DiscoveredProblem = None,
        max_approval_rounds: int = 0,
        resume: bool = False,
        interactive: bool = False,
        session_id: Optional[str] = None,
        dry_run: bool = False,
        scaffold_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete company workflow.

        Args:
            auto_discover: If True, discover problems automatically
            problem: If provided, work on this specific problem
            max_approval_rounds: Max QA+CEO retry rounds (0 = unlimited, retries until CEO approves)
            resume: If True, attempt to resume from last checkpoint
            interactive: If True, pause at key decision points for user approval
            session_id: If provided, resume this specific session
            dry_run: If True, skip code execution, testing, and delivery

        Returns:
            Workflow result with all artifacts
        """
        self._interactive = interactive
        self.state = WorkflowState(started_at=datetime.now())
        self.progress.start()

        # Create or resume a session
        if session_id:
            session = self.session_manager.resume_session(session_id=session_id)
            if session:
                console.info(f"Resumed session: {session.id}")
            else:
                console.warning(f"Session {session_id} not found, creating new session")
                session = self.session_manager.create_session(
                    name=problem.description if problem else "auto-discover"
                )
        elif resume:
            session = self.session_manager.resume_session()
            if session:
                console.info(f"Resumed most recent session: {session.id}")
            else:
                session = self.session_manager.create_session(
                    name=problem.description if problem else "auto-discover"
                )
        else:
            session = self.session_manager.create_session(
                name=problem.description if problem else "auto-discover"
            )

        # Set language early from problem metadata so all phases can access it
        if problem and problem.metadata.get("language"):
            self.state.artifacts["language"] = problem.metadata["language"]

        # Run pre-flight health checks
        console.info("Running pre-flight health checks...")
        health_results = self.health_checker.run_all_checks()
        overall_health = self.health_checker.get_overall_status()
        if overall_health == HealthStatus.UNHEALTHY:
            unhealthy = [name for name, r in health_results.items() if r.status == HealthStatus.UNHEALTHY]
            console.error(f"System unhealthy: {', '.join(unhealthy)}. Workflow may fail.")
        elif overall_health == HealthStatus.DEGRADED:
            console.warning("System health degraded. Proceeding with caution.")
        else:
            console.success("All health checks passed")

        # Start background health monitoring (daemon thread, stops with process)
        self.health_checker.start_background_checks()

        # Start cost tracking session
        cost_session_id = self.cost_tracker.start_session()
        console.info(f"Cost tracking session: {cost_session_id}")

        # Start structured logging context for this workflow run
        wf_log_ctx = self.structured_logger.new_context(workflow_id=cost_session_id)
        self.structured_logger.context = wf_log_ctx
        self.structured_logger.info("Workflow started")

        # Start memory monitoring
        self.memory_monitor.start_monitoring()

        # Trigger session start hook
        from utils.hooks import HookContext, HookEvent
        await self.hooks_manager.session_start(session.id)

        # Adjust agent behavior based on past performance
        self._adjust_agent_behavior()

        # Load lessons from past retrospectives into agent prompts
        self._load_previous_learnings()

        # Run 1:1 performance reviews for underperformers (before main work)
        await self._run_performance_reviews()

        # Attempt to resume from checkpoint
        resumed_phase = None
        if resume and self._load_checkpoint():
            resumed_phase = self.state.phase
            console.info(f"Resumed from checkpoint at phase: {resumed_phase.value}")

        try:
            # Load previous session summary and inject action items into relevant agents
            try:
                import json as _json
                _summary_path = Path(self.data_dir) / "session_summary.json"
                if _summary_path.exists():
                    _prev = _json.loads(_summary_path.read_text())
                    _prev_items = _prev.get("action_items", [])
                    _prev_insights = _prev.get("key_insights", [])
                    if _prev_items or _prev_insights:
                        _prev_note = "\n\n[PREVIOUS SESSION LESSONS]\n"
                        if _prev_items:
                            _prev_note += "Action items from last run:\n" + "\n".join(f"- {a}" for a in _prev_items[:5]) + "\n"
                        if _prev_insights:
                            _prev_note += "Key insights from last run:\n" + "\n".join(f"- {i}" for i in _prev_insights[:3]) + "\n"
                        _prev_note += "Apply these learnings to avoid repeating past mistakes.\n"
                        for _agent in [self.cto, self.developer, self.pm]:
                            if _prev_note not in _agent.system_prompt:
                                _agent.system_prompt += _prev_note
                        console.info(f"Previous session lessons injected ({len(_prev_items)} action items, {len(_prev_insights)} insights)")
            except Exception as _sle:
                console.warning(f"Session summary load failed: {_sle}")

            # Inject cross-run strategy memory into CTO + Developer at start of run
            try:
                cross_run_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                if cross_run_context and len(cross_run_context.strip()) > 20:
                    cross_run_injection = (
                        f"\n\n[CROSS-RUN STRATEGY MEMORY]\n{cross_run_context}\n"
                        "These lessons were learned from previous runs on similar problems. "
                        "Do NOT repeat approaches that previously failed.\n"
                    )
                    for agent in [self.cto, self.developer]:
                        if cross_run_injection not in agent.system_prompt:
                            agent.system_prompt += cross_run_injection
                    console.info("Cross-run strategy memory injected into CTO and Developer prompts")
            except Exception as _sme:
                console.warning(f"Cross-run strategy memory injection failed: {_sme}")

            # CEO morning brief — sets priorities and constraints for the whole run
            await self._run_ceo_morning_brief(problem)

            # Morning standup for team alignment
            problem_name = problem.description if problem else "Auto-Discovery"
            await self._run_morning_standup(problem_name)

            # Phase 1: Research
            if resumed_phase and resumed_phase.value not in ("idle", "research"):
                self.progress.skip_phase("research")
            else:
                self.progress.start_phase("research")
                if auto_discover and not problem:
                    console.section("PHASE 1: RESEARCH")
                    console.info("Discovering problems from web sources...")
                    research_result = await self.error_recovery.retry_async(
                        self._run_research_phase,
                        "research_phase"
                    )
                    if not research_result.success:
                        console.error(f"Research phase failed after retries: {research_result.error}")
                elif problem:
                    self.state.current_problem = problem
                    self.state.artifacts["discovered_problems"] = [problem.to_dict()]
                    self.state.artifacts["user_provided_problem"] = True
                    console.info("Using provided problem/requirement")
                    console.show_problem(problem.to_dict())
                self.progress.complete_phase("research")
                if self._current_sprint_id:
                    try:
                        self.sprint_manager.complete_task(self._current_sprint_id, "research")
                    except ValueError:
                        pass
                self._save_checkpoint()

            if not self.state.current_problem:
                # Part 2b: Expand search scope instead of immediate FAILED
                console.warning("No problems found — expanding search scope")
                expanded = False
                for expand_attempt in range(2):
                    console.info(f"Expanded search attempt {expand_attempt + 1}/2...")
                    # Re-run research with broader web search
                    try:
                        web_problems = await self._discover_from_web_search()
                        if web_problems:
                            top = max(web_problems, key=lambda p: p.score)
                            self.state.current_problem = top
                            self.state.artifacts["discovered_problems"] = [p.to_dict() for p in web_problems]
                            console.success(f"Found problem on expanded search: {top.description[:60]}")
                            expanded = True
                            break
                    except Exception as e:
                        console.warning(f"Expanded search failed: {e}")

                if not expanded:
                    self.state.phase = WorkflowPhase.FAILED
                    self.state.error = "No problem discovered (even after expanded search)"
                    self.progress.fail_phase("research", "No problem discovered")
                    console.error("No problem discovered even after expanded search!")
                    self.health_checker.stop_background_checks()
                    self.memory_monitor.stop_monitoring()
                    self.cost_tracker.end_session()
                    result = self._get_workflow_result()
                    self._persist_result(result)
                    return result

            # Store discovered problem in shared memory
            if self.state.current_problem:
                self.shared_memory.store_problem(
                    description=self.state.current_problem.description,
                    source="auto_discovery" if not problem else "user_input",
                    severity=self.state.current_problem.severity.value if isinstance(self.state.current_problem.severity, ProblemSeverity) else str(self.state.current_problem.severity),
                    domain=self.state.current_problem.domain,
                    created_by="researcher"
                )

            # Publish to message bus
            self.message_bus.publish(
                topic="product",
                sender="Researcher",
                content=f"Discovered problem: {self.state.current_problem.description}",
                priority=3
            )

            # Quality gate: Research
            await self._check_quality_gate("research")

            # Phase 1b: Data Analysis (cross-validate, deduplicate, detect bias)
            # Skipped in scaffold_mode — not meaningful when spec is user-provided
            if scaffold_mode or self.scaffold_mode:
                self.progress.skip_phase("data_analysis")
                console.info("Scaffold mode: skipping DATA ANALYSIS phase")
            elif resumed_phase and resumed_phase.value not in ("idle", "research", "data_analysis"):
                self.progress.skip_phase("data_analysis")
            else:
                self.progress.start_phase("data_analysis")
                console.section("PHASE 1b: DATA ANALYSIS")
                console.info("DataAnalyst cross-validating and deduplicating research findings...")
                await self._run_data_analysis_phase()

                # Part 3c: Feed DataAnalyst cross-validation results back into problem scoring
                cross_val_output = self.state.artifacts.get("cross_validation_verdicts", {})
                discovered = self.state.artifacts.get("discovered_problems", [])
                if cross_val_output and discovered and self.state.current_problem:
                    # Use parsed structured verdicts from DataAnalyst LLM cross-validation
                    from utils.output_parser import StructuredOutputParser
                    parser = StructuredOutputParser()
                    parsed_verdicts = cross_val_output if isinstance(cross_val_output, dict) else parser.parse_data_analyst_verdict(str(cross_val_output))
                    findings_list = parsed_verdicts.get("findings", [])

                    # Map parsed findings back to problems by index
                    validation_results = {}
                    for i, p_dict in enumerate(discovered):
                        pid = p_dict.get("id", "")
                        if i < len(findings_list):
                            finding = findings_list[i]
                            validation_results[pid] = {
                                "verdict": finding.get("verdict", "UNCONFIRMED"),
                                "confidence": finding.get("confidence", 0.5),
                                "evidence_for": finding.get("evidence_for", ""),
                                "evidence_against": finding.get("evidence_against", ""),
                            }

                    if validation_results:
                        # Reconstruct DiscoveredProblem objects for adjustment
                        problem_objects = []
                        for p_dict in discovered:
                            try:
                                prob = DiscoveredProblem(
                                    id=p_dict.get("id", ""),
                                    description=p_dict.get("description", ""),
                                    severity=ProblemSeverity(p_dict.get("severity", "medium")),
                                    frequency=ProblemFrequency(p_dict.get("frequency", "common")),
                                    score=p_dict.get("score", 0.0),
                                    sources=p_dict.get("sources", []),
                                    evidence=p_dict.get("evidence", []),
                                    keywords=p_dict.get("keywords", []),
                                    domain=p_dict.get("domain", "general"),
                                    target_users=p_dict.get("target_users", ""),
                                )
                                problem_objects.append(prob)
                            except Exception as e:
                                console.warning(f"Skipping malformed problem dict during validation: {e}")
                                continue

                        adjusted = self.problem_discoverer.apply_validation_adjustments(
                            problem_objects, validation_results
                        )
                        self.state.artifacts["discovered_problems"] = [p.to_dict() for p in adjusted]

                        # Update current problem if it was adjusted
                        if adjusted:
                            new_top = max(adjusted, key=lambda p: p.score)
                            if new_top.id != self.state.current_problem.id:
                                console.info(f"Top problem changed after validation feedback: {new_top.description[:60]}")
                                self.state.current_problem = new_top

                        console.info(f"Applied validation adjustments to {len(validation_results)} problems")

                # Quality gate: Data Analysis
                await self._check_quality_gate("data_analysis")

                self.progress.complete_phase("data_analysis")
                self._save_checkpoint()

            # Phase 2: Analysis
            if resumed_phase and resumed_phase.value not in ("idle", "research", "analysis"):
                self.progress.skip_phase("analysis")
            else:
                self.progress.start_phase("analysis")
                console.section("PHASE 2: ANALYSIS")
                console.info("Researcher and Product Manager analyzing the problem...")
                analysis_result = await self.error_recovery.retry_async(
                    self._run_analysis_phase,
                    "analysis_phase"
                )
                if not analysis_result.success:
                    console.error(f"Analysis phase failed after retries: {analysis_result.error}")
                self.progress.complete_phase("analysis")
                self._save_checkpoint()

            # Phase 3: Opportunity Evaluation (CEO Decision)
            # Skipped in scaffold_mode — user has already defined the spec
            if scaffold_mode or self.scaffold_mode:
                self.progress.skip_phase("opportunity")
                console.info("Scaffold mode: skipping OPPORTUNITY EVALUATION phase (spec is user-provided)")
            elif resumed_phase and resumed_phase.value not in ("idle", "research", "analysis", "opportunity_evaluation"):
                self.progress.skip_phase("opportunity")
            else:
                self.progress.start_phase("opportunity")
                console.section("PHASE 3: OPPORTUNITY EVALUATION")
                console.info("CEO evaluating the opportunity with team meeting...")
                if not await self._run_opportunity_evaluation():
                    # Pivot: PM narrows scope and re-evaluate once
                    console.warning("Opportunity rejected by CEO. PM narrowing scope...")
                    narrow_task = {
                        "type": "define_requirements",
                        "problem": self.state.current_problem.to_dict(),
                        "target_users": self.state.current_problem.target_users,
                        "constraint": "Narrow the scope to the smallest viable subset that addresses the core pain point. Reduce complexity significantly."
                    }
                    narrow_result = await self.product_manager.execute_task(narrow_task)
                    self.state.artifacts["narrowed_requirements"] = narrow_result.output
                    # Update problem description with narrowed scope
                    if isinstance(narrow_result.output, dict):
                        narrowed_desc = narrow_result.output.get("requirements", "")
                        if narrowed_desc:
                            self.state.current_problem.description = f"[Narrowed] {self.state.current_problem.description}"
                            self.state.current_problem.potential_solution_ideas = [narrowed_desc]

                    # Re-evaluate with narrowed scope
                    console.info("Re-evaluating narrowed opportunity...")
                    if not await self._run_opportunity_evaluation():
                        # Part 2a: Try next problem from discovered list instead of FAILED
                        discovered = self.state.artifacts.get("discovered_problems", [])
                        pivoted = False
                        if len(discovered) > 1:
                            # Remove the rejected problem and try the next best
                            current_id = self.state.current_problem.id if self.state.current_problem else None
                            remaining = [p for p in discovered if p.get("id") != current_id]
                            if remaining:
                                best = max(remaining, key=lambda p: p.get("score", 0))
                                console.info(f"CEO rejected — pivoting to next problem: {best.get('description', '')[:60]}")
                                # Reconstruct DiscoveredProblem from dict
                                self.state.current_problem = DiscoveredProblem(
                                    id=best.get("id", "PIVOT-001"),
                                    description=best.get("description", ""),
                                    severity=ProblemSeverity(best.get("severity", "medium")),
                                    frequency=ProblemFrequency(best.get("frequency", "common")),
                                    target_users=best.get("target_users", ""),
                                    evidence=best.get("evidence", []),
                                    sources=best.get("sources", []),
                                    domain=best.get("domain", "general"),
                                    keywords=best.get("keywords", []),
                                    score=best.get("score", 0.0),
                                )
                                self.state.artifacts["discovered_problems"] = remaining
                                # Re-run opportunity evaluation with new problem
                                if await self._run_opportunity_evaluation():
                                    pivoted = True

                        if not pivoted:
                            self.state.phase = WorkflowPhase.FAILED
                            self.state.error = "Opportunity rejected by CEO (all problems exhausted)"
                            self.progress.fail_phase("opportunity", "Rejected by CEO after pivot")
                            console.error("All problems rejected by CEO")
                            self.health_checker.stop_background_checks()
                            self.memory_monitor.stop_monitoring()
                            self.cost_tracker.end_session()
                            result = self._get_workflow_result()
                            self._persist_result(result)
                            return result
                console.success("Opportunity approved!")

                # Store decision in shared memory
                problem = self.state.current_problem
                last_decision = self.state.decisions[-1] if self.state.decisions else {}
                self.shared_memory.store_decision(
                    decision=f"Approved: {problem.description}",
                    reasoning=last_decision.get("reasoning", ""),
                    made_by="CEO",
                    participants=["CEO", "CTO", "ProductManager"]
                )

                # Announce approval via message bus
                self.message_bus.announce(
                    sender="CEO",
                    announcement=f"Approved project: {problem.description}"
                )

                self.progress.complete_phase("opportunity")
                self._save_checkpoint()

                if not self._interactive_gate("Opportunity Evaluation", f"Problem: {problem.description}"):
                    self.state.phase = WorkflowPhase.FAILED
                    self.state.error = "Aborted by user"
                    result = self._get_workflow_result()
                    self._persist_result(result)
                    return result

            # Phase 4: Technical Design
            if resumed_phase and resumed_phase.value not in ("idle", "research", "analysis", "opportunity_evaluation", "technical_design"):
                self.progress.skip_phase("design")
            else:
                self.progress.start_phase("design")
                console.section("PHASE 4: TECHNICAL DESIGN")
                console.info("CTO designing technical architecture...")
                design_result = await self.error_recovery.retry_async(
                    self._run_technical_design_phase,
                    "technical_design_phase"
                )
                if not design_result.success:
                    console.error(f"Technical design phase failed after retries: {design_result.error}")
                console.success("Technical design complete")

                # Quality gate: Tech Design (root cause must be extracted)
                await self._check_quality_gate("tech_design")

                self.progress.complete_phase("design")
                if self._current_sprint_id:
                    try:
                        self.sprint_manager.complete_task(self._current_sprint_id, "design")
                    except ValueError:
                        pass
                self._save_checkpoint()

                if not self._interactive_gate("Technical Design", "Architecture and requirements defined. Ready to implement."):
                    self.state.phase = WorkflowPhase.FAILED
                    self.state.error = "Aborted by user"
                    result = self._get_workflow_result()
                    self._persist_result(result)
                    return result

            # Phase 4b: Design Review (CTO + Developer + QA review architecture before coding)
            if resumed_phase and resumed_phase.value not in ("idle", "research", "data_analysis", "analysis", "opportunity_evaluation", "technical_design", "design_review"):
                self.progress.skip_phase("design_review")
            else:
                self.progress.start_phase("design_review")
                console.section("PHASE 4b: DESIGN REVIEW")
                console.info("CTO, Developer, and QA reviewing architecture before implementation...")
                await self._run_design_review_phase()
                self.progress.complete_phase("design_review")
                self._save_checkpoint()

            # Phase 5: Implementation
            if resumed_phase and resumed_phase.value not in ("idle", "research", "analysis", "opportunity_evaluation", "technical_design", "implementation"):
                self.progress.skip_phase("implementation")
            else:
                self.progress.start_phase("implementation")
                console.section("PHASE 5: IMPLEMENTATION")
                console.info("Developer implementing the solution...")
                impl_recovery = await self.error_recovery.retry_async(
                    self._run_implementation_phase,
                    "implementation_phase"
                )
                if not impl_recovery.success:
                    console.error(f"Implementation phase failed after retries: {impl_recovery.error}")
                console.success("Implementation complete")
                if self._current_sprint_id:
                    try:
                        self.sprint_manager.complete_task(self._current_sprint_id, "implement")
                    except ValueError:
                        pass

                # Notify message bus
                self.message_bus.publish(
                    topic="testing",
                    sender="Developer",
                    content="Implementation complete. Ready for QA review.",
                    priority=3
                )

                self.progress.complete_phase("implementation")
                self._save_checkpoint()

                # MoA: CEO and CTO independently review the implementation
                try:
                    if hasattr(self, 'ceo') and hasattr(self, 'cto'):
                        from collaboration.moa_aggregator import MoAReviewAggregator
                        _impl_artifact = self.state.artifacts.get("implementation", {})
                        if isinstance(_impl_artifact, dict):
                            impl_code = str(_impl_artifact.get("code", "") or _impl_artifact.get("implementation", ""))[:3000]
                        else:
                            impl_code = str(_impl_artifact)[:3000]
                        if impl_code:
                            ceo_review = await self.ceo.generate_response_async(
                                f"Review this implementation for business requirements alignment and quality:\n\n{impl_code[:2000]}\n\n"
                                f"Problem being solved: {self.state.current_problem.description}\n\n"
                                "Identify top 3 concerns. Be specific."
                            )
                            cto_review = await self.cto.generate_response_async(
                                f"Review this implementation for technical correctness and architecture:\n\n{impl_code[:2000]}\n\n"
                                "Identify top 3 technical issues. Be specific."
                            )
                            aggregator = MoAReviewAggregator(synthesizer_agent=self.qa_engineer)
                            moa_review = await aggregator.aggregate_reviews(
                                artifact=impl_code,
                                reviews={"CEO": ceo_review, "CTO": cto_review},
                                context=f"Problem: {self.state.current_problem.description}"
                            )
                            self.state.artifacts["moa_review"] = moa_review
                            console.info("MoA review complete — synthesized CEO + CTO feedback")

                            # Critic ensemble pass on the implementation
                            try:
                                from collaboration.critic_ensemble import CriticEnsemble
                                critic_ensemble = CriticEnsemble(personas=["skeptic", "pragmatist", "user"])
                                critique_result = await critic_ensemble.critique(
                                    agent=self.qa_engineer,
                                    artifact=impl_code,
                                    task_context=self.state.current_problem.description,
                                    synthesize=True,
                                )
                                self.state.artifacts["critic_ensemble_review"] = critique_result.get("synthesis", "")
                                console.info(f"Critic ensemble review complete — verdict: {critique_result.get('verdict', 'N/A')}")
                            except Exception as _critic_err:
                                console.warning(f"Critic ensemble review skipped: {_critic_err}")
                except Exception as _moa_err:
                    console.warning(f"MoA review skipped: {_moa_err}")

                if not self._interactive_gate("Implementation", "Code generated. Proceeding to testing and QA."):
                    self.state.phase = WorkflowPhase.FAILED
                    self.state.error = "Aborted by user"
                    result = self._get_workflow_result()
                    self._persist_result(result)
                    return result

            # Part 4b: Extract root cause statement from developer output for QA context
            impl_output = self.state.artifacts.get("implementation", {})
            root_cause_statement = ""
            if isinstance(impl_output, dict):
                impl_text = impl_output.get("implementation", "")
            else:
                impl_text = str(impl_output)
            # Look for ROOT CAUSE: line in developer output
            import re as _re
            _rc_match = _re.search(r'ROOT\s*CAUSE:\s*(.+?)(?:\n|$)', impl_text, _re.IGNORECASE)
            if _rc_match:
                root_cause_statement = _rc_match.group(1).strip()
                console.info(f"Root cause identified: {root_cause_statement[:80]}")
                self.state.artifacts["root_cause"] = root_cause_statement

            # Phases 6-8: QA + Security Validation + CEO Approval loop
            # Uses EscalationManager for graduated recovery when enabled
            max_qa_retries = 2
            approval_round = 0
            dev_fix_count = 0
            failure_history: List[str] = []
            # Never stop unless force_stop is set — unlimited rounds
            max_rounds = max_approval_rounds if (max_approval_rounds > 0 and getattr(self, 'force_stop', False)) else 999

            while True:
                approval_round += 1

                # Check token budget limit — allow up to 3 extensions before partial delivery
                if self.cost_tracker.is_over_budget():
                    budget_extensions = getattr(self, '_budget_extensions', 0)
                    self._budget_extensions = budget_extensions + 1
                    pct = max(0.1, 0.3 - (budget_extensions * 0.05))  # Decreasing but never zero
                    console.warning(f"Token budget reached (extension {budget_extensions + 1}). Extending by {int(pct*100)}% and compacting agent memories...")
                    self.cost_tracker.extend_budget(pct)
                    for agent in [self.developer, self.cto, self.product_manager, self.researcher, self.qa_engineer]:
                        if hasattr(agent, 'memory'):
                            agent.memory.compact()
                    # Emergency PM rescope on first hit
                    if budget_extensions == 0 and self.enable_escalation and hasattr(self, 'product_manager'):
                        console.info("Emergency PM rescope to reduce complexity...")
                        rescope_task = {
                            "type": "define_requirements",
                            "problem": self.state.current_problem.to_dict() if self.state.current_problem else {},
                            "target_users": getattr(self.state.current_problem, 'target_users', ''),
                            "constraint": "Budget nearly exhausted. Reduce to absolute minimum viable scope."
                        }
                        rescope_result = await self.product_manager.execute_task(rescope_task)
                        self.state.artifacts["requirements"] = rescope_result.output

                # Check time-based soft stop — allow up to 3 extensions before partial delivery
                if self.max_workflow_minutes > 0 and self.state.started_at:
                    elapsed_minutes = (datetime.now() - self.state.started_at).total_seconds() / 60
                    if elapsed_minutes >= self.max_workflow_minutes:
                        time_extensions = getattr(self, '_time_extensions', 0)
                        self._time_extensions = time_extensions + 1
                        pct = max(0.2, 0.5 - (time_extensions * 0.1))  # Decreasing but never zero
                        extended = int(self.max_workflow_minutes * pct)
                        self.max_workflow_minutes += extended
                        console.warning(f"Time limit reached (extension {time_extensions + 1}). Extending by {extended} minutes. Compacting memories...")
                        for agent in [self.developer, self.cto, self.product_manager, self.researcher, self.qa_engineer]:
                            if hasattr(agent, 'memory'):
                                agent.memory.compact()

                round_label = f"{approval_round}/{max_rounds}"

                # Inject post-mortem context when starting a new escalation cycle
                if approval_round > 1 and (approval_round - 1) % 11 == 0:
                    post_mortem = self.escalation_manager.generate_post_mortem(
                        workflow_state={"phase": self.state.phase.value},
                        failure_history=failure_history,
                    )
                    pm_context = (
                        f"POST-MORTEM AFTER {post_mortem['total_rounds']} ROUNDS:\n"
                        f"Pattern: {post_mortem['failure_pattern']}\n"
                        f"Recommendations: {'; '.join(post_mortem['recommendations'][:3])}\n"
                    )
                    self.state.artifacts["post_mortem_context"] = pm_context
                    console.info(f"Post-mortem injected for cycle-back (round {approval_round})")

                console.section(f"APPROVAL ROUND {round_label}")

                # === PHASE 5b: CODE EXECUTION CHECK ===
                if self.run_code_execution:
                    self.progress.start_phase("code_execution")
                    console.section(f"PHASE 5b: CODE EXECUTION CHECK (Round {round_label})")
                    console.info("Running generated code to check for errors...")
                    execution_results = await self._run_code_execution()
                    self.state.artifacts["execution_results"] = execution_results

                    if not execution_results["success"]:
                        console.error(f"Code execution failed: {execution_results['summary']}")
                        project_dir = self.state.artifacts.get("project_name", self.output_dir)
                        exec_fix_task = {
                            "type": "fix_bug",
                            "bug_description": f"Code execution failed with errors. Fix these runtime errors:\n{execution_results['summary']}",
                            "error_message": "\n".join(execution_results["runtime_errors"] + execution_results["syntax_errors"]),
                            "qa_report": execution_results.get("test_output", ""),
                            "original_requirements": str(self.state.artifacts.get("requirements", "")),
                            "output_dir": project_dir
                        }
                        fix_result = await self.developer.execute_task(exec_fix_task)
                        self.state.artifacts["execution_fix"] = fix_result.output
                        dev_fix_count += 1
                        console.success("Developer applied execution fixes")

                        execution_results = await self._run_code_execution()
                        self.state.artifacts["execution_results"] = execution_results
                        if execution_results["success"]:
                            console.success("Code now passes execution checks")
                        else:
                            console.warning(f"Code still has execution errors: {execution_results['summary']}")
                    else:
                        console.success(f"Code execution check passed: {execution_results['summary']}")
                    self.progress.complete_phase("code_execution")
                else:
                    console.info("Code execution check skipped (disabled in config)")
                    self.progress.skip_phase("code_execution")

                self._save_checkpoint()

                # Phase 5.5: CTO peer review before QA on round >= 3
                if approval_round >= 3:
                    console.section(f"PHASE 5.5: CTO PEER REVIEW (Round {round_label})")
                    console.info("CTO reviewing implementation before QA (escalation catch)...")
                    impl_files = self.state.artifacts.get("implementation", {}).get("files_written", [])
                    impl_code = self.state.artifacts.get("implementation", {}).get("code", "")
                    cto_review_task = {
                        "type": "review_code",
                        "code": impl_code[:3000] if impl_code else "(see files: " + ", ".join(impl_files[:5]) + ")",
                        "file_path": ", ".join(impl_files[:5]) if impl_files else "unknown",
                        "context": (
                            f"Round {approval_round} peer review before QA. "
                            f"Problem: {self.state.current_problem.description if self.state.current_problem else 'unknown'}. "
                            f"Architecture: {self.state.artifacts.get('technical_design', {}).get('architecture', '')[:300]}"
                        ),
                    }
                    try:
                        cto_review_result = await self.cto.execute_task(cto_review_task)
                        cto_peer_review = cto_review_result.output.get("review", cto_review_result.output.get("response", str(cto_review_result.output)))
                        self.state.artifacts["cto_peer_review"] = cto_peer_review
                        console.info(f"CTO peer review complete. Issues noted: {str(cto_peer_review)[:200]}")
                        # Notify Developer of CTO findings
                        self.message_bus.notify_issue(
                            from_agent="CTO",
                            to_agent="Developer",
                            issue=f"CTO peer-review (round {approval_round}): {str(cto_peer_review)[:500]}"
                        )
                    except Exception as _e:
                        console.warning(f"CTO peer review failed: {_e}")
                        self.state.artifacts["cto_peer_review"] = ""

                # Phase 6: QA Validation with inner feedback loop
                qa_passed = False
                self.progress.start_phase("qa_validation")
                for qa_attempt in range(1, max_qa_retries + 1):
                    console.section(f"PHASE 6: QA VALIDATION (Round {round_label}, Attempt {qa_attempt}/{max_qa_retries})")
                    console.info("QA Engineer validating the solution...")
                    qa_passed = await self._run_qa_phase()

                    if qa_passed:
                        break

                    if qa_attempt < max_qa_retries:
                        console.warning(f"QA failed on attempt {qa_attempt}. Sending feedback to Developer...")
                        qa_feedback = self.state.artifacts.get("qa_validation", {})
                        qa_report = self.state.artifacts.get("qa_report", {})

                        console.section(f"PHASE 6a: DEVELOPER FIX (QA Iteration {qa_attempt})")
                        console.info("Developer fixing issues found by QA...")
                        project_dir = self.state.artifacts.get("project_name", self.output_dir)
                        execution_results = self.state.artifacts.get("execution_results", {})
                        # Parse QA result for root cause feedback
                        qa_root_cause_note = ""
                        if isinstance(qa_feedback, dict):
                            validation_text = qa_feedback.get("validation", "")
                            from utils.output_parser import StructuredOutputParser
                            qa_parser = StructuredOutputParser()
                            qa_parsed = qa_parser.parse_qa_result(str(validation_text))
                            if qa_parsed.get("root_cause_addressed") is False:
                                qa_root_cause_note = (
                                    f"\nROOT CAUSE NOT ADDRESSED: {qa_parsed.get('root_cause_explanation', 'QA found solution treats symptoms, not root cause.')}"
                                    f"\nStated root cause: {self.state.artifacts.get('root_cause', 'unknown')}"
                                )

                        fix_task = {
                            "type": "fix_bug",
                            "bug_description": f"QA validation failed. Issues: {str(qa_feedback)}{qa_root_cause_note}",
                            "error_message": execution_results.get("summary", ""),
                            "qa_report": str(qa_report),
                            "original_requirements": str(self.state.artifacts.get("requirements", "")),
                            "output_dir": project_dir
                        }
                        fix_result = await self.developer.execute_task(fix_task)
                        self.state.artifacts["fix_iteration"] = fix_result.output
                        dev_fix_count += 1
                        console.success(f"Developer applied fixes (QA iteration {qa_attempt})")

                        if self.run_code_execution:
                            execution_results = await self._run_code_execution()
                            self.state.artifacts["execution_results"] = execution_results

                if qa_passed:
                    self.progress.complete_phase("qa_validation")
                    if self._current_sprint_id:
                        try:
                            self.sprint_manager.complete_task(self._current_sprint_id, "qa")
                        except ValueError:
                            pass
                else:
                    self.progress.fail_phase("qa_validation", "QA validation failed")

                # Phase 6b: Security Review (runs alongside QA results, both read-only)
                security_passed = True
                if self.enable_security_review and qa_passed:
                    self.progress.start_phase("security_review")
                    console.section(f"PHASE 6b: SECURITY REVIEW (Round {round_label})")
                    console.info("Security Engineer reviewing the solution...")
                    security_passed = await self._run_security_review_phase()
                    if security_passed:
                        self.progress.complete_phase("security_review")
                    else:
                        self.progress.fail_phase("security_review", "Security issues found")
                elif not self.enable_security_review:
                    self.progress.skip_phase("security_review")

                if not qa_passed or not security_passed:
                    # Collect failure feedback for escalation
                    qa_feedback_str = str(self.state.artifacts.get("qa_validation", ""))
                    if not security_passed:
                        qa_feedback_str += " | " + str(self.state.artifacts.get("security_review", ""))
                    failure_history.append(qa_feedback_str)
                    # QA notifies Developer of issues via message bus
                    self.message_bus.notify_issue(
                        "QAEngineer", "Developer",
                        f"Validation failed (round {approval_round}): {qa_feedback_str[:400]}",
                        "high"
                    )
                    # Incident response meeting on persistent failures (round >= 5)
                    if approval_round >= 5:
                        incident_prompt = self.incident_manager.generate_incident_prompt(
                            failure_description=qa_feedback_str[:300],
                            failure_history=failure_history,
                            round_number=approval_round
                        )
                        console.section(f"INCIDENT RESPONSE (Round {approval_round})")
                        console.warning(incident_prompt[:600])
                        self.state.artifacts["incident_analysis"] = incident_prompt

                    # OrgChart-aware failure routing
                    failing_agent = self._detect_failure_owner(qa_feedback_str)
                    supervisor = self.org_chart.get_supervisor(failing_agent)
                    if supervisor:
                        console.info(f"Escalation: {failing_agent}'s area flagged -> supervisor {supervisor} notified")
                        # Actually notify supervisor via message bus (OrgChart routing)
                        severity = "high" if approval_round >= 3 else "medium"
                        self.message_bus.notify_issue(
                            failing_agent, supervisor,
                            f"Round {approval_round} quality failure in {failing_agent}'s area: {qa_feedback_str[:300]}",
                            severity=severity
                        )

                    # Log trust score for the failing agent
                    trust = self.trust_tracker.get_trust("qa_engineer", failing_agent)
                    if trust < 0.3:
                        console.warning(f"Low trust in {failing_agent} ({trust:.2f}). Suggesting earlier escalation.")

                    if self.enable_escalation:
                        # Use EscalationManager to decide next action
                        qa_verdict = "fail" if not qa_passed else "security_fail"
                        action = self.escalation_manager.should_escalate(
                            round=approval_round,
                            qa_verdict=qa_verdict,
                            dev_fix_count=dev_fix_count,
                            failure_history=failure_history
                        )
                        console.info(f"Escalation decision: {action.value}")

                        if action == EscalationAction.DEVELOPER_FIX:
                            console.warning(f"QA/Security failed in round {approval_round}. Developer reworking solution...")
                            project_dir = self.state.artifacts.get("project_name", self.output_dir)
                            qa_feedback = self.state.artifacts.get("qa_validation", {})
                            qa_report = self.state.artifacts.get("qa_report", {})
                            execution_results = self.state.artifacts.get("execution_results", {})
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            incident_context = self.state.artifacts.get("incident_analysis", "")
                            dev_messages = self._drain_messages_for_agent("Developer")
                            rework_task = {
                                "type": "fix_bug",
                                "bug_description": f"QA/Security validation failed. Issues: {str(qa_feedback)}",
                                "error_message": execution_results.get("summary", ""),
                                "qa_report": str(qa_report),
                                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                "output_dir": project_dir,
                                "strategy_context": f"{strategy_context}\n{incident_context}\n{dev_messages}"[:1200] if (incident_context or dev_messages) else strategy_context
                            }
                            rework_result = await self.developer.execute_task(rework_task)
                            self.state.artifacts[f"rework_round_{approval_round}"] = rework_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            dev_fix_count += 1
                            console.success(f"Developer reworked solution (round {approval_round})")
                            continue

                        elif action == EscalationAction.CTO_REDESIGN:
                            console.warning("Escalating to CTO for architecture redesign...")
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            needs_deeper = self.state.artifacts.get("needs_deeper_root_cause", False)
                            deeper_suffix = "\nWARNING: Previous root cause analysis was too shallow. Apply 5-WHY analysis: ask WHY 5 times before stating the root cause." if needs_deeper else ""
                            redesign_task = {
                                "type": "design_architecture",
                                "problem": self.state.current_problem.to_dict(),
                                "requirements": str(self.state.artifacts.get("requirements", "")),
                                "constraint": f"Previous design failed QA {approval_round} times. Failure history: {failure_history[-1]}. Redesign the technical approach from scratch.\n{strategy_context}{deeper_suffix}"
                            }
                            redesign_result = await self.cto.execute_task(redesign_task)
                            self.state.artifacts["architecture"] = redesign_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            console.success("CTO redesigned architecture")
                            # Re-implement with new architecture
                            console.info("Developer re-implementing with new architecture...")
                            await self._run_implementation_phase()
                            console.success("Re-implementation complete")
                            continue

                        elif action == EscalationAction.PM_RESCOPE:
                            console.warning("Escalating to PM for requirements rescoping...")
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            rescope_task = {
                                "type": "define_requirements",
                                "problem": self.state.current_problem.to_dict(),
                                "target_users": self.state.current_problem.target_users,
                                "constraint": f"Previous implementation failed after {approval_round} rounds. Reduce scope to the minimum viable subset. Drop non-essential features.\n{strategy_context}"
                            }
                            rescope_result = await self.product_manager.execute_task(rescope_task)
                            self.state.artifacts["requirements"] = rescope_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            console.success("PM rescoped requirements")
                            # Redesign and re-implement with reduced scope
                            await self._run_technical_design_phase()
                            await self._run_implementation_phase()
                            console.success("Re-implementation with reduced scope complete")
                            continue

                        elif action == EscalationAction.TEAM_BRAINSTORM:
                            console.warning("Escalating to team brainstorm — all agents contribute solutions...")
                            brainstorm_result = await self._team_brainstorm()
                            if brainstorm_result:
                                self.state.artifacts["brainstorm_solutions"] = brainstorm_result
                                await self._run_implementation_phase()
                                console.success("Re-implementation after brainstorm complete")
                            continue

                        elif action == EscalationAction.DECOMPOSE_PROBLEM:
                            console.warning("Decomposing problem into sub-problems...")
                            sub_problems = await self._decompose_problem()
                            if sub_problems:
                                self.state.artifacts["sub_problems"] = sub_problems
                                await self._run_implementation_phase()
                                console.success("Re-implementation after decomposition complete")
                            continue

                        elif action == EscalationAction.PIVOT_APPROACH:
                            console.warning("Pivoting to a completely different approach...")
                            await self._pivot_approach()
                            await self._run_implementation_phase()
                            console.success("Re-implementation after pivot complete")
                            continue

                        elif action == EscalationAction.SIMPLIFY:
                            console.warning("Simplifying to absolute minimum viable solution...")
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            simplify_task = {
                                "type": "define_requirements",
                                "problem": self.state.current_problem.to_dict(),
                                "target_users": self.state.current_problem.target_users,
                                "constraint": f"SIMPLIFY: Reduce to the absolute minimum. 1 file, 1 function, core feature ONLY. Cut everything else.\n{strategy_context}"
                            }
                            simplify_result = await self.product_manager.execute_task(simplify_task)
                            self.state.artifacts["requirements"] = simplify_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            await self._run_technical_design_phase()
                            await self._run_implementation_phase()
                            console.success("Re-implementation with simplified scope complete")
                            continue

                        elif action == EscalationAction.ALTERNATIVE_STACK:
                            console.warning("Trying alternative technology stack...")
                            strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                            alt_stack_task = {
                                "type": "design_architecture",
                                "problem": self.state.current_problem.to_dict(),
                                "requirements": str(self.state.artifacts.get("requirements", "")),
                                "constraint": f"ALTERNATIVE STACK: The current tech stack has failed repeatedly. Try a completely different language/framework.\n{strategy_context}"
                            }
                            alt_stack_result = await self.cto.execute_task(alt_stack_task)
                            self.state.artifacts["architecture"] = alt_stack_result.output
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                            )
                            await self._run_implementation_phase()
                            console.success("Re-implementation with alternative stack complete")
                            continue

                        elif action == EscalationAction.FRESH_START:
                            console.warning("FRESH START: Discarding all previous work and re-analyzing from scratch...")
                            self.escalation_manager.strategy_memory.record_attempt(
                                approach=action.value, round=approval_round,
                                outcome="attempted", feedback="Fresh start triggered"
                            )
                            # Re-run analysis from problem understanding
                            await self._run_analysis_phase()
                            await self._run_technical_design_phase()
                            await self._run_implementation_phase()
                            console.success("Re-implementation after fresh start complete")
                            continue

                    else:
                        # Escalation disabled: use legacy max_approval_rounds behavior
                        has_limit = max_approval_rounds > 0
                        at_limit = has_limit and approval_round >= max_approval_rounds
                        if not at_limit:
                            console.warning(f"QA/Security failed in round {approval_round}. Developer reworking solution...")
                            project_dir = self.state.artifacts.get("project_name", self.output_dir)
                            qa_feedback = self.state.artifacts.get("qa_validation", {})
                            qa_report = self.state.artifacts.get("qa_report", {})
                            execution_results = self.state.artifacts.get("execution_results", {})
                            rework_task = {
                                "type": "fix_bug",
                                "bug_description": f"Validation failed after {max_qa_retries} attempts. Issues: {str(qa_feedback)}",
                                "error_message": execution_results.get("summary", ""),
                                "qa_report": str(qa_report),
                                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                "output_dir": project_dir
                            }
                            rework_result = await self.developer.execute_task(rework_task)
                            self.state.artifacts[f"rework_round_{approval_round}"] = rework_result.output
                            dev_fix_count += 1
                            console.success(f"Developer reworked solution (round {approval_round})")
                            continue
                        else:
                            if getattr(self, 'force_stop', False):
                                console.warning(f"Validation failed after {approval_round} rounds. Force-stop enabled — delivering best available solution.")
                                self.state.phase = WorkflowPhase.COMPLETED
                                self.state.artifacts["partial_delivery"] = True
                                self.state.artifacts["delivery_note"] = f"Solution delivered after {approval_round} rounds. QA issues remain — manual review recommended."
                                break
                            else:
                                # Never stop: rework and continue
                                console.warning(f"QA/Security failed in round {approval_round} (at previous limit). Reworking — never giving up...")
                                project_dir = self.state.artifacts.get("project_name", self.output_dir)
                                qa_feedback = self.state.artifacts.get("qa_validation", {})
                                qa_report = self.state.artifacts.get("qa_report", {})
                                execution_results = self.state.artifacts.get("execution_results", {})
                                rework_task = {
                                    "type": "fix_bug",
                                    "bug_description": f"Validation failed after {approval_round} rounds. Issues: {str(qa_feedback)}",
                                    "error_message": execution_results.get("summary", ""),
                                    "qa_report": str(qa_report),
                                    "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                    "output_dir": project_dir
                                }
                                rework_result = await self.developer.execute_task(rework_task)
                                self.state.artifacts[f"rework_round_{approval_round}"] = rework_result.output
                                dev_fix_count += 1
                                console.success(f"Developer reworked solution (round {approval_round})")
                                continue

                # Publish QA verdict via message bus
                qa_output = self.state.artifacts.get("qa_validation", {})
                verdict = qa_output.get("verdict", "unknown") if isinstance(qa_output, dict) else "unknown"
                self.message_bus.publish(
                    topic="decisions",
                    sender="QAEngineer",
                    content=f"QA verdict: {verdict.upper()}",
                    priority=4
                )

                self._save_checkpoint()

                # Phase 6c: DevOps Validation (after QA+Security pass, before CEO)
                devops_passed = True
                if qa_passed and security_passed:
                    console.section(f"PHASE 6c: DEVOPS VALIDATION (Round {round_label})")
                    console.info("DevOps Engineer validating deployability...")
                    try:
                        devops_task = {
                            "type": "validate_deployment",
                            "description": f"Validate deployment readiness for: {self.state.current_problem.description if self.state.current_problem else 'solution'}",
                            "project_dir": self.state.artifacts.get("project_name", self.output_dir),
                            "files": list(self.state.artifacts.get("generated_files", {}).keys()) if isinstance(self.state.artifacts.get("generated_files"), dict) else [],
                            "requirements": str(self.state.artifacts.get("requirements", "")),
                        }
                        devops_result = await self.devops_engineer.execute_task(devops_task)
                        self.state.artifacts["devops_validation"] = devops_result.output
                        # Track DevOpsEngineer experience and pool performance
                        if hasattr(self.devops_engineer, '_experience'):
                            self.devops_engineer._experience.add_experience("validate_deployment", devops_result.success)
                        self.agent_pool.record_task_outcome("DevOpsEngineer", devops_result.success)
                        self.performance_tracker.record_task(
                            "DevOpsEngineer", success=devops_result.success,
                            response_time_ms=devops_result.execution_time * 1000, tokens=0
                        )
                        # Reflect on DevOps validation (learning)
                        if self.enable_learning:
                            self.devops_engineer.reflect(
                                "validate_deployment",
                                str(devops_result.output),
                                devops_result.success
                            )
                        devops_verdict = str(devops_result.output).upper() if isinstance(devops_result.output, str) else str(devops_result.output.get("verdict", "")).upper() if isinstance(devops_result.output, dict) else ""
                        if "NOT_DEPLOYABLE" in devops_verdict:
                            devops_passed = False
                            console.warning("DevOps: NOT_DEPLOYABLE — Developer must fix before CEO review")
                            # DevOps notifies Developer of deployment issues via message bus
                            self.message_bus.notify_issue(
                                "DevOpsEngineer", "Developer",
                                f"NOT_DEPLOYABLE: {str(devops_result.output)[:400]}",
                                "high"
                            )
                            # Send back to developer for deployment fixes
                            deploy_fix_task = {
                                "type": "fix_bug",
                                "bug_description": f"DevOps validation failed: {str(devops_result.output)}",
                                "error_message": "Fix deployment issues: missing files, unpinned deps, or broken entry point",
                                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                "output_dir": self.state.artifacts.get("project_name", self.output_dir),
                            }
                            deploy_fix = await self.developer.execute_task(deploy_fix_task)
                            self.state.artifacts["devops_fix"] = deploy_fix.output
                            dev_fix_count += 1
                            console.success("Developer applied DevOps fixes")
                            continue  # Re-run QA loop
                        elif "NEEDS_FIXES" in devops_verdict:
                            console.warning("DevOps: NEEDS_FIXES — proceeding with warnings")
                            # DevOps notifies Developer of deployment warnings via message bus
                            self.message_bus.notify_issue(
                                "DevOpsEngineer", "Developer",
                                f"NEEDS_FIXES (non-blocking): {str(devops_result.output)[:300]}",
                                "medium"
                            )
                        else:
                            console.success("DevOps: DEPLOYABLE")
                    except Exception as devops_err:
                        console.warning(f"DevOps validation error (non-blocking): {devops_err}")

                # Phase 7: CEO Final Approval
                self.progress.start_phase("ceo_approval")
                console.section(f"PHASE 7: CEO APPROVAL (Round {round_label})")
                console.info("Requesting CEO final approval...")
                if await self._run_ceo_approval():
                    self.state.phase = WorkflowPhase.COMPLETED
                    console.success("Solution approved by CEO!")
                    self.progress.complete_phase("ceo_approval")
                    if self._current_sprint_id:
                        try:
                            self.sprint_manager.complete_task(self._current_sprint_id, "approval")
                        except ValueError:
                            pass

                    # Store solution in shared memory
                    problem = self.state.current_problem
                    self.shared_memory.store_solution(
                        problem_id=problem.id,
                        solution_description=f"Solution approved: {problem.description}",
                        implementation_path=str(self.state.artifacts.get("project_name", "")),
                        created_by="developer"
                    )

                    self._save_checkpoint()

                    # Delivery phase (skip in dry-run mode)
                    if dry_run:
                        console.info("Dry-run mode: skipping delivery phase")
                        self.progress.skip_phase("delivery")
                    else:
                        self.progress.start_phase("delivery")
                        console.section("PHASE 8: DELIVERY")
                        console.info("Packaging solution for delivery...")
                        delivery = await self._run_delivery_phase()
                        console.success(f"Delivered: {len(delivery.get('files', []))} files in {delivery['project_dir']}")
                        self.progress.complete_phase("delivery")
                        self._save_checkpoint()
                    break

                self.progress.fail_phase("ceo_approval", "Rejected by CEO")

                # CEO rejected - collect feedback for escalation
                rejection = self.state.artifacts.get("final_approval", {})
                rejection_reasoning = rejection.get("reasoning", str(rejection)) if isinstance(rejection, dict) else str(rejection)
                failure_history.append(f"CEO rejection: {rejection_reasoning}")
                # CEO notifies Developer of rejection via message bus
                self.message_bus.notify_issue(
                    "CEO", "Developer",
                    f"CEO rejected (round {approval_round}): {rejection_reasoning[:400]}",
                    "high"
                )

                if self.enable_escalation:
                    action = self.escalation_manager.should_escalate(
                        round=approval_round,
                        qa_verdict="ceo_rejected",
                        dev_fix_count=dev_fix_count,
                        failure_history=failure_history
                    )
                    console.info(f"Escalation decision: {action.value}")

                    if action == EscalationAction.DEVELOPER_FIX:
                        console.warning(f"CEO rejected in round {approval_round}. Developer addressing feedback...")
                        project_dir = self.state.artifacts.get("project_name", self.output_dir)
                        execution_results = self.state.artifacts.get("execution_results", {})
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        dev_messages = self._drain_messages_for_agent("Developer")
                        ceo_fix_task = {
                            "type": "fix_bug",
                            "bug_description": f"CEO rejected the solution. Feedback: {str(rejection_reasoning)}",
                            "error_message": execution_results.get("summary", ""),
                            "qa_report": str(self.state.artifacts.get("qa_report", "")),
                            "original_requirements": str(self.state.artifacts.get("requirements", "")),
                            "output_dir": project_dir,
                            "strategy_context": f"{strategy_context}\n{dev_messages}"[:1000] if dev_messages else strategy_context
                        }
                        ceo_fix_result = await self.developer.execute_task(ceo_fix_task)
                        self.state.artifacts[f"ceo_fix_round_{approval_round}"] = ceo_fix_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        dev_fix_count += 1
                        console.success(f"Developer applied CEO feedback fixes (round {approval_round})")
                        continue
                    elif action == EscalationAction.CTO_REDESIGN:
                        console.warning("CEO rejected. Escalating to CTO for architecture redesign...")
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        redesign_task = {
                            "type": "design_architecture",
                            "problem": self.state.current_problem.to_dict(),
                            "requirements": str(self.state.artifacts.get("requirements", "")),
                            "constraint": f"CEO rejected after {approval_round} rounds. Feedback: {rejection_reasoning}. Redesign approach.\n{strategy_context}"
                        }
                        redesign_result = await self.cto.execute_task(redesign_task)
                        self.state.artifacts["architecture"] = redesign_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        await self._run_implementation_phase()
                        console.success("Re-implementation after CTO redesign complete")
                        continue
                    elif action == EscalationAction.PM_RESCOPE:
                        console.warning("CEO rejected. Escalating to PM for scope reduction...")
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        rescope_task = {
                            "type": "define_requirements",
                            "problem": self.state.current_problem.to_dict(),
                            "target_users": self.state.current_problem.target_users,
                            "constraint": f"CEO rejected after {approval_round} rounds. Reduce scope dramatically.\n{strategy_context}"
                        }
                        rescope_result = await self.product_manager.execute_task(rescope_task)
                        self.state.artifacts["requirements"] = rescope_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        await self._run_technical_design_phase()
                        await self._run_implementation_phase()
                        console.success("Re-implementation with reduced scope complete")
                        continue
                    elif action == EscalationAction.TEAM_BRAINSTORM:
                        console.warning("CEO rejected. Escalating to team brainstorm...")
                        brainstorm_result = await self._team_brainstorm()
                        if brainstorm_result:
                            self.state.artifacts["brainstorm_solutions"] = brainstorm_result
                            await self._run_implementation_phase()
                            console.success("Re-implementation after brainstorm complete")
                        continue
                    elif action == EscalationAction.DECOMPOSE_PROBLEM:
                        console.warning("CEO rejected. Decomposing problem into sub-problems...")
                        sub_problems = await self._decompose_problem()
                        if sub_problems:
                            self.state.artifacts["sub_problems"] = sub_problems
                            await self._run_implementation_phase()
                            console.success("Re-implementation after decomposition complete")
                        continue
                    elif action == EscalationAction.PIVOT_APPROACH:
                        console.warning("CEO rejected. Pivoting to a different approach...")
                        await self._pivot_approach()
                        await self._run_implementation_phase()
                        console.success("Re-implementation after pivot complete")
                        continue
                    elif action == EscalationAction.SIMPLIFY:
                        console.warning("CEO rejected. Simplifying to absolute minimum...")
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        simplify_task = {
                            "type": "define_requirements",
                            "problem": self.state.current_problem.to_dict(),
                            "target_users": self.state.current_problem.target_users,
                            "constraint": f"CEO rejected. SIMPLIFY: 1 file, 1 function, core feature ONLY.\n{strategy_context}"
                        }
                        simplify_result = await self.product_manager.execute_task(simplify_task)
                        self.state.artifacts["requirements"] = simplify_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        await self._run_technical_design_phase()
                        await self._run_implementation_phase()
                        console.success("Re-implementation with simplified scope complete")
                        continue
                    elif action == EscalationAction.ALTERNATIVE_STACK:
                        console.warning("CEO rejected. Trying alternative technology stack...")
                        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
                        alt_stack_task = {
                            "type": "design_architecture",
                            "problem": self.state.current_problem.to_dict(),
                            "requirements": str(self.state.artifacts.get("requirements", "")),
                            "constraint": f"CEO rejected. ALTERNATIVE STACK: Try a completely different approach.\n{strategy_context}"
                        }
                        alt_stack_result = await self.cto.execute_task(alt_stack_task)
                        self.state.artifacts["architecture"] = alt_stack_result.output
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback=failure_history[-1] if failure_history else ""
                        )
                        await self._run_implementation_phase()
                        console.success("Re-implementation with alternative stack complete")
                        continue
                    elif action == EscalationAction.FRESH_START:
                        console.warning("CEO rejected. FRESH START: Re-analyzing from scratch...")
                        self.escalation_manager.strategy_memory.record_attempt(
                            approach=action.value, round=approval_round,
                            outcome="attempted", feedback="Fresh start triggered after CEO rejection"
                        )
                        await self._run_analysis_phase()
                        await self._run_technical_design_phase()
                        await self._run_implementation_phase()
                        console.success("Re-implementation after fresh start complete")
                        continue
                else:
                    # Escalation disabled: use legacy behavior
                    has_limit = max_approval_rounds > 0
                    at_limit = has_limit and approval_round >= max_approval_rounds
                    if not at_limit:
                        console.warning(f"CEO rejected in round {approval_round}. Sending feedback to Developer...")
                        project_dir = self.state.artifacts.get("project_name", self.output_dir)
                        execution_results = self.state.artifacts.get("execution_results", {})
                        ceo_fix_task = {
                            "type": "fix_bug",
                            "bug_description": f"CEO rejected the solution. Feedback: {str(rejection_reasoning)}",
                            "error_message": execution_results.get("summary", ""),
                            "qa_report": str(self.state.artifacts.get("qa_report", "")),
                            "original_requirements": str(self.state.artifacts.get("requirements", "")),
                            "output_dir": project_dir
                        }
                        ceo_fix_result = await self.developer.execute_task(ceo_fix_task)
                        self.state.artifacts[f"ceo_fix_round_{approval_round}"] = ceo_fix_result.output
                        dev_fix_count += 1
                        console.success(f"Developer applied CEO feedback fixes (round {approval_round})")
                    else:
                        if getattr(self, 'force_stop', False):
                            console.warning(f"CEO rejected after {approval_round} rounds. Force-stop enabled — delivering best available solution.")
                            self.state.phase = WorkflowPhase.COMPLETED
                            self.state.artifacts["partial_delivery"] = True
                            self.state.artifacts["delivery_note"] = f"Solution delivered after {approval_round} CEO rejections. Last feedback: {rejection_reasoning}"
                            break
                        else:
                            # Never stop: rework and continue
                            console.warning(f"CEO rejected in round {approval_round} (at previous limit). Reworking — never giving up...")
                            project_dir = self.state.artifacts.get("project_name", self.output_dir)
                            execution_results = self.state.artifacts.get("execution_results", {})
                            ceo_fix_task = {
                                "type": "fix_bug",
                                "bug_description": f"CEO rejected the solution. Feedback: {str(rejection_reasoning)}",
                                "error_message": execution_results.get("summary", ""),
                                "qa_report": str(self.state.artifacts.get("qa_report", "")),
                                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                                "output_dir": project_dir
                            }
                            ceo_fix_result = await self.developer.execute_task(ceo_fix_task)
                            self.state.artifacts[f"ceo_fix_round_{approval_round}"] = ceo_fix_result.output
                            dev_fix_count += 1
                            console.success(f"Developer applied CEO feedback fixes (round {approval_round})")
                            continue

            self.state.completed_at = datetime.now()

            # Phase 9: Retrospective (runs after delivery or failure)
            if self.enable_retrospective:
                self.progress.start_phase("retrospective")
                console.section("PHASE 9: RETROSPECTIVE")
                console.info("All agents reflecting on the workflow run...")
                await self._run_retrospective_phase()
                self.progress.complete_phase("retrospective")
            else:
                self.progress.skip_phase("retrospective")

            if self.state.phase == WorkflowPhase.COMPLETED:
                self._clear_checkpoint()

        except KeyboardInterrupt:
            self.state.phase = WorkflowPhase.FAILED
            self.state.error = "Interrupted by user (Ctrl+C)"
            self.state.completed_at = datetime.now()
            console.warning("\nWorkflow interrupted by user (Ctrl+C)")
            self._save_checkpoint()

        except Exception as e:
            import traceback
            self.state.error = str(e)
            self.state.completed_at = datetime.now()
            console.error(f"Workflow error: {str(e)}")
            console.error(traceback.format_exc())
            self._save_checkpoint()

            # Part 2c: Persistence loop — retry from research on unexpected errors
            pivot_count = getattr(self, '_pivot_count', 0)
            max_pivots = 3
            if pivot_count < max_pivots and self.state.phase != WorkflowPhase.COMPLETED:
                self._pivot_count = pivot_count + 1
                console.warning(f"Workflow error on attempt {pivot_count + 1}/{max_pivots} — resetting to research phase")
                self.state.phase = WorkflowPhase.RESEARCH
                self.state.error = None
                self.state.completed_at = None
                # Recursive retry — re-run the entire workflow
                try:
                    return await self.run_full_workflow(
                        auto_discover=True,
                        problem=None,
                        max_approval_rounds=max_approval_rounds,
                        resume=False,
                        interactive=interactive,
                        dry_run=dry_run,
                    )
                except Exception as retry_err:
                    console.error(f"Retry also failed: {retry_err}")

            self.state.phase = WorkflowPhase.FAILED
            # Trigger error hook
            try:
                from utils.hooks import HookContext, HookEvent
                await self.hooks_manager.execute(HookContext(
                    event=HookEvent.ERROR,
                    metadata={"error": str(e)}
                ))
            except Exception as hook_err:
                console.warning(f"Error hook failed: {hook_err}")

        finally:
            # Trigger session end hook
            try:
                session_id = self.session_manager.current_session.id if self.session_manager.current_session else ""
                await self.hooks_manager.session_end(session_id)
            except Exception as hook_err:
                console.warning(f"Session end hook failed: {hook_err}")

            # Cleanup: always stop background services and persist results
            self.health_checker.stop_background_checks()
            self.memory_monitor.stop_monitoring()

            # End session with appropriate state
            if self.state.phase == WorkflowPhase.COMPLETED:
                self.session_manager.end_session(SessionState.COMPLETED)
            elif self.state.phase == WorkflowPhase.FAILED:
                self.session_manager.end_session(SessionState.FAILED)
            else:
                self.session_manager.end_session(SessionState.PAUSED)

            self.structured_logger.info(f"Workflow ended with status: {self.state.phase.value}")
            self.state.artifacts["task_stats"] = self.task_manager.get_task_stats()

            cost_session = self.cost_tracker.end_session()
            if cost_session:
                self.state.artifacts["cost_summary"] = cost_session.to_dict()

            result = self._get_workflow_result()
            try:
                self._persist_result(result)
            except Exception as persist_err:
                console.warning(f"Could not persist workflow result: {persist_err}")

            if self.state.phase == WorkflowPhase.COMPLETED:
                self._print_completion_summary(result)

            return result

    # ============================================================
    #  QUALITY GATES
    # ============================================================

    async def _check_quality_gate(self, phase: str) -> bool:
        """Check quality gate for a phase. Returns True if passed, False if failed.

        On failure, retries up to 2 times, then proceeds with a degradation flag.
        """
        max_retries = 2

        for attempt in range(max_retries + 1):
            passed, reason = self._evaluate_gate(phase)
            if passed:
                return True

            if attempt < max_retries:
                console.warning(f"Quality gate '{phase}' failed (attempt {attempt + 1}/{max_retries + 1}): {reason}")
                retry_ok = await self._retry_gate(phase, reason)
                if not retry_ok:
                    break
            else:
                console.warning(f"Quality gate '{phase}' failed after {max_retries + 1} attempts: {reason}")

        # Proceed with degradation flag
        self.state.artifacts.setdefault("degraded_phases", []).append(phase)
        console.warning(f"Proceeding with degraded quality for phase: {phase}")
        return False

    def _evaluate_gate(self, phase: str) -> tuple:
        """Evaluate a quality gate. Returns (passed: bool, reason: str)."""
        if phase == "research":
            problems = self.state.artifacts.get("discovered_problems", [])
            user_provided = self.state.artifacts.get("user_provided_problem", False)
            min_required = 1 if user_provided else 2
            if len(problems) < min_required:
                return False, f"Only {len(problems)} problems found (need >= {min_required})"
            max_score = max((p.get("score", 0) if isinstance(p, dict) else getattr(p, "score", 0)) for p in problems)
            if not user_provided and max_score < 0.3:
                return False, f"Max problem score {max_score:.2f} < 0.3"
            return True, ""

        elif phase == "data_analysis":
            verdicts = self.state.artifacts.get("cross_validation_verdicts", {})
            summary = verdicts.get("summary", {}) if isinstance(verdicts, dict) else {}
            confirmed = summary.get("confirmed", 0) + summary.get("partially_confirmed", 0)
            if confirmed < 1:
                return False, "No findings confirmed or partially confirmed by DataAnalyst"
            return True, ""

        elif phase == "tech_design":
            root_cause = self.state.artifacts.get("root_cause", "")
            if not root_cause or len(root_cause.strip()) < 5:
                return False, "Root cause not extracted from CTO design"
            return True, ""

        elif phase == "implementation":
            impl = self.state.artifacts.get("implementation", {})
            if isinstance(impl, dict):
                files = impl.get("files_written", [])
            else:
                files = []
            code_artifacts = self.state.artifacts.get("project", {})
            if isinstance(code_artifacts, dict):
                files = files or code_artifacts.get("files_created", [])
            if not files:
                return False, "No code files in implementation artifacts"
            return True, ""

        elif phase == "qa":
            qa_output = self.state.artifacts.get("qa_validation", {})
            if isinstance(qa_output, dict):
                verdict = qa_output.get("verdict", "uncertain")
            else:
                verdict = "uncertain"
            if verdict == "uncertain":
                return False, "QA verdict is uncertain"
            return True, ""

        return True, ""

    async def _retry_gate(self, phase: str, reason: str) -> bool:
        """Attempt to fix a quality gate failure. Returns True if retry was attempted."""
        if phase == "research":
            # Re-run with expanded keywords
            console.info("Re-running research with expanded keywords...")
            await self._discover_from_web_search()
            return True
        elif phase == "data_analysis":
            # Warn and proceed - can't easily re-run data analysis
            console.warning(f"Data analysis gate: {reason} — proceeding with caution")
            return False
        elif phase == "tech_design":
            # CTO retry
            console.info("Re-running CTO design to extract root cause...")
            problem = self.state.current_problem
            if problem:
                retry_task = {
                    "type": "design_architecture",
                    "problem": problem.to_dict(),
                    "requirements": str(self.state.artifacts.get("requirements", "")),
                    "context_messages": [],
                }
                result = await self.cto.execute_task(retry_task)
                self.state.artifacts["architecture"] = result.output
                # Re-extract root cause
                from utils.output_parser import StructuredOutputParser
                parser = StructuredOutputParser()
                design_text = result.artifacts.get("architecture_doc", str(result.output))
                parsed = parser.parse_cto_design(design_text)
                rc = parsed.get("root_cause", "")
                if rc:
                    self.state.artifacts["root_cause"] = rc
                return True
            return False
        elif phase == "implementation":
            # Developer retry
            console.info("Re-running implementation...")
            return True  # The main loop handles this
        elif phase == "qa":
            # QA retry with clearer instructions
            console.info("Re-running QA with clearer instructions...")
            return True
        return False

    # ============================================================
    #  PHASE 1: RESEARCH
    # ============================================================

    async def _run_research_phase(self) -> None:
        """Run the research phase to discover problems."""
        self.state.phase = WorkflowPhase.RESEARCH

        # Get configured subreddits
        subreddits = self.research_sources.get_reddit_subreddits()
        console.agent_action("Researcher", "Scanning Sources", "Reddit + HN + StackOverflow + GitHub Issues + Web Search")

        # Run research sources sequentially (concurrent calls exhaust RAM on 16GB systems)
        problems = []

        research_tasks = [
            ("Reddit", self.problem_discoverer.discover_from_reddit(subreddits[:1], limit_per_sub=25, time_period="month")),
            ("HackerNews", self.problem_discoverer.discover_from_hacker_news(limit=30)),
            ("StackOverflow", self.problem_discoverer.discover_from_stackoverflow(limit=20)),
            ("GitHub", self.problem_discoverer.discover_from_github_issues(repos=["microsoft/vscode"], limit=20)),
            ("WebSearch", self._discover_from_web_search()),
        ]

        for source_name, coro in research_tasks:
            try:
                result = await coro
                if result:
                    problems.extend(result)
                    console.info(f"{source_name}: {len(result)} problems discovered")
                else:
                    console.warning(f"{source_name}: no problems found")
            except Exception as e:
                console.warning(f"{source_name}: error - {e}")

        # Filter out previously-solved problems
        from memory.shared_memory import MemoryType
        solved_memories = self.shared_memory.get_by_type(MemoryType.SOLUTION)
        if solved_memories and problems:
            solved_descriptions = {m.content.lower() for m in solved_memories}
            original_count = len(problems)
            filtered = []
            for p in problems:
                desc = p.description.lower() if hasattr(p, 'description') else ""
                words = set(desc.split())
                is_solved = any(
                    len(words & set(sd.split())) / max(len(words), 1) > 0.6
                    for sd in solved_descriptions
                ) if words else False
                if not is_solved:
                    filtered.append(p)
            if filtered:
                problems = filtered
                if len(filtered) < original_count:
                    console.info(f"Filtered {original_count - len(filtered)} previously-solved problems")

        # Filter problems with existing open-source solutions
        before_solution_check = len(problems)
        problems = await self.problem_discoverer.filter_problems_with_solutions(problems)
        if len(problems) < before_solution_check:
            console.info(f"Solution check: dropped {before_solution_check - len(problems)} problems with existing solutions")

        # Hard filter: only keep problems from the last 60 days
        max_age_days = 60
        cutoff = datetime.now() - timedelta(days=max_age_days)
        before_filter = len(problems)
        problems = [p for p in problems if p.discovered_at >= cutoff]
        if len(problems) < before_filter:
            console.info(f"Freshness filter: dropped {before_filter - len(problems)} problems older than {max_age_days} days")

        # Get top problem from the already-filtered list
        if problems:
            top_problem = max(problems, key=lambda p: p.score)
            self.state.current_problem = top_problem
            console.show_problem(top_problem.to_dict())

        self.state.artifacts["discovered_problems"] = [p.to_dict() for p in problems]
        console.success(f"Discovered {len(problems)} problems")

    # ============================================================
    #  PHASE 3: ANALYSIS (PM + RESEARCHER)
    # ============================================================

    async def _run_analysis_phase(self) -> None:
        """Run the analysis phase with Product Manager and Researcher."""
        self.state.phase = WorkflowPhase.ANALYSIS
        self.structured_logger.info("Analysis phase started")

        problem = self.state.current_problem
        if not problem:
            console.error("No problem to analyze")
            return

        # Researcher validates problem (tracked by TaskManager)
        console.agent_action("Researcher", "Validating Problem", problem.description)
        tm_task = self.task_manager.create_task(
            task_type="validate_problem",
            description=f"Validate problem: {problem.description}",
            assigned_to="Researcher",
            priority=TaskPriority.HIGH,
        )
        self.task_manager.start_task(tm_task.id)
        validation_task = {
            "type": "validate_problem",
            "problem": problem.to_dict(),
            "evidence": "\n".join(problem.evidence),
            "context_messages": self._drain_messages_for_agent("Researcher")
        }
        validation_result = await self.researcher.execute_task(validation_task)
        self.task_manager.complete_task(tm_task.id, {"status": "done"})
        self.state.artifacts["problem_validation"] = validation_result.output
        # Handle case where output might be a string instead of dict
        validation_status = validation_result.output.get('status', 'analyzing') if isinstance(validation_result.output, dict) else 'analyzing'
        console.agent_thinking("Researcher", f"Validation: {validation_status}...")
        # Track Researcher experience and pool performance
        if hasattr(self.researcher, '_experience'):
            self.researcher._experience.add_experience("validate_problem", validation_result.success)
        self.agent_pool.record_task_outcome("Researcher", validation_result.success)
        self.performance_tracker.record_task(
            "Researcher", success=validation_result.success,
            response_time_ms=validation_result.execution_time * 1000, tokens=0
        )
        # Researcher shares findings with DataAnalyst via message bus
        self.message_bus.share_finding(
            "Researcher", "DataAnalyst",
            f"Problem validation result: {validation_status}. Evidence summary: {str(validation_result.output)[:300]}"
        )

        # Product Manager analyzes problem (tracked by TaskManager)
        console.agent_action("ProductManager", "Analyzing Market Opportunity", "Evaluating problem severity and market fit...")
        pm_task = self.task_manager.create_task(
            task_type="analyze_problem",
            description=f"Analyze market opportunity: {problem.description}",
            assigned_to="ProductManager",
            priority=TaskPriority.HIGH,
        )
        self.task_manager.start_task(pm_task.id)
        analysis_task = {
            "type": "analyze_problem",
            "problem": problem.to_dict(),
            "research_data": str(validation_result.output)
        }
        analysis_result = await self.product_manager.execute_task(analysis_task)
        self.state.artifacts["problem_analysis"] = analysis_result.output
        self.task_manager.complete_task(pm_task.id, {"status": "done"})
        # Track ProductManager experience and pool performance
        if hasattr(self.product_manager, '_experience'):
            self.product_manager._experience.add_experience("analyze_problem", analysis_result.success)
        self.agent_pool.record_task_outcome("ProductManager", analysis_result.success)
        self.performance_tracker.record_task(
            "ProductManager", success=analysis_result.success,
            response_time_ms=analysis_result.execution_time * 1000, tokens=0
        )
        # Reflect on PM problem analysis (learning)
        if self.enable_learning:
            self.product_manager.reflect(
                "analyze_problem",
                str(analysis_result.output),
                analysis_result.success
            )

        # Reflect on validation (learning)
        if self.enable_learning:
            self.researcher.reflect(
                "validate_problem",
                str(validation_result.output),
                validation_result.success
            )

        console.info(f"Problem: {problem.description}")
        final_status = validation_result.output.get('status', 'done') if isinstance(validation_result.output, dict) else 'done'
        console.success(f"Analysis complete - Validation: {final_status}")

    async def _discover_from_web_search(self):
        """Discover problems via DuckDuckGo web search with multi-query expansion."""
        try:
            from research.web_search import WebSearch
            web_search = WebSearch()
            domain = self.research_sources.get_reddit_subreddits()[0] if self.research_sources.get_reddit_subreddits() else "software"
            year = datetime.now().year
            base_query = f"{domain} common problems {year}"

            # Use LLM-powered term expansion for broader coverage
            try:
                search_queries = await self.problem_discoverer.expand_search_terms(base_query, num_alternatives=4)
            except Exception:
                search_queries = [
                    base_query,
                    f"developer frustrations {domain} {year}",
                ]

            all_snippets = []
            seen_titles = set()
            for query in search_queries:
                results = await web_search.search_duckduckgo(query, max_results=5)
                for r in results:
                    # Deduplicate across queries
                    if r.title not in seen_titles:
                        seen_titles.add(r.title)
                        all_snippets.append(f"- {r.title}: {r.snippet}")
            if not all_snippets:
                return []
            content = "\n".join(all_snippets[:25])
            problems = await self.problem_discoverer._analyze_content_for_problems(
                content, source="web_search", domain=domain
            )
            return problems
        except Exception as e:
            console.warning(f"Web search research failed: {e}")
            return []

    # ============================================================
    #  PHASE 4: OPPORTUNITY EVALUATION (CEO)
    # ============================================================

    async def _run_opportunity_evaluation(self) -> bool:
        """CEO evaluates the opportunity with a decision meeting."""
        self.state.phase = WorkflowPhase.OPPORTUNITY_EVALUATION

        problem = self.state.current_problem
        if not problem:
            console.error("No problem to evaluate")
            return False
        analysis = self.state.artifacts.get("problem_analysis", {})

        # Hold a decision meeting with key stakeholders
        meeting_participants = [
            {
                "name": "CEO",
                "role": "Strategic Decision Maker",
                "model": self.model_config.get_model_name("ceo")
            },
            {
                "name": "CTO",
                "role": "Technical Advisor",
                "model": self.model_config.get_model_name("cto")
            },
            {
                "name": "ProductManager",
                "role": "Product Strategy",
                "model": self.model_config.get_model_name("product_manager")
            }
        ]

        # Gather research artifacts to give decision-makers full picture
        bias_flags = self.state.artifacts.get("bias_flags", [])
        bias_summary = "; ".join(
            f"{b.get('type','?')} ({b.get('severity','?')})" for b in bias_flags
        ) if bias_flags else "None detected"
        credibility_raw = self.state.artifacts.get("credibility_analysis", {})
        credibility_summary = (
            str(credibility_raw)[:300] if credibility_raw else "Not available"
        )
        counter_evidence = self.state.artifacts.get("counter_evidence", [])
        counter_summary = "; ".join(
            ce.get("description", "")[:80] for ce in counter_evidence if ce.get("counter_evidence")
        ) or "None found"
        opposing = self.state.artifacts.get("opposing_viewpoints", [])
        opposing_summary = "; ".join(
            f"{o.get('title','')}: {o.get('snippet','')[:60]}" for o in opposing[:3]
        ) or "None found"
        top_score = self.state.artifacts.get("discovered_problems", [{}])[0].get("score", "N/A") if self.state.artifacts.get("discovered_problems") else "N/A"
        freshness = self.state.artifacts.get("discovered_problems", [{}])[0].get("freshness_score", "N/A") if self.state.artifacts.get("discovered_problems") else "N/A"

        meeting_context = f"""
Problem: {problem.description}
Severity: {problem.severity}
Target Users: {problem.target_users}
Composite Score: {top_score}
Data Freshness: {freshness}
Market Analysis: {str(analysis)}
Potential Solutions: {', '.join(problem.potential_solution_ideas) if problem.potential_solution_ideas else 'None specified'}

--- RESEARCH QUALITY FLAGS ---
Bias Detected: {bias_summary}
Credibility Analysis: {credibility_summary}
Counter-Evidence (reasons this may NOT be a real problem): {counter_summary}
Opposing Viewpoints: {opposing_summary}
"""
        if self.enable_meetings:
            meeting_outcome = await quick_meeting(
                topic=f"Should we pursue: {problem.description}?",
                participants=meeting_participants,
                meeting_type=MeetingType.DECISION,
                context=meeting_context
            )

            self.state.artifacts["opportunity_meeting"] = {
                "decision": meeting_outcome.decision,
                "consensus_reached": meeting_outcome.consensus_reached,
                "votes": meeting_outcome.votes,
                "key_insights": meeting_outcome.key_insights
            }
            meeting_decision = meeting_outcome.decision
        else:
            console.info("Meetings disabled — skipping team meeting, CEO decides alone")
            meeting_decision = "Proceed (meetings disabled)"
            self.state.artifacts["opportunity_meeting"] = {"decision": meeting_decision, "meetings_disabled": True}

        # CEO still makes final call based on meeting
        eval_task = {
            "type": "evaluate_opportunity",
            "problem": problem.to_dict(),
            "market_analysis": str(analysis),
            "meeting_outcome": meeting_decision,
            "bias_flags": bias_summary,
            "credibility": credibility_summary,
            "counter_evidence": counter_summary,
            "opposing_viewpoints": opposing_summary,
            "freshness_score": freshness,
        }
        result = await self.ceo.execute_task(eval_task)

        # Handle case where output might be a string instead of dict
        output = result.output if isinstance(result.output, dict) else {"response": str(result.output)}
        decision = output.get("decision", "rejected")
        reasoning = output.get("reasoning", "")
        meeting_consensus = meeting_outcome.consensus_reached if self.enable_meetings else None
        self.state.decisions.append({
            "phase": "opportunity_evaluation",
            "decision": decision,
            "reasoning": reasoning,
            "meeting_consensus": meeting_consensus
        })

        self.state.artifacts["ceo_evaluation"] = result.output
        if self.enable_meetings:
            console.info(f"Meeting Consensus: {meeting_consensus}")
        console.agent_decision("CEO", decision.upper(), reasoning if reasoning else "")

        approved = decision == "approved"

        # Devil's Advocate gate: challenge approved decisions
        if self.enable_meetings and approved:
            console.info("Running Devil's Advocate review...")
            da_participants = [
                {"name": "ProductManager", "role": "Advocate", "model": self.model_config.get_model_name("product_manager")},
                {"name": "SecurityEngineer", "role": "Critic", "model": self.model_config.get_model_name("security_engineer")},
                {"name": "CEO", "role": "Judge", "model": self.model_config.get_model_name("ceo")},
            ]
            da_outcome = await quick_meeting(
                topic=f"Devil's Advocate: {problem.description}",
                participants=da_participants,
                meeting_type=MeetingType.DEVILS_ADVOCATE,
                context=f"Initial evaluation: APPROVED\nReasoning: {reasoning}",
            )
            self.state.artifacts["devils_advocate"] = {
                "decision": da_outcome.decision,
                "votes": da_outcome.votes,
                "key_insights": da_outcome.key_insights,
            }
            if da_outcome.decision and "reject" in da_outcome.decision.lower():
                console.warning("Devil's advocate overturned initial approval")
                approved = False

        return approved

    # ============================================================
    #  PHASE 5: TECHNICAL DESIGN (CTO)
    # ============================================================

    async def _run_technical_design_phase(self) -> None:
        """CTO designs the technical architecture."""
        self.state.phase = WorkflowPhase.TECHNICAL_DESIGN
        self.structured_logger.info("Technical design phase started")

        problem = self.state.current_problem
        if not problem:
            console.error("No problem for technical design")
            return
        _analysis = self.state.artifacts.get("problem_analysis", {})  # Available for future use

        # Refine the problem statement before feeding to PM and CTO
        try:
            from research.problem_statement_refiner import ProblemStatementRefiner
            _refiner = ProblemStatementRefiner()
            _refined = _refiner.refine(problem.description)
            refined_desc = _refiner.format_refined_statement(_refined)
            self.state.artifacts["refined_problem"] = refined_desc
            console.info(
                f"Problem refined: type={_refined.problem_type.value}, "
                f"actionable={_refined.is_actionable}, confidence={_refined.confidence:.0%}"
            )
            if not _refined.is_actionable:
                for note in _refined.refinement_notes:
                    console.warning(f"Problem refinement note: {note}")
            # Augment the problem dict passed to PM with refined description
            problem_dict_refined = problem.to_dict()
            problem_dict_refined["refined_description"] = refined_desc
        except Exception as _re:
            console.warning(f"Problem statement refiner failed: {_re}")
            problem_dict_refined = problem.to_dict()

        # Define requirements
        console.agent_action("ProductManager", "Defining Requirements", "Creating detailed specifications...")
        problem_analysis = self.state.artifacts.get("problem_analysis", {})
        req_task = {
            "type": "define_requirements",
            "problem": problem_dict_refined,
            "target_users": problem.target_users,
            "market_analysis": str(problem_analysis)[:600] if problem_analysis else ""
        }
        req_result = await self.product_manager.execute_task(req_task)
        self.state.artifacts["requirements"] = req_result.output
        # Store structured RequirementsDoc alongside raw output
        req_doc_dict = req_result.artifacts.get("requirements_doc") or (
            req_result.output.get("requirements_doc") if isinstance(req_result.output, dict) else None
        )
        if req_doc_dict:
            self.state.artifacts["requirements_doc"] = req_doc_dict
            ac_count = len(req_doc_dict.get("acceptance_criteria", []))
            console.info(f"RequirementsDoc: {ac_count} acceptance criteria extracted")
        console.agent_thinking("ProductManager", "Requirements defined")
        # Track PM experience for requirements definition
        if hasattr(self.product_manager, '_experience'):
            self.product_manager._experience.add_experience("define_requirements", req_result.success)
        self.agent_pool.record_task_outcome("ProductManager", req_result.success)

        # Technical design
        console.agent_action("CTO", "Designing Architecture", "Creating technical blueprint...")
        # Handle case where output might be a string instead of dict
        req_output = req_result.output if isinstance(req_result.output, dict) else {"response": str(req_result.output)}
        design_task = {
            "type": "design_architecture",
            "problem": problem_dict_refined,
            "requirements": str(req_output.get("requirements", "")),
            "requirements_doc": self.state.artifacts.get("requirements_doc", {}),
            "market_analysis": str(problem_analysis)[:400] if problem_analysis else "",
            "context_messages": self._drain_messages_for_agent("CTO")
        }
        # Debate: for complex tasks, run CEO vs CTO debate before finalizing architecture
        design_prompt = (
            f"Architecture for: {problem.description}\n\n"
            f"Requirements: {str(req_output.get('requirements', ''))[:400]}"
        )
        compute = self.cto._get_compute_config(design_prompt)
        if compute["consistency_samples"] >= 5:  # complex task — run debate
            try:
                from collaboration.debate import DebateOrchestrator
                debate = DebateOrchestrator(max_rounds=2)
                outcome = await debate.run_debate(
                    agent_a=self.ceo,
                    agent_b=self.cto,
                    topic=f"Architecture for: {self.state.current_problem.description}",
                    synthesizer=self.cto,
                )
                self.state.artifacts["architecture_debate"] = outcome.final_position
                console.info("Architecture debate complete — skipping HyperTree to avoid redundant LLM calls")
                # Inject debate outcome into the CTO's final design task
                design_task["requirements"] = (
                    design_task["requirements"]
                    + f"\n\nARCHITECTURE DEBATE OUTCOME:\n{outcome.final_position[:1000]}"
                )
                # HyperTree is skipped when Debate runs — both fire on complex tasks
                # and together produce 10+ LLM calls. Debate already covers the same ground.
            except Exception as _debate_err:
                console.warning(f"Architecture debate skipped: {_debate_err}")

        design_result = await self.cto.execute_task(design_task)
        self.state.artifacts["architecture"] = design_result.output
        # Store structured ArchitectureNote + Stories
        arch_note_dict = design_result.artifacts.get("architecture_note") or (
            design_result.output.get("architecture_note") if isinstance(design_result.output, dict) else None
        )
        if arch_note_dict:
            self.state.artifacts["architecture_note"] = arch_note_dict
            stories = arch_note_dict.get("stories", [])
            self.state.artifacts["stories"] = stories
            adr_count = len(arch_note_dict.get("architecture_decisions", []))
            console.info(f"ArchitectureNote: {adr_count} ADRs, {len(stories)} stories sharded")
        console.agent_thinking("CTO", "Architecture design complete")
        # Track CTO experience and pool performance
        if hasattr(self.cto, '_experience'):
            self.cto._experience.add_experience("design_architecture", design_result.success)
        self.agent_pool.record_task_outcome("CTO", design_result.success)
        self.performance_tracker.record_task(
            "CTO", success=design_result.success,
            response_time_ms=design_result.execution_time * 1000, tokens=0
        )
        # CTO shares architecture context with Developer via message bus
        arch_summary = str(design_result.output)[:300] if design_result.output else "Architecture designed"
        self.message_bus.share_finding("CTO", "Developer", f"Architecture ready: {arch_summary}")

        # Extract root cause from CTO's structured output
        from utils.output_parser import StructuredOutputParser
        design_parser = StructuredOutputParser()
        design_text = design_result.artifacts.get("architecture_doc", "")
        if not design_text and isinstance(design_result.output, dict):
            design_text = design_result.output.get("architecture", "")
        parsed_design = design_parser.parse_cto_design(str(design_text))
        root_cause = parsed_design.get("root_cause", "")
        if root_cause:
            self.state.artifacts["root_cause"] = root_cause
            console.info(f"Root cause identified: {root_cause}")
        else:
            console.warning("CTO did not provide a structured ROOT_CAUSE")
        # Score root cause depth — ensures deep first-principles thinking
        rcd_score = design_parser.score_root_cause_depth(root_cause or design_text[:500])
        self.state.artifacts["root_cause_depth"] = rcd_score
        console.info(f"Root cause analysis depth: {rcd_score}/4 (1=symptom, 2=proximate, 3=root, 4=systemic)")
        if rcd_score < 2:
            console.warning("Shallow root cause analysis — CTO will be prompted for deeper analysis on next redesign")
            self.state.artifacts["needs_deeper_root_cause"] = True

        # Reflect on CTO architecture design (learning)
        if self.enable_learning:
            self.cto.reflect(
                "design_architecture",
                str(design_result.output),
                design_result.success
            )

        # Feasibility assessment
        console.agent_action("CTO", "Assessing Feasibility", "Evaluating technical constraints...")
        design_output = design_result.output if isinstance(design_result.output, dict) else {"response": str(design_result.output)}
        feasibility_task = {
            "type": "assess_feasibility",
            "solution": design_output.get("architecture", ""),
            "constraints": [],
            "timeline": "MVP"
        }
        feasibility_result = await self.cto.execute_task(feasibility_task)
        self.state.artifacts["feasibility"] = feasibility_result.output

        # Handle case where output might be a string instead of dict
        feas_output = feasibility_result.output if isinstance(feasibility_result.output, dict) else {"response": str(feasibility_result.output)}
        feasibility = feas_output.get('feasibility', 'unknown')
        console.success(f"Feasibility Assessment: {feasibility}")

        # Create execution plan for remaining phases via PlanManager
        exec_plan = self.plan_manager.create_plan(
            name=f"solution-{problem.id}",
            description=f"Execution plan for: {problem.description}",
            tags=["auto-generated"],
        )
        exec_plan.add_step("Implementation", "Implement the solution based on architecture")
        exec_plan.add_step("Code Execution", "Run and verify generated code")
        exec_plan.add_step("QA Validation", "Validate solution quality")
        exec_plan.add_step("CEO Approval", "Get final approval from CEO")
        exec_plan.add_step("Delivery", "Package and deliver the solution")
        self.plan_manager.approve_plan(exec_plan, approved_by="CTO")
        self.plan_manager.start_execution(exec_plan)
        self.state.artifacts["execution_plan"] = exec_plan.to_dict()
        console.info(f"Execution plan created: {exec_plan.name} ({len(exec_plan.steps)} steps)")

    # ============================================================
    #  PHASE 7: IMPLEMENTATION (DEVELOPER)
    # ============================================================

    async def _run_implementation_phase(self) -> None:
        """Developer implements the solution."""
        self.state.phase = WorkflowPhase.IMPLEMENTATION
        self.structured_logger.info("Implementation phase started")

        # Track plan step
        if self.plan_manager.current_plan:
            self.plan_manager.start_step("step_1")

        problem = self.state.current_problem
        if not problem:
            console.error("No problem for implementation")
            return

        architecture = self.state.artifacts.get("architecture", {})
        requirements = self.state.artifacts.get("requirements", {})

        # Get language from problem metadata (default to python)
        language = problem.metadata.get("language", "python")
        self.state.artifacts["language"] = language
        console.info(f"Target language: {language}")

        # Determine project location
        # "Direct output" mode: output_dir is "." or matches workspace_root
        # In this mode, files go directly into the target directory (like Claude Code)

        is_direct_output = (
            self.output_dir == "."
            or Path(self.output_dir).resolve() == Path(self.workspace_root).resolve()
        )

        if is_direct_output:
            # Direct output mode - like Claude Code, files go into workspace_root
            project_name = str(Path(self.workspace_root).resolve())
            console.info(f"Generating files in: {project_name} (Claude Code style)")
        else:
            # Isolated project mode - create a subfolder per solution
            project_name = f"{self.output_dir}/solution_{problem.id}"
            console.info(f"Output folder: {project_name}/")

        # Store project_name so QA feedback loop can reference it
        self.state.artifacts["project_name"] = project_name

        # Create project
        console.agent_action("Developer", "Creating Project", f"Setting up {language} project structure...")
        create_task = {
            "type": "create_project",
            "name": project_name,
            "project_type": language,  # Use the target language
            "description": problem.description,
            "features": problem.potential_solution_ideas,
            "output_dir": project_name
        }
        create_result = await self.developer.execute_task(create_task)
        self.state.artifacts["project"] = create_result.output

        # Handle case where output might be a string instead of dict
        create_output = create_result.output if isinstance(create_result.output, dict) else {"response": str(create_result.output)}
        files_created = create_output.get('files_created', [])
        console.success(f"Project created with {len(files_created)} files")

        # Implement feature
        console.agent_action("Developer", "Implementing Features", "Writing code based on specifications...")
        # Handle case where architecture might be a string instead of dict
        arch_data = architecture if isinstance(architecture, dict) else {"response": str(architecture)}
        time_context = self._get_time_context()
        spec_text = str(requirements)
        if time_context:
            spec_text = f"{time_context}\n{spec_text}"
        impl_task = {
            "type": "implement_feature",
            "specification": spec_text,
            "architecture": str(arch_data.get("architecture", "")),
            "file_structure": {},
            "language": language,
            "output_dir": project_name,
            "stories": self.state.artifacts.get("stories", []),
            "requirements_doc": self.state.artifacts.get("requirements_doc", {}),
            "context_messages": self._drain_messages_for_agent("Developer")
        }
        impl_result = await self.developer.execute_task(impl_task)
        # Wire first_principles_score from agent onto TaskResult
        impl_result.first_principles_score = getattr(self.developer, '_last_fp_score', 1.0)
        self.state.artifacts["implementation"] = impl_result.output
        self.state.artifacts["impl_fp_score"] = impl_result.first_principles_score

        # RAG: index this implementation for future retrieval
        if self.state.artifacts.get("implementation"):
            impl_summary = str(self.state.artifacts["implementation"])[:2000]
            self.rag_store.add(
                content=f"Problem: {self.state.current_problem.description}\n\nSolution:\n{impl_summary}",
                tags=[self.state.current_problem.domain],
                source="solution"
            )
        if impl_result.first_principles_score < 0.6:
            console.warning(
                f"Developer first-principles compliance LOW ({impl_result.first_principles_score:.0%}). "
                "Solution may be a surface-level fix rather than addressing root cause."
            )

        # Parse developer confidence score from response text
        import re as _re_conf
        _impl_text = str(impl_result.output)
        _conf_match = _re_conf.search(r'CONFIDENCE:\s*([0-9]\.[0-9]+)', _impl_text, _re_conf.IGNORECASE)
        dev_confidence = float(_conf_match.group(1)) if _conf_match else None
        if dev_confidence is not None:
            self.state.artifacts["developer_confidence"] = dev_confidence
            if dev_confidence < 0.6:
                console.warning(
                    f"Developer self-reported CONFIDENCE={dev_confidence:.0%}. "
                    "QA should focus extra attention on flagged uncertain areas."
                )
            else:
                console.info(f"Developer confidence: {dev_confidence:.0%}")

        # Handle case where output might be a string instead of dict
        impl_output = impl_result.output if isinstance(impl_result.output, dict) else {"response": str(impl_result.output)}
        impl_files = impl_output.get('files_written', [])
        console.success(f"Implementation complete - {len(impl_files)} files written")

        # Reflect on implementation (learning)
        if self.enable_learning:
            self.developer.reflect(
                "implement_feature",
                str(impl_result.output),
                impl_result.success
            )

        # Complete plan step
        if self.plan_manager.current_plan:
            self.plan_manager.complete_step("step_1", output="Implementation complete")

    # ============================================================
    #  PHASE 9-10: QA + SECURITY VALIDATION + CEO APPROVAL LOOP
    # ============================================================

    async def _run_qa_phase(self) -> bool:
        """QA Engineer validates the solution with a review meeting."""
        self.state.phase = WorkflowPhase.QA_VALIDATION
        self.structured_logger.info("QA validation phase started")

        # Track plan step (step_3 = QA Validation)
        if self.plan_manager.current_plan:
            self.plan_manager.start_step("step_3")

        problem = self.state.current_problem
        if not problem:
            console.error("No problem for QA phase")
            return False

        implementation = self.state.artifacts.get("implementation", {})
        requirements = self.state.artifacts.get("requirements", {})

        # Create test plan
        test_plan_task = {
            "type": "create_test_plan",
            "feature": {
                "name": problem.description,
                "description": problem.description
            },
            "requirements": str(requirements)
        }
        test_plan_result = await self.qa_engineer.execute_task(test_plan_task)
        self.state.artifacts["test_plan"] = test_plan_result.output

        # Hold a review meeting
        review_participants = [
            {
                "name": "QAEngineer",
                "role": "Quality Assurance",
                "model": self.model_config.get_model_name("qa_engineer")
            },
            {
                "name": "Developer",
                "role": "Implementation",
                "model": self.model_config.get_model_name("developer")
            },
            {
                "name": "CTO",
                "role": "Technical Review",
                "model": self.model_config.get_model_name("cto")
            }
        ]

        execution_results = self.state.artifacts.get("execution_results", {})
        execution_summary = execution_results.get("summary", "Not run")

        dev_confidence = self.state.artifacts.get("developer_confidence")
        confidence_note = (
            f"\nDeveloper Self-Confidence: {dev_confidence:.0%} — "
            "FOCUS testing on areas the developer flagged as uncertain."
            if dev_confidence is not None and dev_confidence < 0.75 else ""
        )

        cto_peer_review = self.state.artifacts.get("cto_peer_review", "")
        cto_peer_note = f"\nCTO Peer Review Issues: {cto_peer_review[:400]}" if cto_peer_review else ""
        review_context = f"""
Problem Solved: {problem.description}
Requirements: {str(requirements)}
Implementation Summary: {str(implementation)}
Test Plan: {str(test_plan_result.output)}
Code Execution Results: {execution_summary}{confidence_note}{cto_peer_note}
"""
        if self.enable_meetings:
            review_outcome = await quick_meeting(
                topic="Code Review and Quality Assessment",
                participants=review_participants,
                meeting_type=MeetingType.REVIEW,
                context=review_context
            )

            self.state.artifacts["qa_review_meeting"] = {
                "decision": review_outcome.decision,
                "votes": review_outcome.votes,
                "action_items": review_outcome.action_items
            }

            # Store meeting in shared memory
            self.shared_memory.store_meeting(
                topic="Code Review and Quality Assessment",
                participants=["QAEngineer", "Developer", "CTO"],
                summary=review_outcome.decision,
                action_items=review_outcome.action_items
            )
            review_feedback = review_outcome.decision
        else:
            console.info("Meetings disabled — skipping QA review meeting")
            review_feedback = "Review meeting skipped (meetings disabled)"
            review_outcome = None
            self.state.artifacts["qa_review_meeting"] = {"decision": review_feedback, "meetings_disabled": True}

        # Determine review depth based on trust between QA and Developer
        review_depth = self.trust_tracker.get_review_threshold("QAEngineer", "Developer")
        depth_instructions = {
            "light": "Quick review — focus on critical functionality only. The developer has a strong track record.",
            "standard": "Standard review — check functionality and code quality.",
            "thorough": "Deep review — check edge cases, error handling, and logic flow. Previous work had quality issues.",
        }
        depth_note = depth_instructions.get(review_depth, depth_instructions["standard"])
        console.info(f"Review depth: {review_depth}")

        # Part 4b: Include root cause context for QA verification
        root_cause = self.state.artifacts.get("root_cause", "")
        qa_requirements_str = str(requirements)
        if root_cause:
            qa_requirements_str += f"\n\nVerify the implementation addresses this root cause: {root_cause}"

        # Validate solution
        dev_confidence_val = self.state.artifacts.get("developer_confidence")
        validation_task = {
            "type": "validate_solution",
            "solution": {
                "description": problem.description,
                "implementation": str(implementation)
            },
            "requirements": qa_requirements_str,
            "requirements_doc": self.state.artifacts.get("requirements_doc", {}),
            "problem": problem.to_dict(),
            "review_feedback": review_feedback,
            "review_depth": review_depth,
            "review_depth_instruction": depth_note,
            "execution_results": self.state.artifacts.get("execution_results", {}),
            "developer_confidence": dev_confidence_val,
            "cto_peer_review": cto_peer_review,
            "context_messages": self._drain_messages_for_agent("QAEngineer")
        }
        validation_result = await self.qa_engineer.execute_task(validation_task)
        # Wire first_principles_score from QA onto result
        validation_result.first_principles_score = getattr(self.qa_engineer, '_last_fp_score', 1.0)
        self.state.artifacts["qa_fp_score"] = validation_result.first_principles_score
        self.state.artifacts["qa_validation"] = validation_result.output
        # Store typed QAReport if produced
        if isinstance(validation_result.output, dict) and "qa_report" in validation_result.output:
            self.state.artifacts["qa_report_typed"] = validation_result.output["qa_report"]
            qa_report_dict = validation_result.output["qa_report"]
            passed = len([r for r in qa_report_dict.get("criterion_results", []) if r.get("status") == "PASS"])
            total = len(qa_report_dict.get("criterion_results", []))
            if total:
                console.info(f"QAReport: {passed}/{total} acceptance criteria passed")

        # Generate QA report (execution_results already fetched above)
        report_task = {
            "type": "generate_qa_report",
            "project": problem.id,
            "test_results": test_plan_result.output,
            "issues": review_outcome.action_items if review_outcome else [],
            "execution_results": execution_results
        }
        report_result = await self.qa_engineer.execute_task(report_task)
        self.state.artifacts["qa_report"] = report_result.output

        # Handle case where output might be a string instead of dict
        qa_output = validation_result.output if isinstance(validation_result.output, dict) else {"response": str(validation_result.output)}
        verdict = qa_output.get("verdict", "fail")
        if self.enable_meetings and review_outcome:
            approvals = sum(1 for v in review_outcome.votes.values() if v == "approve")
            console.info(f"Review Meeting Approvals: {approvals}/{len(review_outcome.votes)}")
        console.info(f"QA Verdict: {verdict.upper()}")

        # Track performance: QA task and Developer's work quality
        qa_passed = verdict in ["pass", "pass_with_issues"]
        self.performance_tracker.record_task(
            "QAEngineer", success=True,
            response_time_ms=validation_result.execution_time * 1000,
            tokens=0
        )
        self.performance_tracker.record_approval("Developer", approved=qa_passed)
        if not qa_passed:
            self.performance_tracker.record_rework("Developer")

        # Update trust: QA reviewed Developer's work
        # High-confidence reviews are a stronger signal for trust
        qa_confidence = validation_result.confidence
        if qa_confidence >= CONFIDENCE_HIGH:
            self.trust_tracker.update_trust("QAEngineer", "Developer", positive=qa_passed)
        else:
            self.trust_tracker.update_trust("QAEngineer", "Developer", positive=qa_passed)

        # Update experience
        if hasattr(self.qa_engineer, '_experience'):
            self.qa_engineer._experience.add_experience("validate_solution", True)
        if hasattr(self.developer, '_experience'):
            self.developer._experience.add_experience("implementation", qa_passed)

        # Graduated confidence response (OrgChart integration)
        if qa_confidence < CONFIDENCE_CRITICAL:
            supervisor = self.org_chart.get_supervisor("QAEngineer")
            console.warning(f"QA confidence CRITICAL ({qa_confidence:.2f}) — supervisor {supervisor} will re-review")
            self.performance_tracker.record_task("QAEngineer", success=False, response_time_ms=0, tokens=0)
        elif qa_confidence < CONFIDENCE_LOW:
            supervisor = self.org_chart.get_supervisor("QAEngineer")
            if supervisor:
                console.info(f"QA confidence low ({qa_confidence:.2f}) — supervisor {supervisor} notified")
        elif qa_confidence >= CONFIDENCE_HIGH:
            console.info(f"QA high-confidence verdict ({qa_confidence:.2f})")

        # Reflect on QA validation (learning)
        if self.enable_learning:
            self.qa_engineer.reflect(
                "validate_solution",
                str(validation_result.output),
                qa_passed
            )

        # Update plan step
        if self.plan_manager.current_plan:
            if qa_passed:
                self.plan_manager.complete_step("step_3", output=f"QA verdict: {verdict}")
            else:
                self.plan_manager.fail_step("step_3", error=f"QA verdict: {verdict}")

        return qa_passed

    async def _run_ceo_approval(self) -> bool:
        """CEO gives final approval."""
        self.state.phase = WorkflowPhase.CEO_APPROVAL
        self.structured_logger.info("CEO approval phase started")

        # Track plan step (step_4 = CEO Approval)
        if self.plan_manager.current_plan:
            self.plan_manager.start_step("step_4")

        problem = self.state.current_problem
        if not problem:
            console.error("No problem for CEO approval")
            return False

        execution_results = self.state.artifacts.get("execution_results", {})
        execution_summary = execution_results.get("summary", "Not run")
        # Include DevOps verdict in CEO context
        devops_validation = self.state.artifacts.get("devops_validation", "Not run")
        devops_summary = str(devops_validation)[:500] if devops_validation else "Not run"
        # Include Security review findings
        security_review = self.state.artifacts.get("security_review", None)
        security_summary = str(security_review)[:500] if security_review else "Not run"

        impl_fp = self.state.artifacts.get("impl_fp_score", 1.0)
        qa_fp = self.state.artifacts.get("qa_fp_score", 1.0)
        quality_score = execution_results.get("quality_score", "N/A")
        quality_details = execution_results.get("quality_details", {})
        quality_str = (
            f"Quality={quality_score:.0%} "
            f"(TODOs={quality_details.get('todo_count',0)}, "
            f"long_funcs={quality_details.get('long_functions',0)}, "
            f"hardcoded={quality_details.get('hardcoded_values',0)})"
            if isinstance(quality_score, float) else f"Quality=N/A"
        )
        approval_task = {
            "type": "approve_solution",
            "solution": {
                "description": problem.description,
                "implementation": str(self.state.artifacts.get("implementation", {}))
            },
            "qa_report": str(self.state.artifacts.get("qa_report", {})),
            "security_report": security_summary,
            "original_problem": problem.to_dict(),
            "execution_summary": (
                f"{execution_summary}\nDevOps Validation: {devops_summary}\n"
                f"Security Review: {security_summary}\n"
                f"First-Principles Scores: Developer={impl_fp:.0%}, QA={qa_fp:.0%}\n"
                f"Code {quality_str}"
            ),
            "root_cause": self.state.artifacts.get("root_cause", ""),
        }
        result = await self.ceo.execute_task(approval_task)

        # Handle case where output might be a string instead of dict
        output = result.output if isinstance(result.output, dict) else {"response": str(result.output)}
        decision = output.get("decision", "rejected")
        reasoning = output.get("reasoning", "")
        self.state.decisions.append({
            "phase": "final_approval",
            "decision": decision,
            "reasoning": reasoning
        })

        self.state.artifacts["final_approval"] = result.output
        console.info(f"Final Approval: {decision.upper()}")

        # Track performance: CEO task and Developer approval
        approved = decision == "approved"
        self.performance_tracker.record_task(
            "CEO", success=True,
            response_time_ms=result.execution_time * 1000,
            tokens=0
        )
        self.performance_tracker.record_approval("Developer", approved=approved)
        if not approved:
            self.performance_tracker.record_rework("Developer")
        # Track CEO experience and pool performance
        if hasattr(self.ceo, '_experience'):
            self.ceo._experience.add_experience("approve_solution", approved)
        self.agent_pool.record_task_outcome("CEO", approved)
        # Reflect on CEO approval decision (learning)
        if self.enable_learning:
            self.ceo.reflect(
                "approve_solution",
                str(result.output),
                approved
            )

        # Update trust: CEO reviewed Developer's work
        self.trust_tracker.update_trust("CEO", "Developer", positive=approved)

        # Graduated confidence response (OrgChart integration)
        ceo_confidence = result.confidence
        if ceo_confidence < CONFIDENCE_CRITICAL:
            console.warning(f"CEO confidence CRITICAL ({ceo_confidence:.2f}) — decision may be unreliable")
        elif ceo_confidence < CONFIDENCE_LOW and not approved:
            console.info(f"CEO confidence low ({ceo_confidence:.2f}) — reviewing rejection reasoning")
        elif ceo_confidence >= CONFIDENCE_HIGH and approved:
            console.info(f"CEO high-confidence approval ({ceo_confidence:.2f})")

        # Update plan step
        if self.plan_manager.current_plan:
            if approved:
                self.plan_manager.complete_step("step_4", output=f"CEO: {decision}")
            else:
                self.plan_manager.fail_step("step_4", error=f"CEO: {decision} - {reasoning}")

        return approved

    # ============================================================
    #  ESCALATION STRATEGIES
    # ============================================================

    async def _team_brainstorm(self) -> Optional[Dict[str, Any]]:
        """All agents brainstorm solutions for a stuck problem.

        Uses a brainstorm meeting with CTO, PM, and Developer to
        generate fresh ideas, then picks the most promising one.
        """
        from collaboration.meeting import AgentMeeting
        from collaboration.meeting import MeetingType as MT
        problem = self.state.current_problem
        if not problem:
            return None

        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()

        meeting = AgentMeeting(MT.BRAINSTORM, f"Brainstorm solutions for: {problem.description}")
        meeting.add_participant("CTO", "cto", self.model_config.get_model_name("cto"))
        meeting.add_participant("ProductManager", "product_manager", self.model_config.get_model_name("product_manager"))
        meeting.add_participant("Developer", "developer", self.model_config.get_model_name("developer"))

        context = (
            f"Problem: {problem.description}\n"
            f"Previous attempts have failed.\n{strategy_context}\n"
            f"We need a DIFFERENT approach. Think creatively."
        )
        outcome = await meeting.run(context)

        # Record in strategy memory
        self.escalation_manager.strategy_memory.record_attempt(
            approach="team_brainstorm", round=len(self.escalation_manager.history),
            outcome="brainstorm_complete",
            feedback=str(outcome.key_insights[:3]) if outcome.key_insights else ""
        )

        # Use brainstorm ideas to inform next design
        if outcome.key_insights:
            brainstorm_ideas = "\n".join(outcome.key_insights[:5])
            redesign_task = {
                "type": "design_architecture",
                "problem": problem.to_dict(),
                "requirements": str(self.state.artifacts.get("requirements", "")),
                "constraint": f"Use these brainstormed ideas as inspiration:\n{brainstorm_ideas}"
            }
            redesign_result = await self.cto.execute_task(redesign_task)
            self.state.artifacts["architecture"] = redesign_result.output
            return {"ideas": outcome.key_insights, "new_architecture": redesign_result.output}

        return None

    async def _decompose_problem(self) -> Optional[List[Dict]]:
        """Break an unsolvable problem into smaller sub-problems.

        Uses PM to identify independently-solvable pieces.
        """
        problem = self.state.current_problem
        if not problem:
            return None

        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()

        decompose_task = {
            "type": "define_requirements",
            "problem": problem.to_dict(),
            "target_users": problem.target_users,
            "constraint": (
                f"The full problem has proven too complex to solve in one attempt.\n"
                f"{strategy_context}\n\n"
                f"Break it into 2-3 INDEPENDENT sub-problems that can each be solved separately.\n"
                f"Each sub-problem should be self-contained and testable on its own.\n"
                f"Focus on the MINIMUM viable solution for each sub-problem."
            )
        }
        decompose_result = await self.product_manager.execute_task(decompose_task)

        # Record in strategy memory
        self.escalation_manager.strategy_memory.record_attempt(
            approach="decompose_problem",
            round=len(self.escalation_manager.history),
            outcome="decomposed",
            feedback=str(decompose_result.output) if decompose_result.output else ""
        )

        # Use decomposed requirements for next implementation
        self.state.artifacts["requirements"] = decompose_result.output
        await self._run_technical_design_phase()

        return [{"decomposed_requirements": decompose_result.output}]

    async def _pivot_approach(self) -> None:
        """Try a completely different technical approach.

        CTO proposes alternative architectures/tech stacks.
        Strategy memory ensures we don't repeat failed approaches.
        """
        problem = self.state.current_problem
        if not problem:
            return

        strategy_context = self.escalation_manager.strategy_memory.summarize_for_next_attempt()
        tried_archs = self.escalation_manager.strategy_memory.tried_architectures

        pivot_task = {
            "type": "design_architecture",
            "problem": problem.to_dict(),
            "requirements": str(self.state.artifacts.get("requirements", "")),
            "constraint": (
                f"ALL previous approaches have failed. You MUST propose a COMPLETELY "
                f"different technical approach.\n\n"
                f"{strategy_context}\n\n"
                f"Previously tried architectures: {', '.join(tried_archs) if tried_archs else 'None recorded'}\n\n"
                f"Consider: different language, different framework, simpler architecture, "
                f"different data structures, or a fundamentally different algorithm.\n"
                f"Do NOT reuse any approach that has already been tried."
            )
        }
        pivot_result = await self.cto.execute_task(pivot_task)
        self.state.artifacts["architecture"] = pivot_result.output

        # Record in strategy memory
        arch_desc = str(pivot_result.output) if pivot_result.output else "unknown"
        self.escalation_manager.strategy_memory.record_architecture(arch_desc)
        self.escalation_manager.strategy_memory.record_attempt(
            approach="pivot_approach",
            round=len(self.escalation_manager.history),
            outcome="pivoted",
            feedback=arch_desc
        )

    # ============================================================
    #  HELPER METHODS
    # ============================================================

    # Token budget allocation per agent (fraction of total budget)
    _TOKEN_BUDGETS = {
        "CEO": 0.10, "CTO": 0.15, "ProductManager": 0.10,
        "Researcher": 0.15, "Developer": 0.25, "QAEngineer": 0.10,
        "DevOpsEngineer": 0.05, "DataAnalyst": 0.05, "SecurityEngineer": 0.05,
    }

    def _check_token_budget(self, agent_name: str) -> int:
        """Return adjusted max_tokens based on remaining agent budget."""
        session = self.cost_tracker.get_current_session()
        if not session or not getattr(self.cost_tracker, '_budget_tokens', None):
            return 4096  # No budget set
        budget_pct = self._TOKEN_BUDGETS.get(agent_name, 0.10)
        agent_budget = int(self.cost_tracker._budget_tokens * budget_pct)
        agent_summary = session.agents.get(agent_name)
        used = agent_summary.total_tokens if agent_summary else 0
        remaining = agent_budget - used
        if remaining < 512:
            console.warning(f"{agent_name} near token budget ({used}/{agent_budget})")
            return 512  # Minimum viable response
        return min(4096, remaining)

    def _get_time_context(self) -> str:
        """Get time budget context for agent prompts."""
        if not self.max_workflow_minutes or not self.state.started_at:
            return ""
        elapsed = (datetime.now() - self.state.started_at).total_seconds() / 60
        remaining = self.max_workflow_minutes - elapsed
        if remaining < 5:
            return "[TIME CRITICAL: <5 min remaining. Simplest viable solution only.]"
        elif remaining < 15:
            return f"[Time remaining: {remaining:.0f}min. Prioritize core functionality.]"
        return ""

    def _detect_failure_owner(self, feedback: str) -> str:
        """Determine which agent 'owns' the failure based on feedback content."""
        feedback_lower = feedback.lower()
        if any(w in feedback_lower for w in ["requirement", "scope", "feature missing", "spec"]):
            return "ProductManager"
        if any(w in feedback_lower for w in ["architecture", "design", "structure", "pattern"]):
            return "CTO"
        if any(w in feedback_lower for w in ["security", "vulnerability", "injection", "secret"]):
            return "SecurityEngineer"
        return "Developer"

    def _adjust_agent_behavior(self) -> None:
        """Adjust agent parameters based on KPI performance from previous runs.

        Called at workflow start. Uses PerformanceTracker data to:
        - Lower temperature for underperformers (make more conservative)
        - Increase temperature for high performers (allow more creativity)
        - Add supervision note for agents with low scores
        - Log adjustments for transparency
        """
        all_kpis = self.performance_tracker.get_all_kpis()
        if not all_kpis:
            return  # No historical data yet

        for agent_name, kpis in all_kpis.items():
            score = kpis.calculate_score()
            agent = self.agents.get(agent_name)
            if not agent:
                continue

            if score < 40:
                # Underperformer: lower temperature, use reconfiguration recommendation
                agent.config.temperature = max(0.1, agent.config.temperature - 0.15)
                from company.organization import AgentManager
                recommendation = AgentManager.recommend_reconfiguration(agent_name, score)
                agent._supervision_note = (
                    f"Performance score: {score:.0f}/100. {recommendation}"
                )
                console.info(f"Reconfiguration for {agent_name}: {recommendation}")
                self.structured_logger.info(
                    f"Adjusted {agent_name}: temp lowered to {agent.config.temperature:.2f} (score={score:.0f})"
                )
            elif score > 80:
                # High performer: slight temp boost for creativity
                agent.config.temperature = min(1.0, agent.config.temperature + 0.05)
                self.structured_logger.info(
                    f"Adjusted {agent_name}: temp raised to {agent.config.temperature:.2f} (score={score:.0f})"
                )

    def _drain_messages_for_agent(self, agent_name: str) -> str:
        """Drain pending messages and return summary for agent context."""
        pending = self.message_bus.peek_messages(agent_name)
        if pending == 0:
            return ""
        msgs = self.message_bus.get_messages(agent_name, limit=5)
        if not msgs:
            return ""
        lines = []
        for msg in msgs:
            sender = getattr(msg, 'sender', 'unknown')
            content = getattr(msg, 'content', str(msg))
            lines.append(f"From {sender}: {content}")
        return "Pending team messages:\n" + "\n".join(lines)

    async def _run_ceo_morning_brief(self, problem) -> None:
        """CEO sets 2-3 priorities and constraints for the run.

        The brief is injected into all agent system prompts so the team
        operates under shared constraints throughout the workflow.
        """
        prob_desc = problem.description if problem else "auto-discover a problem"
        brief_prompt = (
            f"We are about to work on: {prob_desc}\n\n"
            "As CEO, write a 3-bullet morning brief for the team. Each bullet is one constraint or priority:\n"
            "- Bullet 1: primary user/outcome focus (what matters most to the end user)\n"
            "- Bullet 2: technical constraint (e.g., prefer stdlib, single file, no external services)\n"
            "- Bullet 3: quality bar (e.g., must work on first run, zero external dependencies)\n\n"
            "Keep each bullet ≤ 15 words. No fluff."
        )
        try:
            compute = self.ceo._get_compute_config(brief_prompt)
            if compute["consistency_samples"] >= 5:  # complex
                brief = await self.ceo.generate_with_tot(
                    brief_prompt,
                    evaluation_criteria="strategic clarity, risk identification, and actionability"
                )
            else:
                brief = await self.ceo.generate_response_async(brief_prompt, use_first_principles=False)
            brief = brief.strip()
            self.state.artifacts["ceo_morning_brief"] = brief
            console.info(f"CEO Morning Brief:\n{brief}")
            # Inject brief into all agent system prompts for this run
            brief_injection = f"\n\n[CEO MORNING BRIEF]\n{brief}\n"
            for agent in self.agents.values():
                if brief_injection not in agent.system_prompt:
                    agent.system_prompt += brief_injection
        except Exception as e:
            console.warning(f"CEO morning brief skipped: {e}")

    async def _run_morning_standup(self, problem_name: str) -> None:
        """Create sprint and run standup meeting for team alignment."""
        from company.sprint import SprintTask

        sprint_tasks = [
            SprintTask(id="research", name="Research & Discovery", assigned_to="Researcher"),
            SprintTask(id="design", name="Technical Design", assigned_to="CTO"),
            SprintTask(id="implement", name="Implementation", assigned_to="Developer"),
            SprintTask(id="qa", name="QA Validation", assigned_to="QAEngineer"),
            SprintTask(id="approval", name="CEO Approval", assigned_to="CEO"),
        ]
        sprint = self.sprint_manager.create_sprint(
            name=f"Sprint: {problem_name}",
            goal=problem_name,
            tasks=sprint_tasks,
        )
        self.sprint_manager.start_sprint(sprint.id)
        self._current_sprint_id = sprint.id

        if not self.enable_meetings:
            return

        standup_context = f"Sprint goal: {problem_name}\n"
        for agent in self.agents.values():
            if hasattr(agent, '_retrospective_lessons') and agent._retrospective_lessons:
                standup_context += f"Lessons from last run:\n{agent._retrospective_lessons}\n"
                break

        standup_participants = [
            {"name": "CEO", "role": "Leadership", "model": self.model_config.get_model_name("ceo")},
            {"name": "CTO", "role": "Technical Lead", "model": self.model_config.get_model_name("cto")},
            {"name": "Developer", "role": "Implementation", "model": self.model_config.get_model_name("developer")},
        ]
        console.section("MORNING STANDUP")
        standup_outcome = await quick_meeting(
            topic=f"Sprint Kickoff: {problem_name}",
            participants=standup_participants,
            meeting_type=MeetingType.STANDUP,
            context=standup_context,
        )
        self.state.artifacts["standup"] = {
            "action_items": standup_outcome.action_items,
            "insights": standup_outcome.key_insights,
        }
        # Record structured standup minutes via StandupManager
        standup_reports = {}
        for i, insight in enumerate(standup_outcome.key_insights or []):
            participant = standup_participants[i % len(standup_participants)]
            standup_reports[participant["name"]] = insight
        standup_minutes = self.standup_manager.record_standup(
            attendees=[p["name"] for p in standup_participants],
            phase="sprint_kickoff",
            reports=standup_reports,
        )
        if standup_minutes.blockers:
            console.warning(f"Standup blockers identified: {len(standup_minutes.blockers)}")
            for blocker in standup_minutes.blockers:
                console.warning(f"  BLOCKER: {blocker[:200]}")

    def _load_previous_learnings(self) -> None:
        """Load retrospective action items from shared memory into agent context."""
        from memory.shared_memory import MemoryType
        retro_memories = self.shared_memory.get_by_type(MemoryType.MEETING)
        # Filter for retrospectives
        retro_mems = [m for m in retro_memories if "retrospective" in m.content.lower()]
        if not retro_mems:
            return
        # Take most recent 3
        retro_mems.sort(key=lambda m: m.created_at, reverse=True)
        lessons = []
        for mem in retro_mems[:3]:
            items = mem.metadata.get("action_items", [])
            lessons.extend(items[:3])
        if not lessons:
            return
        lessons_text = "\n".join(f"- {l}" for l in lessons[:8])
        for agent in self.agents.values():
            agent._retrospective_lessons = lessons_text
        console.info(f"Loaded {len(lessons[:8])} lessons from past retrospectives")

        # Also load learning system insights from past runs
        insight_memories = self.shared_memory.get_by_type(MemoryType.INSIGHT)
        learning_insights = [m for m in insight_memories if "learning" in (m.tags or [])]
        if learning_insights:
            learning_insights.sort(key=lambda m: m.created_at, reverse=True)
            for mem in learning_insights[:3]:
                agent_name = mem.created_by
                if agent_name in self.agents:
                    existing = getattr(self.agents[agent_name], '_retrospective_lessons', '') or ''
                    self.agents[agent_name]._retrospective_lessons = (
                        existing + f"\n{mem.content}"
                    )
            console.info(f"Loaded {min(3, len(learning_insights))} learning insights from past runs")

    async def _run_performance_reviews(self) -> None:
        """Run 1:1 meetings for underperforming agents."""
        if not self.enable_meetings:
            return
        underperformers = self.performance_tracker.get_underperformers(threshold=40.0)
        if not underperformers:
            return

        for agent_name in underperformers[:2]:  # Max 2 reviews per run
            supervisor_name = self.org_chart.get_supervisor(agent_name)
            if not supervisor_name or supervisor_name not in self.agents:
                continue

            agent = self.agents[agent_name]
            supervisor = self.agents[supervisor_name]
            kpis = self.performance_tracker.get_kpis(agent_name)

            context = (
                f"Agent: {agent_name}\n"
                f"Performance Score: {kpis.calculate_score():.0f}/100\n"
                f"Approval Rate: {kpis.approval_rate:.0%}\n"
                f"Rework Rate: {kpis.rework_rate:.0%}\n"
            )

            console.info(f"1:1 Review: {supervisor_name} -> {agent_name} (score: {kpis.calculate_score():.0f})")
            review_outcome = await quick_meeting(
                topic=f"Performance Review: {agent_name}",
                participants=[
                    {"name": supervisor_name, "role": "Supervisor", "model": supervisor.model},
                    {"name": agent_name, "role": "Report", "model": agent.model},
                ],
                meeting_type=MeetingType.ONE_ON_ONE,
                context=context,
            )
            if review_outcome.action_items:
                from company.organization import AgentManager
                reconfig = AgentManager.recommend_reconfiguration(agent_name, kpis.calculate_score())
                agent._supervision_note = (
                    f"Feedback from {supervisor_name}: "
                    + "; ".join(review_outcome.action_items[:3])
                    + f"\nSystem recommendation: {reconfig}"
                )

    # ============================================================
    #  DELIVERY + RETROSPECTIVE
    # ============================================================

    async def _run_delivery_phase(self) -> Dict[str, Any]:
        """Package the approved solution for delivery."""
        self.structured_logger.info("Delivery phase started")

        # Track plan step (step_5 = Delivery)
        if self.plan_manager.current_plan:
            self.plan_manager.start_step("step_5")

        project_dir = self.state.artifacts.get("project_name", self.output_dir)
        project_path = Path(project_dir)
        problem = self.state.current_problem

        # Generate README via Developer
        readme_task = {
            "type": "implement_feature",
            "specification": (
                f"Create a README.md file for this project.\n"
                f"Project: {problem.description}\n"
                f"Include: Overview, Installation, Usage, Architecture.\n"
                f"Keep it concise and practical."
            ),
            "architecture": str(self.state.artifacts.get("architecture", "")),
            "file_structure": {},
            "language": self.state.artifacts.get("language", "python"),
            "output_dir": project_dir
        }
        await self.developer.execute_task(readme_task)

        # Generate Dockerfile for easy deployment
        language = self.state.artifacts.get("language", "python")
        dockerfile_task = {
            "type": "implement_feature",
            "specification": (
                f"Create a Dockerfile for this {language} project.\n"
                f"Project: {problem.description}\n"
                f"Requirements:\n"
                f"- Use a slim/alpine base image for {language}\n"
                f"- Install dependencies (requirements.txt or package.json if present)\n"
                f"- Set the correct WORKDIR and ENTRYPOINT\n"
                f"- Add a .dockerignore file excluding .git, __pycache__, node_modules, venv, .env\n"
                f"- Keep it simple and production-ready\n"
                f"Output ONLY the Dockerfile and .dockerignore files."
            ),
            "architecture": "",
            "file_structure": {},
            "language": language,
            "output_dir": project_dir
        }
        try:
            await self.developer.execute_task(dockerfile_task)
            console.success("Dockerfile generated for easy deployment")
        except Exception as e:
            console.warning(f"Dockerfile generation skipped: {e}")

        # Format generated code before delivery
        from tools.code_formatter import CodeFormatter
        try:
            formatter = CodeFormatter(timeout=30)
            format_result = formatter.format_directory(
                project_dir,
                recursive=True,
                exclude_patterns=["node_modules", "__pycache__", ".git", "venv", ".venv", ".deps", "dist", "build"]
            )
            if format_result.files_formatted:
                console.success(f"Formatted {len(format_result.files_formatted)} files")
            if format_result.errors:
                for err in format_result.errors[:3]:
                    console.warning(f"Format warning: {err}")
        except Exception as e:
            console.warning(f"Code formatting skipped: {e}")

        # Initialize git repository for the delivered project
        try:
            project_git = GitTools(repo_path=project_dir)
            if not project_git.is_repo():
                import subprocess as _sp
                _sp.run(["git", "init"], cwd=project_dir, capture_output=True, timeout=10)
                project_git.stage_all()
                project_git.commit(
                    message=f"feat: initial solution for {problem.description}",
                    add_co_author=True,
                )
                console.success("Git repository initialized with initial commit")
            else:
                console.info("Git repository already exists, skipping init")
        except Exception as git_err:
            console.warning(f"Git init skipped: {git_err}")

        # Install dependencies for the delivered project
        import subprocess as _sub
        try:
            req_file = project_path / "requirements.txt"
            pkg_file = project_path / "package.json"
            if req_file.exists():
                console.info("Installing Python dependencies...")
                _sub.run(
                    ["python3", "-m", "pip", "install", "-q", "-r", str(req_file)],
                    capture_output=True, text=True, timeout=120, cwd=project_dir
                )
                console.success("Python dependencies installed")
            elif pkg_file.exists():
                console.info("Installing Node.js dependencies...")
                npm_cmd = "npm" if _sub.run(["which", "npm"], capture_output=True).returncode == 0 else None
                if npm_cmd:
                    _sub.run(
                        ["npm", "install", "--silent"],
                        capture_output=True, text=True, timeout=120, cwd=project_dir
                    )
                    console.success("Node.js dependencies installed")
                else:
                    console.warning("npm not found, skipping dependency install")
        except Exception as dep_err:
            console.warning(f"Dependency installation skipped: {dep_err}")

        # Build delivery summary
        delivery = {
            "project_dir": project_dir,
            "problem": problem.description,
            "language": self.state.artifacts.get("language", "python"),
            "files": [str(f.relative_to(project_path))
                      for f in self._filter_project_files(list(project_path.rglob("*")))
                      if f.is_file()] if project_path.exists() else [],
            "execution_status": self.state.artifacts.get("execution_results", {}).get("summary", "Not run"),
        }

        self.state.artifacts["delivery"] = delivery

        # Complete plan step
        if self.plan_manager.current_plan:
            self.plan_manager.complete_step("step_5", output=f"Delivered {len(delivery.get('files', []))} files")

        return delivery

    # ============================================================
    #  PHASE 2: DATA ANALYSIS
    # ============================================================

    async def _run_data_analysis_phase(self) -> None:
        """DataAnalyst cross-validates, deduplicates, and detects bias in research findings."""
        self.state.phase = WorkflowPhase.DATA_ANALYSIS
        self.structured_logger.info("Data analysis phase started")

        problems = self.state.artifacts.get("discovered_problems", [])
        if not problems:
            console.info("No discovered problems to analyze, skipping data analysis")
            return

        # Step 1: Deduplicate
        console.agent_action("DataAnalyst", "Deduplicating Findings", f"Checking {len(problems)} problems for duplicates...")
        deduped = self.cross_validator.deduplicate(problems)
        removed_count = len(problems) - len(deduped)
        if removed_count > 0:
            console.success(f"Deduplicated: removed {removed_count} duplicate(s), {len(deduped)} unique problems remain")
        self.state.artifacts["discovered_problems"] = deduped

        # Step 2: Cross-validate
        console.agent_action("DataAnalyst", "Cross-Validating", "Scoring multi-source confirmation...")
        validated = self.cross_validator.cross_validate(deduped)
        self.state.artifacts["discovered_problems"] = validated

        # Apply freshness scoring — downweight stale problems before LLM analysis
        from research.credibility import score_freshness
        freshness_penalized = 0
        for p in validated:
            disc_at = p.get("discovered_at")
            if isinstance(disc_at, str):
                try:
                    disc_at = datetime.fromisoformat(disc_at)
                except (ValueError, TypeError):
                    disc_at = None
            elif not isinstance(disc_at, datetime):
                disc_at = None
            freshness = score_freshness(discovered_at=disc_at)
            p["freshness_score"] = freshness
            if freshness < 0.8:
                p["score"] = round(p.get("score", 0.0) * freshness, 4)
                freshness_penalized += 1
        if freshness_penalized:
            console.info(f"Freshness scoring: {freshness_penalized} problem(s) penalized for staleness")
        else:
            console.info("Freshness scoring: all problems are recent (no penalty)")

        # Step 2b: LLM-powered cross-validation (DataAnalyst agent)
        console.agent_action("DataAnalyst", "LLM Cross-Validation", "Agent cross-validating findings with structured verdicts...")
        cross_val_task = {
            "type": "cross_validate_research",
            "findings": validated[:5],  # Top 5 for token efficiency
            "sources": list({s for p in validated for s in p.get("sources", [])}),
            "topic": self.state.current_problem.description if self.state.current_problem else "research findings",
        }
        cross_val_result = await self.data_analyst.execute_task(cross_val_task)
        # Track DataAnalyst experience and pool performance
        if hasattr(self.data_analyst, '_experience'):
            self.data_analyst._experience.add_experience("cross_validate_research", cross_val_result.success)
        self.agent_pool.record_task_outcome("DataAnalyst", cross_val_result.success)
        self.performance_tracker.record_task(
            "DataAnalyst", success=cross_val_result.success,
            response_time_ms=cross_val_result.execution_time * 1000, tokens=0
        )
        # Reflect on DataAnalyst cross-validation (learning)
        if self.enable_learning:
            self.data_analyst.reflect(
                "cross_validate_research",
                str(cross_val_result.output),
                cross_val_result.success
            )
        # DataAnalyst shares validation results back to Researcher via message bus
        self.message_bus.share_finding(
            "DataAnalyst", "Researcher",
            f"Cross-validation complete for {len(validated)} findings. See cross_validation_verdicts artifact."
        )

        # Parse structured verdicts
        from utils.output_parser import StructuredOutputParser
        verdict_parser = StructuredOutputParser()
        cross_val_text = cross_val_result.artifacts.get("cross_validation_report", "")
        parsed_verdicts = verdict_parser.parse_data_analyst_verdict(cross_val_text)
        self.state.artifacts["cross_validation_verdicts"] = parsed_verdicts

        summary = parsed_verdicts.get("summary", {})
        console.info(
            f"LLM cross-validation: {summary.get('confirmed', 0)} confirmed, "
            f"{summary.get('partially_confirmed', 0)} partial, "
            f"{summary.get('unconfirmed', 0)} unconfirmed, "
            f"{summary.get('contradicted', 0)} contradicted"
        )

        # Step 3: Detect bias
        console.agent_action("DataAnalyst", "Detecting Bias", "Checking for systematic research biases...")
        bias_flags = self.cross_validator.detect_bias(validated)
        if bias_flags:
            for flag in bias_flags:
                console.warning(f"Bias detected: {flag.bias_type.value} ({flag.severity}) - {flag.description}")
            self.state.artifacts["bias_flags"] = [
                {"type": bf.bias_type.value, "description": bf.description, "severity": bf.severity}
                for bf in bias_flags
            ]

        # Step 3b: Bias-reactive corrections
        if bias_flags:
            for flag in bias_flags:
                flag_data = {"type": getattr(flag.bias_type, 'value', str(flag.bias_type)),
                             "severity": flag.severity}
                if flag.severity == "high":
                    if "community" in flag_data["type"].lower():
                        # Downweight single-source findings
                        for p in validated:
                            sources = p.get("sources", [])
                            if len(sources) <= 1:
                                old_score = p.get("score", 0)
                                p["score"] = old_score * 0.7
                                console.info(f"Downweighted single-source finding: {p.get('description', '')}")
                    if "recency" in flag_data["type"].lower():
                        console.warning("Recency bias detected: all findings from last 7 days. Consider historical context.")
            self.state.artifacts["discovered_problems"] = validated

        # Step 3b+: Mandatory opposing-viewpoint search when high-severity bias detected
        high_bias_flags = [f for f in bias_flags if getattr(f, 'severity', '') == "high"]
        if high_bias_flags and validated:
            top_desc = validated[0].get("description", "")
            if top_desc:
                console.info(
                    f"High-severity bias detected ({len(high_bias_flags)} flags). "
                    "Running mandatory opposing-viewpoint search..."
                )
                try:
                    from research.web_search import WebSearch as _WS
                    _counter_searcher = _WS()
                    _opposing_queries = [
                        f"why {top_desc[:70]} is not a real problem",
                        f"{top_desc[:60]} criticism overhyped",
                        f"problems with {top_desc[:60]} solution",
                    ]
                    _opposing_results = []
                    for _q in _opposing_queries[:2]:
                        _res = await _counter_searcher.search_duckduckgo(_q, max_results=3, recency_days=90)
                        _opposing_results.extend(_res)
                    if _opposing_results:
                        self.state.artifacts["opposing_viewpoints"] = [
                            {"title": r.title, "snippet": r.snippet, "url": r.url}
                            for r in _opposing_results[:6]
                        ]
                        console.info(
                            f"Opposing viewpoints: {len(_opposing_results)} results found to balance bias"
                        )
                except Exception as _e:
                    console.warning(f"Opposing viewpoint search failed: {_e}")

        # Step 3c: Counter-evidence search for top problems
        try:
            problem_objects = []
            for p in validated[:3]:
                try:
                    prob = DiscoveredProblem(
                        id=p.get("id", ""),
                        description=p.get("description", ""),
                        severity=ProblemSeverity(p.get("severity", "medium")),
                        score=p.get("score", 0.0),
                        keywords=p.get("keywords", []),
                        sources=p.get("sources", []),
                    )
                    problem_objects.append(prob)
                except Exception as e:
                    console.warning(f"Skipping malformed problem in data analysis phase: {e}")
                    continue
            if problem_objects:
                counter_evidence = await self.problem_discoverer.discover_counter_evidence(problem_objects, top_n=3)
                if counter_evidence:
                    self.state.artifacts["counter_evidence"] = counter_evidence
                    for ce in counter_evidence:
                        if ce.get("counter_evidence"):
                            console.info(f"Counter-evidence for '{ce['description'][:50]}': {len(ce['counter_evidence'])} items")
        except Exception as e:
            console.warning(f"Counter-evidence search failed: {e}")

        # Step 4: Use DataAnalyst agent for LLM-powered credibility scoring
        console.agent_action("DataAnalyst", "Scoring Credibility", "LLM-powered credibility assessment...")
        credibility_task = {
            "type": "score_credibility",
            "findings": str(validated[:5]),  # Top 5 for LLM analysis (token-efficient)
            "sources": list({s for p in validated for s in p.get("sources", [])})
        }
        cred_result = await self.data_analyst.execute_task(credibility_task)
        self.state.artifacts["credibility_analysis"] = cred_result.output

        # Re-rank problems by final composite score (bias + freshness adjustments applied)
        if validated:
            validated.sort(key=lambda p: p.get("score", 0.0), reverse=True)
            self.state.artifacts["discovered_problems"] = validated
            top = validated[0]
            top_score = top.get("score", 0.0)
            console.info(f"Final top problem score after full analysis: {top_score:.3f}")
            # Update current_problem if re-ranking changed which problem is top
            if self.state.current_problem and top.get("id") != self.state.current_problem.id:
                console.info(f"Top problem changed after scoring: '{top.get('description', '')[:60]}...'")
                try:
                    new_top = DiscoveredProblem(
                        id=top.get("id", ""),
                        description=top.get("description", ""),
                        severity=ProblemSeverity(top.get("severity", "medium")),
                        score=top_score,
                        keywords=top.get("keywords", []),
                        sources=top.get("sources", []),
                    )
                    self.state.current_problem = new_top
                    console.success(f"Current problem updated to highest-scored: {new_top.description[:80]}")
                except Exception as e:
                    console.warning(f"Could not update current problem after re-ranking: {e}")

        console.success(f"Data analysis complete: {len(validated)} problems, {len(bias_flags)} bias flags")

    # ============================================================
    #  PHASE 6: DESIGN REVIEW
    # ============================================================

    async def _run_design_review_phase(self) -> None:
        """Quick meeting: CTO + Developer + QA review architecture before coding."""
        self.state.phase = WorkflowPhase.DESIGN_REVIEW
        self.structured_logger.info("Design review phase started")

        architecture = self.state.artifacts.get("architecture", {})
        requirements = self.state.artifacts.get("requirements", {})
        problem = self.state.current_problem

        if not self.enable_meetings:
            console.info("Meetings disabled — skipping design review")
            self.state.artifacts["design_review"] = {"decision": "skipped", "meetings_disabled": True}
            return

        review_participants = [
            {
                "name": "CTO",
                "role": "Architecture Author",
                "model": self.model_config.get_model_name("cto")
            },
            {
                "name": "Developer",
                "role": "Implementation Feasibility",
                "model": self.model_config.get_model_name("developer")
            },
            {
                "name": "QAEngineer",
                "role": "Testability Review",
                "model": self.model_config.get_model_name("qa_engineer")
            }
        ]

        review_context = f"""
Problem: {problem.description if problem else 'N/A'}
Architecture: {str(architecture)}
Requirements: {str(requirements)}

Review this architecture BEFORE implementation begins. Focus on:
1. Is the design implementable with available resources?
2. Are there obvious design flaws or missing components?
3. Is the design testable? Can QA validate it effectively?
4. Are there simpler alternatives that achieve the same goal?
"""
        review_outcome = await quick_meeting(
            topic="Design Review: Architecture Feasibility Check",
            participants=review_participants,
            meeting_type=MeetingType.REVIEW,
            context=review_context
        )

        self.state.artifacts["design_review"] = {
            "decision": review_outcome.decision,
            "votes": review_outcome.votes,
            "action_items": review_outcome.action_items
        }

        if review_outcome.decision == "needs_revision":
            console.warning("Design review flagged issues. CTO revising architecture...")
            # CTO revises based on feedback
            revise_task = {
                "type": "design_architecture",
                "problem": problem.to_dict() if problem else {},
                "requirements": str(requirements),
                "constraint": f"Design review feedback: {'; '.join(review_outcome.action_items[:3])}. Address these concerns."
            }
            revised = await self.cto.execute_task(revise_task)
            self.state.artifacts["architecture"] = revised.output
            console.success("Architecture revised based on design review feedback")
        else:
            console.success("Design review passed")

    async def _run_security_review_phase(self) -> bool:
        """Security Engineer reviews the solution code for vulnerabilities."""
        self.state.phase = WorkflowPhase.SECURITY_REVIEW
        self.structured_logger.info("Security review phase started")

        implementation = self.state.artifacts.get("implementation", {})
        project_dir = self.state.artifacts.get("project_name", self.output_dir)

        # Collect code files for review
        files = {}
        project_path = Path(project_dir)
        if project_path.exists():
            py_files = self._filter_project_files(list(project_path.rglob("*.py")))
            for f in py_files[:10]:  # Limit to 10 files for token efficiency
                try:
                    files[str(f.relative_to(project_path))] = f.read_text()
                except Exception:
                    pass

        # Security review
        console.agent_action("SecurityEngineer", "Security Review", f"Reviewing {len(files)} files...")
        review_task = {
            "type": "security_review",
            "code": str(implementation),
            "files": files,
            "file_path": project_dir,
            "context": f"Problem: {self.state.current_problem.description if self.state.current_problem else 'N/A'}"
        }
        review_result = await self.security_engineer.execute_task(review_task)
        self.state.artifacts["security_review"] = review_result.output
        # Track SecurityEngineer experience and pool performance
        if hasattr(self.security_engineer, '_experience'):
            self.security_engineer._experience.add_experience("security_review", review_result.success)
        self.agent_pool.record_task_outcome("SecurityEngineer", review_result.success)
        self.performance_tracker.record_task(
            "SecurityEngineer", success=review_result.success,
            response_time_ms=review_result.execution_time * 1000, tokens=0
        )
        # Reflect on Security review (learning)
        if self.enable_learning:
            self.security_engineer.reflect(
                "security_review",
                str(review_result.output),
                review_result.success
            )

        # Parse verdict
        output = review_result.output if isinstance(review_result.output, dict) else {}
        verdict = output.get("verdict", "has_issues")
        critical_count = output.get("critical_count", 0)

        console.info(f"Security verdict: {verdict.upper()}, {critical_count} critical vulnerabilities")

        if verdict == "critical_vulnerabilities":
            console.error("Critical security vulnerabilities found! Must be fixed before deployment.")
            # Feed security issues to Developer for fixing
            project_dir = self.state.artifacts.get("project_name", self.output_dir)
            sec_fix_task = {
                "type": "fix_bug",
                "bug_description": f"Critical security vulnerabilities found. Fix these: {output.get('review', '')}",
                "error_message": "",
                "qa_report": "",
                "original_requirements": str(self.state.artifacts.get("requirements", "")),
                "output_dir": project_dir
            }
            fix_result = await self.developer.execute_task(sec_fix_task)
            self.state.artifacts["security_fix"] = fix_result.output
            console.success("Developer applied security fixes")

            # Re-run security review
            review_result = await self.security_engineer.execute_task(review_task)
            self.state.artifacts["security_review"] = review_result.output
            output = review_result.output if isinstance(review_result.output, dict) else {}
            verdict = output.get("verdict", "has_issues")
            if verdict == "critical_vulnerabilities":
                return False  # Still has critical issues
            console.success("Security issues resolved after fix")

        return verdict != "critical_vulnerabilities"

    async def _run_retrospective_phase(self) -> None:
        """All agents reflect on the workflow run - what worked, what didn't, what to improve."""
        self.state.phase = WorkflowPhase.RETROSPECTIVE
        self.structured_logger.info("Retrospective phase started")

        if not self.enable_meetings:
            console.info("Meetings disabled — skipping retrospective")
            self.state.artifacts["retrospective"] = {"decision": "skipped", "meetings_disabled": True}
            return

        # Gather workflow summary for context
        status = self.state.phase.value
        duration = None
        if self.state.started_at:
            duration = (datetime.now() - self.state.started_at).total_seconds()

        duration_str = f"{duration:.0f}s" if duration else "N/A"
        retro_context = f"""
Workflow Status: {status}
Duration: {duration_str}
Problem: {self.state.current_problem.description if self.state.current_problem else 'N/A'}
Decisions Made: {len(self.state.decisions)}
Phases Completed: {', '.join(k for k, v in self.state.artifacts.items() if v)}
Error: {self.state.error or 'None'}
"""

        retro_participants = [
            {"name": "CEO", "role": "Strategic Leadership", "model": self.model_config.get_model_name("ceo")},
            {"name": "CTO", "role": "Technical Leadership", "model": self.model_config.get_model_name("cto")},
            {"name": "ProductManager", "role": "Product Strategy", "model": self.model_config.get_model_name("product_manager")},
            {"name": "Developer", "role": "Implementation", "model": self.model_config.get_model_name("developer")},
        ]

        retro_outcome = await quick_meeting(
            topic="Workflow Retrospective: What worked, what didn't, what to improve?",
            participants=retro_participants,
            meeting_type=MeetingType.RETROSPECTIVE,
            context=retro_context
        )

        self.state.artifacts["retrospective"] = {
            "decision": retro_outcome.decision,
            "key_insights": retro_outcome.key_insights,
            "action_items": retro_outcome.action_items
        }

        # Store learnings in shared memory
        self.shared_memory.store_meeting(
            topic="Workflow Retrospective",
            participants=[p["name"] for p in retro_participants],
            summary=retro_outcome.decision,
            action_items=retro_outcome.action_items
        )

        # Activate learning system analysis for each agent
        from memory.shared_memory import MemoryType
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'learning'):
                perf_analysis = agent.learning.analyze_performance()
                if perf_analysis.get("status") != "no_data":
                    suggestions = agent.learning.get_improvement_suggestions()
                    if suggestions:
                        summary = agent.learning.export_learning_summary()
                        console.info(
                            f"Learning: {agent_name} — "
                            f"{perf_analysis.get('total_lessons', 0)} lessons, "
                            f"{perf_analysis.get('success_rate', 0):.0%} success"
                        )
                        self.shared_memory.store(
                            content=f"Learning for {agent_name}: {summary}",
                            memory_type=MemoryType.INSIGHT,
                            created_by=agent_name,
                            tags=["learning", "retrospective", agent_name],
                            importance=0.6,
                        )

        console.success(f"Retrospective complete: {len(retro_outcome.key_insights)} insights, {len(retro_outcome.action_items)} action items")

        # HR review: fire underperformers, log hiring needs
        underperformers = self.agent_pool.get_underperformers(threshold=0.3, min_tasks=3)
        for under in underperformers:
            self.agent_pool.fire_agent(
                under.name,
                reason=(
                    f"Performance score {under.performance_score:.0%} "
                    f"({under.tasks_failed} failures / {under.tasks_completed + under.tasks_failed} tasks) "
                    "below 30% threshold after retrospective review"
                )
            )
            # Log hiring need to backfill the role
            self.agent_pool.hire_agent(
                under.role,
                reason=f"Backfill after {under.name} termination — maintaining team capacity"
            )
        if underperformers:
            console.info(
                f"HR action: {len(underperformers)} agent(s) terminated and backfilled"
            )

        # Persist action items as next-run constraints in strategy memory
        action_items = retro_outcome.action_items if retro_outcome.action_items else []
        if action_items:
            constraints_text = "RETROSPECTIVE CONSTRAINTS (from last run): " + "; ".join(str(a) for a in action_items[:5])
            problem_key = (
                self.state.current_problem.description[:80]
                if self.state.current_problem else "general"
            )
            self.escalation_manager.strategy_memory.record_attempt(
                problem=problem_key,
                approach="retrospective_constraints",
                outcome="action_items",
                details=constraints_text,
                success=True
            )
            console.info(f"Stored {len(action_items)} retrospective action items as next-run constraints")

        # Write structured session summary JSON to disk for cross-run learning
        import json as _json
        summary_path = Path(self.data_dir) / "session_summary.json"
        session_summary = {
            "timestamp": datetime.now().isoformat(),
            "problem": self.state.current_problem.description if self.state.current_problem else "N/A",
            "duration_s": duration,
            "phases_completed": list(self.state.artifacts.keys()),
            "decisions": self.state.decisions[-20:],  # last 20 decisions
            "action_items": action_items,
            "key_insights": retro_outcome.key_insights if retro_outcome.key_insights else [],
            "error": self.state.error,
        }
        try:
            summary_path.write_text(_json.dumps(session_summary, indent=2, default=str))
            console.info(f"Session summary saved → {summary_path}")
        except Exception as _e:
            console.warning(f"Could not save session summary: {_e}")

    def _save_checkpoint(self) -> None:
        """Save workflow state checkpoint after each phase."""
        checkpoint_dir = Path(self.data_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "phase": self.state.phase.value,
            "problem": self.state.current_problem.to_dict() if self.state.current_problem else None,
            "artifacts": {},
            "decisions": self.state.decisions,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "error": self.state.error,
        }

        # Filter artifacts to only JSON-serializable values
        for key, value in self.state.artifacts.items():
            try:
                json.dumps(value)
                checkpoint["artifacts"][key] = value
            except (TypeError, ValueError):
                checkpoint["artifacts"][key] = str(value)

        checkpoint_file = checkpoint_dir / "latest.json"
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            # Also save per-session checkpoint so older checkpoints aren't lost
            if self.session_manager.current_session:
                run_file = checkpoint_dir / f"run_{self.session_manager.current_session.id}.json"
                with open(run_file, "w") as f:
                    json.dump(checkpoint, f, indent=2)
        except OSError as e:
            console.warning(f"Could not save checkpoint: {e}")

        # Also save session state with a checkpoint marker
        if self.session_manager.current_session:
            session = self.session_manager.current_session
            session.workflow_state = checkpoint
            session.create_checkpoint(
                name=self.state.phase.value,
                metadata={"phase": self.state.phase.value}
            )
            self.session_manager.save_current()

    def _load_checkpoint(self) -> bool:
        """Load workflow from last checkpoint. Returns True if loaded."""
        checkpoint_file = Path(self.data_dir) / "checkpoints" / "latest.json"
        if not checkpoint_file.exists():
            return False

        try:
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)

            self.state.phase = WorkflowPhase(checkpoint["phase"])
            self.state.artifacts = checkpoint.get("artifacts", {})
            self.state.decisions = checkpoint.get("decisions", [])

            if checkpoint.get("started_at"):
                self.state.started_at = datetime.fromisoformat(checkpoint["started_at"])

            if checkpoint.get("problem"):
                prob_data = checkpoint["problem"]
                self.state.current_problem = DiscoveredProblem(
                    id=prob_data["id"],
                    description=prob_data["description"],
                    severity=ProblemSeverity(prob_data["severity"]),
                    frequency=ProblemFrequency(prob_data.get("frequency", "common")),
                    target_users=prob_data.get("target_users", ""),
                    evidence=prob_data.get("evidence", []),
                    sources=prob_data.get("sources", []),
                    domain=prob_data.get("domain", "software"),
                    keywords=prob_data.get("keywords", []),
                    potential_solution_ideas=prob_data.get("potential_solution_ideas", []),
                    validation_status=prob_data.get("validation_status", "unvalidated"),
                    score=prob_data.get("score", 0.0),
                    metadata=prob_data.get("metadata", {}),
                )

            self.state.error = checkpoint.get("error")
            return True
        except Exception as e:
            console.warning(f"Could not load checkpoint: {e}")
            return False

    def _clear_checkpoint(self) -> None:
        """Clear checkpoint after successful completion."""
        checkpoint_file = Path(self.data_dir) / "checkpoints" / "latest.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    # ============================================================
    #  RESULT PERSISTENCE
    # ============================================================

    def _persist_result(self, result: Dict[str, Any]) -> None:
        """Persist workflow result to history database (JSON file)."""
        results_file = Path(self.data_dir) / "results" / "workflow_history.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results
        history = []
        if results_file.exists():
            try:
                with open(results_file) as f:
                    history = json.load(f)
            except Exception as e:
                console.warning(f"Could not load workflow history: {e}")

        # Append new result (with serializable data only)
        entry = {
            "id": len(history) + 1,
            "status": result.get("status"),
            "problem": result.get("problem", {}).get("description", "") if result.get("problem") else "",
            "domain": result.get("problem", {}).get("domain", "") if result.get("problem") else "",
            "language": result.get("artifacts", {}).get("language", "python"),
            "started_at": result.get("started_at"),
            "completed_at": result.get("completed_at"),
            "duration_seconds": result.get("duration_seconds"),
            "decisions": result.get("decisions", []),
            "delivery_dir": result.get("artifacts", {}).get("delivery", {}).get("project_dir", ""),
        }
        history.append(entry)

        try:
            with open(results_file, "w") as f:
                json.dump(history, f, indent=2)
        except OSError as e:
            console.warning(f"Could not write workflow history: {e}")

    def _get_workflow_result(self) -> Dict[str, Any]:
        """Get the complete workflow result."""
        duration = None
        if self.state.started_at and self.state.completed_at:
            duration = (self.state.completed_at - self.state.started_at).total_seconds()

        # Collect KPIs for all agents
        all_kpis = {}
        for agent_name in self.agents:
            kpis = self.performance_tracker.get_kpis(agent_name)
            if kpis.tasks_completed + kpis.tasks_failed > 0:
                all_kpis[agent_name] = {
                    "score": kpis.calculate_score(),
                    "tasks_completed": kpis.tasks_completed,
                    "tasks_failed": kpis.tasks_failed,
                    "approval_rate": round(kpis.approval_rate, 2),
                    "rework_rate": round(kpis.rework_rate, 2),
                }

        # Persist agent experience to disk
        for agent_name, agent in self.agents.items():
            if hasattr(agent, '_experience'):
                exp_path = self._experience_dir / f"{agent_name}.json"
                agent._experience.save(exp_path)

        # Persist performance, trust, and strategy trackers to disk
        perf_path = Path(self.data_dir) / "performance" / "kpis.json"
        self.performance_tracker.save(perf_path)
        trust_path = Path(self.data_dir) / "trust" / "trust_scores.json"
        self.trust_tracker.save(trust_path)
        strat_path = Path(self.data_dir) / "escalation" / "strategy_memory.json"
        self.escalation_manager.save_strategy_memory(strat_path)

        # Log KPIs
        if all_kpis:
            console.info("Agent KPI Summary:")
            for name, kpi_data in all_kpis.items():
                console.info(f"  {name}: score={kpi_data['score']:.1f}, "
                             f"approval={kpi_data['approval_rate']:.0%}")

        return {
            "status": self.state.phase.value,
            "problem": self.state.current_problem.to_dict() if self.state.current_problem else None,
            "decisions": self.state.decisions,
            "artifacts": self.state.artifacts,
            "error": self.state.error,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "completed_at": self.state.completed_at.isoformat() if self.state.completed_at else None,
            "duration_seconds": duration,
            "agent_kpis": all_kpis,
            "trust_scores": self.trust_tracker.get_all_scores(),
        }

    # ============================================================
    #  INTERACTIVE APPROVAL
    # ============================================================

    def _interactive_gate(self, phase_name: str, details: str = "") -> bool:
        """Pause for user approval in interactive mode. Returns True to continue, False to abort."""
        if not getattr(self, '_interactive', False):
            return True

        from ui.interactive import AskUserQuestion, QuestionType

        asker = AskUserQuestion(color_enabled=True)
        prompt = f"Phase complete: {phase_name}"
        if details:
            prompt += f"\n{details}"
        prompt += "\nContinue to next phase?"

        result = asker.ask(
            question=prompt,
            question_type=QuestionType.YES_NO,
            header="Approval",
            default="y",
        )

        if result.skipped or result.timed_out:
            return False

        if not result.answer:
            console.warning("Workflow aborted by user")
            return False

        return True

    # ============================================================
    #  COMPLETION SUMMARY
    # ============================================================

    def _print_scaffold_summary(self, result: Dict[str, Any]) -> None:
        """Print the scaffold delivery summary with TODO counts and next steps."""
        delivery = self.state.artifacts.get("delivery", {})
        project_dir = delivery.get("project_dir", "")
        files = delivery.get("files", [])

        # Count # TODO occurrences in output directory
        todo_count = 0
        if project_dir and Path(project_dir).exists():
            for p in Path(project_dir).rglob("*"):
                if p.is_file() and p.suffix in (".py", ".js", ".ts", ".md"):
                    try:
                        todo_count += p.read_text(encoding="utf-8", errors="ignore").count("# TODO")
                    except Exception:
                        pass

        border = "\u2501" * 40
        console.info(border)
        console.info(" SCAFFOLD GENERATED \u2014 HUMAN REVIEW NEEDED")
        console.info(border)
        console.info(f"Files generated: {len(files)}")
        console.info(f"TODOs to complete: {todo_count}")
        console.info("Next steps:")
        console.info(f"  1. Review generated files in {project_dir or '(output dir)'}")
        console.info("  2. Complete all # TODO sections")
        console.info("  3. Run tests and fix failures")
        console.info("  4. Remove draft headers when production-ready")
        console.info(border)

    def _print_completion_summary(self, result: Dict[str, Any]) -> None:
        """Print a clear summary after successful workflow completion."""
        delivery = self.state.artifacts.get("delivery", {})
        project_dir = delivery.get("project_dir", "")
        files = delivery.get("files", [])
        language = delivery.get("language", "python")
        problem_desc = ""
        if self.state.current_problem:
            problem_desc = self.state.current_problem.description

        # If scaffold mode, print the specialised scaffold summary instead
        if self.scaffold_mode:
            self._print_scaffold_summary(result)
            return

        if not project_dir or not files:
            console.section("WORKFLOW COMPLETE")
            console.warning("Delivery artifacts missing - solution may be incomplete.")
            console.info(f"  Status: {result.get('status', 'unknown').upper()}")
            if problem_desc:
                console.info(f"  Problem: {problem_desc}")
            return

        duration = result.get("duration_seconds")
        duration_str = f"{duration:.0f}s" if duration else "N/A"

        cost_summary = self.state.artifacts.get("cost_summary", {})
        total_tokens = cost_summary.get("total_tokens", 0) if isinstance(cost_summary, dict) else 0

        console.section("WORKFLOW COMPLETE")
        lines = [
            f"  Problem:    {problem_desc}",
            f"  Status:     {result.get('status', 'unknown').upper()}",
            f"  Duration:   {duration_str}",
            f"  Tokens:     {total_tokens:,}" if total_tokens else None,
            f"  Project:    {project_dir}",
            f"  Files:      {len(files)}",
        ]
        for line in lines:
            if line:
                console.info(line)

        # Show file tree (top-level only, grouped)
        if files:
            console.info("\n  File tree:")
            # Show up to 20 files
            for f in sorted(files)[:20]:
                console.info(f"    {f}")
            if len(files) > 20:
                console.info(f"    ... and {len(files) - 20} more files")

        # Show how to run
        console.info("\n  Quick start:")
        if language == "python":
            console.info(f"    cd {project_dir}")
            if any("requirements.txt" in f for f in files):
                console.info("    pip install -r requirements.txt")
            if any("main.py" in f for f in files):
                console.info("    python main.py")
            elif any("app.py" in f for f in files):
                console.info("    python app.py")
            if any("test" in f.lower() for f in files):
                console.info("    pytest")
        elif language in ("javascript", "typescript"):
            console.info(f"    cd {project_dir}")
            if any("package.json" in f for f in files):
                console.info("    npm install")
                console.info("    npm start")
        elif language == "go":
            console.info(f"    cd {project_dir}")
            console.info("    go run .")
        elif language == "rust":
            console.info(f"    cd {project_dir}")
            console.info("    cargo run")
        else:
            console.info(f"    cd {project_dir}")

        # Show Docker hint if Dockerfile was generated
        if any("Dockerfile" in f for f in files):
            console.info("\n  Docker:")
            console.info(f"    docker build -t solution {project_dir}")
            console.info("    docker run solution")

        console.info("")

    # Directories to skip when scanning for project source files
    _IGNORED_DIRS = {
        "__pycache__", ".pytest_cache", ".venv", "venv", "env",
        "node_modules", ".git", "dist", "build", ".mypy_cache",
        ".ruff_cache", ".tox", ".eggs",
    }
    # Suffixes that mark directories to skip (e.g., package.egg-info)
    _IGNORED_SUFFIXES = (".egg-info",)

    def _filter_project_files(self, files: list) -> list:
        """Filter out files inside ignored directories."""
        return [
            f for f in files
            if not any(
                part in self._IGNORED_DIRS
                or part.endswith(self._IGNORED_SUFFIXES)
                for part in f.parts
            )
        ]

    # ============================================================
    #  PHASE 8: CODE EXECUTION
    # ============================================================

    async def _run_code_execution(self) -> Dict[str, Any]:
        """Run generated code to catch runtime errors."""
        import os
        import subprocess


        self.state.phase = WorkflowPhase.CODE_EXECUTION
        self.structured_logger.info("Code execution phase started")

        # Track plan step (step_2 = Code Execution)
        if self.plan_manager.current_plan:
            self.plan_manager.start_step("step_2")

        project_dir = self.state.artifacts.get("project_name", self.output_dir)
        language = self.state.artifacts.get("language", "python")

        results = {
            "success": True,
            "syntax_errors": [],
            "runtime_errors": [],
            "test_output": "",
            "summary": ""
        }

        # Only support Python execution for now
        if language != "python":
            results["summary"] = f"Code execution not yet supported for {language}"
            return results

        project_path = Path(project_dir)
        if not project_path.exists():
            results["success"] = False
            results["summary"] = f"Project directory not found: {project_dir}"
            return results

        # Step 1: Find all Python files (excluding caches, venvs, etc.)
        py_files = self._filter_project_files(list(project_path.rglob("*.py")))
        if not py_files:
            results["success"] = False
            results["summary"] = "No Python files found in project"
            return results

        console.info(f"Found {len(py_files)} Python files to check")

        # Step 2: Syntax check all files
        for py_file in py_files:
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(py_file)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    error = result.stderr.strip() or result.stdout.strip()
                    results["syntax_errors"].append(f"{py_file.name}: {error}")
                    console.error(f"Syntax error in {py_file.name}")
            except subprocess.TimeoutExpired:
                console.warning(f"Syntax check timed out for {py_file.name}")

        if results["syntax_errors"]:
            results["success"] = False
            results["summary"] = f"Syntax errors in {len(results['syntax_errors'])} files"
            return results

        console.success("All files pass syntax check")

        # Step 2b: Install dependencies if requirements.txt exists
        req_file = project_path / "requirements.txt"
        if req_file.exists():
            console.info("Installing dependencies from requirements.txt...")
            try:
                result = subprocess.run(
                    ["python3", "-m", "pip", "install", "-q",
                     "--target", str(project_path / ".deps"),
                     "-r", str(req_file)],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    console.success("Dependencies installed")
                else:
                    console.warning(f"Some dependencies failed to install: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                console.warning("Could not install dependencies (pip unavailable or timed out)")

        # Build PYTHONPATH so import checks find installed deps
        extra_paths = []
        deps_dir = project_path / ".deps"
        if deps_dir.exists():
            extra_paths.append(str(deps_dir))
        env = dict(os.environ)
        if extra_paths:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))

        # Step 3: Try running entry point to catch import/runtime errors
        entry_points = ["main.py", "app.py", "cli.py", "__main__.py"]
        entry_file = None
        for ep in entry_points:
            # Prefer top-level file, fall back to nested
            top_level = project_path / ep
            if top_level.exists():
                entry_file = top_level
                break
            nested = self._filter_project_files(list(project_path.rglob(ep)))
            if nested:
                entry_file = nested[0]
                break

        if entry_file:
            console.info(f"Testing entry point: {entry_file.name} (sandboxed)")
            try:
                from utils.sandbox import (SandboxExecutor,
                                           create_development_sandbox)
                sandbox_config = create_development_sandbox(str(project_path))
                sandbox_config.timeout = 15
                sandbox = SandboxExecutor(sandbox_config)

                # Build PYTHONPATH for sandbox env
                sandbox_cmd = (
                    f'PYTHONPATH="{":".join(extra_paths)}" '
                    if extra_paths else ""
                )
                sandbox_cmd += (
                    f'python3 -c "'
                    f"import sys, importlib.util; "
                    f"spec = importlib.util.spec_from_file_location('mod', '{entry_file}'); "
                    f"mod = importlib.util.module_from_spec(spec); "
                    f'spec.loader.exec_module(mod)"'
                )

                sandbox_result = await sandbox.execute(sandbox_cmd)
                sandbox.cleanup()

                if sandbox_result.blocked:
                    error = f"Sandbox blocked: {sandbox_result.block_reason}"
                    results["runtime_errors"].append(error)
                    results["success"] = False
                    console.warning(f"Sandbox blocked execution of {entry_file.name}: {sandbox_result.block_reason}")
                elif not sandbox_result.success:
                    error = sandbox_result.stderr.strip()
                    results["runtime_errors"].append(error)
                    results["success"] = False
                    console.error(f"Runtime error in {entry_file.name}")
                else:
                    console.success(f"Entry point {entry_file.name} loads without errors")
            except Exception as sandbox_err:
                # Fallback to direct execution if sandbox fails
                console.warning(f"Sandbox unavailable ({sandbox_err}), falling back to direct execution")
                try:
                    result = subprocess.run(
                        ["python3", "-c",
                         "import sys, importlib.util; "
                         "spec = importlib.util.spec_from_file_location('mod', sys.argv[1]); "
                         "mod = importlib.util.module_from_spec(spec); "
                         "spec.loader.exec_module(mod)",
                         str(entry_file)],
                        capture_output=True, text=True, timeout=15,
                        cwd=str(project_path), env=env
                    )
                    if result.returncode != 0:
                        error = result.stderr.strip()
                        results["runtime_errors"].append(error)
                        results["success"] = False
                        console.error(f"Runtime error in {entry_file.name}")
                    else:
                        console.success(f"Entry point {entry_file.name} loads without errors")
                except subprocess.TimeoutExpired:
                    results["runtime_errors"].append(f"{entry_file.name}: Timed out (15s)")
                    results["success"] = False

        # Step 4: Run tests using the multi-language TestRunner
        from tools.test_runner import TestRunner
        test_runner = TestRunner(workspace_root=str(project_path), timeout=60)
        detected_framework = test_runner.detect_framework(str(project_path))
        if detected_framework:
            console.info(f"Running tests with {detected_framework.value}...")
            test_result = test_runner.run_tests(str(project_path), framework=detected_framework)
            results["test_output"] = test_result.output + test_result.error_output
            results["test_details"] = {
                "passed": test_result.passed,
                "failed": test_result.failed,
                "skipped": test_result.skipped,
                "errors": test_result.errors,
                "total": test_result.total,
                "framework": test_result.framework,
            }
            if not test_result.success:
                results["success"] = False
                console.warning(f"Tests failed: {test_result.failed} failed, {test_result.errors} errors")
                if results["test_output"]:
                    console.warning(f"Output:\n{results['test_output']}")
            else:
                console.success(f"All tests passed ({test_result.passed} passed)")
        else:
            # Fallback: check for test files manually (no framework detected)
            test_files = [f for f in py_files
                          if f.name.startswith("test_") or f.name.endswith("_test.py")]
            if test_files:
                console.info(f"Running {len(test_files)} test files with pytest...")
                try:
                    result = subprocess.run(
                        ["python3", "-m", "pytest", "-x", "--tb=short", "-q",
                         str(project_path)],
                        capture_output=True, text=True, timeout=60,
                        cwd=str(project_path), env=env
                    )
                    results["test_output"] = result.stdout + result.stderr
                    if result.returncode != 0:
                        results["success"] = False
                        console.warning(f"Tests failed:\n{results['test_output']}")
                    else:
                        console.success("All tests passed")
                except subprocess.TimeoutExpired:
                    results["test_output"] = "pytest timed out after 60s"
                    results["success"] = False
                except FileNotFoundError:
                    results["test_output"] = "pytest not installed"

        # Build summary
        has_tests = bool(results.get("test_output") or results.get("test_details"))
        if results["success"]:
            results["summary"] = "All checks passed: syntax OK, imports OK" + (
                ", tests passed" if has_tests else "")
        else:
            parts = []
            if results["syntax_errors"]:
                parts.append(f"Syntax errors: {'; '.join(results['syntax_errors'])}")
            if results["runtime_errors"]:
                parts.append(f"Runtime errors: {'; '.join(results['runtime_errors'])}")
            if results["test_output"]:
                parts.append(f"Test failures: {results['test_output']}")
            results["summary"] = " | ".join(parts)

        # Code quality metrics (static, no LLM call)
        try:
            todo_count = 0
            long_funcs = 0
            hardcoded_count = 0
            total_lines = 0
            import re as _re
            _hardcoded_pat = _re.compile(r'(?<![#\'""])["\'][/~][^"\']{5,}["\']|(?<!\w)(\d{1,3}\.){3}\d{1,3}(?!\w)')
            for py_file in py_files:
                try:
                    text = py_file.read_text(errors="replace")
                    lines = text.splitlines()
                    total_lines += len(lines)
                    todo_count += sum(1 for l in lines if "TODO" in l or "FIXME" in l or "STUB" in l)
                    # Count functions >30 lines (simple heuristic)
                    in_func = False
                    func_start = 0
                    for i, l in enumerate(lines):
                        if l.strip().startswith("def ") or l.strip().startswith("async def "):
                            if in_func and (i - func_start) > 30:
                                long_funcs += 1
                            in_func = True
                            func_start = i
                    if in_func and (len(lines) - func_start) > 30:
                        long_funcs += 1
                    hardcoded_count += len(_hardcoded_pat.findall(text))
                except Exception:
                    pass
            # Quality score: starts at 1.0, deducted for issues
            quality_score = 1.0
            if todo_count > 0:
                quality_score -= min(0.3, todo_count * 0.05)
            if long_funcs > 0:
                quality_score -= min(0.2, long_funcs * 0.04)
            if hardcoded_count > 3:
                quality_score -= min(0.2, (hardcoded_count - 3) * 0.04)
            quality_score = max(0.0, round(quality_score, 2))
            results["quality_score"] = quality_score
            results["quality_details"] = {
                "todo_count": todo_count,
                "long_functions": long_funcs,
                "hardcoded_values": hardcoded_count,
                "total_lines": total_lines,
            }
            q_label = "EXCELLENT" if quality_score >= 0.9 else "GOOD" if quality_score >= 0.7 else "NEEDS_WORK"
            console.info(
                f"Code quality: {q_label} ({quality_score:.0%}) — "
                f"TODOs={todo_count}, long funcs={long_funcs}, hardcoded={hardcoded_count}"
            )
        except Exception as _qe:
            console.warning(f"Code quality check failed: {_qe}")
            results["quality_score"] = 0.5

        # Update plan step (step_2 = Code Execution)
        if self.plan_manager.current_plan:
            if results["success"]:
                self.plan_manager.complete_step("step_2", output=results["summary"])
            else:
                self.plan_manager.fail_step("step_2", error=results["summary"])

        return results

    def get_agent_statuses(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {name: agent.get_status() for name, agent in self.agents.items()}

    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        return {
            "phase": self.state.phase.value,
            "problem": self.state.current_problem.to_dict() if self.state.current_problem else None,
            "decisions": self.state.decisions,
            "error": self.state.error
        }

    def reset_workflow(self) -> None:
        """Reset workflow to initial state, clearing inter-run state."""
        if self.state.phase != WorkflowPhase.IDLE:
            self.workflow_history.append(self.state)
        self.state = WorkflowState()

        # Clear message bus and shared memory to prevent state pollution
        # between continuous mode runs
        self.message_bus.clear_history()
        self.message_bus.clear_all_queues()

        # Reinitialize progress tracker for the next run
        self.progress = ProgressTracker(
            metrics_file=Path(self.data_dir) / "metrics" / "phase_metrics.json"
        )
        self.progress.add_phases([
            {"id": "research", "name": "Research"},
            {"id": "data_analysis", "name": "Data Analysis"},
            {"id": "analysis", "name": "Analysis"},
            {"id": "opportunity", "name": "Opportunity Eval"},
            {"id": "design", "name": "Technical Design"},
            {"id": "design_review", "name": "Design Review"},
            {"id": "implementation", "name": "Implementation"},
            {"id": "code_execution", "name": "Code Execution"},
            {"id": "qa_validation", "name": "QA Validation"},
            {"id": "security_review", "name": "Security Review"},
            {"id": "ceo_approval", "name": "CEO Approval"},
            {"id": "delivery", "name": "Delivery"},
            {"id": "retrospective", "name": "Retrospective"},
        ])

        # Reset task manager for fresh tracking
        self.task_manager = TaskManager()

        # Clear current plan
        self.plan_manager.current_plan = None

        # Reset escalation manager for fresh tracking
        self.escalation_manager.reset()

        # Stop memory monitoring (will be restarted in next run)
        self.memory_monitor.stop_monitoring()

        # Reset structured logger context
        self.structured_logger.context = None

    async def run_continuous(
        self,
        delay_between_runs: int = 60,
        max_iterations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the workflow continuously, discovering and solving problems in a loop.

        Args:
            delay_between_runs: Seconds to wait between workflow runs
            max_iterations: Maximum number of iterations (None for infinite)

        Returns:
            List of all workflow results
        """
        import asyncio as _asyncio

        results = []
        iteration = 0

        console.section("CONTINUOUS MODE")
        console.info(f"Delay between runs: {delay_between_runs}s")
        if max_iterations:
            console.info(f"Max iterations: {max_iterations}")
        else:
            console.info("Running indefinitely (Ctrl+C to stop)")

        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            console.section(f"CONTINUOUS RUN #{iteration}")

            try:
                self.reset_workflow()
                result = await self.run_full_workflow(auto_discover=True)
                results.append(result)

                status = result.get("status", "unknown")
                console.info(f"Run #{iteration} completed with status: {status}")

                if max_iterations is None or iteration < max_iterations:
                    console.info(f"Waiting {delay_between_runs}s before next run...")
                    await _asyncio.sleep(delay_between_runs)

            except KeyboardInterrupt:
                console.info("Continuous mode stopped by user")
                break
            except Exception as e:
                console.error(f"Run #{iteration} failed: {e}")
                results.append({"status": "error", "error": str(e), "iteration": iteration})
                await _asyncio.sleep(delay_between_runs)

        console.section("CONTINUOUS MODE SUMMARY")
        console.info(f"Total runs: {len(results)}")
        completed = sum(1 for r in results if r.get("status") == "completed")
        console.info(f"Successful: {completed}/{len(results)}")

        return results
