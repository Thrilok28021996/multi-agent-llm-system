"""Configuration file loader for Company AGI."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

yaml = None
YAML_AVAILABLE = False
try:
    import yaml as yaml_module
    yaml = yaml_module
    YAML_AVAILABLE = True
except ImportError:
    pass

from ui.console import console


@dataclass
class WorkflowConfig:
    """Configuration for the workflow."""
    auto_discover: bool = True
    max_research_sources: int = 5
    enable_meetings: bool = True
    enable_learning: bool = True
    interactive_approval: bool = False
    token_budget: Optional[int] = None
    enhance_dir: str = ""


@dataclass
class OutputConfig:
    """Configuration for output."""
    solutions_dir: str = "output/solutions"
    logs_dir: str = "output/logs"
    reports_dir: str = "output/reports"
    format_code: bool = True
    run_tests: bool = True


@dataclass
class LLMConfig:
    """Configuration for LLM behavior."""
    temperature: float = 0.7
    max_tokens: int = 4096
    streaming: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class AgentModels:
    """Model assignments for each agent.

    Defaults match config/models.py MODEL_CONFIGS (optimized for 16GB RAM).
    """
    ceo: str = "qwen3-8b"
    cto: str = "qwen3-8b"
    product_manager: str = "ministral-8b"
    researcher: str = "ministral-8b"
    developer: str = "qwen2.5-coder-7b"
    qa_engineer: str = "qwen2.5-coder-7b"
    devops_engineer: str = "qwen2.5-coder-7b"
    data_analyst: str = "qwen3-8b"
    security_engineer: str = "qwen2.5-coder-7b"


@dataclass
class AppConfig:
    """Main application configuration."""
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    models: Optional[AgentModels] = None  # None = use ModelConfig defaults; set when config file loaded

    # Problem settings (can be overridden by CLI)
    problem: str = ""
    domain: str = "software"
    language: str = "python"


class ConfigLoader:
    """Loads configuration from YAML or JSON files."""

    DEFAULT_CONFIG_PATHS = [
        "config.yaml",
        "config.yml",
        "config.json",
        ".multi-agent-llm-company-system.yaml",
        ".multi-agent-llm-company-system.json",
    ]

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = AppConfig()

    def load(self) -> AppConfig:
        """Load configuration from file if it exists."""
        config_file = self._find_config_file()

        if config_file:
            console.info(f"Loading config from: {config_file}")
            data = self._load_file(config_file)
            self._apply_config(data)

        # Also check environment variables
        self._apply_env_vars()

        return self.config

    def _find_config_file(self) -> Optional[Path]:
        """Find the configuration file."""
        if self.config_path:
            path = Path(self.config_path)
            if path.exists():
                return path
            else:
                console.warning(f"Config file not found: {self.config_path}")
                return None

        # Check default paths
        for default_path in self.DEFAULT_CONFIG_PATHS:
            path = Path(default_path)
            if path.exists():
                return path

        return None

    def _load_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        content = path.read_text(encoding="utf-8")

        if path.suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE or yaml is None:
                console.warning("PyYAML not installed. Install with: pip install pyyaml")
                return {}
            return yaml.safe_load(content) or {}
        elif path.suffix == ".json":
            return json.loads(content)
        else:
            # Try YAML first, then JSON
            if YAML_AVAILABLE and yaml is not None:
                try:
                    return yaml.safe_load(content) or {}
                except Exception as e:
                    console.warning(f"YAML parse failed for {path}, trying JSON: {e}")
            try:
                return json.loads(content)
            except Exception:
                console.warning(f"Could not parse config file: {path}")
                return {}

    def _apply_config(self, data: Dict[str, Any]) -> None:
        """Apply configuration data to config object."""
        if not data:
            return

        # Workflow settings
        if "workflow" in data:
            wf = data["workflow"]
            self.config.workflow.auto_discover = wf.get("auto_discover", True)
            self.config.workflow.max_research_sources = wf.get("max_research_sources", 5)
            self.config.workflow.enable_meetings = wf.get("enable_meetings", True)
            self.config.workflow.enable_learning = wf.get("enable_learning", True)
            self.config.workflow.interactive_approval = wf.get("interactive_approval", False)
            self.config.workflow.token_budget = wf.get("token_budget", None)
            self.config.workflow.enhance_dir = wf.get("enhance_dir", "")

        # Output settings
        if "output" in data:
            out = data["output"]
            self.config.output.solutions_dir = out.get("solutions_dir", "output/solutions")
            self.config.output.logs_dir = out.get("logs_dir", "output/logs")
            self.config.output.reports_dir = out.get("reports_dir", "output/reports")
            self.config.output.format_code = out.get("format_code", True)
            self.config.output.run_tests = out.get("run_tests", True)

        # LLM settings
        if "llm" in data:
            llm = data["llm"]
            self.config.llm.temperature = llm.get("temperature", 0.7)
            self.config.llm.max_tokens = llm.get("max_tokens", 4096)
            self.config.llm.streaming = llm.get("streaming", True)
            self.config.llm.retry_attempts = llm.get("retry_attempts", 3)
            self.config.llm.retry_delay = llm.get("retry_delay", 1.0)

        # Model assignments - only set when config file specifies models
        if "models" in data:
            defaults = AgentModels()
            models = data["models"]
            self.config.models = AgentModels(
                ceo=models.get("ceo", defaults.ceo),
                cto=models.get("cto", defaults.cto),
                product_manager=models.get("product_manager", defaults.product_manager),
                researcher=models.get("researcher", defaults.researcher),
                developer=models.get("developer", defaults.developer),
                qa_engineer=models.get("qa_engineer", defaults.qa_engineer),
                devops_engineer=models.get("devops_engineer", defaults.devops_engineer),
                data_analyst=models.get("data_analyst", defaults.data_analyst),
                security_engineer=models.get("security_engineer", defaults.security_engineer),
            )

        # Problem settings
        self.config.problem = data.get("problem", "")
        self.config.domain = data.get("domain", "software")
        self.config.language = data.get("language", "python")

    def _apply_env_vars(self) -> None:
        """Apply environment variable overrides."""
        # Model overrides (create AgentModels if env vars are set)
        ceo_model = os.getenv("COMPANY_AGI_CEO_MODEL")
        cto_model = os.getenv("COMPANY_AGI_CTO_MODEL")
        dev_model = os.getenv("COMPANY_AGI_DEVELOPER_MODEL")

        if ceo_model or cto_model or dev_model:
            if self.config.models is None:
                self.config.models = AgentModels()
            if ceo_model:
                self.config.models.ceo = ceo_model
            if cto_model:
                self.config.models.cto = cto_model
            if dev_model:
                self.config.models.developer = dev_model

        # Output overrides
        output_dir = os.getenv("COMPANY_AGI_OUTPUT_DIR")
        if output_dir:
            self.config.output.solutions_dir = output_dir

        run_tests_env = os.getenv("COMPANY_AGI_RUN_TESTS")
        if run_tests_env is not None:
            self.config.output.run_tests = run_tests_env.lower() in ("true", "1", "yes")

        # Workflow overrides
        approve_env = os.getenv("COMPANY_AGI_APPROVE")
        if approve_env is not None:
            self.config.workflow.interactive_approval = approve_env.lower() in ("true", "1", "yes")

        token_budget_env = os.getenv("COMPANY_AGI_TOKEN_BUDGET")
        if token_budget_env:
            try:
                self.config.workflow.token_budget = int(token_budget_env)
            except ValueError:
                console.warning(f"Invalid COMPANY_AGI_TOKEN_BUDGET value: {token_budget_env}")

        # LLM overrides
        streaming_env = os.getenv("COMPANY_AGI_STREAMING")
        if streaming_env:
            self.config.llm.streaming = streaming_env.lower() == "true"

    def save_default_config(self, path: str = "config.yaml") -> None:
        """Save default configuration to file."""
        config_dict = {
            "workflow": {
                "auto_discover": True,
                "max_research_sources": 5,
                "enable_meetings": True,
                "enable_learning": True,
                "interactive_approval": False,
                "token_budget": None,
                "enhance_dir": ""
            },
            "output": {
                "solutions_dir": "output/solutions",
                "logs_dir": "output/logs",
                "reports_dir": "output/reports",
                "format_code": True,
                "run_tests": True
            },
            "llm": {
                "temperature": 0.7,
                "max_tokens": 4096,
                "streaming": True,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "models": {
                "ceo": "qwen3-8b",
                "cto": "qwen3-8b",
                "product_manager": "ministral-8b",
                "researcher": "ministral-8b",
                "developer": "qwen2.5-coder-7b",
                "qa_engineer": "qwen2.5-coder-7b",
                "devops_engineer": "qwen2.5-coder-7b",
                "data_analyst": "qwen3-8b",
                "security_engineer": "qwen2.5-coder-7b"
            },
            "problem": "",
            "domain": "software",
            "language": "python"
        }

        path_obj = Path(path)
        if path_obj.suffix in [".yaml", ".yml"]:
            if YAML_AVAILABLE and yaml is not None:
                content = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
            else:
                console.warning("PyYAML not available, saving as JSON")
                path = path.replace(".yaml", ".json").replace(".yml", ".json")
                content = json.dumps(config_dict, indent=2)
        else:
            content = json.dumps(config_dict, indent=2)

        Path(path).write_text(content, encoding="utf-8")
        console.info(f"Default config saved to: {path}")


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Convenience function to load configuration."""
    loader = ConfigLoader(config_path)
    return loader.load()
