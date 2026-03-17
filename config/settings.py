"""Global settings and configuration for Company AGI."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMBackendConfig:
    """LLM backend configuration for Ollama."""

    backend: str = "ollama"
    # Ollama server address (override with OLLAMA_HOST env var)
    ollama_host: str = field(default_factory=lambda: os.getenv(
        "OLLAMA_HOST", "http://localhost:11434"
    ))
    timeout: int = 120
    num_ctx: int = 65536
    temperature: float = 0.7


@dataclass
class ResearchConfig:
    """Research and scraping configuration."""

    rate_limit_delay: float = 2.0  # Seconds between requests
    max_concurrent_requests: int = 3
    max_results_per_source: int = 50
    cache_duration_hours: int = 24


@dataclass
class WorkflowConfig:
    """Workflow orchestration configuration."""

    max_iterations_per_phase: int = 5
    meeting_max_rounds: int = 10
    decision_timeout_seconds: int = 300
    enable_logging: bool = True
    log_level: str = "INFO"
    max_workflow_minutes: int = 0  # 0 = no time limit
    enable_escalation: bool = True
    enable_retrospective: bool = True
    enable_security_review: bool = True
    force_stop: bool = False  # Only way to stop besides success


@dataclass
class MemoryConfig:
    """Memory and storage configuration."""

    chroma_persist_dir: str = "./output/memory"
    max_memory_items: int = 1000
    embedding_model: str = "nomic-embed-text"


@dataclass
class Settings:
    """Main settings container for Company AGI."""

    llm: LLMBackendConfig = field(default_factory=LLMBackendConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Output directories
    output_dir: str = "./output"
    solutions_dir: str = "./output/solutions"
    reports_dir: str = "./output/reports"
    logs_dir: str = "./output/logs"

    # Company mission (used by CEO for alignment)
    company_mission: str = "Discover real-world problems and create innovative solutions that provide genuine value to users."

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        settings = cls()

        # Override with environment variables if present
        if os.getenv("OLLAMA_HOST"):
            settings.llm.ollama_host = os.getenv("OLLAMA_HOST")
        if os.getenv("RATE_LIMIT_DELAY"):
            settings.research.rate_limit_delay = float(os.getenv("RATE_LIMIT_DELAY"))
        if os.getenv("LOG_LEVEL"):
            settings.workflow.log_level = os.getenv("LOG_LEVEL")

        return settings


# Global settings instance
settings = Settings.from_env()
