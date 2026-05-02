"""Model configuration and assignments for each agent role.

Pull models with Ollama:
  ollama pull goekdenizguelmez/JOSIEFIED-Qwen3:8b
  ollama pull ministral-3:8b
  ollama pull thealxlabs/lumen:latest

Default config uses three models:
  - JOSIEFIED-Qwen3 8B — strategic reasoning with thinking mode
  - Ministral 3 8B     — agentic tasks, function calling
  - Lumen (latest)     — code generation and review

Context: 65536 tokens (64k). Ollama manages model loading/unloading automatically.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from config.roles import AgentRole
from ui.console import console


@dataclass
class ModelSpec:
    """Specification for a model — supports Ollama and LM Studio backends."""

    name: str                     # Display name
    ollama_model: str             # Ollama model tag (e.g. "qwen3:8b")
    lmstudio_model: str = ""      # LM Studio model ID (e.g. "lmstudio-community/Qwen2.5-7B-Instruct-GGUF")
                                  # If empty, falls back to ollama_model value (useful when IDs match)
    context_window: int = 65536
    temperature: float = 0.7
    description: str = ""

    def model_id(self, backend: str) -> str:
        """Return the correct model identifier for the given backend.

        Any backend other than 'ollama' is treated as LM Studio compatible
        (covers 'lmstudio', 'mlx', and future variants).
        """
        if backend != "ollama" and self.lmstudio_model:
            return self.lmstudio_model
        return self.ollama_model


# =============================================================================
# DEFAULT MODEL CONFIG
# =============================================================================

MODEL_CONFIGS: Dict[AgentRole, ModelSpec] = {
    # Strategic roles — JOSIEFIED-Qwen3 8B (thinking mode)
    AgentRole.CEO: ModelSpec(
        name="goekdenizguelmez/JOSIEFIED-Qwen3:8b",
        ollama_model="goekdenizguelmez/JOSIEFIED-Qwen3:8b",
        context_window=65536,
        temperature=0.6,
        description="Strategic decision-making, vision alignment, first principles",
    ),
    AgentRole.CTO: ModelSpec(
        name="goekdenizguelmez/JOSIEFIED-Qwen3:8b",
        ollama_model="goekdenizguelmez/JOSIEFIED-Qwen3:8b",
        context_window=65536,
        temperature=0.5,
        description="Technical architecture, feasibility, system design",
    ),
    AgentRole.DATA_ANALYST: ModelSpec(
        name="goekdenizguelmez/JOSIEFIED-Qwen3:8b",
        ollama_model="goekdenizguelmez/JOSIEFIED-Qwen3:8b",
        context_window=65536,
        temperature=0.4,
        description="Cross-validation, bias detection, credibility scoring",
    ),

    # Agentic roles — Ministral 3 8B
    AgentRole.PRODUCT_MANAGER: ModelSpec(
        name="ministral-3:8b",
        ollama_model="ministral-3:8b",
        context_window=65536,
        temperature=0.7,
        description="Product strategy, user needs, requirements definition",
    ),
    AgentRole.RESEARCHER: ModelSpec(
        name="ministral-3:8b",
        ollama_model="ministral-3:8b",
        context_window=65536,
        temperature=0.8,
        description="Problem discovery, market analysis, evidence gathering",
    ),

    # Technical execution — Lumen (latest)
    AgentRole.DEVELOPER: ModelSpec(
        name="thealxlabs/lumen:latest",
        ollama_model="thealxlabs/lumen:latest",
        context_window=65536,
        temperature=0.2,
        description="Code implementation, debugging, refactoring",
    ),
    AgentRole.QA_ENGINEER: ModelSpec(
        name="thealxlabs/lumen:latest",
        ollama_model="thealxlabs/lumen:latest",
        context_window=65536,
        temperature=0.3,
        description="Test generation, code review, bug detection",
    ),
    AgentRole.DEVOPS_ENGINEER: ModelSpec(
        name="thealxlabs/lumen:latest",
        ollama_model="thealxlabs/lumen:latest",
        context_window=65536,
        temperature=0.3,
        description="Deployment validation, CI/CD, dependency auditing",
    ),
    AgentRole.SECURITY_ENGINEER: ModelSpec(
        name="thealxlabs/lumen:latest",
        ollama_model="thealxlabs/lumen:latest",
        context_window=65536,
        temperature=0.2,
        description="Security review, secrets scanning, threat modeling",
    ),
}


class ModelConfig:
    """Model configuration manager.

    Priority (highest → lowest):
      1. custom_configs passed directly
      2. Environment variables  (MODEL_<ROLE> and LMSTUDIO_MODEL_<ROLE>)
      3. Hardcoded MODEL_CONFIGS defaults

    Environment variable names:
      MODEL_CEO, MODEL_CTO, MODEL_PRODUCT_MANAGER, MODEL_RESEARCHER,
      MODEL_DEVELOPER, MODEL_QA_ENGINEER, MODEL_DEVOPS_ENGINEER,
      MODEL_SECURITY_ENGINEER, MODEL_DATA_ANALYST

    LM Studio equivalents (only needed when IDs differ from Ollama tags):
      LMSTUDIO_MODEL_CEO, LMSTUDIO_MODEL_CTO, ... (same pattern)

    Example .env:
      MODEL_DEVELOPER=qwen2.5-coder:7b
      LMSTUDIO_MODEL_DEVELOPER=lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF
    """

    # Maps AgentRole → env var suffix
    _ROLE_ENV: Dict[AgentRole, str] = {
        AgentRole.CEO:               "CEO",
        AgentRole.CTO:               "CTO",
        AgentRole.PRODUCT_MANAGER:   "PRODUCT_MANAGER",
        AgentRole.RESEARCHER:        "RESEARCHER",
        AgentRole.DEVELOPER:         "DEVELOPER",
        AgentRole.QA_ENGINEER:       "QA_ENGINEER",
        AgentRole.DEVOPS_ENGINEER:   "DEVOPS_ENGINEER",
        AgentRole.SECURITY_ENGINEER: "SECURITY_ENGINEER",
        AgentRole.DATA_ANALYST:      "DATA_ANALYST",
    }

    def __init__(self, custom_configs: Optional[Dict[AgentRole, ModelSpec]] = None):
        import os
        self.configs = MODEL_CONFIGS.copy()

        # Apply env var overrides
        for role, suffix in self._ROLE_ENV.items():
            ollama_override = os.getenv(f"MODEL_{suffix}")
            lmstudio_override = os.getenv(f"LMSTUDIO_MODEL_{suffix}")
            if ollama_override or lmstudio_override:
                spec = self.configs[role]
                self.configs[role] = ModelSpec(
                    name=ollama_override or spec.ollama_model,
                    ollama_model=ollama_override or spec.ollama_model,
                    lmstudio_model=lmstudio_override or spec.lmstudio_model,
                    context_window=spec.context_window,
                    temperature=spec.temperature,
                    description=spec.description,
                )

        # custom_configs win over everything
        if custom_configs:
            self.configs.update(custom_configs)

    def get_model(self, role: AgentRole) -> ModelSpec:
        return self.configs.get(role, MODEL_CONFIGS[AgentRole.RESEARCHER])

    def get_model_name(self, role_name: str) -> str:
        from config.llm_client import _get_backend
        role_map = {r.value: r for r in AgentRole}
        role = role_map.get(role_name.lower())
        if role and role in self.configs:
            return self.configs[role].model_id(_get_backend())
        return "thealxlabs/lumen:latest"

    def set_model(self, role: AgentRole, model_spec: ModelSpec) -> None:
        self.configs[role] = model_spec

    def set_model_for_role(self, role_name: str, model_name: str, lmstudio_model: str = "") -> None:
        """Update the model tag for a role. Optionally set a separate LM Studio model ID."""
        role_map = {r.value: r for r in AgentRole}
        role = role_map.get(role_name.lower())
        if role and role in self.configs:
            spec = self.configs[role]
            self.configs[role] = ModelSpec(
                name=model_name,
                ollama_model=model_name,
                lmstudio_model=lmstudio_model,
                context_window=spec.context_window,
                temperature=spec.temperature,
                description=spec.description,
            )

    def list_required_models(self) -> list[str]:
        """List unique Ollama model tags required."""
        return sorted({spec.ollama_model for spec in self.configs.values()})

    def print_config(self) -> None:
        console.section("Model Configuration")
        for role, spec in self.configs.items():
            console.info(f"  {role.value:20} -> {spec.ollama_model}")
        console.info(f"Required models: {', '.join(self.list_required_models())}")
