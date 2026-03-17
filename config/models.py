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
    """Specification for an Ollama model."""

    name: str                     # Display name
    ollama_model: str             # Ollama model tag (e.g. "qwen3:8b")
    context_window: int = 65536
    temperature: float = 0.7
    description: str = ""


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
    """Model configuration manager."""

    def __init__(self, custom_configs: Optional[Dict[AgentRole, ModelSpec]] = None):
        self.configs = MODEL_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)

    def get_model(self, role: AgentRole) -> ModelSpec:
        return self.configs.get(role, MODEL_CONFIGS[AgentRole.RESEARCHER])

    def get_model_name(self, role_name: str) -> str:
        role_map = {r.value: r for r in AgentRole}
        role = role_map.get(role_name.lower())
        if role and role in self.configs:
            return self.configs[role].name
        return "thealxlabs/lumen:latest"

    def set_model(self, role: AgentRole, model_spec: ModelSpec) -> None:
        self.configs[role] = model_spec

    def set_model_for_role(self, role_name: str, model_name: str) -> None:
        """Update the Ollama model tag for a role."""
        role_map = {r.value: r for r in AgentRole}
        role = role_map.get(role_name.lower())
        if role and role in self.configs:
            spec = self.configs[role]
            self.configs[role] = ModelSpec(
                name=model_name,
                ollama_model=model_name,
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
