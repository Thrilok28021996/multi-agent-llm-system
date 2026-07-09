"""Agent module for Company AGI."""

from config.roles import AgentRole
from .base_agent import BaseAgent
from .ceo import CEOAgent
from .cto import CTOAgent
from .product_manager import ProductManagerAgent
from .researcher import ResearcherAgent
from .developer import DeveloperAgent
from .qa_engineer import QAEngineerAgent
from .devops_engineer import DevOpsEngineerAgent
from .data_analyst import DataAnalystAgent
from .security_engineer import SecurityEngineerAgent
from .thinking import (
    ThinkingEngine,
    ThinkingBlock,
    ThinkingStep,
    ThinkingConfig,
    ThinkingDepth,
    ThinkingPhase,
    get_thinking_engine,
    reset_thinking_engine,
)
from .personality import AgentPersonality, AgentExperience, DEFAULT_PERSONALITIES

__all__ = [
    # Roles
    "AgentRole",
    # Agents
    "BaseAgent",
    "CEOAgent",
    "CTOAgent",
    "ProductManagerAgent",
    "ResearcherAgent",
    "DeveloperAgent",
    "QAEngineerAgent",
    "DevOpsEngineerAgent",
    "DataAnalystAgent",
    "SecurityEngineerAgent",
    # Thinking mode
    "ThinkingEngine",
    "ThinkingBlock",
    "ThinkingConfig",
    "ThinkingDepth",
    "ThinkingPhase",
    "ThinkingStep",
    "get_thinking_engine",
    "reset_thinking_engine",
    # Personality
    "AgentPersonality",
    "AgentExperience",
    "DEFAULT_PERSONALITIES",
]
