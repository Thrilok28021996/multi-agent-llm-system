"""Agent role definitions (shared between config and agents packages)."""

from enum import Enum


class AgentRole(Enum):
    """Available agent roles in the company."""

    CEO = "ceo"
    CTO = "cto"
    PRODUCT_MANAGER = "product_manager"
    RESEARCHER = "researcher"
    DEVELOPER = "developer"
    QA_ENGINEER = "qa_engineer"
    DEVOPS_ENGINEER = "devops_engineer"
    DATA_ANALYST = "data_analyst"
    SECURITY_ENGINEER = "security_engineer"
