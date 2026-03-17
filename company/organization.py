"""
Declarative company org chart, department definitions, and agent management helpers.

This module is pure data and logic -- zero runtime overhead, no LLM calls.

Standard org chart
------------------
CEO (Executive)
 |-- CTO (Engineering)
 |    |-- Developer
 |    |-- DevOpsEngineer
 |    |-- SecurityEngineer
 |    |-- QAEngineer
 |-- ProductManager (Product)
      |-- Researcher
      |-- DataAnalyst
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Department(Enum):
    """Top-level organisational departments."""

    EXECUTIVE = "executive"
    ENGINEERING = "engineering"
    PRODUCT = "product"
    RESEARCH = "research"
    OPERATIONS = "operations"


@dataclass
class DepartmentInfo:
    """Detailed department definition with responsibilities and authority."""
    head: str
    members: List[str]
    responsibilities: List[str]
    decision_authority: str


DEPARTMENT_DEFINITIONS: Dict[str, DepartmentInfo] = {
    "executive": DepartmentInfo(
        head="CEO",
        members=["CEO"],
        responsibilities=["Strategic direction", "Final approval", "Resource allocation"],
        decision_authority="company-wide"
    ),
    "engineering": DepartmentInfo(
        head="CTO",
        members=["CTO", "Developer", "DevOps", "SecurityEngineer"],
        responsibilities=["Architecture", "Implementation", "Deployment", "Security"],
        decision_authority="technical"
    ),
    "product": DepartmentInfo(
        head="ProductManager",
        members=["ProductManager", "Researcher", "DataAnalyst"],
        responsibilities=["Requirements", "Research", "Data validation", "User advocacy"],
        decision_authority="product-scope"
    ),
    "quality": DepartmentInfo(
        head="QAEngineer",
        members=["QAEngineer"],
        responsibilities=["Testing", "Quality gates", "Release readiness"],
        decision_authority="quality-gates"
    ),
}


# Decision authority matrix
DECISION_AUTHORITY: Dict[str, Dict[str, str]] = {
    "CEO": {
        "scope": "company-wide",
        "can_approve": "solutions, company direction, override any decision",
        "can_reject": "any solution, any architecture, any scope",
        "cannot": "write code directly"
    },
    "CTO": {
        "scope": "technical",
        "can_approve": "architectures, tech stack, implementation approaches",
        "can_reject": "technical designs, technology choices",
        "cannot": "approve final solutions (CEO only), change requirements (PM only)"
    },
    "ProductManager": {
        "scope": "product-scope",
        "can_approve": "scope changes, requirement definitions, feature priorities",
        "can_reject": "scope creep, unclear requirements",
        "cannot": "approve final solutions (CEO only), choose tech stack (CTO only)"
    },
    "QAEngineer": {
        "scope": "quality-gates",
        "can_approve": "release readiness, quality standards met",
        "can_reject": "releases with critical bugs, untested solutions",
        "cannot": "change requirements (PM only), redesign architecture (CTO only)"
    },
    "Developer": {
        "scope": "implementation",
        "can_approve": "nothing (recommends only)",
        "can_reject": "nothing (flags issues only)",
        "cannot": "approve solutions, change scope, choose architecture"
    },
    "Researcher": {
        "scope": "research",
        "can_approve": "nothing (recommends only)",
        "can_reject": "nothing (flags issues only)",
        "cannot": "approve solutions, define requirements"
    },
    "DataAnalyst": {
        "scope": "analysis",
        "can_approve": "nothing (recommends only)",
        "can_reject": "nothing (flags issues only)",
        "cannot": "approve solutions, define requirements"
    },
    "DevOpsEngineer": {
        "scope": "deployment",
        "can_approve": "deployment readiness",
        "can_reject": "undeployable solutions",
        "cannot": "approve final solutions, change requirements"
    },
    "SecurityEngineer": {
        "scope": "security",
        "can_approve": "security posture",
        "can_reject": "solutions with critical vulnerabilities",
        "cannot": "approve final solutions, change requirements"
    },
}


@dataclass
class OrgChart:
    """Immutable snapshot of the company org structure.

    Parameters
    ----------
    departments:
        Mapping from *Department* to the list of agent names that belong to it.
    reporting_lines:
        Mapping from an agent name to its direct supervisor's name.
    escalation_paths:
        Mapping from an agent name to the ordered chain of command above it
        (immediate supervisor first, CEO last).
    """

    departments: Dict[Department, List[str]] = field(default_factory=dict)
    reporting_lines: Dict[str, str] = field(default_factory=dict)
    escalation_paths: Dict[str, List[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_supervisor(self, agent_name: str) -> Optional[str]:
        """Return the direct supervisor of *agent_name*, or ``None``."""
        return self.reporting_lines.get(agent_name)

    def get_department(self, agent_name: str) -> Optional[Department]:
        """Return the department that *agent_name* belongs to, or ``None``."""
        for dept, members in self.departments.items():
            if agent_name in members:
                return dept
        return None

    def get_escalation_path(self, agent_name: str) -> List[str]:
        """Return the escalation chain for *agent_name* (may be empty)."""
        return list(self.escalation_paths.get(agent_name, []))

    def get_department_info(self, department_name: str) -> Optional["DepartmentInfo"]:
        """Return detailed info for a department by name."""
        return DEPARTMENT_DEFINITIONS.get(department_name.lower())

    def get_decision_maker(self, decision_type: str) -> Optional[str]:
        """Return the agent who has authority for a given decision type.

        Args:
            decision_type: One of 'final_approval', 'technical', 'product',
                          'quality', 'deployment', 'security'.

        Returns:
            Agent name with authority, or None.
        """
        decision_map = {
            "final_approval": "CEO",
            "company_direction": "CEO",
            "technical": "CTO",
            "architecture": "CTO",
            "tech_stack": "CTO",
            "product": "ProductManager",
            "requirements": "ProductManager",
            "scope": "ProductManager",
            "quality": "QAEngineer",
            "release": "QAEngineer",
            "deployment": "DevOpsEngineer",
            "security": "SecurityEngineer",
        }
        return decision_map.get(decision_type.lower())

    def get_agent_authority(self, agent_name: str) -> Optional[Dict[str, str]]:
        """Return the decision authority for a specific agent."""
        return DECISION_AUTHORITY.get(agent_name)

    def communication_protocol(self, sender: str, receiver: str) -> str:
        """Determine the communication protocol between two agents.

        Returns:
            A string describing how the communication should be routed:
            - 'direct': Same department or direct report
            - 'cc_head': Cross-department, CC the department head
            - 'escalation': Goes up the chain of command
        """
        sender_dept = self.get_department(sender)
        receiver_dept = self.get_department(receiver)

        # Direct supervisor relationship
        if self.reporting_lines.get(sender) == receiver:
            return "direct"
        if self.reporting_lines.get(receiver) == sender:
            return "direct"

        # Same department
        if sender_dept and sender_dept == receiver_dept:
            return "direct"

        # Cross-department communication
        if sender_dept != receiver_dept:
            return "cc_head"

        return "direct"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "OrgChart":
        """Return the standard company org chart."""

        departments: Dict[Department, List[str]] = {
            Department.EXECUTIVE: ["CEO"],
            Department.ENGINEERING: [
                "CTO",
                "Developer",
                "DevOpsEngineer",
                "SecurityEngineer",
                "QAEngineer",
            ],
            Department.PRODUCT: ["ProductManager"],
            Department.RESEARCH: ["Researcher", "DataAnalyst"],
            Department.OPERATIONS: [],
        }

        reporting_lines: Dict[str, str] = {
            "CTO": "CEO",
            "ProductManager": "CEO",
            "Developer": "CTO",
            "DevOpsEngineer": "CTO",
            "SecurityEngineer": "CTO",
            "QAEngineer": "CTO",
            "Researcher": "ProductManager",
            "DataAnalyst": "ProductManager",
        }

        escalation_paths: Dict[str, List[str]] = {
            "Developer": ["CTO", "CEO"],
            "DevOpsEngineer": ["CTO", "CEO"],
            "SecurityEngineer": ["CTO", "CEO"],
            "QAEngineer": ["CTO", "CEO"],
            "Researcher": ["ProductManager", "CEO"],
            "DataAnalyst": ["ProductManager", "CEO"],
            "CTO": ["CEO"],
            "ProductManager": ["CEO"],
            "CEO": [],
        }

        return cls(
            departments=departments,
            reporting_lines=reporting_lines,
            escalation_paths=escalation_paths,
        )


# ============================================================
#  ESCALATION CHAINS & ROLE DESCRIPTIONS
# ============================================================

ESCALATION_CHAINS: Dict[str, List[str]] = {
    "Developer": ["CTO", "ProductManager", "CEO"],
    "QAEngineer": ["CTO", "CEO"],
    "Researcher": ["ProductManager", "CEO"],
    "DataAnalyst": ["ProductManager", "CEO"],
    "DevOpsEngineer": ["CTO", "CEO"],
    "SecurityEngineer": ["CTO", "CEO"],
    "CTO": ["CEO"],
    "ProductManager": ["CEO"],
    "CEO": [],
}


def get_escalation_chain(agent_name: str) -> List[str]:
    """Return the escalation chain for a given agent.

    Args:
        agent_name: Name of the agent.

    Returns:
        Ordered list of agents to escalate to (immediate supervisor first).
    """
    return list(ESCALATION_CHAINS.get(agent_name, ["CEO"]))


ROLE_DESCRIPTIONS: Dict[str, str] = {
    "CEO": "Makes final decisions on what ships. Accountable for product quality.",
    "CTO": "Designs architecture. Accountable for technical soundness.",
    "ProductManager": "Defines requirements. Accountable for problem-solution fit.",
    "Developer": "Implements solutions. Accountable for code quality and correctness.",
    "QAEngineer": "Validates solutions. Accountable for catching bugs before shipping.",
    "Researcher": "Discovers problems. Accountable for finding real, validated pain points.",
    "DataAnalyst": "Cross-validates research. Accountable for unbiased data analysis.",
    "DevOpsEngineer": "Ensures deployability. Accountable for reproducible builds.",
    "SecurityEngineer": "Reviews security. Accountable for identifying vulnerabilities.",
}


class AgentManager:
    """Utility helpers for workforce management decisions.

    All methods are deterministic -- no LLM calls.
    """

    @staticmethod
    def get_underperformers(
        kpis: Dict[str, float],
        threshold: float = 40.0,
    ) -> List[str]:
        """Return agent names whose composite KPI score falls below *threshold*.

        Parameters
        ----------
        kpis:
            Mapping of agent name to its composite score (0-100).
        threshold:
            Minimum acceptable score.  Agents below this value are flagged.
        """
        return sorted(
            name for name, score in kpis.items() if score < threshold
        )

    @staticmethod
    def recommend_reconfiguration(agent_name: str, kpi_score: float) -> str:
        """Return a plain-text recommendation for an underperforming agent.

        The recommendation is template-based and deterministic.
        """
        if kpi_score >= 70.0:
            return (
                f"{agent_name} is performing well (score={kpi_score:.1f}). "
                "No reconfiguration needed."
            )

        if kpi_score >= 40.0:
            return (
                f"{agent_name} is below target (score={kpi_score:.1f}). "
                "Recommendation: review prompt templates, increase oversight, "
                "and schedule a performance check-in next sprint."
            )

        if kpi_score >= 20.0:
            return (
                f"{agent_name} is significantly underperforming "
                f"(score={kpi_score:.1f}). "
                "Recommendation: reassign current tasks, pair with a senior "
                "agent for supervised work, and re-evaluate after 2 sprints."
            )

        return (
            f"{agent_name} is critically underperforming "
            f"(score={kpi_score:.1f}). "
            "Recommendation: suspend autonomous task assignment, rebuild "
            "agent configuration from scratch, and require supervisor "
            "approval for all outputs."
        )
