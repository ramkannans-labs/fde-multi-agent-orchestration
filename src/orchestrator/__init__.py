"""Multi-Agent Orchestration - FDE Demo Project.

A demonstration of multi-agent orchestration patterns using PydanticAI.
"""

from __future__ import annotations

from .config import get_settings, settings
from .schemas import (
    AgentMessage,
    AgentRole,
    AgentState,
    HumanApprovalRequest,
    HumanApprovalResponse,
    MessageType,
    OrchestrationResult,
    Plan,
    Task,
)
from .orchestrator import Orchestrator, demo_orchestration
from .agents import (
    BaseAgent,
    CoordinatorAgent,
    PlannerAgent,
    ResearcherAgent,
    CoderAgent,
    ReviewerAgent,
)

__all__ = [
    # Config
    "get_settings",
    "settings",
    # Schemas
    "AgentMessage",
    "AgentRole",
    "AgentState",
    "HumanApprovalRequest",
    "HumanApprovalResponse",
    "MessageType",
    "OrchestrationResult",
    "Plan",
    "Task",
    # Orchestrator
    "Orchestrator",
    "demo_orchestration",
    # Agents
    "BaseAgent",
    "CoordinatorAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "CoderAgent",
    "ReviewerAgent",
]

__version__ = "0.1.0"