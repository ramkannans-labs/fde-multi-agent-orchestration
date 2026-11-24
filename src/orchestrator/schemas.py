"""Pydantic schemas for multi-agent orchestration."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class AgentRole(str, Enum):
    """Specialized agent roles in the orchestration."""

    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"


class MessageType(str, Enum):
    """Types of messages in the agent communication system."""

    TASK = "task"
    RESULT = "result"
    QUESTION = "question"
    ANSWER = "answer"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    ERROR = "error"
    STATUS = "status"


class AgentMessage(BaseModel):
    """Message passed between agents."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    from_agent: AgentRole
    to_agent: AgentRole
    type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(default="", description="Groups related messages")
    requires_response: bool = False


class Task(BaseModel):
    """A task assigned to an agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    assigned_to: AgentRole
    created_by: AgentRole
    status: Literal["pending", "in_progress", "completed", "failed", "awaiting_approval"] = "pending"
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, description="Task IDs this depends on")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class Plan(BaseModel):
    """Execution plan created by the planner."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    tasks: list[Task] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["draft", "approved", "executing", "completed", "failed"] = "draft"

    @field_validator("tasks")
    @classmethod
    def validate_no_circular_deps(cls, v: list[Task]) -> list[Task]:
        """Basic check for circular dependencies."""
        # Build adjacency list
        adj: dict[str, list[str]] = {t.id: [] for t in v}
        for task in v:
            for dep in task.depends_on:
                if dep in adj:
                    adj[dep].append(task.id)

        # DFS cycle detection
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for task in v:
            if task.id not in visited:
                if has_cycle(task.id):
                    raise ValueError(f"Circular dependency detected involving task {task.id}")
        return v


class AgentState(BaseModel):
    """Current state of an agent."""

    role: AgentRole
    status: Literal["idle", "working", "waiting", "error"] = "idle"
    current_task_id: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)
    message_history: list[AgentMessage] = Field(default_factory=list)


class OrchestrationResult(BaseModel):
    """Final result of a multi-agent orchestration."""

    plan_id: str
    goal: str
    status: Literal["success", "partial", "failed"]
    completed_tasks: list[Task]
    failed_tasks: list[Task]
    final_output: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_turns: int = 0


class HumanApprovalRequest(BaseModel):
    """Request for human approval at a checkpoint."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    checkpoint: str
    question: str
    context: dict[str, Any] = Field(default_factory=dict)
    options: list[str] = Field(default_factory=lambda: ["Approve", "Reject", "Modify"])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    responded_at: Optional[datetime] = None
    response: Optional[str] = None


class HumanApprovalResponse(BaseModel):
    """Response to a human approval request."""

    request_id: str
    response: str  # One of the options from the request
    comment: Optional[str] = None
    responded_at: datetime = Field(default_factory=datetime.utcnow)