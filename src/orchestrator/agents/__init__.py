"""Agent implementations for multi-agent orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from pydantic_ai import Agent
from pydantic_ai.models import Model

from ..config import get_settings
from ..schemas import (
    AgentMessage,
    AgentRole,
    AgentState,
    HumanApprovalRequest,
    HumanApprovalResponse,
    MessageType,
    Task,
)


class BaseAgent(ABC):
    """Base class for all specialized agents."""

    def __init__(self, role: AgentRole, model: Model | None = None):
        self.role = role
        self.state = AgentState(role=role)
        self._model = model
        self._agent: Agent | None = None

    @property
    def model(self) -> Model:
        """Get or create the model."""
        if self._model is None:
            settings = get_settings()
            if settings.anthropic_api_key:
                from pydantic_ai.models.anthropic import AnthropicModel
                self._model = AnthropicModel(
                    settings.llm_model.replace("anthropic:", ""),
                    api_key=settings.anthropic_api_key,
                )
            else:
                # Test model fallback
                from pydantic_ai.models.test import TestModel
                self._model = TestModel()
        return self._model

    @property
    def agent(self) -> Agent:
        """Get or create the PydanticAI agent."""
        if self._agent is None:
            self._agent = Agent(
                model=self.model,
                system_prompt=self.get_system_prompt(),
                output_type=self.get_output_type(),
            )
        return self._agent

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def get_output_type(self) -> type:
        """Return the expected output type."""
        pass

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process an incoming message and optionally return a response."""
        self.state.message_history.append(message)

        if message.type == MessageType.TASK:
            return await self.execute_task(message)
        elif message.type == MessageType.QUESTION:
            return await self.answer_question(message)
        elif message.type == MessageType.APPROVAL_RESPONSE:
            return await self.handle_approval_response(message)

        return None

    @abstractmethod
    async def execute_task(self, message: AgentMessage) -> AgentMessage | None:
        """Execute a task assigned to this agent."""
        pass

    async def answer_question(self, message: AgentMessage) -> AgentMessage:
        """Answer a question from another agent."""
        prompt = f"Another agent asks: {message.payload.get('question', '')}\nContext: {message.payload.get('context', {})}"
        result = await self.agent.run(prompt)
        return AgentMessage(
            from_agent=self.role,
            to_agent=message.from_agent,
            type=MessageType.ANSWER,
            payload={"answer": result.output, "original_question": message.payload.get("question")},
            correlation_id=message.correlation_id,
        )

    async def handle_approval_response(self, message: AgentMessage) -> AgentMessage | None:
        """Handle human approval response."""
        # Override in subclasses if needed
        return None

    def create_task_message(self, task: Task, correlation_id: str = "") -> AgentMessage:
        """Create a task message for another agent."""
        return AgentMessage(
            from_agent=self.role,
            to_agent=task.assigned_to,
            type=MessageType.TASK,
            payload={"task": task.model_dump()},
            correlation_id=correlation_id or task.id,
            requires_response=True,
        )

    def create_approval_request(
        self,
        task_id: str,
        checkpoint: str,
        question: str,
        context: dict[str, Any],
        options: list[str] | None = None,
    ) -> AgentMessage:
        """Create a human approval request message."""
        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            type=MessageType.APPROVAL_REQUEST,
            payload={
                "request": HumanApprovalRequest(
                    task_id=task_id,
                    checkpoint=checkpoint,
                    question=question,
                    context=context,
                    options=options or ["Approve", "Reject", "Modify"],
                ).model_dump()
            },
            correlation_id=task_id,
            requires_response=True,
        )


class PlannerAgent(BaseAgent):
    """Agent responsible for creating execution plans."""

    def __init__(self, model: Model | None = None):
        super().__init__(AgentRole.PLANNER, model)

    def get_system_prompt(self) -> str:
        return """You are a Planning Agent. Your job is to break down complex goals into executable tasks.

Given a goal, create a plan with:
1. Discrete, actionable tasks
2. Clear task assignments to agent roles (researcher, coder, reviewer)
3. Dependencies between tasks
4. Success criteria for each task

Output a structured Plan with tasks. Each task should have:
- description: What needs to be done
- assigned_to: One of [researcher, coder, reviewer]
- depends_on: List of task IDs this depends on
- input_data: Any context needed

Think step by step. Be specific and actionable."""

    def get_output_type(self) -> type:
        from ..schemas import Plan
        return Plan

    async def execute_task(self, message: AgentMessage) -> AgentMessage | None:
        task_data = message.payload.get("task", {})
        goal = task_data.get("description", message.payload.get("goal", ""))

        prompt = f"""Create an execution plan for this goal: {goal}

Break it down into tasks for specialized agents:
- researcher: Information gathering, API lookups, documentation search
- coder: Writing code, scripts, configurations, data transformations
- reviewer: Code review, validation, testing, quality checks

Return a Plan with tasks, dependencies, and assignments."""

        result = await self.agent.run(prompt)
        plan = result.output

        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            type=MessageType.RESULT,
            payload={"plan": plan.model_dump()},
            correlation_id=message.correlation_id,
        )


class ResearcherAgent(BaseAgent):
    """Agent responsible for information gathering."""

    def __init__(self, model: Model | None = None):
        super().__init__(AgentRole.RESEARCHER, model)

    def get_system_prompt(self) -> str:
        return """You are a Research Agent. Your job is to gather information to support other agents.

You can:
- Search documentation (simulate with knowledge)
- Look up API references
- Find best practices
- Gather context from existing codebases
- Summarize technical information

Be thorough but concise. Cite sources when possible. Return structured findings."""

    def get_output_type(self) -> type:
        from ..schemas import Task
        return Task

    async def execute_task(self, message: AgentMessage) -> AgentMessage | None:
        task_data = message.payload.get("task", {})
        task = Task(**task_data)

        prompt = f"""Research task: {task.description}

Context: {task.input_data}

Provide findings that will help the coder and reviewer. Focus on:
- Key APIs, libraries, or patterns to use
- Common pitfalls to avoid
- Relevant examples or references"""

        result = await self.agent.run(prompt)
        task.output_data = {"findings": result.output}
        task.status = "completed"

        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            type=MessageType.RESULT,
            payload={"task": task.model_dump()},
            correlation_id=message.correlation_id,
        )


class CoderAgent(BaseAgent):
    """Agent responsible for writing code and implementations."""

    def __init__(self, model: Model | None = None):
        super().__init__(AgentRole.CODER, model)

    def get_system_prompt(self) -> str:
        return """You are a Coding Agent. Your job is to write clean, working code.

Given a task and research findings, produce:
- Complete, runnable code
- Proper error handling
- Type hints and documentation
- Tests if appropriate

Follow best practices for the language/framework. Make code production-ready."""

    def get_output_type(self) -> type:
        from ..schemas import Task
        return Task

    async def execute_task(self, message: AgentMessage) -> AgentMessage | None:
        task_data = message.payload.get("task", {})
        task = Task(**task_data)

        # Include research findings if available
        research_context = task.input_data.get("research_findings", "")

        prompt = f"""Implement this task: {task.description}

Research findings: {research_context}

Input data: {task.input_data}

Write complete, production-ready code with proper structure, error handling, and type hints."""

        result = await self.agent.run(prompt)
        task.output_data = {"code": result.output}
        task.status = "completed"

        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            type=MessageType.RESULT,
            payload={"task": task.model_dump()},
            correlation_id=message.correlation_id,
        )


class ReviewerAgent(BaseAgent):
    """Agent responsible for reviewing and validating work."""

    def __init__(self, model: Model | None = None):
        super().__init__(AgentRole.REVIEWER, model)

    def get_system_prompt(self) -> str:
        return """You are a Review Agent. Your job is to validate work meets quality standards.

Check for:
- Correctness: Does the code/logic actually solve the problem?
- Completeness: Are all requirements met?
- Quality: Type hints, error handling, documentation, tests
- Security: No obvious vulnerabilities
- Maintainability: Clean structure, good naming, separation of concerns

Be thorough but fair. Provide specific, actionable feedback."""

    def get_output_type(self) -> type:
        from ..schemas import Task
        return Task

    async def execute_task(self, message: AgentMessage) -> AgentMessage | None:
        task_data = message.payload.get("task", {})
        task = Task(**task_data)

        # Get the artifact to review
        artifact = task.input_data.get("artifact", "")
        artifact_type = task.input_data.get("artifact_type", "code")
        original_task = task.input_data.get("original_task", "")

        prompt = f"""Review this {artifact_type}:

Original task: {original_task}
Artifact:
{artifact}

Provide:
1. Verdict: PASS / NEEDS_CHANGES / FAIL
2. Specific issues found
3. Suggested fixes
4. Overall quality score (1-10)"""

        result = await self.agent.run(prompt)
        task.output_data = {"review": result.output}
        task.status = "completed"

        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            type=MessageType.RESULT,
            payload={"task": task.model_dump()},
            correlation_id=message.correlation_id,
        )


class CoordinatorAgent(BaseAgent):
    """Agent responsible for orchestrating the multi-agent workflow."""

    def __init__(self, model: Model | None = None):
        super().__init__(AgentRole.COORDINATOR, model)
        self.plan: Plan | None = None
        self.task_queue: list[Task] = []
        self.completed_tasks: list[Task] = []
        self.failed_tasks: list[Task] = []
        self.human_approvals_pending: dict[str, HumanApprovalRequest] = {}
        self.turn_count = 0
        self.max_turns = get_settings().max_turns

    def get_system_prompt(self) -> str:
        return """You are the Coordinator Agent. You orchestrate the multi-agent workflow.

Your responsibilities:
1. Receive goals and delegate to Planner
2. Execute plans by assigning tasks to agents
3. Track task dependencies and completion
4. Handle human-in-the-loop approvals
5. Aggregate results and report final outcome

Manage the workflow state and ensure tasks execute in the right order."""

    def get_output_type(self) -> type:
        from ..schemas import OrchestrationResult
        return OrchestrationResult

    async def execute_task(self, message: AgentMessage) -> AgentMessage | None:
        # Coordinator doesn't execute tasks directly; it orchestrates
        return None

    async def start_orchestration(self, goal: str) -> OrchestrationResult:
        """Start a full orchestration for a given goal."""
        from ..schemas import OrchestrationResult, Plan
        import datetime

        started_at = datetime.datetime.utcnow()

        # Step 1: Create plan via Planner
        planner = PlannerAgent()
        plan_msg = AgentMessage(
            from_agent=AgentRole.COORDINATOR,
            to_agent=AgentRole.PLANNER,
            type=MessageType.TASK,
            payload={"goal": goal},
            correlation_id=str(uuid4()),
            requires_response=True,
        )
        plan_response = await planner.process_message(plan_msg)
        if plan_response and plan_response.payload.get("plan"):
            self.plan = Plan(**plan_response.payload["plan"])

        if not self.plan:
            return OrchestrationResult(
                plan_id="",
                goal=goal,
                status="failed",
                completed_tasks=[],
                failed_tasks=[],
                started_at=started_at,
                completed_at=datetime.datetime.utcnow(),
            )

        # Step 2: Execute plan
        self.task_queue = self.plan.tasks.copy()
        return await self._execute_plan(goal, started_at)

        # Step 1: Create plan via Planner
        planner = PlannerAgent()
        plan_msg = AgentMessage(
            from_agent=AgentRole.COORDINATOR,
            to_agent=AgentRole.PLANNER,
            type=MessageType.TASK,
            payload={"goal": goal},
            correlation_id=str(uuid4()),
            requires_response=True,
        )
        plan_response = await planner.process_message(plan_msg)
        if plan_response and plan_response.payload.get("plan"):
            self.plan = Plan(**plan_response.payload["plan"])

        if not self.plan:
            return OrchestrationResult(
                plan_id="",
                goal=goal,
                status="failed",
                completed_tasks=[],
                failed_tasks=[],
                started_at=started_at,
                completed_at=datetime.datetime.utcnow(),
            )

        # Step 2: Execute plan
        self.task_queue = self.plan.tasks.copy()
        return await self._execute_plan(goal, started_at)

    async def _execute_plan(self, goal: str, started_at: datetime.datetime) -> OrchestrationResult:
        """Execute the plan by running tasks in dependency order."""
        from ..schemas import OrchestrationResult

        agents = {
            AgentRole.RESEARCHER: ResearcherAgent(),
            AgentRole.CODER: CoderAgent(),
            AgentRole.REVIEWER: ReviewerAgent(),
        }

        completed_ids: set[str] = set()

        while self.task_queue and self.turn_count < self.max_turns:
            self.turn_count += 1

            # Find tasks whose dependencies are met
            ready_tasks = [
                t for t in self.task_queue
                if all(dep in completed_ids for dep in t.depends_on)
            ]

            if not ready_tasks:
                # Check for circular deps or missing deps
                break

            # Execute ready tasks (in parallel would be better, but sequential for now)
            for task in ready_tasks:
                task.status = "in_progress"
                task.started_at = datetime.datetime.utcnow()

                agent = agents.get(task.assigned_to)
                if not agent:
                    task.status = "failed"
                    task.error = f"No agent for role {task.assigned_to}"
                    self.failed_tasks.append(task)
                    completed_ids.add(task.id)
                    self.task_queue.remove(task)
                    continue

                # Add context from completed tasks
                context = {}
                for ct in self.completed_tasks:
                    context[ct.id] = ct.output_data

                task.input_data["context"] = context

                msg = AgentMessage(
                    from_agent=AgentRole.COORDINATOR,
                    to_agent=task.assigned_to,
                    type=MessageType.TASK,
                    payload={"task": task.model_dump()},
                    correlation_id=task.id,
                    requires_response=True,
                )

                response = await agent.process_message(msg)

                if response and response.payload.get("task"):
                    updated_task = Task(**response.payload["task"])
                    if updated_task.status == "completed":
                        self.completed_tasks.append(updated_task)
                        completed_ids.add(updated_task.id)
                    else:
                        self.failed_tasks.append(updated_task)
                        completed_ids.add(updated_task.id)

                self.task_queue.remove(task)

        # Check for human approval if enabled
        settings = get_settings()
        if settings.enable_human_in_loop and self.plan:
            # Could pause here for human review of plan
            pass

        import datetime
        status = "success" if not self.failed_tasks else ("partial" if self.completed_tasks else "failed")

        return OrchestrationResult(
            plan_id=self.plan.id,
            goal=goal,
            status=status,
            completed_tasks=self.completed_tasks,
            failed_tasks=self.failed_tasks,
            final_output={"plan": self.plan.model_dump()},
            started_at=started_at,
            completed_at=datetime.datetime.utcnow(),
            total_turns=self.turn_count,
        )