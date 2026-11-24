"""Main orchestration engine."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic_ai.models import Model

from .config import get_settings
from .schemas import (
    AgentRole,
    OrchestrationResult,
    Plan,
    Task,
)
from .agents import (
    CoordinatorAgent,
    PlannerAgent,
    ResearcherAgent,
    CoderAgent,
    ReviewerAgent,
)


class Orchestrator:
    """Main entry point for multi-agent orchestration."""

    def __init__(self, model: Model | None = None):
        self.settings = get_settings()
        self.coordinator = CoordinatorAgent(model)
        self.planner = PlannerAgent(model)
        self.researcher = ResearcherAgent(model)
        self.coder = CoderAgent(model)
        self.reviewer = ReviewerAgent(model)

    async def run(self, goal: str) -> OrchestrationResult:
        """Run a full multi-agent orchestration for the given goal."""
        return await self.coordinator.start_orchestration(goal)

    async def run_with_plan(self, plan: Plan) -> OrchestrationResult:
        """Run orchestration with a pre-defined plan."""
        self.coordinator.plan = plan
        self.coordinator.task_queue = plan.tasks.copy()
        self.coordinator.completed_tasks = []
        self.coordinator.failed_tasks = []
        self.coordinator.turn_count = 0

        started_at = datetime.utcnow()
        return await self.coordinator._execute_plan(plan.goal, started_at)

    def create_plan(self, goal: str) -> Plan:
        """Synchronously create a plan (for inspection before execution)."""
        # This would need a sync version of the planner
        # For now, return a basic plan structure
        return Plan(goal=goal, tasks=[])


async def demo_orchestration(goal: str) -> OrchestrationResult:
    """Run a demo orchestration."""
    orchestrator = Orchestrator()
    return await orchestrator.run(goal)


if __name__ == "__main__":
    import sys

    goal = sys.argv[1] if len(sys.argv) > 1 else "Create a simple REST API with health check endpoint"

    async def main():
        print(f"🎯 Goal: {goal}")
        print("=" * 60)

        result = await demo_orchestration(goal)

        print(f"\n📊 Status: {result.status.upper()}")
        print(f"✅ Completed: {len(result.completed_tasks)} tasks")
        print(f"❌ Failed: {len(result.failed_tasks)} tasks")
        print(f"🔄 Turns: {result.total_turns}")

        if result.completed_tasks:
            print("\n📋 Completed Tasks:")
            for task in result.completed_tasks:
                print(f"  - [{task.assigned_to.value}] {task.description}")

        if result.failed_tasks:
            print("\n⚠️  Failed Tasks:")
            for task in result.failed_tasks:
                print(f"  - [{task.assigned_to.value}] {task.description}: {task.error}")

    asyncio.run(main())