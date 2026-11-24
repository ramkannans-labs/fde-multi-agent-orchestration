#!/usr/bin/env python
"""CLI entry point for multi-agent orchestration."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator, demo_orchestration
from orchestrator.config import get_settings
from orchestrator.schemas import OrchestrationResult

app = typer.Typer(
    name="orchestrate",
    help="Multi-Agent Orchestration CLI - FDE Demo",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    goal: str = typer.Argument(..., help="The goal for the multi-agent system to achieve"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Run a multi-agent orchestration for the given goal."""

    settings = get_settings()

    # Check for API key
    if not settings.anthropic_api_key:
        console.print(
            Panel(
                "[yellow]No ANTHROPIC_API_KEY found in environment.[/yellow]\n"
                "Running in [bold]mock mode[/bold] with TestModel.\n"
                "Set ANTHROPIC_API_KEY in .env for real LLM calls.",
                title="⚠️  Configuration",
                border_style="yellow",
            )
        )

    console.print(Panel(f"[bold]{goal}[/bold]", title="🎯 Goal", border_style="blue"))

    async def _run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running orchestration...", total=None)
            result = await demo_orchestration(goal)
            progress.update(task, completed=True)
        return result

    result = asyncio.run(_run())

    # Display results
    _display_results(result, verbose)


@app.command()
def plan(
    goal: str = typer.Argument(..., help="The goal to create a plan for"),
):
    """Create and display an execution plan without running it."""

    from orchestrator import PlannerAgent
    from orchestrator.schemas import AgentMessage, AgentRole, MessageType
    import uuid

    console.print(Panel(f"[bold]{goal}[/bold]", title="🎯 Goal", border_style="blue"))

    async def _plan():
        planner = PlannerAgent()
        msg = AgentMessage(
            from_agent=AgentRole.COORDINATOR,
            to_agent=AgentRole.PLANNER,
            type=MessageType.TASK,
            payload={"goal": goal},
            correlation_id=str(uuid.uuid4()),
            requires_response=True,
        )
        response = await planner.process_message(msg)
        return response

    response = asyncio.run(_plan())

    if response and response.payload.get("plan"):
        from orchestrator.schemas import Plan
        plan = Plan(**response.payload["plan"])
        _display_plan(plan)
    else:
        console.print("[red]Failed to create plan[/red]")


@app.command()
def check():
    """Check configuration and environment."""
    settings = get_settings()

    table = Table(title="Configuration Check")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")

    # API Key
    api_key_status = "✅ Set" if settings.anthropic_api_key else "❌ Missing (mock mode)"
    table.add_row("ANTHROPIC_API_KEY", "***" if settings.anthropic_api_key else "(empty)", api_key_status)

    # Model
    table.add_row("LLM_MODEL", settings.llm_model, "✅")

    # Other settings
    table.add_row("MAX_TURNS", str(settings.max_turns), "✅")
    table.add_row("HUMAN_IN_LOOP", str(settings.enable_human_in_loop), "✅")
    table.add_row("LOG_LEVEL", settings.log_level, "✅")

    console.print(table)

    if not settings.anthropic_api_key:
        console.print("\n[yellow]Tip: Copy .env.example to .env and add your ANTHROPIC_API_KEY[/yellow]")


def _display_results(result: OrchestrationResult, verbose: bool):
    """Display orchestration results."""

    # Status panel
    status_color = {"success": "green", "partial": "yellow", "failed": "red"}.get(result.status, "white")
    console.print(
        Panel(
            f"Status: [{status_color}]{result.status.upper()}[/{status_color}]\n"
            f"Completed: {len(result.completed_tasks)} tasks\n"
            f"Failed: {len(result.failed_tasks)} tasks\n"
            f"Turns: {result.total_turns}",
            title="📊 Result",
            border_style=status_color,
        )
    )

    # Completed tasks
    if result.completed_tasks:
        table = Table(title="✅ Completed Tasks")
        table.add_column("Role", style="cyan")
        table.add_column("Description")
        table.add_column("Output Preview")

        for task in result.completed_tasks:
            output_preview = str(task.output_data)[:100] + "..." if len(str(task.output_data)) > 100 else str(task.output_data)
            table.add_row(task.assigned_to.value, task.description, output_preview)

        console.print(table)

    # Failed tasks
    if result.failed_tasks:
        table = Table(title="❌ Failed Tasks")
        table.add_column("Role", style="cyan")
        table.add_column("Description")
        table.add_column("Error", style="red")

        for task in result.failed_tasks:
            table.add_row(task.assigned_to.value, task.description, task.error or "Unknown error")

        console.print(table)

    # Verbose output
    if verbose and result.final_output:
        console.print("\n[bold]Final Output:[/bold]")
        console.print(result.final_output)


def _display_plan(plan):
    """Display an execution plan."""
    console.print(Panel(f"Plan ID: {plan.id}\nGoal: {plan.goal}", title="📋 Plan", border_style="blue"))

    table = Table(title="Tasks")
    table.add_column("ID", style="dim")
    table.add_column("Assigned To", style="cyan")
    table.add_column("Description")
    table.add_column("Dependencies")
    table.add_column("Status")

    for task in plan.tasks:
        deps = ", ".join(task.depends_on) if task.depends_on else "—"
        table.add_row(task.id[:8], task.assigned_to.value, task.description, deps, task.status)

    console.print(table)


if __name__ == "__main__":
    app()