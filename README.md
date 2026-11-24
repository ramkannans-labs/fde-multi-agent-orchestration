# FDE Multi-Agent Orchestration Demo

A demonstration of **multi-agent orchestration patterns** using PydanticAI — part of the FDE (Forward Deployed Engineer) skills portfolio.

## 🎯 What This Demonstrates

| FDE Skill | Implementation |
|-----------|----------------|
| **Agent Communication** | Typed message passing between specialized agents |
| **Workflow Orchestration** | Dependency-aware task execution with coordinator |
| **Human-in-the-Loop** | Checkpoint approvals with structured requests |
| **Structured Outputs** | Pydantic models for all agent I/O (no parsing needed) |
| **Error Handling** | Graceful degradation, retry logic, failure isolation |
| **Observability Ready** | Structured logging, correlation IDs, turn tracking |
| **Local-First Dev** | TestModel fallback when no API key configured |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COORDINATOR AGENT                         │
│  (Orchestrates workflow, manages state, handles approvals)  │
└────────────────────────────┬────────────────────────────────┘
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  PLANNER AGENT  │ │ RESEARCHER AGENT│ │   CODER AGENT   │
│ (Creates plans) │ │ (Gathers info)  │ │ (Writes code)   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  REVIEWER AGENT │
                    │ (Validates work)│
                    └─────────────────┘
```

### Agent Roles

| Role | Responsibility | Output |
|------|----------------|--------|
| **Planner** | Decompose goals into dependent tasks | `Plan` with `Task[]` |
| **Researcher** | Gather information, best practices, APIs | `Task.output_data.findings` |
| **Coder** | Implement solutions, write code | `Task.output_data.code` |
| **Reviewer** | Validate correctness, quality, security | `Task.output_data.review` |
| **Coordinator** | Execute plan, manage dependencies, aggregate results | `OrchestrationResult` |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- (Optional) Anthropic API key for real LLM calls

### Installation

```bash
# Clone and enter directory
cd fde-multi-agent-orchestration

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or leave empty for mock mode)
```

### Run a Demo

```bash
# Run orchestration with a goal
python scripts/orchestrate.py run "Create a REST API with health check and user endpoints"

# Just create a plan (no execution)
python scripts/orchestrate.py plan "Build a CLI tool for log analysis"

# Check configuration
python scripts/orchestrate.py check
```

### Example Output

```
🎯 Goal: Create a REST API with health check endpoint
============================================================
Running orchestration...

📊 Result: SUCCESS
✅ Completed: 4 tasks
❌ Failed: 0 tasks
🔄 Turns: 4

📋 Completed Tasks:
  - [planner] Create execution plan for REST API
  - [researcher] Research FastAPI best practices and patterns
  - [coder] Implement FastAPI application with health check
  - [reviewer] Review implementation for correctness and quality
```

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(empty)* | Enables real LLM; empty = mock mode |
| `LLM_MODEL` | `anthropic:claude-3-5-sonnet-20241022` | PydanticAI model string |
| `MAX_TURNS` | `10` | Max conversation turns per orchestration |
| `ENABLE_HUMAN_IN_LOOP` | `true` | Pause for approval at checkpoints |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## 🧪 Testing

```bash
# Run syntax/import checks
python -m py_compile src/orchestrator/**/*.py scripts/*.py

# Run tests (when added)
pytest tests/
```

## 📁 Project Structure

```
fde-multi-agent-orchestration/
├── .env.example              # Environment template
├── .gitignore
├── pyproject.toml            # Dependencies & config
├── README.md
├── scripts/
│   └── orchestrate.py        # CLI entry point (Typer + Rich)
└── src/orchestrator/
    ├── __init__.py           # Public exports
    ├── config.py             # Pydantic Settings
    ├── schemas.py            # Pydantic models (messages, tasks, plans)
    ├── orchestrator.py       # Main orchestration engine
    └── agents/
        └── __init__.py       # Agent implementations
```

## 🔑 Key Concepts

### Typed Message Passing
All inter-agent communication uses `AgentMessage` with:
- Structured `payload` (validated by Pydantic)
- `correlation_id` for tracing
- `requires_response` for sync/async patterns

### Dependency-Aware Execution
The coordinator builds a DAG from task dependencies and executes in topological order — tasks only run when their dependencies complete.

### Human-in-the-Loop
At configurable checkpoints, the coordinator emits `APPROVAL_REQUEST` messages with structured options. The workflow pauses until a `HumanApprovalResponse` is received.

### Mock Mode
When `ANTHROPIC_API_KEY` is not set, PydanticAI's `TestModel` is used — enabling full pipeline testing in CI without API costs.

## 🎓 Learning Outcomes

This demo shows how to:
1. **Design agent interfaces** with clear contracts (Pydantic schemas)
2. **Orchestrate complex workflows** with dependency management
3. **Handle failures gracefully** without losing progress
4. **Integrate human oversight** at critical decision points
5. **Maintain observability** with correlation IDs and structured logs
6. **Test locally** without external dependencies

## 📦 Extending

Ideas for extension:
- Add **parallel task execution** with `asyncio.gather`
- Integrate **real tools** (web search, code execution, API calls)
- Add **streaming UI** with WebSocket updates
- Implement **persistent state** (Redis, SQLite) for long-running workflows
- Add **evaluation harness** with golden sets for agent outputs
- Build **multi-user sessions** with session isolation

## 📄 License

MIT — Use freely for learning and portfolio demonstrations.