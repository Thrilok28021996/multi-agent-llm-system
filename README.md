# multi-agent-llm-company-system

**Autonomous Multi-Agent System for Software Development**

multi-agent-llm-company-system is a fully autonomous AI system that simulates a complete software company using local LLMs (via Ollama). Nine specialized agents — CEO, CTO, Product Manager, Researcher, Developer, QA Engineer, DevOps Engineer, Security Engineer, and Data Analyst — collaborate through a structured workflow to discover real-world problems and deliver working software from scratch.

No cloud dependencies. Everything runs locally.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Agents](#agents)
- [Agent Interaction Workflow](#agent-interaction-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Mode](#interactive-mode)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Example Outputs](#example-outputs)
- [Supported Languages](#supported-languages)
- [Architecture Overview](#architecture-overview)
- [Troubleshooting](#troubleshooting)

---

## How It Works

At its core, multi-agent-llm-company-system orchestrates a pipeline of specialized agents that hand off work to one another — just like a real software team. Each agent has a defined role, a dedicated LLM, and a strict set of responsibilities. The orchestrator manages the lifecycle, handles failures through an escalation system, and persists state across sessions.

**Two file generation modes:**

| Mode | Command | Output Location | Best For |
|------|---------|----------------|----------|
| **Default (Isolated)** | `python main.py --run --problem "..."` | `output/solutions/solution_*/` | New projects, experiments |
| **Current Directory** | `python main.py --run --current-dir --problem "..."` | Your working directory | Enhancing existing codebases |

---

## Agents

### CEO
**Model:** `qwen3:8b` | **Temperature:** 0.7

The final decision-maker. Evaluates solutions using the **MTRR Framework**: Market fit, Technical soundness, Resource efficiency, and Risk assessment. Deliberately rejects 30–40% of first submissions to enforce quality. Provides structured fix lists on rejection — not vague feedback. Escalates after round 4+ if systemic failures are detected.

Decisions: `APPROVE` | `REJECT` | `REQUEST_MORE_INFO`

### CTO
**Model:** `qwen3:8b` | **Temperature:** 0.5

Designs system architecture and makes technology decisions. Operates with a **Simplicity Budget** (max 5 files for MVP) and always presents both a simple and a scalable design option. Produces Architecture Decision Records (ADRs) and a Handoff Contract detailed enough that the Developer needs no clarifying questions. Rates tech debt on a 1–5 scale with a payoff plan.

### Product Manager
**Model:** `qwen3:8b` | **Temperature:** 0.7

Translates problems into structured requirements using **GIVEN/WHEN/THEN acceptance criteria**. Performs Jobs-to-be-done analysis, defines out-of-scope boundaries, and validates market fit before any technical work begins.

### Researcher
**Model:** `qwen3:8b` | **Temperature:** 0.7

Discovers and validates real-world problems from Reddit and HackerNews. Scores problems by severity (LOW → CRITICAL) and frequency (RARE → VERY_FREQUENT). Collects evidence and cross-validates findings before passing to the PM.

### Developer
**Model:** `qwen2.5-coder:7b` | **Temperature:** 0.3

Writes complete, runnable code — never partial snippets or pseudocode. Follows a strict discipline: **Make it work → Make it clear → Make it complete**. Performs mental line-by-line execution before submitting. Declares a confidence score (0.0–1.0) at the end of every implementation. Supports multiple file format markers for structured output.

### QA Engineer
**Model:** `qwen3:8b` | **Temperature:** 0.5

Creates test plans, maps results to PM acceptance criteria, and runs optional code execution. Uses the **Critic Ensemble** system (five personas: Skeptic, Optimist, Pragmatist, Security, User) to review code from multiple angles before issuing a PASS or FAIL verdict.

### DevOps Engineer
**Model:** `qwen3:8b` | **Temperature:** 0.5

Handles deployment planning, generates Dockerfiles, manages dependencies, and configures monitoring. Runs `pip install` / `npm install` as part of the delivery phase.

### Security Engineer
**Model:** `qwen3:8b` | **Temperature:** 0.5

Performs vulnerability scanning, risk assessment, and best-practice review. Runs as an optional phase after QA validation and before CEO approval.

### Data Analyst
**Model:** `qwen3:8b` | **Temperature:** 0.5

Analyzes problem frequency, market opportunity, and solution viability. Provides data-driven recommendations to support the PM and CEO decision phases.

---

## Agent Interaction Workflow

The following shows how agents interact from problem input to code delivery.

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│   --problem "Build a CLI tool"   OR   --run (auto-discover)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 · RESEARCH                                              │
│  Agent: Researcher                                               │
│  ─────────────────────────────────────────────────────────────  │
│  • Scrapes Reddit, HackerNews for pain points                   │
│  • Scores problems by severity & frequency                      │
│  • Returns: DiscoveredProblem with evidence and sources         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 · DATA ANALYSIS                                         │
│  Agent: Data Analyst                                             │
│  ─────────────────────────────────────────────────────────────  │
│  • Analyzes market size, frequency, and viability               │
│  • Returns: market opportunity score + recommendation           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3 · REQUIREMENTS ANALYSIS                                 │
│  Agent: Product Manager                                          │
│  ─────────────────────────────────────────────────────────────  │
│  • Converts problem into RequirementsDoc                        │
│  • Defines GIVEN/WHEN/THEN acceptance criteria                  │
│  • Clarifies out-of-scope items                                 │
│  • Returns: structured RequirementsDoc artifact                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4 · OPPORTUNITY EVALUATION                                │
│  Agents: CEO + full team meeting                                 │
│  ─────────────────────────────────────────────────────────────  │
│  • Team meeting: all agents discuss go/no-go                    │
│  • CEO applies MTRR framework                                   │
│  • Decision: APPROVE → continue │ REJECT → stop                │
└────────────────────────┬────────────────────────────────────────┘
                         │ APPROVED
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5 · TECHNICAL DESIGN                                      │
│  Agent: CTO                                                      │
│  ─────────────────────────────────────────────────────────────  │
│  • Designs architecture with ADRs                               │
│  • Selects tech stack (prefers proven over trendy)              │
│  • Produces Handoff Contract for Developer                      │
│  • Returns: ArchitectureNote artifact                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6 · DESIGN REVIEW                                         │
│  Agents: CTO + PM + QA meeting                                   │
│  ─────────────────────────────────────────────────────────────  │
│  • Cross-team validation of architecture                        │
│  • PM checks design covers all acceptance criteria              │
│  • QA flags testability concerns                                │
│  • Issues: CTO redesigns │ No issues: continue                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 7 · IMPLEMENTATION                                        │
│  Agent: Developer                                                │
│  ─────────────────────────────────────────────────────────────  │
│  • Writes complete, runnable code                               │
│  • Generates full file/directory structure                      │
│  • Includes tests, requirements.txt / package.json             │
│  • Declares confidence score (0.0–1.0)                          │
│  • Returns: code files + confidence                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 8 · CODE EXECUTION  (optional, --run-code)               │
│  Agent: Developer (runs the code)                                │
│  ─────────────────────────────────────────────────────────────  │
│  • Executes entry point via sandbox                             │
│  • Captures stdout/stderr                                       │
│  • Feeds output into QA context                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 9 · QA VALIDATION                                         │
│  Agent: QA Engineer + Critic Ensemble                            │
│  ─────────────────────────────────────────────────────────────  │
│  • Maps code to acceptance criteria → PASS / FAIL / UNTESTED   │
│  • Critic Ensemble runs 5 personas (Skeptic, Optimist,          │
│    Pragmatist, Security, User)                                  │
│  • Issues list + suggestions                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  QA FAIL → Escalation System                            │   │
│  │  Round 1–3: Developer fixes code                        │   │
│  │  Round 4:   Decompose problem into sub-problems         │   │
│  │  Round 5:   CTO redesigns architecture                  │   │
│  │  Round 6:   PM rescopes requirements                    │   │
│  │  Round 7:   Full team brainstorm                        │   │
│  │  Round 8:   Pivot to completely different approach      │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │ QA PASS
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 10 · SECURITY REVIEW  (optional, enabled by default)     │
│  Agent: Security Engineer                                        │
│  ─────────────────────────────────────────────────────────────  │
│  • Scans for OWASP-style vulnerabilities                        │
│  • Rates risk and provides mitigation steps                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 11 · CEO APPROVAL                                         │
│  Agent: CEO                                                      │
│  ─────────────────────────────────────────────────────────────  │
│  • Audits evidence: execution proof, QA results, confidence     │
│  • Checks problem-solution fit                                  │
│  • APPROVE → Delivery                                           │
│  • REJECT  → Same escalation ladder as QA FAIL                 │
│  • REQUEST_MORE_INFO → Targeted re-investigation                │
└────────────────────────┬────────────────────────────────────────┘
                         │ APPROVED
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 12 · DELIVERY                                             │
│  Agents: DevOps + Developer                                      │
│  ─────────────────────────────────────────────────────────────  │
│  • Formats code (black / prettier / gofmt)                      │
│  • Generates Dockerfile                                         │
│  • Runs pip install / npm install                               │
│  • Auto-runs git init + initial commit                          │
│  • Writes solution to output directory                          │
│  • Prints file tree + quick-start commands                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 13 · RETROSPECTIVE  (optional)                            │
│  Agents: full team                                               │
│  ─────────────────────────────────────────────────────────────  │
│  • What worked well                                             │
│  • What didn't work                                             │
│  • Improvements for the next run                                │
│  • Persisted to agent memory for future sessions                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                    ✅ COMPLETED
          Workflow report saved to ~/.multi-agent-llm-company-system/reports/
```

### Inter-Agent Communication

Agents do not call each other directly. All communication flows through two systems:

- **Message Bus** — Priority queue per agent with pub/sub topic subscriptions. Agents publish events (e.g., `qa.failed`, `ceo.approved`) and subscribers react accordingly.
- **Shared Memory** — A persistent company-wide knowledge base. Agents write typed entries (problem, solution, decision, insight, artifact) tagged by importance. Any agent can query relevant past context before acting.

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

```bash
# Verify Ollama
ollama --version
ollama serve   # Start the Ollama server if not running
```

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/multi-agent-llm-company-system
cd multi-agent-llm-company-system

# Install Python dependencies
pip install -r requirements.txt

# Or install as a package

pip install -e .

# Check which models are installed
python main.py --check-models

# Pull any missing models
python main.py --install-models
```

**Required models** (pulled automatically with `--install-models`):
- `qwen3:8b` — CEO, CTO, PM, Researcher, QA, DevOps, Security, DataAnalyst
- `qwen2.5-coder:7b` — Developer (coding-specialized)

---

## Usage

### Run with a specific problem

```bash
python main.py --run --problem "Build a CLI todo app with SQLite"
```

### Autonomous discovery mode

```bash
# AI discovers problems from Reddit/HackerNews and solves them
python main.py --run
```

### Target a specific language

```bash
python main.py --run --problem "Build a REST API" --language typescript
python main.py --run --problem "Build a gRPC service" --language go
```

### Enhance an existing project

```bash
python main.py --enhance ./my-flask-app "Add JWT authentication middleware"
python main.py --enhance ./my-api "Add rate limiting" --approve
```

### Skip research, go straight to design and code

```bash
# Scaffold mode: PM → CTO → Developer (no web research)
python main.py --scaffold "FastAPI app for managing todos"
```

### Continuous mode

```bash
# Discover and solve problems in a loop
python main.py --continuous --max-iterations 5
```

### Interactive approval at each phase

```bash
python main.py --run --problem "Build a dashboard" --approve
```

### Dry run (design only, no code execution or file writes)

```bash
python main.py --dry-run --problem "Build a payment service"
```

### Offline mode (skip web research, use local knowledge only)

```bash
python main.py --offline --problem "Build a CLI calculator"
```

### Solution history and management

```bash
python main.py --list-solutions          # View all past runs
python main.py --rerun 3                 # Re-run solution #3
python main.py --feedback 3 good         # Rate solution #3
python main.py --export 3                # Export #3 as a zip
python main.py --list-sessions           # View past sessions
python main.py --session-id abc123       # Resume a session
python main.py --resume                  # Resume last interrupted workflow
```

### Token and time limits

```bash
python main.py --run --problem "..." --max-tokens 50000
python main.py --run --problem "..." --max-workflow-minutes 30
python main.py --run --problem "..." --max-rounds 5
```

### Pipe from stdin

```bash
echo "Build a URL shortener with Redis" | python main.py
cat requirements.md | python main.py --language typescript
```

### Custom output directory

```bash
python main.py --run --problem "Build Flask API" --output-dir "~/projects/my-app"
```

### Verbosity control

```bash
python main.py --verbose --problem "..."   # Debug-level output
python main.py --quiet --problem "..."     # Errors only
```

---

## Interactive Mode

`interactive_mode.py` provides a conversational coding assistant backed by the agent system, operating directly on your working directory.

### Generate code in current directory

```bash
python interactive_mode.py "Add user authentication with JWT"
```

### Review specific files

```bash
python interactive_mode.py --review app.py models.py auth.py
```

### Fix a bug in a file

```bash
python interactive_mode.py --fix "Memory leak in connection pool" --file db.py
```

### Explain code

```bash
python interactive_mode.py --explain complex_algorithm.py
```

### Multi-turn conversational mode

```bash
python interactive_mode.py --chat
# Then type requests naturally:
# > add rate limiting to the API
# > now write tests for it
# > explain the middleware
```

---

## Advanced Features

### Escalation System

The system never gives up on a failing workflow. When QA fails or the CEO rejects, the Escalation Manager rotates through strategies rather than repeating the same attempt:

| Round | Strategy | Description |
|-------|----------|-------------|
| 1–3 | Developer Fix | Developer addresses specific issues |
| 4 | Decompose | Break problem into smaller sub-problems |
| 5 | CTO Redesign | Completely redesign the architecture |
| 6 | PM Rescope | Narrow the requirements |
| 7 | Team Brainstorm | Full team meeting to generate new ideas |
| 8 | Pivot | Entirely different technical approach |
| 9+ | Simplify / Alternative Stack | Reduce complexity or switch technologies |

The escalation system tracks attempted approaches and prevents repeating failed strategies.

### Collaboration Systems

**Agent Meetings** — Structured multi-agent discussions used at the Opportunity Evaluation, Design Review, and Retrospective phases. Meeting types: BRAINSTORM, DECISION, REVIEW, PLANNING, DEBATE, STANDUP, RETROSPECTIVE, DEVILS_ADVOCATE.

**Critic Ensemble** — Five sequential code review personas applied to the Developer's output during QA:
- **Skeptic** — Finds every flaw
- **Optimist** — Constructive improvements
- **Pragmatist** — Real-world viability
- **Security** — Vulnerability detection
- **User** — Usability from end-user perspective

**Debate Orchestrator** — Two agents debate a topic, argue positions over two rounds, then a synthesizer produces a unified recommendation.

**Mixture-of-Agents (MoA) Aggregator** — Collects independent reviews from multiple agents, surfaces consensus issues vs. single-reviewer flags, and produces a final PASS / CONDITIONAL_PASS / FAIL verdict.

### Advanced Reasoning Agents

**Tree-of-Thoughts** (`agents/tree_of_thoughts.py`) — Generates N diverse approaches, scores each, picks the best branch, then produces the final response. Default: 3 branches = 5 LLM calls total.

**HyperTree Planner** (`agents/hypertree_planner.py`) — Decomposes complex tasks into a subtask tree (max depth 2), executes subtasks sequentially with self-reflection, then synthesizes the result.

**ReAct Loop** (`agents/react_loop.py`) — A Reason + Act tool-use loop. The LLM generates a response, the parser extracts `<tool_use>` blocks, tools execute, results inject back into context, and the loop continues until no tool calls remain or max iterations are hit.

### Memory System

**Agent Memory** — Each agent maintains its own persistent memory: conversation history (last 50 turns), recorded experiences (action → outcome → lessons learned), working memory for current task, and learned patterns. Saved to `~/.multi-agent-llm-company-system/memory/{AgentName}_memory.json`.

**Shared Memory** — Company-wide knowledge base with typed entries (problem, solution, decision, insight, meeting, artifact). Supports importance weighting, tag queries, and linked memory entries.

**RAG Store** (`memory/rag_store.py`) — Local TF-IDF retrieval-augmented generation with cosine similarity. No GPU needed. ~20–30MB RAM for 500 documents. Injects relevant past solutions into agent prompts.

**Session Manager** — Saves and resumes full workflow sessions. Supports session forking for parallel exploration.

### Tools (13 built-in)

All agents have access to Claude Code-style tools via `AgentToolsMixin`:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Create or overwrite files |
| `edit_file` | Replace text in a file |
| `bash_execute` | Run shell commands in a sandbox |
| `search_files` | Regex search across files |
| `list_files` | Glob pattern file listing |
| `web_search` | Search the web |
| `web_fetch` | Fetch a URL |
| `git_*` | Init, commit, push |
| `image_tools` | Image analysis |
| `pdf_tools` | PDF manipulation |
| `code_formatter` | Format code (black, prettier, gofmt, rustfmt) |
| `test_runner` | Run tests (pytest, npm test, go test) |

### Infrastructure

| System | Purpose |
|--------|---------|
| Message Bus | Inter-agent pub/sub communication with priority queues |
| Shared Memory | Cross-agent company-wide knowledge base |
| Agent Memory | Per-agent persistent learning and history |
| RAG Store | Local TF-IDF retrieval for past solutions |
| Escalation Manager | Never-give-up strategy rotation |
| Task Manager | Task lifecycle: PENDING → IN_PROGRESS → COMPLETED |
| Cost Tracker | Per-model and per-agent token usage tracking |
| Health Checker | Pre-flight validation (Ollama, models, disk, permissions) |
| Memory Monitor | Auto-compact triggers to prevent memory bloat |
| Progress Tracker | Phase-level progress with ETA |
| Error Recovery | Exponential backoff with circuit breaker pattern |
| Structured Logger | JSON logging with correlation IDs |
| Sandbox Executor | Resource-limited safe command execution |
| Enhanced LSP | Real language servers (pylsp, gopls, rust-analyzer) |
| MCP Protocol | Model Context Protocol STDIO/HTTP/SSE support |
| Output Validator | JSON schema and verdict validation for LLM outputs |
| Prompt Compressor | TF-IDF + position-weighted compression (30–60% reduction) |
| Artifact Parser | Parses LLM outputs into typed RequirementsDoc, ArchitectureNote, QAReport |

---

## Configuration

### Command-line overrides

```bash
# Use a single model for all agents
python main.py --run --single-model "mistral:latest"

# Use lightweight models (faster, lower RAM)
python main.py --run --lightweight

# Custom LLM host
python main.py --run --llm-host "http://192.168.1.100:11434"
```

### Config file

```bash
# Generate a default config.yaml
python main.py --generate-config

# Edit config.yaml to set models, paths, feature flags, etc.
# Then run with it:
python main.py --run --config config.yaml
```

Key config fields:

```yaml
llm:
  backend: ollama
  host: http://localhost:11434
  timeout: 120
  num_ctx: 65536

workflow:
  max_iterations_per_phase: 5
  enable_escalation: true
  enable_security_review: true
  enable_retrospective: true

memory:
  max_memory_items: 1000
  persist_dir: ~/.multi-agent-llm-company-system/memory

research:
  rate_limit_delay: 2.0
  max_results_per_source: 50
```

---

## Output Structure

### Default mode (isolated projects)

```
output/solutions/solution_PROB-20260317-0001/
├── src/
│   ├── main.py
│   └── utils.py
├── tests/
│   └── test_main.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

### Internal data directory

```
~/.multi-agent-llm-company-system/
├── memory/
│   ├── Developer_memory.json
│   ├── CEO_memory.json
│   ├── CTO_memory.json
│   └── ...
├── reports/
│   └── workflow_20260317_120000.json
├── logs/
│   └── interactive.jsonl
├── costs/
│   └── cost_history.json
├── results/
│   └── workflow_history.json
└── sessions/
    └── session_abc123def456.json
```

### Workflow report

Each run saves a detailed JSON report at `~/.multi-agent-llm-company-system/reports/workflow_YYYYMMDD_HHMMSS.json` containing:
- Complete execution log with timestamps
- All agent decisions and reasoning
- Generated code
- QA results and criterion mapping
- CEO approval chain
- Token usage and cost breakdown

---

## Example Outputs

### Terminal output during a workflow run

```
╔══════════════════════════════════════════════════════╗
║     multi-agent-llm-company-system - Autonomous Mode ║
╚══════════════════════════════════════════════════════╝

[RESEARCH] Researcher scanning Reddit, HackerNews...
  ✓ Found 12 problems. Top problem: "CLI tools lack good progress reporting"
  Severity: HIGH | Frequency: FREQUENT | Score: 84/100

[ANALYSIS] Product Manager defining requirements...
  Acceptance Criteria:
  • GIVEN a long-running command WHEN executed THEN show a progress bar
  • GIVEN completion WHEN finished THEN display elapsed time and summary
  Out of scope: GUI, web interface, Windows-only features

[DESIGN] CTO designing architecture...
  Stack: Python + rich (progress bars) + click (CLI)
  Files: main.py, progress.py, formatters.py, cli.py, tests/test_progress.py
  ADR-001: Use rich over tqdm — richer output, active maintenance
  Tech debt: 1/5 (minimal)

[MEETING] Opportunity Evaluation — CEO + Team
  Researcher: evidence strong, 847 upvotes across 12 threads
  Data Analyst: market score 82/100, low competition
  CEO: APPROVED ✓

[IMPLEMENTATION] Developer writing code...
  Confidence: 0.91
  Files written: 5

[QA] QA Engineer running validation...
  Critic Ensemble:
    Skeptic: missing edge case for empty input
    Security: no issues
    User: clean UX, clear output
  Criteria: 4/5 PASS, 1/5 FAIL (empty input handling)
  → Escalating to Developer (round 1)

[FIX] Developer addressing QA issues...
  Fixed: empty input guard added
  Confidence: 0.95

[QA] Re-validation...
  Criteria: 5/5 PASS ✓

[SECURITY] Security Engineer reviewing...
  No vulnerabilities found. Risk: LOW ✓

[APPROVAL] CEO final review...
  Evidence audit: execution proof ✓, QA results ✓, confidence 0.95 ✓
  APPROVED ✓

[DELIVERY] DevOps packaging solution...
  ✓ Code formatted (black)
  ✓ Dockerfile generated
  ✓ pip install completed
  ✓ git init + initial commit

══════════════════════════════════════════════════════
  ✅ COMPLETED in 4m 32s · 14,821 tokens
══════════════════════════════════════════════════════

Output: output/solutions/solution_PROB-20260317-0001/
├── src/
│   ├── main.py
│   ├── progress.py
│   ├── formatters.py
│   └── cli.py
├── tests/
│   └── test_progress.py
├── requirements.txt
├── Dockerfile
└── README.md

Quick start:
  cd output/solutions/solution_PROB-20260317-0001
  pip install -r requirements.txt
  python src/main.py --help
```

### `--list-solutions` output

```
Past Solutions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 #   Problem                            Status     Date
────────────────────────────────────────────────────────
 1   CLI progress reporting library     ✅ DONE    2026-03-17
 2   FastAPI todo app with SQLite        ✅ DONE    2026-03-16
 3   React dashboard for metrics         ❌ FAILED  2026-03-15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use --rerun N to retry, --export N to export, --feedback N good/bad to rate
```

### Generated project structure (Python CLI tool)

```
solution_PROB-20260317-0001/
├── src/
│   ├── main.py           # Entry point
│   ├── progress.py       # Core progress bar logic
│   ├── formatters.py     # Output formatters
│   └── cli.py            # Click CLI definitions
├── tests/
│   └── test_progress.py  # pytest test suite
├── requirements.txt      # rich>=13.0.0, click>=8.0.0
├── Dockerfile
├── README.md             # Quick start + usage
└── .gitignore
```

### Interactive mode session

```
$ python interactive_mode.py --chat

multi-agent-llm-company-system Interactive Mode
Type your request or 'exit' to quit.

> add rate limiting to the Flask API

[Developer] Reading existing files...
  Read: app.py, routes/api.py

[Developer] Implementing rate limiting...
  Writing: middleware/rate_limiter.py
  Editing: app.py (register middleware)
  Writing: tests/test_rate_limiter.py

Done. 3 files modified/created.

> write integration tests for the rate limiter

[QA Engineer] Generating test suite...
  Writing: tests/integration/test_rate_limiting.py

Done. Tests cover: per-IP limiting, burst handling, header validation.

> exit
Session saved. Resume with: python interactive_mode.py --session-id a3f91c
```

---

## Supported Languages

Specify with `--language`:

- `python` (default)
- `javascript`
- `typescript`
- `go`
- `rust`
- `java`
- `csharp`

---

## Architecture Overview

```
multi-agent-llm-company-system
│
├── main.py                    # CLI entry point, workflow launcher
├── interactive_mode.py        # Interactive coding session
│
├── agents/                    # Specialized AI agents
│   ├── base_agent.py          # BaseAgent, AgentConfig, TaskResult, retry logic
│   ├── agent_tools_mixin.py   # 13 Claude Code-style tools, first-principles thinking
│   ├── ceo.py                 # CEO: MTRR decisions, evidence audit
│   ├── cto.py                 # CTO: architecture, ADRs, handoff contract
│   ├── product_manager.py     # PM: GIVEN/WHEN/THEN criteria, JTBD
│   ├── researcher.py          # Researcher: Reddit/HN discovery, scoring
│   ├── developer.py           # Developer: complete code generation
│   ├── qa_engineer.py         # QA: criterion mapping, critic ensemble
│   ├── devops_engineer.py     # DevOps: Dockerfile, deployment
│   ├── security_engineer.py   # Security: vulnerability scan
│   ├── data_analyst.py        # DataAnalyst: market analysis
│   ├── tree_of_thoughts.py    # Advanced: branch-and-score reasoning
│   ├── hypertree_planner.py   # Advanced: subtask decomposition
│   └── react_loop.py          # Advanced: reason + act tool loop
│
├── orchestrator/              # Workflow management
│   ├── workflow.py            # CompanyWorkflow: 13-phase pipeline
│   ├── task_manager.py        # Task lifecycle tracking
│   ├── escalation.py          # Never-give-up strategy rotation
│   ├── message_bus.py         # Inter-agent pub/sub communication
│   ├── task_queue.py          # Async priority task queue
│   ├── plan_mode.py           # Human approval workflow
│   ├── artifact_parser.py     # Parse LLM output → typed artifacts
│   └── artifacts.py           # RequirementsDoc, ArchitectureNote, QAReport
│
├── collaboration/             # Multi-agent collaboration
│   ├── meeting.py             # Structured agent meetings (8 meeting types)
│   ├── critic_ensemble.py     # 5-persona code critique
│   ├── debate.py              # Structured two-agent debate
│   └── moa_aggregator.py      # Mixture-of-Agents synthesis
│
├── memory/                    # Persistence and learning
│   ├── agent_memory.py        # Per-agent: history, experiences, patterns
│   ├── shared_memory.py       # Company-wide typed knowledge base
│   ├── session.py             # Session save/resume/fork
│   ├── learning.py            # Learn from successes and failures
│   ├── context_manager.py     # Context window management
│   └── rag_store.py           # Local TF-IDF RAG retrieval
│
├── tools/                     # Agent-accessible tools
│   ├── unified_tools.py       # All 13 tools in one interface
│   ├── file_operations.py     # Read, write, edit, delete, move
│   ├── command_executor.py    # Sandboxed shell execution
│   ├── git_tools.py           # Git init, commit, push
│   ├── code_formatter.py      # black, prettier, gofmt, rustfmt
│   ├── test_runner.py         # pytest, npm test, go test
│   ├── enhanced_lsp.py        # pylsp, gopls, rust-analyzer
│   ├── mcp.py                 # Model Context Protocol support
│   └── tool_registry.py       # Tool schemas for LLM injection
│
├── research/                  # Problem discovery
│   ├── problem_discoverer.py  # Reddit + HackerNews scraping
│   ├── web_search.py          # Cached web search
│   ├── web_scraper.py         # Article text extraction
│   ├── cross_validator.py     # Solution validation
│   ├── credibility.py         # Source credibility scoring
│   ├── problem_statement_refiner.py  # Clarify vague problems
│   └── sources.py             # Source configuration
│
├── config/                    # Configuration
│   ├── settings.py            # LLMBackendConfig, WorkflowConfig, MemoryConfig
│   ├── models.py              # Agent-to-model mapping
│   ├── llm_client.py          # Ollama client singleton
│   ├── config_loader.py       # YAML/JSON config loading
│   ├── roles.py               # AgentRole definitions
│   └── validation.py          # Config validation
│
├── company/                   # Company simulation
│   ├── organization.py        # Company hierarchy
│   ├── backlog.py             # Multi-problem backlog management
│   ├── sprint.py              # Sprint planning
│   ├── meetings.py            # Meeting scheduling
│   ├── culture.py             # Company values
│   ├── hiring.py              # Agent role assignment
│   ├── performance.py         # Agent performance tracking
│   └── trust.py               # Inter-agent trust scoring
│
├── ui/                        # Output and display
│   ├── console.py             # Rich console: agent colors, phase tracking
│   ├── streaming.py           # Real-time LLM output streaming
│   ├── interactive.py         # Interactive command interface
│   └── logger.py              # File logging
│
└── utils/                     # Supporting utilities
    ├── cost_tracker.py        # Per-model/agent token and cost tracking
    ├── error_recovery.py      # Exponential backoff, circuit breaker
    ├── health_check.py        # Pre-flight validation
    ├── memory_monitor.py      # Memory usage and auto-compact
    ├── progress_tracker.py    # Phase-level progress with ETA
    ├── structured_logging.py  # JSON logging with correlation IDs
    ├── output_validator.py    # LLM output format validation
    ├── prompt_compressor.py   # TF-IDF token-saving compression
    ├── enhanced_code_parser.py  # Extract multi-file code from LLM output
    ├── file_lock.py           # Atomic file operations
    ├── sandbox.py             # Resource-limited command execution
    ├── input_validation.py    # CLI input validation
    └── hooks.py               # Pre/post tool execution hooks
```

---

## Troubleshooting

### Ollama not running

```bash
ollama serve
# Then retry your command
```

### Required models not found

```bash
python main.py --check-models
python main.py --install-models
```

### Files not appearing in output

```bash
# Check the workflow report
cat ~/.multi-agent-llm-company-system/reports/workflow_*.json | python -m json.tool | grep files_written
```

### Workflow interrupted mid-run

```bash
python main.py --resume
# Or resume a specific session:
python main.py --list-sessions
python main.py --session-id <id>
```

### High memory usage

```bash
# Enable auto-compact (enabled by default)
# Or reduce context window in config:
#   llm.num_ctx: 32768
```

### Run taking too long

```bash
python main.py --run --problem "..." --max-workflow-minutes 20
python main.py --run --problem "..." --lightweight
```

---

## License

MIT License
