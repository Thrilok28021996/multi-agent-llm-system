# Company AGI - Usage Guide

## File Generation Modes

Company AGI supports two file generation modes, similar to Claude Code, Codex, and Gemini CLI:

### 1. **Default Mode: Isolated Projects** (Recommended)
```bash
python main.py --run
```

**Files generated in:** `output/solutions/solution_PROB-XXXXXXXX/`

**Use when:**
- Testing new ideas
- Autonomous problem discovery
- You want projects organized separately
- Running experiments

**Example output structure:**
```
output/solutions/solution_PROB-20260126-0001/
├── backend/
│   ├── src/
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       └── App.js
├── README.md
└── tests/
    └── test_main.py
```

### 2. **Current Directory Mode: Like Claude Code**
```bash
python main.py --run --current-dir --problem "Add authentication"
```

**Files generated in:** `.` (current working directory)

**Use when:**
- Working on existing project
- Want files generated directly here
- Similar workflow to Claude Code/Copilot

**Example: Enhancing existing Flask app**
```bash
cd my-flask-app/
python /path/to/Company-AGI/main.py --run --current-dir \
    --problem "Add JWT authentication to existing Flask app" \
    --language python
```

**Result:**
```
my-flask-app/
├── app.py              # (existing)
├── auth.py             # (generated)
├── middleware.py       # (generated)
├── requirements.txt    # (updated)
└── tests/
    └── test_auth.py    # (generated)
```

## Command Reference

### Basic Usage

```bash
# Discover problems and build solution (default output folder)
python main.py --run

# Build specific solution (default output folder)
python main.py --run --problem "Create a REST API for tasks"

# Build in current directory (like Claude Code)
python main.py --run --current-dir --problem "Add feature X"

# Custom output directory
python main.py --run --output-dir "~/projects/my-app"
```

### Session & History Management

```bash
# List past workflow runs
python main.py --list-solutions

# Re-run a past solution
python main.py --rerun 3

# Rate a solution (feeds into agent learning)
python main.py --feedback 3 good

# Export a solution as a zip bundle
python main.py --export 3

# List past workflow sessions
python main.py --list-sessions

# Resume a specific session
python main.py --run --session-id abc123def456

# Preview what agents would build without executing code
python main.py --run --problem "Build a CLI tool" --dry-run

# Work offline (no web research)
python main.py --run --problem "Build a calculator" --offline

# Verbose output for debugging
python main.py --run --problem "Build an API" --verbose

# Quiet mode (errors only)
python main.py --run --problem "Build an API" --quiet

# Resume last checkpoint
python main.py --run --resume
```

### Language Support

```bash
# Python (default)
python main.py --run --problem "Build CLI tool" --language python

# JavaScript/TypeScript
python main.py --run --problem "Build React dashboard" --language typescript

# Go
python main.py --run --problem "Build microservice" --language go

# Rust
python main.py --run --problem "Build system tool" --language rust
```

### Model Configuration

```bash
# Default models (optimized for 16GB RAM)
python main.py --run

# Lightweight models (faster, less RAM)
python main.py --run --lightweight

# Single model for all agents
python main.py --run --single-model "mistral:latest"
```

## Workflow Phases

Company AGI follows a complete software company workflow:

1. **Research Phase** (Researcher Agent)
   - Discovers problems from Reddit, HackerNews
   - OR uses your provided requirement

2. **Analysis Phase** (PM + Researcher)
   - Validates problem
   - Analyzes market opportunity

3. **Opportunity Evaluation** (CEO + Team Meeting)
   - Strategic decision to proceed or not

4. **Technical Design** (CTO)
   - Designs architecture
   - Assesses feasibility

5. **Implementation** (Developer)
   - Generates code files
   - Follows architecture specs

6. **QA Validation** (QA Engineer + Review Meeting)
   - Creates test plan
   - Validates implementation
   - Code review

7. **CEO Approval** (CEO)
   - Final go/no-go decision

## File Generation Details

### How Files Are Created

1. **Developer Agent** generates code based on:
   - CTO's architecture design
   - PM's requirements
   - Target language
   - Existing codebase patterns (if current-dir mode)

2. **File Parser** extracts files from LLM response:
   - Supports multiple format markers
   - Handles code blocks properly
   - Creates directory structure

3. **File Writer** saves to disk:
   - Creates parent directories automatically
   - Works with current directory or isolated folder
   - Logs all file operations

### File Formats Supported

The parser recognizes these file markers:

```
**File Path:** `backend/src/main.py`
```python
# code here
```
```

```
=== FILE: frontend/App.js ===
```javascript
// code here
```
```

```
#### File: tests/test_api.py
```python
# code here
```
```

## Examples

### Example 1: Build New Project (Default Mode)

```bash
# Let AI discover and solve a problem
python main.py --run

# Specify what to build
python main.py --run --problem "Build a todo list CLI app with SQLite"
```

**Output:** `output/solutions/solution_PROB-*/`

### Example 2: Enhance Existing Project (Claude Code Style)

```bash
cd ~/projects/my-api/
python /path/to/Company-AGI/main.py --run --current-dir \
    --problem "Add rate limiting middleware" \
    --language python
```

**Output:** Files generated in `~/projects/my-api/`

### Example 3: Build TypeScript Project

```bash
python main.py --run \
    --problem "Build Express REST API with TypeScript" \
    --language typescript \
    --output-dir "~/projects/my-api"
```

**Output:** `~/projects/my-api/solution_PROB-*/`

### Example 4: Review Existing Code

```bash
# Using interactive mode
python interactive_mode.py --review app.py models.py auth.py

# Get explanations
python interactive_mode.py --explain complex_algorithm.py
```

### Example 5: Fix Issues

```bash
# Fix specific issue in file
python interactive_mode.py --fix "Memory leak in user session" --file auth.py

# General fix (auto-finds relevant files)
python interactive_mode.py --fix "SQL injection vulnerability"
```

## Output Files

### Workflow Report
**Location:** `output/reports/workflow_{timestamp}.json`

Contains:
- Complete workflow execution log
- All agent decisions and reasoning
- Code generated
- QA results
- CEO approvals

**Size:** ~60-110 KB per run

### Agent Memory
**Location:** `output/memory/{AgentName}/`

Contains:
- Per-agent learning data
- Past experiences
- Pattern recognition

### Logs
**Location:** `output/logs/`

Contains:
- Structured logs with correlation IDs
- Agent communications
- Tool usage

## Configuration

### Via CLI Flags (Highest Priority)
```bash
python main.py --run --lightweight --output-dir "."
```

### Via Config File
```bash
# Generate default config
python main.py --generate-config

# Edit config.yaml, then:
python main.py --run --config config.yaml
```

### Via Environment
Set in `config/config.yaml`:
```yaml
output:
  solutions_dir: "./projects"
  reports_dir: "./output/reports"

models:
  developer: "codellama:34b"
  qa_engineer: "mistral:latest"
```

## Comparison with Claude Code

| Feature | Company AGI | Claude Code |
|---------|-------------|-------------|
| **File Generation** | ✅ Both modes | ✅ Current dir |
| **Current Dir Mode** | `--current-dir` | Default |
| **Isolated Projects** | Default | N/A |
| **Architecture** | Multi-agent workflow | Single agent |
| **Auto Discovery** | ✅ Autonomous | Manual prompts |
| **Code Review** | ✅ QA Agent | ✅ Built-in |
| **Decision Making** | ✅ CEO/CTO | N/A |
| **Meetings** | ✅ Team consensus | N/A |

## Best Practices

### When to Use Default Mode
- Exploring new ideas
- Autonomous problem solving
- Building standalone projects
- Testing different approaches

### When to Use Current Dir Mode
- Enhancing existing codebase
- Adding features to current project
- Working interactively
- Quick code generation

### Combining Both Modes
```bash
# 1. Let AI build initial solution in isolation
python main.py --run --problem "Build task management API"

# 2. Review generated code
cd output/solutions/solution_PROB-*/

# 3. If good, move to your project
cp -r output/solutions/solution_PROB-*/* ~/myproject/

# 4. Then use current-dir mode for enhancements
cd ~/myproject/
python /path/to/Company-AGI/main.py --run --current-dir \
    --problem "Add WebSocket support"
```

## Troubleshooting

### Files Not Generated?
1. Check `output/reports/workflow_*.json` for errors
2. Look at `artifacts.implementation.files_written`
3. Verify file parser recognized format
4. Check logs in `output/logs/`

### Wrong Output Directory?
```bash
# Check what directory will be used
python main.py --run --problem "test" --output-dir "." # Current dir
python main.py --run --problem "test"                  # output/solutions/
```

### LLM Not Generating Correct Format?
- The parser now supports multiple formats
- Check the Developer agent prompts for formatting rules
- Update prompts in `agents/developer.py` if needed

## Quick Start

```bash
# 1. Check models
python main.py --check-models

# 2. Install missing models
python main.py --install-models

# 3. Run in default mode (isolated project)
python main.py --run --problem "Build a CLI calculator"

# 4. Check generated files
ls -la output/solutions/solution_PROB-*/

# 5. OR run in current directory mode
cd my-project/
python /path/to/Company-AGI/main.py --run --current-dir \
    --problem "Add logging to all functions"

# 6. Review what was generated
git diff  # if using git
```

## Summary

**Default Behavior (Recommended):**
- Keeps projects organized in `output/solutions/`
- Easy to review before integrating
- Clean separation of concerns

**Current Directory Mode (Claude Code Style):**
- Use `--current-dir` flag
- Generates files directly in working directory
- Perfect for enhancing existing projects

Both modes keep the full Company AGI hierarchical workflow:
**CEO → CTO → Product Manager → Researcher → Developer → QA Engineer**
