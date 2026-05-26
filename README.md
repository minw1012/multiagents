# Multiagents: A Model-Driven Tool-Using Agent Runtime

This repository contains the runtime implementation of a multi-agent terminal system built around a canonical tool-calling loop.

TLDR: the model decides when to call tools and when to stop; the runtime executes tool calls, enforces policy, records observations, and loops until completion.

## Abstract
Modern agent systems often fail because control logic is overfit to brittle intent routing, or because tool execution lacks robust safety and recovery semantics. This project implements a model-driven runtime where the language model is the controller and the harness is the executor. The harness integrates typed tools, stateful memory, execution traces, and permission-aware policy checks. The system supports document analysis, tabular workflows, local knowledge lookup, code-oriented actions, and optional network/database/browser utilities within explicit trust boundaries. A deterministic fallback path is included for environments where model-native tool control is unavailable.

## Summary Figure

![Publication-style summary figure](model_driven_tool_runtime.svg)

The model is responsible for control decisions.
The runtime is responsible for execution, safety checks, and state updates.
When `Need tool call now?` is `No`, the controller can either finish (`FINAL`) or return a clarification question (`CLARIFY/CHAT`) if information is still missing.

## System

### Harness Definition

```text
Harness = Tools + Knowledge + Observation + Action Interfaces + Permissions
```

### Runtime Components
- `DynamicLoopOrchestrator`: primary controller with model-native tool loop and fallback execution path.
- `MCTS plan optimizer`: searches multiple candidate plans and selects higher-scoring execution workflows before runtime.
- `Constraint/Verifier/Selector pipeline`:
  - `constraint_agent` builds per-step contracts (preconditions/postconditions).
  - `selector_agent` dynamically chooses reasoning/tool/code/clarification actions.
  - `verifier_agent` validates each step output and emits fix suggestions.
- `ToolRegistry`: typed tool catalog with schemas, permissions, ownership, retries, and timeouts.
- `ExecutionPolicy`: risk-aware gating, approval checks, trust boundaries for network/filesystem actions.
- `MemoryStore`: session history, workflow events, knowledge documents, and experience records.
- `ExperienceAgent`: captures solved runs and distills reusable local skills under `skills/distilled/`.
- `terminal_chat.py`: interactive terminal interface for human-in-the-loop execution.

### Main Code Locations
- Runtime and tools: `multi_agent_system.py`
- Terminal interface: `terminal_chat.py`
- Modular entrypoints (phase 1): `src/core`, `src/tools`, `src/agents`, `src/policy`, `src/skills`
- Architecture docs: `docs/architecture.md`, `docs/tools.md`, `docs/dev-guide.md`

## Main Features

### Task Families
- Knowledge lookup from local workspace content and ingested references.
- Document reading and summarization for `.pdf`, `.docx`, `.txt`, `.md`.
- Tabular analysis for `.csv`, `.xlsx`.
- ML workflow blocks: preprocessing, model suggestion, tuning, training, evaluation, reporting.
- Code-task actions: read/search/edit files and run shell commands.
- Optional utility actions: network HTTP calls, sqlite queries, browser-like page inspection.

### Representative Tools
- File/search: `list_workspace_files`, `search_workspace_text`, `read_text_file`, `write_text_file`
- Document: `read_document_file`, `summarize_text`, `extract_key_points`
- Tabular: `read_spreadsheet_preview`, `profile_tabular_columns`, `analyze_tabular_with_python`
- Code: `read_code_file`, `read_code_span`, `replace_text_in_file`, `run_shell_command`
- ML: `process_data`, `model_suggest`, `tune_models`, `train_models`, `evaluate_models`, `generate_report`
- Knowledge: `kb_search`, `knowledge_ingest_workspace_docs`, `knowledge_list_sources`, `knowledge_get_doc`
- Skills: `skill_install_from_git`, `skill_list_installed`

## Code

### Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Optional (recommended for model-native tool loop):

```bash
export OPENAI_API_KEY="<your_api_key>"
```

Without `OPENAI_API_KEY`, the system runs in deterministic/local fallback mode.

## How to Run

### 1) Start interactive terminal

```bash
python3 terminal_chat.py --workspace . --model gpt-4o
```

`--workspace .` scopes default file and tool operations to the current directory.

### 2) Use built-in commands
- `/help`
- `/tools`
- `/skills list`
- `/skills install <repo_url> [alias] [ref]`
- `/file summarize <path_to_docx_or_pdf>`
- `/raw on|off`
- `/exit`

### 3) Example prompts
- `could you walk through this workspace?`
- `please summarize README.md`
- `analyze data/logreg_dataset.csv`
- `do you find this key d5bbc8180dba11ecb1e81171463288e9 in the json file`
- `could you help me download a data we can run it for logistic regression model?`

## Reliability and Safety

### Failure Recovery
The runtime includes explicit recovery mechanisms:
- failure classification (`missing_input`, `parse_error`, `timeout`, `policy_block`, `command_missing`, `json_error`)
- recovery step synthesis and dynamic insertion
- reflection logging and repeat-guard behavior
- focused clarification when recovery budget is exhausted
- step-level contract precheck and post-execution verification (`precondition -> execute -> verify -> recover/replan`)
- verifier notes attached to execution output for traceability

### Experience Distillation
- each completed run can be summarized into an experience entry (`skills/experience_catalog.json`)
- experience is distilled into a local skill (`skills/distilled/<skill_name>/SKILL.md`)
- distilled skills are auto-registered in `skills/skills_manifest.json` with `repo_url=local_distilled`

### Policy Model
- permission-level risk mapping (`low`, `medium`, `high`)
- explicit approval requirement for high-risk actions
- trusted domain checks for network access
- filesystem boundary checks for database paths

## Reproducibility Notes
- Python package requirements are listed in `requirements.txt`.
- Runtime behavior is deterministic in fallback mode and model-dependent in LLM mode.
- Tool results, observations, and event phases are stored in workflow state for traceability.
