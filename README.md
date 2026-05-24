# Multiagents

Adaptive multi-agent terminal system for:
- knowledge lookup
- document summary
- tabular data analysis
- code-oriented execution

The runtime is designed as a dynamic loop, not a fixed keyword router.

## Quick Start

```bash
python3 terminal_chat.py --workspace .
```

`--workspace .` means all file search/read/write/tool execution is scoped to the current directory.

## Core Runtime Loop

The orchestrator follows:

1. Goal Understanding
2. Plan
3. Tool Call
4. Observation
5. Reflection
6. Recovery/Replan (if needed)
7. Continue or Finish

Main implementation: `DynamicLoopOrchestrator` in `multi_agent_system.py`.

## New Reliability Logic (Reflect + Recovery + Replan)

The system now includes explicit failure handling instead of stopping after repetitive errors.

- Failure classification (`missing_input`, `parse_error`, `timeout`, `policy_block`, etc.)
- Recovery step synthesis based on failure category
- Plan injection (insert recovery steps into current plan and continue)
- Reflection logging in results (`reflections`) and event stream (`reflect` phase)
- Clarify only after recovery attempts are exhausted

Key methods:
- `_classify_tool_failure(...)`
- `_build_recovery_steps(...)`
- `_inject_recovery_steps(...)`
- `_reflect_and_recover(...)`

These methods are in `multi_agent_system.py` under `DynamicLoopOrchestrator`.

## Rule-Based Fast Path (Deterministic Intents)

For high-frequency lookup tasks (for example JSON key search), the orchestrator uses a deterministic pre-plan before LLM planning.

Current rule:
- detect token/key lookup intent
- run:
  - `list_workspace_files` for `*.json`
  - `search_workspace_text` for the target token

This avoids fragile repeated trial calls and improves consistency.

## Tooling Highlights

- `read_document_file`: parse `.docx/.pdf/.txt/.md`
- `read_spreadsheet_preview` / `profile_tabular_columns`: CSV/XLSX preview and profiling
- `analyze_tabular_with_python`: writes and executes a temporary Python analyzer script, then returns structured results
- `read_code_file` / `read_code_span` / `replace_text_in_file` / `run_shell_command`: code-task workflow

## Terminal UX

`terminal_chat.py` has English-first startup/help/fallback messages and renders:
- execution summary
- observations
- reflections
- event phases

## Example Queries

- `do you find this key d5bbc8180dba11ecb1e81171463288e9 in the json file`
- `analyze this file ./sample_data.xlsx`
- `summarize /absolute/path/to/file.pdf`
- `check multi_agent_system.py and run python3 -m py_compile multi_agent_system.py`

## Notes

- This project executes commands locally in the workspace (not containerized by default).
- Safety checks are enforced for risky shell patterns and policy-blocked tool calls.
- High-risk actions can require explicit approval (`approved=true` depending on tool permission policy).
