# Multiagents: A Model-Driven Tool-Using Agent Harness for Terminal Workflows

## Abstract
This repository implements a practical agent runtime centered on a single principle: the language model decides *when* to call tools and *when* to stop, while the runtime executes requested actions, records observations, enforces policy, and returns control to the model. The system targets mixed terminal tasks including document analysis, tabular data workflows, code-oriented operations, local knowledge lookup, and controlled network/database/browser-style actions. The implementation combines an LLM-native tool loop with a deterministic fallback path for robustness when model access is unavailable.

## 1. Problem Statement
Most task agents fail for one of two reasons:
1. Routing logic is overfit to brittle keyword rules.
2. Tool execution is under-specified (weak safety boundaries, poor recovery after failures).

This project addresses both by building around a canonical iterative loop and a domain harness that provides:
- operational tools,
- memory and state,
- observation streams,
- safety/permission constraints,
- failure reflection and recovery.

## 2. Core Agent Pattern

```text
                    THE AGENT PATTERN
                    =================

    User --> messages[] --> LLM --> response
                                      |
                            stop_reason == "tool_use"?
                           /                          \
                         yes                           no
                          |                             |
                    execute tools                    return text
                    append results
                    loop back -----------------> messages[]
```

### 2.1 Operational Interpretation
- The model is the controller.
- The harness is the executor.
- The loop terminates only when the model emits a non-tool answer (or safety/iteration limits trigger termination).

## 3. Method

### 3.1 System Definition
Let the runtime state be:
- `M_t`: message history at iteration `t`
- `T`: tool registry (name, schema, handler, permission)
- `P`: execution policy and trust boundaries
- `S_t`: workflow state (observations, tool outputs, events, recovery stats)

At each step, the model receives `M_t` (+ tool schemas) and returns either:
- a set of tool calls, or
- final text.

Tool calls are validated against policy `P`, executed, converted to structured tool results, appended to `M_{t+1}`, and fed back to the model.

### 3.2 Algorithm (Runtime Skeleton)

```python
def agent_loop(messages, tools, policy):
    while True:
        response = llm(messages=messages, tools=tools)
        messages.append(response.as_assistant_message())

        if response.stop_reason != "tool_use":
            return response.text

        results = []
        for call in response.tool_calls:
            if not policy.allow(call):
                output = {"ok": False, "error": "blocked by policy"}
            else:
                output = execute_tool(call)
            results.append(tool_result(call.id, output))

        messages.append(user_message(results))
```

## 4. Harness Architecture

```text
Harness = Tools + Knowledge + Observation + Action Interfaces + Permissions
```

### 4.1 Tools
Implemented in `build_tool_registry(...)` in `multi_agent_system.py`.

Representative groups:
- File/search: `list_workspace_files`, `search_workspace_text`, `read_text_file`, `write_text_file`
- Document: `read_document_file`, `summarize_text`, `extract_key_points`
- Tabular: `read_spreadsheet_preview`, `profile_tabular_columns`, `analyze_tabular_with_python`
- Code: `read_code_file`, `read_code_span`, `replace_text_in_file`, `run_shell_command`
- ML blocks: `process_data`, `model_suggest`, `tune_models`, `train_models`, `evaluate_models`, `generate_report`
- Knowledge: `kb_search`, `knowledge_ingest_workspace_docs`, `knowledge_list_sources`, `knowledge_get_doc`
- Skills: `skill_install_from_git`, `skill_list_installed`
- Network/DB/Browser: `network_http_request`, `network_download_file`, `sqlite_query`, `browser_open_page`, `browser_click_link`
- Observation: `observe_git_diff`, `observe_error_logs`, `observe_recent_events`, `observe_browser_state`

### 4.2 Memory and State
`MemoryStore` (same file) maintains:
- `session`: short-turn chat history
- `workflow`: trace events and artifacts
- `knowledge`: ingested documents and metadata
- `experience`: solved-problem summaries and reusable skill candidates
- `profile`: user preferences

### 4.3 Orchestration Layers
`DynamicLoopOrchestrator` executes two-tier control:
1. **Primary path**: model-native tool loop (`_run_model_tool_loop`) using OpenAI Responses API tools interface.
2. **Fallback path**: local planning/decision/recovery path when model access is unavailable or fails.

### 4.4 Policy and Trust Boundaries
`ExecutionPolicy` provides permission-aware gating:
- risk levels by permission class (`low`, `medium`, `high`)
- explicit approval requirement for high-risk actions
- network trust checks (scheme, private hosts, trusted domain policy)
- filesystem boundary checks for database paths

## 5. Reliability Mechanisms

### 5.1 Reflection and Recovery
On tool failure, the runtime performs:
1. failure classification (`missing_input`, `parse_error`, `timeout`, `policy_block`, etc.)
2. recovery-step synthesis
3. dynamic plan injection
4. continuation or targeted clarification

Key methods in `multi_agent_system.py`:
- `_classify_tool_failure(...)`
- `_build_recovery_steps(...)`
- `_inject_recovery_steps(...)`
- `_reflect_and_recover(...)`

### 5.2 Repeat-Guard
Repeated identical failing calls are bounded; the system requests focused clarification instead of endlessly retrying.

## 6. Reproducibility

### 6.1 Environment
- Python 3
- Dependencies in `requirements.txt`:
  - `openai>=1.40.0`
  - `pypdf>=5.0.0`

### 6.2 Setup
```bash
python3 -m pip install -r requirements.txt
```

Optional (enables model-native loop):
```bash
export OPENAI_API_KEY="<your_key>"
```

### 6.3 Run
```bash
python3 terminal_chat.py --workspace .
```

`--workspace .` scopes default file/tool operations to the current repository directory.

## 7. Evaluation Protocol (Recommended)
For rigorous validation, evaluate at least four classes of scenarios:
1. **Capability queries**: expected to return precise, non-hallucinated capability boundaries.
2. **Executable tasks**: expected to call appropriate tools and produce concrete outputs.
3. **Failure handling**: expected to recover or ask targeted clarifications (not generic retries).
4. **Safety constraints**: expected to block unsafe operations unless explicit approval is provided.

Suggested metrics:
- task success rate
- mean tool calls per successful task
- recovery success rate after first failure
- policy-violation rate (should be zero)
- clarification precision (focused vs generic)

## 8. Limitations
- This is a harness-oriented implementation, not a benchmark-trained autonomous agent.
- Quality depends on model availability and tool coverage.
- Network/database actions are policy-controlled and intentionally constrained.
- Local fallback remains conservative by design.

## 9. Project Structure
- Runtime and tools: `multi_agent_system.py`
- Terminal interface: `terminal_chat.py`
- Redesign notes: `system_redesign.md`
- Communication notes: `multi_agent_commu.md`

## 10. Quick Terminal Commands
- `/help`
- `/tools`
- `/skills list`
- `/skills install <repo_url> [alias] [ref]`
- `/file summarize <path_to_docx_or_pdf>`
- `/raw on|off`
- `/exit`
