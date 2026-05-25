# Developer Guide

## Project Entry Points

- Runtime: `multi_agent_system.py`
- Terminal UI: `terminal_chat.py`
- Modular interfaces: `src/`

## Adding a New Tool

1. Implement function in runtime code.
2. Register it via `ToolSpec` in `build_tool_registry`.
3. Assign permission and timeout.
4. Update docs (`docs/tools.md`).

## Adding a New Agent

1. Subclass `BaseAgent`.
2. Implement `handle(message, state)`.
3. Wire the agent in orchestrator/supervisor assembly.
4. Add exports in `src/agents/__init__.py`.

## Refactor Rule

During migration, prefer no behavior change:

- move code in small slices
- keep old imports working
- validate with compile/run checks after each slice
