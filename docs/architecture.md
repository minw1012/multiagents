# Architecture

## Current Runtime (Phase 1)

The runtime remains implemented in `multi_agent_system.py` to avoid behavior changes.

A modular import surface is now available under `src/`:

- `src/core`: orchestrator, state, scheduler, supervisor
- `src/tools`: tool registry and tool builder
- `src/agents`: agent interfaces and concrete agents
- `src/policy`: execution policy
- `src/skills`: skill store

Phase 2 progress:

- `src/tools/registry.py` now owns `ToolSpec` and `ToolRegistry`.
- `src/skills/store.py` now owns `SkillStore`.
- `src/policy/execution.py` and `src/policy/risk.py` now own policy logic and risk map.
- `multi_agent_system.py` imports these components instead of defining them inline.

## Target Runtime (Phase 2)

Code movement plan:

1. Move pure utility functions first.
2. Move `ToolSpec`, `ToolRegistry`, `build_tool_registry` into `src/tools`.
3. Move `ExecutionPolicy` into `src/policy`.
4. Move agents into `src/agents`.
5. Keep a thin compatibility layer in `multi_agent_system.py` until migration is complete.

## Runtime Loop

1. User message enters orchestrator.
2. Router/planner decides next action.
3. Tool calls are validated by policy.
4. Results are appended to workflow state.
5. Loop continues until final user response.
