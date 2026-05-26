# skill_search_workspace_text

## Purpose
Reusable workflow learned from solved user requests.

## When To Use
Use when a new task has similar goal and tool pattern.

## Workflow
1. {'step': 'Gather workspace context', 'tool': 'search_workspace_text', 'arguments': {'query': 'find key 123abc in json files', 'top_k': 30}}
2. {'step': 'List JSON files in workspace', 'tool': 'list_workspace_files', 'arguments': {'pattern': '*.json', 'recursive': True, 'limit': 500}}
3. {'step': 'Search key across workspace text', 'tool': 'search_workspace_text', 'arguments': {'query': 'find key 123abc in json files', 'top_k': 80}}

## Common Failure Patterns
- No notable failure pattern captured yet.

## Tool Chain
- `search_workspace_text`

## Provenance
- experience_id: `exp_1779773898685`
- trace_id: `trace_1779773898671`
- intent: `GENERAL_CHAT`
- created_at_ms: `1779773898685`
