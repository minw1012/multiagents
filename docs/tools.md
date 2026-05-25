# Tools Overview

## Tool Layers

- Registry: `ToolSpec`, `ToolRegistry`
- Construction: `build_tool_registry(memory, workspace)`
- Execution controls: timeout, retries, permission labels

## Key Tool Groups

- File/search: workspace listing, text search, read/write
- Document: read/summarize/extract
- Tabular: preview/profile/Python analyzer
- Code: file operations and shell execution
- ML workflow: preprocess/suggest/train/evaluate/report
- Skills: install/list local skills

## Permissions

Risk is determined by permission labels (low/medium/high) and enforced by policy checks.
