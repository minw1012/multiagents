# Multi-Agent System Redesign (Knowledge + ML Workflow)

## 1. Quick Clarification on Naming

The old names were:
- `KB_QUERY`: query historical information from the knowledge base.
- `DS_PIPELINE`: run data science workflow steps end-to-end.

To make intent names easier to understand, this design uses:
- `KNOWLEDGE_LOOKUP` (instead of `KB_QUERY`)
- `ML_WORKFLOW` (instead of `DS_PIPELINE`)

Important:
- These are **primary route labels**, not isolated silos.
- `ML_WORKFLOW` can optionally call knowledge retrieval when user requirements need historical/policy/context information.

## 2. Direct Answers to Your Three Questions

### 1) Should we download the latest skills?
- Yes. Tools should support downloading and managing skills from online repositories.
- This enables a code agent to dynamically extend capabilities per task.
- In this implementation, skill operations are exposed as tools (`skill_install_from_git`, `skill_list_installed`).

### 2) Terminal chat + agent interaction + ChatGPT 4o
- Implemented in `terminal_chat.py`.
- Default model is `gpt-4o` (configurable via `--model`).
- Fallback supported: if `OPENAI_API_KEY` or SDK is unavailable, the app returns local structured output.

### 3) Redesign summary
- Core implementation is in `multi_agent_system.py`.
- Two primary intent branches are supported:
  - `KNOWLEDGE_LOOKUP`: retrieve historical knowledge snippets.
  - `ML_WORKFLOW`: dynamically decomposed workflow driven by user requirements.
  - `ML_WORKFLOW` can be knowledge-enriched before planning.

## 3. Overall Architecture

Core components:
- `Message`: one-step communication payload.
- `State`: workflow-level shared context.
- `Supervisor`: centralized routing and transition logging.
- `Agent`: isolated business unit; agents do not call each other directly.
- `PlannerAgent`: compiles a dynamic step list from user requirements.
- `WorkflowControllerAgent`: executes the dynamic step list step-by-step.
- `ToolRegistry`: unified tool registration, permission, and invocation.
- `MemoryStore`: four-layer memory management.

Main execution path:
1. User input enters `IntentRouterAgent`.
2. Router chooses a primary intent and whether knowledge enrichment is needed.
3. If enrichment is needed, route to `kb_retriever` first, then return to `planner`.
4. `planner` compiles steps based on constraints (not a fixed pipeline).
5. `workflow_controller` dispatches each step to the right capability agent.
6. `Supervisor` keeps dispatching until `receiver == user`.

### Diagram (High-Level)

```mermaid
flowchart TD
    U[User in Terminal] --> T[terminal_chat.py]
    T --> S[Supervisor]
    S --> I[IntentRouterAgent]
    I -->|KNOWLEDGE_LOOKUP direct| KB[KBRetrieverAgent]
    I -->|ML_WORKFLOW without KB| P[PlannerAgent]
    I -->|ML_WORKFLOW + KB needed| KB
    KB -->|knowledge_context_ready| P
    P --> C[WorkflowControllerAgent]

    C --> D[DataAgent]
    C --> M[ModelAgent]
    C --> TR[TrainerAgent]
    C --> E[EvaluatorAgent]
    C --> R[ReporterAgent]
    D --> C
    M --> C
    TR --> C
    E --> C
    R --> C

    KB --> OUT[Final Response to User]
    C --> OUT

    S -. reads/writes .-> ST[(State)]
    S -. reads/writes .-> MEM[(MemoryStore)]
    KB -. tool call .-> TOOLS[(ToolRegistry)]
    D -. tool call .-> TOOLS
    M -. tool call .-> TOOLS
    TR -. tool call .-> TOOLS
    E -. tool call .-> TOOLS
    R -. tool call .-> TOOLS
```

## 4. Memory Design

Four namespaces:
- `session`: short-term conversation context.
- `workflow`: task-level state, artifacts, and trace records.
- `knowledge`: long-term documents and searchable snippets.
- `profile`: user preferences and policy settings (reserved).

Unified API:
- `put(namespace, key, value)`
- `get(namespace, key)`
- `search(namespace, query, top_k)`

### Diagram (Memory Layers)

```mermaid
flowchart LR
    APP[Agents / Supervisor] --> SESSION[(session)]
    APP --> WORKFLOW[(workflow)]
    APP --> KNOWLEDGE[(knowledge)]
    APP --> PROFILE[(profile)]

    SESSION --> S1[Short-term conversation context]
    WORKFLOW --> W1[Task state and artifacts]
    KNOWLEDGE --> K1[Long-term documents and search]
    PROFILE --> P1[User preferences and policies]
```

## 5. Tool Design

Unified registry pattern:
- `ToolSpec(name, input_schema, output_schema, timeout_s, retry, permission, owner_agent)`
- `ToolRegistry.register(...)`
- `ToolRegistry.execute(name, **kwargs)`
- `ToolRegistry.list_tools()`

Current tool groups:
- Knowledge: `kb_search`
- Data: `process_data`, `feature_plan`
- Model: `model_suggest`, `tune_models`, `train_models`, `evaluate_models`, `error_analyze`
- Report: `generate_report`
- SkillOps: `skill_install_from_git`, `skill_list_installed`

### Diagram (Tool Invocation)

```mermaid
flowchart TD
    A[Agent] --> X[ToolRegistry.execute]
    X --> K[kb_search]
    X --> D[process_data]
    X --> F[feature_plan]
    X --> M[model_suggest]
    X --> U[tune_models]
    X --> T[train_models]
    X --> E[evaluate_models]
    X --> EA[error_analyze]
    X --> R[generate_report]
    X --> SI[skill_install_from_git]
    X --> SL[skill_list_installed]
    K --> RET[Structured Tool Result]
    D --> RET
    F --> RET
    M --> RET
    U --> RET
    T --> RET
    E --> RET
    EA --> RET
    R --> RET
    SI --> RET
    SL --> RET
    RET --> A
```

## 6. Intent Routing Strategy

Current `IntentRouterAgent` strategy:
- Use an LLM-based request understanding engine first (structured JSON output + code validation).
- Do not use keyword-based routing heuristics in the execution decision path.
- Determine a **primary intent**:
  - Knowledge-only requests -> `KNOWLEDGE_LOOKUP`.
  - Modeling/training/evaluation requests -> `ML_WORKFLOW`.
  - Casual or unclear requests -> `GENERAL_CHAT` (ask for clarification, do not trigger workflow).
- Determine whether `needs_knowledge` is true.
- If primary intent is `ML_WORKFLOW` and `needs_knowledge=true`, call `kb_retriever` first and route back to `planner`.

Fallback behavior:
- If LLM understanding is unavailable or parsing fails, do **not** auto-trigger execution.
- In fallback mode, the router asks for clarification by default.
- Fallback can still execute only when the user provides an explicit structured router command (JSON with `intent` and optional `requirements`).
- Low-confidence understanding is downgraded to `GENERAL_CHAT` to ask a focused clarification.

Current `PlannerAgent` strategy:
- Use validated requirement flags from semantic router output.
- Compile a dynamic step list.
- Send the plan to `WorkflowControllerAgent`.

Recommended evolution:
1. Keep LLM-first understanding with strict schema validation.
2. Add richer uncertainty calibration and confirmation prompts before expensive runs.
3. Add tool capability graph so router can plan around installed skills dynamically.

## 7. Business Flows (Composable)

### A. KNOWLEDGE_LOOKUP
- `intent_router -> kb_retriever -> user`
- Output: knowledge snippets + source references.

```mermaid
sequenceDiagram
    participant U as User
    participant S as Supervisor
    participant I as IntentRouter
    participant K as KBRetriever
    participant TR as ToolRegistry
    participant MEM as MemoryStore

    U->>S: user_request(task)
    S->>I: dispatch(message)
    I-->>S: intent=KNOWLEDGE_LOOKUP, receiver=kb_retriever
    S->>K: dispatch(message)
    K->>TR: kb_search(query)
    TR-->>K: snippets + source
    K->>MEM: write workflow/session records
    K-->>S: final_result(receiver=user)
    S-->>U: answer + citations
```

### B. ML_WORKFLOW (Dynamic)
- Main path:
  - `intent_router -> planner -> workflow_controller -> [dynamic steps] -> user`
- Knowledge-enriched path:
  - `intent_router -> kb_retriever -> planner -> workflow_controller -> [dynamic steps] -> user`
- Output: executed steps, best model, key metrics, and optional report markdown.

```mermaid
sequenceDiagram
    participant U as User
    participant S as Supervisor
    participant I as IntentRouter
    participant K as KBRetriever
    participant P as Planner
    participant C as WorkflowController
    participant D as DataAgent
    participant M as ModelAgent
    participant T as Trainer
    participant E as Evaluator
    participant R as Reporter
    participant TR as ToolRegistry

    U->>S: user_request(task)
    S->>I: dispatch
    I-->>S: intent=ML_WORKFLOW + needs_knowledge?
    alt knowledge needed
        S->>K: retrieve knowledge context
        K->>TR: kb_search
        TR-->>K: snippets + sources
        K-->>S: knowledge_context_ready(receiver=planner)
        S->>P: compile dynamic plan with knowledge context
    else no knowledge needed
        S->>P: compile dynamic plan
    end
    P-->>S: workflow_plan_ready
    S->>C: start workflow execution
    loop per planned step
        C->>S: step_request(target agent)
        S->>D: if step=data_prep/feature_engineering
        S->>M: if step=model_selection
        S->>T: if step=tuning/training
        S->>E: if step=evaluation/error_analysis
        S->>R: if step=reporting
        D-->>S: step_done
        M-->>S: step_done
        T-->>S: step_done
        E-->>S: step_done
        R-->>S: step_done
        S->>C: continue
    end
    C-->>S: final_result(receiver=user)
    S-->>U: executed_steps + metrics + optional report + optional knowledge sources
```

## 8. How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API key (optional but recommended):
```bash
export OPENAI_API_KEY="your_key"
```

3. Start terminal chat:
```bash
python terminal_chat.py --model gpt-4o --workspace .
```

4. Available commands:
- `/help`
- `/tools`
- `/skills list`
- `/skills install <repo_url> [alias] [ref]`
- `/raw on`
- `/exit`

## 9. Next Enhancements

1. Upgrade `knowledge` retrieval from keyword matching to vector search (`pgvector` or `milvus`).
2. Add strict tool input/output validation (`pydantic`).
3. Replace mock `train_models` with real executors (`sklearn`, `xgboost`, or `ray`).
4. Add HTML/PDF rendering and artifact persistence for reports.
5. Add async task queue (`Celery` or `Arq`) for long-running training jobs.
