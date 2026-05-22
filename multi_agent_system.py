from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import heapq
import os
import time


Message = Dict[str, Any]
State = Dict[str, Any]
ToolFunc = Callable[..., Dict[str, Any]]


def now_ms() -> int:
    return int(time.time() * 1000)


def stable_score(seed_text: str, low: float, high: float) -> float:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return round(low + (high - low) * value, 4)


class MemoryStore:
    """
    4-layer memory:
    - session: short-turn history
    - workflow: task-level artifacts
    - knowledge: long-term docs/snippets
    - profile: user preferences
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {
            "session": {},
            "workflow": {},
            "knowledge": {},
            "profile": {},
        }

    def put(self, namespace: str, key: str, value: Dict[str, Any]) -> None:
        self._must_namespace(namespace)
        self._data[namespace][key] = value

    def get(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        self._must_namespace(namespace)
        return self._data[namespace].get(key)

    def append_session_message(self, session_id: str, role: str, content: str) -> None:
        row = self._data["session"].setdefault(session_id, {"messages": []})
        row["messages"].append({"role": role, "content": content, "timestamp_ms": now_ms()})

    def put_knowledge_doc(self, doc_id: str, title: str, text: str, source: str) -> None:
        self._data["knowledge"][doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "source": source,
            "updated_at_ms": now_ms(),
        }

    def search(self, namespace: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self._must_namespace(namespace)
        if namespace != "knowledge":
            return []
        q = query.lower().strip()
        rows = list(self._data["knowledge"].values())
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for row in rows:
            text = f"{row.get('title', '')}\n{row.get('text', '')}".lower()
            if not q:
                score = 0
            else:
                score = text.count(q)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def _must_namespace(self, namespace: str) -> None:
        if namespace not in self._data:
            raise ValueError(f"unknown namespace: {namespace}")


@dataclass
class ToolSpec:
    name: str
    func: ToolFunc
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    timeout_s: int = 60
    retry: int = 0
    permission: str = "default"
    owner_agent: str = "supervisor"


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def execute(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name not in self.tools:
            raise ValueError(f"tool not found: {name}")
        spec = self.tools[name]
        return spec.func(**kwargs)

    def list_tools(self) -> List[Dict[str, Any]]:
        out = []
        for spec in self.tools.values():
            out.append(
                {
                    "name": spec.name,
                    "permission": spec.permission,
                    "owner_agent": spec.owner_agent,
                    "timeout_s": spec.timeout_s,
                    "retry": spec.retry,
                }
            )
        return out


class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def handle(self, message: Message, state: State) -> Message:
        raise NotImplementedError


def compute_priority(task: Dict[str, Any]) -> int:
    score = task.get("base_priority", 60)
    if task.get("user_blocking"):
        score += 20
    if task.get("safety_related"):
        score += 30
    if task.get("blocks_other_tasks"):
        score += 15
    if task.get("deadline_ms", 999_999) < 3000:
        score += 10
    if task.get("retry_count", 0) > 0:
        score += 5
    if task.get("estimated_cost") == "high":
        score -= 5
    return max(0, min(100, score))


def is_ready(task: Dict[str, Any], completed_tasks: set[str]) -> bool:
    return all(dep in completed_tasks for dep in task.get("depends_on", []))


class Scheduler:
    def __init__(self):
        self.heap: List[Tuple[int, float, int, Dict[str, Any]]] = []
        self.counter = 0
        self.waiting: List[Dict[str, Any]] = []
        self.completed_tasks: set[str] = set()

    def submit(self, task: Dict[str, Any]) -> None:
        if is_ready(task, self.completed_tasks):
            self.counter += 1
            p = compute_priority(task)
            heapq.heappush(self.heap, (-p, task.get("created_at", time.time()), self.counter, task))
        else:
            self.waiting.append(task)

    def pop_next(self) -> Optional[Dict[str, Any]]:
        self._promote_ready_waiting()
        if not self.heap:
            return None
        return heapq.heappop(self.heap)[3]

    def mark_done(self, task_id: str) -> None:
        self.completed_tasks.add(task_id)
        self._promote_ready_waiting()

    def _promote_ready_waiting(self) -> None:
        still_waiting: List[Dict[str, Any]] = []
        for task in self.waiting:
            if is_ready(task, self.completed_tasks):
                self.counter += 1
                p = compute_priority(task)
                heapq.heappush(self.heap, (-p, task.get("created_at", time.time()), self.counter, task))
            else:
                still_waiting.append(task)
        self.waiting = still_waiting


class IntentRouterAgent(BaseAgent):
    def __init__(self):
        super().__init__("intent_router")

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        lowered = task.lower()
        kb_keywords = ["历史", "知识库", "之前", "文档", "kb", "history", "knowledge"]
        ds_keywords = ["训练", "模型", "评估", "数据", "pipeline", "train", "evaluate", "report"]

        kb_hit = any(k in lowered for k in kb_keywords)
        ds_hit = any(k in lowered for k in ds_keywords)

        if kb_hit and not ds_hit:
            intent = "KNOWLEDGE_LOOKUP"
            receiver = "kb_retriever"
        else:
            intent = "ML_WORKFLOW"
            receiver = "planner"

        state["intent"] = intent
        return {
            "sender": self.name,
            "receiver": receiver,
            "type": "intent_routed",
            "priority": 80,
            "content": {"task": task, "intent": intent},
            "metadata": {"trace_id": state["trace_id"]},
        }


class KBRetrieverAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("kb_retriever")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        result = self.tools.execute("kb_search", query=task, top_k=5)
        snippets = result.get("snippets", [])
        return {
            "sender": self.name,
            "receiver": "user",
            "type": "final_result",
            "priority": 75,
            "content": {
                "intent": "KNOWLEDGE_LOOKUP",
                "summary": "已检索知识库相关历史信息。",
                "snippets": snippets,
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("planner")

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        plan = [
            "clarify target and constraints",
            "prepare and clean data",
            "select candidate models",
            "train and compare",
            "evaluate and generate report",
        ]
        state["plan"] = plan
        return {
            "sender": self.name,
            "receiver": "data_agent",
            "type": "plan_ready",
            "priority": 70,
            "content": {"task": task, "plan": plan},
            "metadata": {"trace_id": state["trace_id"]},
        }


class DataAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("data_agent")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        data_result = self.tools.execute("process_data", task=task)
        state["data"] = data_result
        return {
            "sender": self.name,
            "receiver": "model_agent",
            "type": "data_ready",
            "priority": 68,
            "content": {"task": task, "data_profile": data_result},
            "metadata": {"trace_id": state["trace_id"]},
        }


class ModelAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("model_agent")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        data_profile = message["content"]["data_profile"]
        picked = self.tools.execute("model_suggest", task=task, data_profile=data_profile)
        state["model_candidates"] = picked
        return {
            "sender": self.name,
            "receiver": "trainer",
            "type": "model_selected",
            "priority": 66,
            "content": {"task": task, "models": picked["models"]},
            "metadata": {"trace_id": state["trace_id"]},
        }


class TrainerAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("trainer")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        models = message["content"]["models"]
        train_result = self.tools.execute("train_models", task=task, models=models)
        state["train_result"] = train_result
        return {
            "sender": self.name,
            "receiver": "evaluator",
            "type": "train_done",
            "priority": 64,
            "content": {"task": task, "train_result": train_result},
            "metadata": {"trace_id": state["trace_id"]},
        }


class EvaluatorAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("evaluator")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        train_result = message["content"]["train_result"]
        eval_result = self.tools.execute("evaluate_models", task=task, train_result=train_result)
        state["evaluation_result"] = eval_result
        return {
            "sender": self.name,
            "receiver": "reporter",
            "type": "evaluation_done",
            "priority": 62,
            "content": {"task": task, "evaluation_result": eval_result},
            "metadata": {"trace_id": state["trace_id"]},
        }


class ReporterAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("reporter")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        evaluation_result = message["content"]["evaluation_result"]
        report = self.tools.execute("generate_report", task=task, evaluation_result=evaluation_result)
        state["report"] = report
        return {
            "sender": self.name,
            "receiver": "user",
            "type": "final_result",
            "priority": 80,
            "content": {
                "intent": "ML_WORKFLOW",
                "summary": "已完成数据处理、模型训练评估，并生成报告。",
                "report": report["report_markdown"],
                "best_model": report["best_model"],
                "metrics": report["metrics"],
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class Supervisor:
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents

    def dispatch(self, message: Message, state: State) -> Message:
        receiver = message["receiver"]
        state["messages"].append(message)
        state["step_count"] += 1

        if receiver == "user":
            return message

        if receiver not in self.agents:
            err = {
                "sender": "supervisor",
                "receiver": "user",
                "type": "error",
                "priority": 100,
                "content": {"error": f"unknown receiver: {receiver}"},
                "metadata": {"trace_id": state["trace_id"]},
            }
            state["errors"].append(err["content"]["error"])
            state["messages"].append(err)
            return err

        response = self.agents[receiver].handle(message, state)
        state["messages"].append(response)
        return response


def build_tool_registry(memory: MemoryStore) -> ToolRegistry:
    tools = ToolRegistry()

    def kb_search(query: str, top_k: int = 5) -> Dict[str, Any]:
        hits = memory.search("knowledge", query=query, top_k=top_k)
        snippets = []
        for row in hits:
            text = row["text"][:280].replace("\n", " ")
            snippets.append({"doc_id": row["doc_id"], "source": row["source"], "snippet": text})
        return {"snippets": snippets}

    def process_data(task: str) -> Dict[str, Any]:
        mode = "classification"
        lowered = task.lower()
        if any(x in lowered for x in ["预测价格", "回归", "regression", "price", "sales"]):
            mode = "regression"
        return {
            "mode": mode,
            "cleaning_steps": ["drop_duplicates", "impute_missing", "standardize_numeric"],
            "split": {"train": 0.7, "valid": 0.15, "test": 0.15},
        }

    def model_suggest(task: str, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        if data_profile["mode"] == "classification":
            models = ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"]
        else:
            models = ["LinearRegression", "RandomForestRegressor", "XGBoostRegressor"]
        return {"models": models}

    def train_models(task: str, models: List[str]) -> Dict[str, Any]:
        rows = []
        for m in models:
            if "Classifier" in m or m == "LogisticRegression":
                metric = stable_score(task + m, 0.78, 0.95)
                rows.append({"model": m, "metric_name": "f1", "metric_value": metric})
            else:
                metric = stable_score(task + m, 0.05, 0.30)
                rows.append({"model": m, "metric_name": "rmse", "metric_value": metric})
        return {"runs": rows}

    def evaluate_models(task: str, train_result: Dict[str, Any]) -> Dict[str, Any]:
        runs = train_result["runs"]
        cls_mode = any(r["metric_name"] == "f1" for r in runs)
        if cls_mode:
            best = max(runs, key=lambda r: r["metric_value"])
            objective = "maximize"
        else:
            best = min(runs, key=lambda r: r["metric_value"])
            objective = "minimize"
        return {"best": best, "objective": objective, "all_metrics": runs}

    def generate_report(task: str, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        best = evaluation_result["best"]
        lines = [
            "# Model Report",
            "",
            f"- Task: {task}",
            f"- Objective: {evaluation_result['objective']}",
            f"- Best Model: {best['model']}",
            f"- Metric: {best['metric_name']}={best['metric_value']}",
            "",
            "## Candidate Results",
        ]
        for row in evaluation_result["all_metrics"]:
            lines.append(f"- {row['model']}: {row['metric_name']}={row['metric_value']}")
        report_md = "\n".join(lines)
        return {"report_markdown": report_md, "best_model": best["model"], "metrics": evaluation_result["all_metrics"]}

    tools.register(ToolSpec(name="kb_search", func=kb_search, permission="kb_read", owner_agent="kb_retriever"))
    tools.register(ToolSpec(name="process_data", func=process_data, permission="data_exec", owner_agent="data_agent"))
    tools.register(ToolSpec(name="model_suggest", func=model_suggest, permission="ml_plan", owner_agent="model_agent"))
    tools.register(ToolSpec(name="train_models", func=train_models, permission="ml_train", owner_agent="trainer"))
    tools.register(ToolSpec(name="evaluate_models", func=evaluate_models, permission="ml_eval", owner_agent="evaluator"))
    tools.register(ToolSpec(name="generate_report", func=generate_report, permission="report_write", owner_agent="reporter"))
    return tools


def load_markdown_as_knowledge(memory: MemoryStore, folder: str) -> int:
    count = 0
    for name in os.listdir(folder):
        if not name.endswith(".md"):
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_id = f"doc_{count+1:03d}"
        memory.put_knowledge_doc(doc_id=doc_id, title=name, text=text, source=path)
        count += 1
    return count


class MultiAgentSystem:
    def __init__(self, workspace: str):
        self.memory = MemoryStore()
        loaded = load_markdown_as_knowledge(self.memory, workspace)
        self.memory.put("workflow", "bootstrap", {"knowledge_docs_loaded": loaded})
        self.tools = build_tool_registry(self.memory)
        self.agents: Dict[str, BaseAgent] = {
            "intent_router": IntentRouterAgent(),
            "kb_retriever": KBRetrieverAgent(self.tools),
            "planner": PlannerAgent(),
            "data_agent": DataAgent(self.tools),
            "model_agent": ModelAgent(self.tools),
            "trainer": TrainerAgent(self.tools),
            "evaluator": EvaluatorAgent(self.tools),
            "reporter": ReporterAgent(self.tools),
        }
        self.supervisor = Supervisor(self.agents)

    def run(self, user_text: str, session_id: str = "default") -> Tuple[Message, State]:
        trace_id = f"trace_{now_ms()}"
        self.memory.append_session_message(session_id, "user", user_text)
        state: State = {
            "trace_id": trace_id,
            "session_id": session_id,
            "messages": [],
            "plan": None,
            "data": None,
            "model_candidates": None,
            "train_result": None,
            "evaluation_result": None,
            "report": None,
            "errors": [],
            "retry_count": {},
            "step_count": 0,
            "execution_metadata": {"started_at_ms": now_ms()},
        }
        message: Message = {
            "sender": "user",
            "receiver": "intent_router",
            "type": "user_request",
            "priority": 80,
            "content": {"task": user_text},
            "metadata": {"trace_id": trace_id},
        }
        max_steps = 24
        for _ in range(max_steps):
            if message["receiver"] == "user":
                break
            message = self.supervisor.dispatch(message, state)
        state["execution_metadata"]["ended_at_ms"] = now_ms()
        self.memory.put("workflow", trace_id, {"state": state, "final_message": message})
        self.memory.append_session_message(session_id, "assistant_structured", str(message["content"]))
        return message, state
