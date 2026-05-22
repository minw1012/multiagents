from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import heapq
import json
import os
import re
import subprocess
import time
from pathlib import Path


Message = Dict[str, Any]
State = Dict[str, Any]
ToolFunc = Callable[..., Dict[str, Any]]


def now_ms() -> int:
    return int(time.time() * 1000)


def stable_score(seed_text: str, low: float, high: float) -> float:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return round(low + (high - low) * value, 4)


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


DEFAULT_CLARIFICATION_REPLY = (
    "I need a semantically clear request before running execution. "
    "Please describe your goal, data/source scope, and constraints."
)

DEFAULT_CLARIFICATION_SUGGESTIONS = [
    "Run ML_WORKFLOW for churn prediction using /path/to/data.csv, include report",
    "Run KNOWLEDGE_LOOKUP for historical policy updates about onboarding",
    'Use explicit JSON command: {"intent":"ML_WORKFLOW","mode":"EXECUTE","requirements":{...}}',
]

UNSUPPORTED_EXECUTION_REPLY = (
    "This request is outside the currently executable pipeline. "
    "Current executable scope: KNOWLEDGE_LOOKUP and tabular ML workflow."
)


def default_general_chat_payload() -> Dict[str, Any]:
    return {
        "reply": DEFAULT_CLARIFICATION_REPLY,
        "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
    }


REQUIREMENT_BOOL_FIELDS = [
    "needs_data_prep",
    "needs_feature_engineering",
    "needs_model_selection",
    "needs_tuning",
    "needs_training",
    "needs_evaluation",
    "needs_error_analysis",
    "needs_report",
]


def default_workflow_requirements() -> Dict[str, Any]:
    return {
        "needs_data_prep": True,
        "needs_feature_engineering": False,
        "needs_model_selection": True,
        "model_hint": None,
        "needs_tuning": False,
        "needs_training": True,
        "needs_evaluation": True,
        "needs_error_analysis": False,
        "needs_report": True,
    }


def normalize_requirements(requirements: Optional[Dict[str, Any]], task: str) -> Dict[str, Any]:
    base = default_workflow_requirements()
    if isinstance(requirements, dict):
        for key in REQUIREMENT_BOOL_FIELDS:
            if key in requirements:
                base[key] = bool(requirements.get(key))
        model_hint = requirements.get("model_hint")
        if isinstance(model_hint, str) and model_hint.strip():
            base["model_hint"] = model_hint.strip()

    if not base["needs_model_selection"] and base["needs_training"] and not base.get("model_hint"):
        base["needs_model_selection"] = True
    if base["needs_error_analysis"] and not base["needs_evaluation"]:
        base["needs_evaluation"] = True
    return base


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def parse_explicit_router_command(task: str) -> Optional[Dict[str, Any]]:
    parsed = _extract_json_object(task)
    if not isinstance(parsed, dict):
        return None

    intent = str(parsed.get("intent", "")).upper()
    if intent not in {"GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW"}:
        return None

    mode_raw = str(parsed.get("mode", "EXECUTE")).upper()
    mode = "EXECUTE" if mode_raw in {"EXECUTE", "RUN", "WORKFLOW"} else "ANSWER_ONLY"
    needs_knowledge = parse_bool(parsed.get("needs_knowledge"), default=False)
    requirements = normalize_requirements(parsed.get("requirements"), task="")

    return {
        "intent": intent,
        "mode": mode,
        "confidence": 0.98,
        "needs_knowledge": needs_knowledge,
        "task_domain": str(parsed.get("task_domain", "general")).lower(),
        "supported_execution": True,
        "reply": str(parsed.get("reply", "")).strip(),
        "suggestions": parsed.get("suggestions", []),
        "requirements": requirements,
        "source": "explicit_command",
    }


class RequestUnderstandingEngine:
    """
    LLM-first request understanding.
    Falls back to conservative heuristic understanding when LLM is unavailable.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None

    def available(self) -> bool:
        return self.client is not None

    def understand(self, task: str) -> Dict[str, Any]:
        if self.client is None:
            return self._heuristic_understand(task, source="heuristic_fallback")

        system_prompt = (
            "You are a request understanding engine for a multi-agent platform. "
            "Infer user intent semantically, not via brittle keyword matching. "
            "Return strict JSON only. "
            "Executable scope currently includes: "
            "(1) knowledge lookup from loaded docs/history, "
            "(2) tabular ML workflow (preprocess, model select, tune, train, evaluate, report). "
            "Computer vision/object detection is not executable in current pipeline."
        )
        user_prompt = {
            "task": task,
            "allowed_intents": ["GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW"],
            "requirement_fields": REQUIREMENT_BOOL_FIELDS + ["model_hint"],
            "instructions": [
                "For casual chat or capability questions, use GENERAL_CHAT.",
                "For historical/doc retrieval requests, use KNOWLEDGE_LOOKUP.",
                "For data/model/training/evaluation/report work, use ML_WORKFLOW.",
                "Distinguish answer-only vs execution: capability inquiry should be answer-only.",
                "Do not claim unsupported execution capabilities.",
                "Set needs_knowledge=true when ML task should reference knowledge/history/policy/docs.",
                "If uncertain, use GENERAL_CHAT and ask a focused clarification.",
                "Do not hallucinate constraints.",
            ],
            "output_schema_hint": {
                "intent": "GENERAL_CHAT | KNOWLEDGE_LOOKUP | ML_WORKFLOW",
                "mode": "ANSWER_ONLY | EXECUTE",
                "confidence": "0.0-1.0",
                "needs_knowledge": "bool",
                "task_domain": "tabular_ml | knowledge | cv | nlp | general",
                "supported_execution": "bool",
                "reply": "string",
                "suggestions": ["string"],
                "requirements": {
                    "needs_data_prep": "bool",
                    "needs_feature_engineering": "bool",
                    "needs_model_selection": "bool",
                    "model_hint": "string|null",
                    "needs_tuning": "bool",
                    "needs_training": "bool",
                    "needs_evaluation": "bool",
                    "needs_error_analysis": "bool",
                    "needs_report": "bool",
                },
            },
        }

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
                temperature=0,
            )
            text = getattr(resp, "output_text", "") or ""
            parsed = _extract_json_object(text)
            if not parsed:
                return self._heuristic_understand(task, source="heuristic_after_parse_fail")
            return self._validate_understanding(parsed, task, source=f"llm:{self.model}")
        except Exception:
            return self._heuristic_understand(task, source="heuristic_after_llm_fail")

    def _validate_understanding(self, parsed: Dict[str, Any], task: str, source: str) -> Dict[str, Any]:
        intent = str(parsed.get("intent", "GENERAL_CHAT")).upper()
        if intent not in {"GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW"}:
            intent = "GENERAL_CHAT"

        mode_raw = str(parsed.get("mode", "ANSWER_ONLY")).upper()
        if mode_raw in {"EXECUTE", "RUN", "WORKFLOW"}:
            mode = "EXECUTE"
        else:
            mode = "ANSWER_ONLY"

        confidence = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        needs_knowledge = bool(parsed.get("needs_knowledge", False))
        task_domain = str(parsed.get("task_domain", "general")).lower().strip()
        if task_domain not in {"tabular_ml", "knowledge", "cv", "nlp", "general"}:
            task_domain = "general"
        supported_execution = parse_bool(
            parsed.get("supported_execution"),
            default=task_domain in {"tabular_ml", "knowledge", "general"},
        )

        reply = str(parsed.get("reply", "")).strip()
        suggestions = parsed.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = []
        suggestions = [str(x).strip() for x in suggestions if str(x).strip()][:3]

        req = normalize_requirements(parsed.get("requirements"), task)

        if mode == "EXECUTE" and not supported_execution:
            mode = "ANSWER_ONLY"
            intent = "GENERAL_CHAT"
            reply = UNSUPPORTED_EXECUTION_REPLY
            suggestions = DEFAULT_CLARIFICATION_SUGGESTIONS

        if confidence < 0.4 and intent != "GENERAL_CHAT":
            intent = "GENERAL_CHAT"
            mode = "ANSWER_ONLY"
            reply = (
                "I need one clarification before running the workflow. "
                "Do you want knowledge lookup, ML execution, or both?"
            )

        if mode == "ANSWER_ONLY":
            intent = "GENERAL_CHAT"

        if intent == "GENERAL_CHAT" and not reply:
            reply = DEFAULT_CLARIFICATION_REPLY
            if not suggestions:
                suggestions = DEFAULT_CLARIFICATION_SUGGESTIONS

        return {
            "intent": intent,
            "mode": mode,
            "confidence": confidence,
            "needs_knowledge": needs_knowledge,
            "task_domain": task_domain,
            "supported_execution": supported_execution,
            "reply": reply,
            "suggestions": suggestions,
            "requirements": req,
            "source": source,
        }

    def _heuristic_understand(self, task: str, source: str) -> Dict[str, Any]:
        explicit = parse_explicit_router_command(task)
        if explicit:
            return explicit

        return {
            "intent": "GENERAL_CHAT",
            "mode": "ANSWER_ONLY",
            "confidence": 0.35,
            "needs_knowledge": False,
            "task_domain": "general",
            "supported_execution": False,
            "reply": DEFAULT_CLARIFICATION_REPLY,
            "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
            "requirements": default_workflow_requirements(),
            "source": source,
        }

def build_dynamic_workflow_steps(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []

    if requirements.get("needs_data_prep"):
        steps.append({"kind": "data_prep", "agent": "data_agent", "name": "Prepare and clean data"})
    if requirements.get("needs_feature_engineering"):
        steps.append({"kind": "feature_engineering", "agent": "data_agent", "name": "Feature engineering"})
    if requirements.get("needs_model_selection"):
        steps.append({"kind": "model_selection", "agent": "model_agent", "name": "Select candidate models"})
    if requirements.get("needs_tuning"):
        steps.append({"kind": "hyperparameter_tuning", "agent": "trainer", "name": "Tune model hyperparameters"})
    if requirements.get("needs_training"):
        steps.append({"kind": "training", "agent": "trainer", "name": "Train model candidates"})
    if requirements.get("needs_evaluation"):
        steps.append({"kind": "evaluation", "agent": "evaluator", "name": "Evaluate model performance"})
    if requirements.get("needs_error_analysis"):
        steps.append({"kind": "error_analysis", "agent": "evaluator", "name": "Analyze model errors"})
    if requirements.get("needs_report"):
        steps.append({"kind": "reporting", "agent": "reporter", "name": "Generate final report"})

    if not steps:
        steps.append({"kind": "reporting", "agent": "reporter", "name": "Generate final report"})
    return steps


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
        terms = [t for t in re.split(r"[\s,.;:!?，。；：！？、/\\-]+", q) if len(t) >= 2]
        if not terms and q:
            terms = [q]
        rows = list(self._data["knowledge"].values())
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for row in rows:
            text = f"{row.get('title', '')}\n{row.get('text', '')}".lower()
            score = sum(text.count(term) for term in terms)
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


class SkillStore:
    """
    Stores downloaded skills and a small local manifest.
    Skills can be pulled from online git repositories and then exposed to agents as tools.
    """

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "skills_manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"skills": []}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("skills"), list):
                return data
        except Exception:
            pass
        return {"skills": []}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_skills(self) -> List[Dict[str, Any]]:
        return list(self._manifest.get("skills", []))

    def install_from_git(self, repo_url: str, ref: Optional[str] = None, alias: Optional[str] = None) -> Dict[str, Any]:
        if not repo_url or not isinstance(repo_url, str):
            return {"ok": False, "error": "repo_url is required"}

        safe_name = (alias or Path(repo_url.rstrip("/")).stem or f"skill_{now_ms()}").replace(" ", "_")
        target = self.root / safe_name

        if target.exists() and (target / ".git").exists():
            pull_cmd = ["git", "-C", str(target), "pull", "--ff-only"]
            pull = subprocess.run(pull_cmd, capture_output=True, text=True)
            if pull.returncode != 0:
                return {"ok": False, "error": pull.stderr.strip() or pull.stdout.strip()}
        elif target.exists() and not (target / ".git").exists():
            return {"ok": False, "error": f"target exists and is not a git repo: {target}"}
        else:
            clone_cmd = ["git", "clone", repo_url, str(target)]
            clone = subprocess.run(clone_cmd, capture_output=True, text=True)
            if clone.returncode != 0:
                return {"ok": False, "error": clone.stderr.strip() or clone.stdout.strip()}

        if ref:
            checkout_cmd = ["git", "-C", str(target), "checkout", ref]
            checkout = subprocess.run(checkout_cmd, capture_output=True, text=True)
            if checkout.returncode != 0:
                return {"ok": False, "error": checkout.stderr.strip() or checkout.stdout.strip()}

        rev_cmd = ["git", "-C", str(target), "rev-parse", "HEAD"]
        rev = subprocess.run(rev_cmd, capture_output=True, text=True)
        commit = rev.stdout.strip() if rev.returncode == 0 else ""

        record = {
            "name": safe_name,
            "repo_url": repo_url,
            "path": str(target),
            "ref": ref or "default",
            "commit": commit,
            "installed_at_ms": now_ms(),
        }
        skills = [s for s in self._manifest.get("skills", []) if s.get("name") != safe_name]
        skills.append(record)
        self._manifest["skills"] = skills
        self._save_manifest()
        return {"ok": True, "skill": record}


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
    def __init__(self, understanding_engine: RequestUnderstandingEngine):
        super().__init__("intent_router")
        self.understanding_engine = understanding_engine

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        understanding = self.understanding_engine.understand(task)
        intent = understanding.get("intent", "GENERAL_CHAT")
        mode = understanding.get("mode", "ANSWER_ONLY")
        needs_knowledge = bool(understanding.get("needs_knowledge", False))
        confidence = float(understanding.get("confidence", 0.5))
        requirements = normalize_requirements(understanding.get("requirements"), task)
        task_domain = understanding.get("task_domain", "general")
        supported_execution = bool(understanding.get("supported_execution", True))

        if intent == "GENERAL_CHAT" or mode != "EXECUTE":
            state["intent"] = intent
            state["needs_knowledge"] = False
            state["router_confidence"] = confidence
            state["router_source"] = understanding.get("source", "unknown")
            state["router_mode"] = mode
            state["task_domain"] = task_domain
            state["supported_execution"] = supported_execution
            reply = understanding.get("reply", "").strip()
            suggestions = understanding.get("suggestions", [])
            if not reply:
                fallback = default_general_chat_payload()
                reply = fallback["reply"]
                suggestions = fallback["suggestions"]
            return {
                "sender": self.name,
                "receiver": "user",
                "type": "final_result",
                "priority": 40,
                "content": {
                    "intent": "GENERAL_CHAT",
                    "reply": reply,
                    "suggestions": suggestions,
                    "router_confidence": confidence,
                    "router_source": understanding.get("source", "unknown"),
                    "router_mode": mode,
                    "task_domain": task_domain,
                    "supported_execution": supported_execution,
                },
                "metadata": {"trace_id": state["trace_id"]},
            }

        if intent == "KNOWLEDGE_LOOKUP":
            receiver = "kb_retriever"
            route_mode = "direct_answer"
            return_to = "user"
        elif needs_knowledge:
            receiver = "kb_retriever"
            route_mode = "enrich_workflow"
            return_to = "planner"
        else:
            receiver = "planner"
            route_mode = "direct_workflow"
            return_to = "planner"

        state["intent"] = intent
        state["needs_knowledge"] = needs_knowledge
        state["router_confidence"] = confidence
        state["router_source"] = understanding.get("source", "unknown")
        state["router_mode"] = mode
        state["task_domain"] = task_domain
        state["supported_execution"] = supported_execution
        state["requirements"] = requirements
        return {
            "sender": self.name,
            "receiver": receiver,
            "type": "intent_routed",
            "priority": 80,
            "content": {
                "task": task,
                "intent": intent,
                "needs_knowledge": needs_knowledge,
                "route_mode": route_mode,
                "return_to": return_to,
                "requirements": requirements,
                "router_confidence": confidence,
                "router_source": understanding.get("source", "unknown"),
                "router_mode": mode,
                "task_domain": task_domain,
                "supported_execution": supported_execution,
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class KBRetrieverAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("kb_retriever")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        intent = message["content"].get("intent", "KNOWLEDGE_LOOKUP")
        route_mode = message["content"].get("route_mode", "direct_answer")
        return_to = message["content"].get("return_to", "user")
        requirements = message["content"].get("requirements")
        router_confidence = message["content"].get("router_confidence", 0.0)
        router_source = message["content"].get("router_source", "unknown")
        result = self.tools.execute("kb_search", query=task, top_k=5)
        snippets = result.get("snippets", [])
        state["knowledge_context"] = snippets

        if route_mode == "enrich_workflow":
            return {
                "sender": self.name,
                "receiver": return_to,
                "type": "knowledge_context_ready",
                "priority": 72,
                "content": {
                    "task": task,
                    "intent": intent,
                    "knowledge_context": snippets,
                    "needs_knowledge": True,
                    "requirements": requirements,
                    "router_confidence": router_confidence,
                    "router_source": router_source,
                },
                "metadata": {"trace_id": state["trace_id"]},
            }

        return {
            "sender": self.name,
            "receiver": "user",
            "type": "final_result",
            "priority": 75,
            "content": {
                "intent": "KNOWLEDGE_LOOKUP",
                "summary": "Knowledge lookup finished.",
                "snippets": snippets,
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("planner")

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        knowledge_context = message["content"].get("knowledge_context", state.get("knowledge_context", []))
        incoming_requirements = message["content"].get("requirements")
        requirements = normalize_requirements(incoming_requirements, task)
        steps = build_dynamic_workflow_steps(requirements)

        state["requirements"] = requirements
        state["workflow_plan"] = steps
        state["workflow_cursor"] = 0
        state["workflow_task"] = task
        state["plan"] = [step["name"] for step in steps]
        state["knowledge_context_used"] = bool(knowledge_context)

        return {
            "sender": self.name,
            "receiver": "workflow_controller",
            "type": "workflow_plan_ready",
            "priority": 70,
            "content": {
                "task": task,
                "plan": state["plan"],
                "steps": steps,
                "requirements": requirements,
                "knowledge_context": knowledge_context,
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class WorkflowControllerAgent(BaseAgent):
    def __init__(self):
        super().__init__("workflow_controller")

    def handle(self, message: Message, state: State) -> Message:
        msg_type = message["type"]
        if msg_type == "workflow_plan_ready":
            state["workflow_cursor"] = 0
            state["workflow_task"] = message["content"]["task"]
            state.setdefault("artifacts", {})
            state.setdefault("executed_steps", [])
        elif msg_type == "step_done":
            step = message["content"].get("step", {})
            state["workflow_cursor"] = state.get("workflow_cursor", 0) + 1
            if step.get("kind"):
                state.setdefault("executed_steps", []).append(step["kind"])
            updates = message["content"].get("artifact_updates", {})
            if updates:
                state.setdefault("artifacts", {}).update(updates)
        else:
            return {
                "sender": self.name,
                "receiver": "user",
                "type": "error",
                "priority": 90,
                "content": {"error": f"unsupported message type for controller: {msg_type}"},
                "metadata": {"trace_id": state["trace_id"]},
            }

        steps: List[Dict[str, Any]] = state.get("workflow_plan", [])
        cursor = state.get("workflow_cursor", 0)
        task = state.get("workflow_task", message["content"].get("task", ""))

        if cursor >= len(steps):
            artifacts = state.get("artifacts", {})
            evaluation_result = artifacts.get("evaluation_result", {})
            report = artifacts.get("report", {})
            metrics = evaluation_result.get("all_metrics", report.get("metrics", []))
            best_model = (
                evaluation_result.get("best", {}).get("model")
                or report.get("best_model")
                or (artifacts.get("selected_models", [None])[0])
                or "N/A"
            )

            knowledge_sources = []
            for row in state.get("knowledge_context", []):
                source = row.get("source")
                if source and source not in knowledge_sources:
                    knowledge_sources.append(source)

            return {
                "sender": self.name,
                "receiver": "user",
                "type": "final_result",
                "priority": 80,
                "content": {
                    "intent": "ML_WORKFLOW",
                    "summary": "Dynamic ML workflow finished based on user requirements.",
                    "executed_steps": state.get("executed_steps", []),
                    "requirements": state.get("requirements", {}),
                    "best_model": best_model,
                    "metrics": metrics,
                    "report": report.get("report_markdown", ""),
                    "knowledge_context_used": bool(knowledge_sources),
                    "knowledge_sources": knowledge_sources[:5],
                },
                "metadata": {"trace_id": state["trace_id"]},
            }

        next_step = steps[cursor]
        return {
            "sender": self.name,
            "receiver": next_step["agent"],
            "type": "step_request",
            "priority": 68,
            "content": {
                "task": task,
                "step": next_step,
                "requirements": state.get("requirements", {}),
                "knowledge_context": state.get("knowledge_context", []),
                "artifacts": state.get("artifacts", {}),
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class DataAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("data_agent")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        step = message["content"].get("step", {"kind": "data_prep"})
        step_kind = step.get("kind")
        task = message["content"]["task"]
        artifacts = message["content"].get("artifacts", {})

        updates: Dict[str, Any] = {}
        if step_kind == "data_prep":
            data_result = self.tools.execute("process_data", task=task)
            state["data"] = data_result
            updates["data_profile"] = data_result
        elif step_kind == "feature_engineering":
            feature_result = self.tools.execute("feature_plan", task=task, data_profile=artifacts.get("data_profile"))
            updates["feature_result"] = feature_result

        return {
            "sender": self.name,
            "receiver": "workflow_controller",
            "type": "step_done",
            "priority": 68,
            "content": {
                "task": task,
                "step": step,
                "artifact_updates": updates,
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class ModelAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("model_agent")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        step = message["content"].get("step", {"kind": "model_selection"})
        step_kind = step.get("kind")
        task = message["content"]["task"]
        artifacts = message["content"].get("artifacts", {})
        requirements = message["content"].get("requirements", {})

        updates: Dict[str, Any] = {}
        if step_kind == "model_selection":
            data_profile = artifacts.get("data_profile", {"mode": "classification"})
            picked = self.tools.execute(
                "model_suggest",
                task=task,
                data_profile=data_profile,
                model_hint=requirements.get("model_hint"),
            )
            state["model_candidates"] = picked
            updates["model_candidates"] = picked
            updates["selected_models"] = picked["models"]

        return {
            "sender": self.name,
            "receiver": "workflow_controller",
            "type": "step_done",
            "priority": 66,
            "content": {
                "task": task,
                "step": step,
                "artifact_updates": updates,
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class TrainerAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("trainer")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        step = message["content"].get("step", {"kind": "training"})
        step_kind = step.get("kind")
        task = message["content"]["task"]
        artifacts = message["content"].get("artifacts", {})

        updates: Dict[str, Any] = {}
        if step_kind == "hyperparameter_tuning":
            models = artifacts.get("selected_models", [])
            tuning_result = self.tools.execute("tune_models", task=task, models=models)
            updates["tuning_result"] = tuning_result
            updates["selected_models"] = tuning_result.get("recommended_models", models)
        elif step_kind == "training":
            models = artifacts.get("selected_models", [])
            train_result = self.tools.execute("train_models", task=task, models=models)
            state["train_result"] = train_result
            updates["train_result"] = train_result

        return {
            "sender": self.name,
            "receiver": "workflow_controller",
            "type": "step_done",
            "priority": 64,
            "content": {
                "task": task,
                "step": step,
                "artifact_updates": updates,
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class EvaluatorAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("evaluator")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        step = message["content"].get("step", {"kind": "evaluation"})
        step_kind = step.get("kind")
        task = message["content"]["task"]
        artifacts = message["content"].get("artifacts", {})

        updates: Dict[str, Any] = {}
        if step_kind == "evaluation":
            train_result = artifacts.get("train_result", {"runs": []})
            eval_result = self.tools.execute("evaluate_models", task=task, train_result=train_result)
            state["evaluation_result"] = eval_result
            updates["evaluation_result"] = eval_result
        elif step_kind == "error_analysis":
            eval_result = artifacts.get("evaluation_result", {})
            error_analysis = self.tools.execute("error_analyze", task=task, evaluation_result=eval_result)
            updates["error_analysis"] = error_analysis

        return {
            "sender": self.name,
            "receiver": "workflow_controller",
            "type": "step_done",
            "priority": 62,
            "content": {
                "task": task,
                "step": step,
                "artifact_updates": updates,
            },
            "metadata": {"trace_id": state["trace_id"]},
        }


class ReporterAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("reporter")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        step = message["content"].get("step", {"kind": "reporting"})
        task = message["content"]["task"]
        artifacts = message["content"].get("artifacts", {})
        evaluation_result = artifacts.get("evaluation_result", {"best": {}, "objective": "maximize", "all_metrics": []})
        report = self.tools.execute(
            "generate_report",
            task=task,
            evaluation_result=evaluation_result,
            executed_steps=state.get("executed_steps", []),
        )
        knowledge_context = state.get("knowledge_context", [])
        knowledge_sources = []
        for row in knowledge_context:
            source = row.get("source")
            if source and source not in knowledge_sources:
                knowledge_sources.append(source)

        state["report"] = report
        return {
            "sender": self.name,
            "receiver": "workflow_controller",
            "type": "step_done",
            "priority": 60,
            "content": {
                "task": task,
                "step": step,
                "artifact_updates": {
                    "report": report,
                    "knowledge_sources": knowledge_sources[:5],
                },
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


def build_tool_registry(memory: MemoryStore, workspace: str) -> ToolRegistry:
    tools = ToolRegistry()
    skill_store = SkillStore(root=os.path.join(workspace, ".skills"))

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

    def feature_plan(task: str, data_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mode = (data_profile or {}).get("mode", "classification")
        if mode == "classification":
            features = ["interaction_features", "target_frequency_encoding"]
        else:
            features = ["lag_features", "rolling_mean_features"]
        return {"mode": mode, "suggested_features": features}

    def model_suggest(task: str, data_profile: Dict[str, Any], model_hint: Optional[str] = None) -> Dict[str, Any]:
        if data_profile["mode"] == "classification":
            models = ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"]
        else:
            models = ["LinearRegression", "RandomForestRegressor", "XGBoostRegressor"]

        if model_hint:
            canonical = {
                "logisticregression": "LogisticRegression",
                "randomforestclassifier": "RandomForestClassifier",
                "xgboostclassifier": "XGBoostClassifier",
                "linearregression": "LinearRegression",
                "randomforestregressor": "RandomForestRegressor",
                "xgboostregressor": "XGBoostRegressor",
            }.get(model_hint.lower().replace(" ", ""), model_hint)
            models = [canonical] + [m for m in models if m != canonical]
        return {"models": models}

    def tune_models(task: str, models: List[str]) -> Dict[str, Any]:
        if not models:
            models = ["LogisticRegression", "RandomForestClassifier"]
        tuned = []
        for model in models:
            score = stable_score(task + model + "_tuned", 0.01, 0.20)
            tuned.append({"model": model, "expected_gain": score})
        tuned.sort(key=lambda x: x["expected_gain"], reverse=True)
        recommended = [x["model"] for x in tuned[:2]]
        return {"recommended_models": recommended, "tuning_candidates": tuned}

    def train_models(task: str, models: List[str]) -> Dict[str, Any]:
        if not models:
            models = ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"]
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
        runs = train_result.get("runs", [])
        if not runs:
            return {
                "best": {"model": "N/A", "metric_name": "n/a", "metric_value": None},
                "objective": "maximize",
                "all_metrics": [],
                "warning": "No training runs available for evaluation.",
            }
        cls_mode = any(r["metric_name"] == "f1" for r in runs)
        if cls_mode:
            best = max(runs, key=lambda r: r["metric_value"])
            objective = "maximize"
        else:
            best = min(runs, key=lambda r: r["metric_value"])
            objective = "minimize"
        return {"best": best, "objective": objective, "all_metrics": runs}

    def error_analyze(task: str, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        best = evaluation_result.get("best", {})
        metric = best.get("metric_name", "n/a")
        value = best.get("metric_value")
        findings = []
        if metric == "f1" and isinstance(value, (int, float)) and value < 0.85:
            findings.append("Model may underperform on difficult classes; review class imbalance handling.")
        if metric == "rmse" and isinstance(value, (int, float)) and value > 0.2:
            findings.append("RMSE is relatively high; consider better features and robust models.")
        if not findings:
            findings.append("No critical issue detected from aggregate metrics. Validate with slice-level checks.")
        return {"findings": findings}

    def generate_report(
        task: str,
        evaluation_result: Dict[str, Any],
        executed_steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        best = evaluation_result.get("best") or {"model": "N/A", "metric_name": "n/a", "metric_value": None}
        lines = [
            "# Model Report",
            "",
            f"- Task: {task}",
            f"- Objective: {evaluation_result.get('objective', 'maximize')}",
            f"- Best Model: {best['model']}",
            f"- Metric: {best['metric_name']}={best['metric_value']}",
            "",
        ]
        if executed_steps:
            lines.append("## Executed Steps")
            for step in executed_steps:
                lines.append(f"- {step}")
            lines.append("")
        lines.append("## Candidate Results")
        for row in evaluation_result.get("all_metrics", []):
            lines.append(f"- {row['model']}: {row['metric_name']}={row['metric_value']}")
        report_md = "\n".join(lines)
        return {"report_markdown": report_md, "best_model": best["model"], "metrics": evaluation_result.get("all_metrics", [])}

    def skill_install_from_git(repo_url: str, ref: Optional[str] = None, alias: Optional[str] = None) -> Dict[str, Any]:
        return skill_store.install_from_git(repo_url=repo_url, ref=ref, alias=alias)

    def skill_list_installed() -> Dict[str, Any]:
        return {"skills": skill_store.list_skills()}

    tools.register(ToolSpec(name="kb_search", func=kb_search, permission="kb_read", owner_agent="kb_retriever"))
    tools.register(ToolSpec(name="process_data", func=process_data, permission="data_exec", owner_agent="data_agent"))
    tools.register(ToolSpec(name="feature_plan", func=feature_plan, permission="data_exec", owner_agent="data_agent"))
    tools.register(ToolSpec(name="model_suggest", func=model_suggest, permission="ml_plan", owner_agent="model_agent"))
    tools.register(ToolSpec(name="tune_models", func=tune_models, permission="ml_train", owner_agent="trainer"))
    tools.register(ToolSpec(name="train_models", func=train_models, permission="ml_train", owner_agent="trainer"))
    tools.register(ToolSpec(name="evaluate_models", func=evaluate_models, permission="ml_eval", owner_agent="evaluator"))
    tools.register(ToolSpec(name="error_analyze", func=error_analyze, permission="ml_eval", owner_agent="evaluator"))
    tools.register(ToolSpec(name="generate_report", func=generate_report, permission="report_write", owner_agent="reporter"))
    tools.register(
        ToolSpec(
            name="skill_install_from_git",
            func=skill_install_from_git,
            permission="skill_install",
            owner_agent="supervisor",
            timeout_s=300,
            retry=0,
        )
    )
    tools.register(
        ToolSpec(
            name="skill_list_installed",
            func=skill_list_installed,
            permission="skill_read",
            owner_agent="supervisor",
        )
    )
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
        router_model = os.getenv("ROUTER_MODEL", "gpt-4o")
        self.understanding_engine = RequestUnderstandingEngine(model=router_model)
        self.tools = build_tool_registry(self.memory, workspace=workspace)
        self.agents: Dict[str, BaseAgent] = {
            "intent_router": IntentRouterAgent(self.understanding_engine),
            "kb_retriever": KBRetrieverAgent(self.tools),
            "planner": PlannerAgent(),
            "workflow_controller": WorkflowControllerAgent(),
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
            "intent": None,
            "needs_knowledge": False,
            "router_confidence": 0.0,
            "router_source": "",
            "router_mode": "",
            "task_domain": "",
            "supported_execution": True,
            "knowledge_context": [],
            "requirements": {},
            "workflow_plan": [],
            "workflow_cursor": 0,
            "workflow_task": "",
            "artifacts": {},
            "executed_steps": [],
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
        max_steps = 48
        for _ in range(max_steps):
            if message["receiver"] == "user":
                break
            message = self.supervisor.dispatch(message, state)
        state["execution_metadata"]["ended_at_ms"] = now_ms()
        self.memory.put("workflow", trace_id, {"state": state, "final_message": message})
        self.memory.append_session_message(session_id, "assistant_structured", str(message["content"]))
        return message, state
