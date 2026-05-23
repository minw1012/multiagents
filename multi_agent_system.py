from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import csv
import hashlib
import heapq
import json
import logging
import os
import re
import subprocess
import time
import zipfile
from html import unescape
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
    "Run DOC_SUMMARY for /path/to/file.docx or /path/to/file.pdf",
    "/file summarize /path/to/file.docx",
    "/file summarize /path/to/file.pdf",
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


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_document_path_from_text(task: str, workspace: Optional[str] = None) -> Optional[str]:
    text = task.strip()
    if not text:
        return None

    # Absolute path with supported extension.
    abs_match = re.search(r"(/[^\"'\s]+?\.(?:pdf|docx|txt|md))", text, flags=re.IGNORECASE)
    if abs_match:
        p = Path(abs_match.group(1)).expanduser().resolve()
        if p.exists() and p.is_file():
            return str(p)

    # Quoted path with spaces.
    quoted = re.findall(r"[\"']([^\"']+\.(?:pdf|docx|txt|md))[\"']", text, flags=re.IGNORECASE)
    for q in quoted:
        p = Path(q).expanduser()
        if not p.is_absolute() and workspace:
            p = Path(workspace) / p
        p = p.resolve()
        if p.exists() and p.is_file():
            return str(p)

    if not workspace:
        return None

    ws = Path(workspace).expanduser().resolve()
    if not ws.exists():
        return None

    candidates = list(ws.glob("*.pdf")) + list(ws.glob("*.docx")) + list(ws.glob("*.txt")) + list(ws.glob("*.md"))
    if not candidates:
        return None

    low = text.lower()
    for p in candidates:
        if p.name.lower() in low:
            return str(p.resolve())

    norm_task = _normalize_name(text)
    stem_hits = []
    for p in candidates:
        if _normalize_name(p.stem) and _normalize_name(p.stem) in norm_task:
            stem_hits.append(p)
    if len(stem_hits) == 1:
        return str(stem_hits[0].resolve())
    return None


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
    if intent not in {"GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW", "DOC_SUMMARY"}:
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
        "file_path": str(parsed.get("file_path", "")).strip() or None,
        "reply": str(parsed.get("reply", "")).strip(),
        "suggestions": parsed.get("suggestions", []),
        "requirements": requirements,
        "source": "explicit_command",
    }


def truncate_text(value: Any, max_chars: int = 1200) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def to_json_text(value: Any, max_chars: int = 2000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        text = str(value)
    return truncate_text(text, max_chars=max_chars)


def ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _extract_json_object(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


class RequestUnderstandingEngine:
    """
    LLM-first request understanding.
    Falls back to conservative heuristic understanding when LLM is unavailable.
    """

    def __init__(self, model: str = "gpt-4o", workspace: Optional[str] = None):
        self.model = model
        self.workspace = workspace
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

    def understand(self, task: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if self.client is None:
            return self._heuristic_understand(task, source="heuristic_fallback")

        system_prompt = (
            "You are a request understanding engine for a multi-agent platform. "
            "Infer user intent semantically, not via brittle keyword matching. "
            "Return strict JSON only. "
            "Executable scope currently includes: "
            "(1) knowledge lookup from loaded docs/history, "
            "(2) tabular ML workflow (preprocess, model select, tune, train, evaluate, report), "
            "(3) DOC_SUMMARY for reading and summarizing .docx/.pdf files by file path. "
            "Computer vision/object detection is not executable in current pipeline."
        )
        user_prompt = {
            "task": task,
            "recent_history": history or [],
            "allowed_intents": ["GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW", "DOC_SUMMARY"],
            "requirement_fields": REQUIREMENT_BOOL_FIELDS + ["model_hint"],
            "instructions": [
                "For casual chat or capability questions, use GENERAL_CHAT.",
                "For historical/doc retrieval requests, use KNOWLEDGE_LOOKUP.",
                "For data/model/training/evaluation/report work, use ML_WORKFLOW.",
                "For requests to summarize/read a Word/PDF file, use DOC_SUMMARY and extract file_path when present.",
                "Distinguish answer-only vs execution: capability inquiry should be answer-only.",
                "Do not claim unsupported execution capabilities.",
                "Set needs_knowledge=true when ML task should reference knowledge/history/policy/docs.",
                "If uncertain, use GENERAL_CHAT and ask a focused clarification.",
                "Do not hallucinate constraints.",
            ],
            "output_schema_hint": {
                "intent": "GENERAL_CHAT | KNOWLEDGE_LOOKUP | ML_WORKFLOW | DOC_SUMMARY",
                "mode": "ANSWER_ONLY | EXECUTE",
                "confidence": "0.0-1.0",
                "needs_knowledge": "bool",
                "task_domain": "tabular_ml | knowledge | cv | nlp | general",
                "supported_execution": "bool",
                "file_path": "string|null",
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
        if intent not in {"GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW", "DOC_SUMMARY"}:
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
        file_path = str(parsed.get("file_path", "")).strip() or None
        task_domain = str(parsed.get("task_domain", "general")).lower().strip()
        if task_domain not in {"tabular_ml", "knowledge", "cv", "nlp", "doc", "general"}:
            task_domain = "general"
        supported_execution = parse_bool(
            parsed.get("supported_execution"),
            default=task_domain in {"tabular_ml", "knowledge", "doc", "general"},
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

        if intent == "DOC_SUMMARY" and mode == "EXECUTE" and not file_path:
            mode = "ANSWER_ONLY"
            intent = "GENERAL_CHAT"
            reply = "Please provide the .docx file path so I can read and summarize it."
            suggestions = [
                '/file summarize /absolute/path/to/file.docx',
                '{"intent":"DOC_SUMMARY","mode":"EXECUTE","file_path":"/absolute/path/to/file.docx"}',
            ]

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
            "file_path": file_path,
            "reply": reply,
            "suggestions": suggestions,
            "requirements": req,
            "source": source,
        }

    def _heuristic_understand(self, task: str, source: str) -> Dict[str, Any]:
        explicit = parse_explicit_router_command(task)
        if explicit:
            return explicit

        inferred_doc_path = resolve_document_path_from_text(task, workspace=self.workspace)
        if inferred_doc_path:
            return {
                "intent": "DOC_SUMMARY",
                "mode": "EXECUTE",
                "confidence": 0.9,
                "needs_knowledge": False,
                "task_domain": "doc",
                "supported_execution": True,
                "file_path": inferred_doc_path,
                "reply": "",
                "suggestions": [],
                "requirements": default_workflow_requirements(),
                "source": f"{source}:file_path_resolver",
            }

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

    def get_session_messages(self, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        row = self._data["session"].get(session_id, {"messages": []})
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            return []
        return messages[-max(1, limit):]

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
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
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
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "examples": spec.examples,
                    "permission": spec.permission,
                    "owner_agent": spec.owner_agent,
                    "timeout_s": spec.timeout_s,
                    "retry": spec.retry,
                }
            )
        return out

    def catalog_for_planner(self) -> List[Dict[str, Any]]:
        rows = []
        for spec in self.tools.values():
            rows.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "permission": spec.permission,
                }
            )
        return rows


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


class DynamicLoopOrchestrator:
    """
    Goal -> Understand -> Plan -> Tool Calling -> Observation -> Memory/State -> Continue/Finish
    """

    def __init__(self, model: str, tools: ToolRegistry, memory: MemoryStore, workspace: str):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.workspace = Path(workspace).expanduser().resolve()
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

    def _llm_json(self, system_prompt: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0,
            )
            parsed = _extract_json_object(getattr(resp, "output_text", "") or "")
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

    def _fallback_understand_goal(self, task: str) -> Dict[str, Any]:
        explicit = parse_explicit_router_command(task)
        if explicit:
            intent = explicit.get("intent")
            if intent == "DOC_SUMMARY":
                path = explicit.get("file_path")
                if path:
                    return {
                        "mode": "EXECUTE",
                        "goal": f"Read and summarize the document at {path}.",
                        "success_criteria": ["Provide a concise summary with key points."],
                        "constraints": [],
                        "clarification_question": "",
                        "final_reply": "",
                        "plan": [
                            {"step": "Read document file", "tool": "read_document_file", "arguments": {"path": path}},
                            {"step": "Summarize document text", "tool": "summarize_text", "arguments": {"text_from": "read_document_file.text"}},
                        ],
                    }
            if intent == "KNOWLEDGE_LOOKUP":
                return {
                    "mode": "EXECUTE",
                    "goal": "Retrieve relevant knowledge snippets for the user request.",
                    "success_criteria": ["Return grounded snippets with sources."],
                    "constraints": [],
                    "clarification_question": "",
                    "final_reply": "",
                    "plan": [
                        {"step": "Search knowledge base", "tool": "kb_search", "arguments": {"query": task, "top_k": 5}},
                    ],
                }
            if intent == "ML_WORKFLOW":
                req = normalize_requirements(explicit.get("requirements"), task)
                plan = []
                if req.get("needs_data_prep"):
                    plan.append({"step": "Prepare data", "tool": "process_data", "arguments": {"task": task}})
                if req.get("needs_feature_engineering"):
                    plan.append({"step": "Design feature plan", "tool": "feature_plan", "arguments": {"task": task}})
                if req.get("needs_model_selection"):
                    plan.append({"step": "Select candidate models", "tool": "model_suggest", "arguments": {"task": task}})
                if req.get("needs_tuning"):
                    plan.append({"step": "Tune candidate models", "tool": "tune_models", "arguments": {"models_from": "model_suggest.models", "task": task}})
                if req.get("needs_training"):
                    plan.append({"step": "Train models", "tool": "train_models", "arguments": {"models_from": "model_suggest.models", "task": task}})
                if req.get("needs_evaluation"):
                    plan.append({"step": "Evaluate models", "tool": "evaluate_models", "arguments": {"train_result_from": "train_models", "task": task}})
                if req.get("needs_error_analysis"):
                    plan.append({"step": "Analyze model errors", "tool": "error_analyze", "arguments": {"evaluation_result_from": "evaluate_models", "task": task}})
                if req.get("needs_report"):
                    plan.append({"step": "Generate final report", "tool": "generate_report", "arguments": {"evaluation_result_from": "evaluate_models", "task": task}})
                if not plan:
                    plan.append({"step": "Generate final report", "tool": "generate_report", "arguments": {"task": task}})
                return {
                    "mode": "EXECUTE",
                    "goal": "Run an adaptive ML workflow aligned to the user objective.",
                    "success_criteria": ["Deliver model outcome and concise report."],
                    "constraints": [],
                    "clarification_question": "",
                    "final_reply": "",
                    "plan": plan,
                }
            return {
                "mode": "CHAT",
                "goal": "Provide a helpful response.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": explicit.get("reply") or DEFAULT_CLARIFICATION_REPLY,
                "plan": [],
            }

        inferred_doc_path = resolve_document_path_from_text(task, workspace=str(self.workspace))
        if inferred_doc_path:
            return {
                "mode": "EXECUTE",
                "goal": f"Read and summarize the document at {inferred_doc_path}.",
                "success_criteria": ["Provide concise summary and highlights."],
                "constraints": [],
                "clarification_question": "",
                "final_reply": "",
                "plan": [
                    {"step": "Read document file", "tool": "read_document_file", "arguments": {"path": inferred_doc_path}},
                    {"step": "Summarize document text", "tool": "summarize_text", "arguments": {"text_from": "read_document_file.text"}},
                ],
            }

        return {
            "mode": "CHAT",
            "goal": "Clarify user objective before execution.",
            "success_criteria": [],
            "constraints": [],
            "clarification_question": (
                "Please share your goal and expected output, and include file/data path if you want me to execute tools."
            ),
            "final_reply": (
                "I can run a dynamic loop: understand your goal, plan steps, call tools, observe results, and iterate. "
                "I can work with knowledge lookup, document reading, CSV/data profiling, ML workflow blocks, report generation, and skill installation. "
                "Tell me the concrete task and any file/data path."
            ),
            "plan": [],
        }

    def understand_goal(self, task: str, recent_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_prompt = (
            "You are the Goal Understanding module. "
            "Understand user intent semantically, then propose an adaptive execution plan. "
            "Do not force fixed workflows. "
            "Return strict JSON."
        )
        payload = {
            "task": task,
            "recent_messages": recent_messages[-8:],
            "tool_catalog": self.tools.catalog_for_planner(),
            "output_schema": {
                "mode": "CHAT | EXECUTE",
                "goal": "string",
                "success_criteria": ["string"],
                "constraints": ["string"],
                "clarification_question": "string",
                "final_reply": "string",
                "plan": [
                    {
                        "step": "string",
                        "tool": "tool_name from catalog or empty",
                        "arguments": "dict",
                    }
                ],
            },
            "rules": [
                "Use EXECUTE when tool calling is needed; use CHAT for pure discussion or clarification.",
                "Plan must be dynamic and task-specific. Avoid fixed template plans.",
                "Only use tools from tool_catalog.",
                "If information is insufficient, set mode=CHAT and ask one focused clarification.",
            ],
        }
        parsed = self._llm_json(system_prompt, payload)
        if not isinstance(parsed, dict):
            return self._fallback_understand_goal(task)

        mode = str(parsed.get("mode", "CHAT")).upper()
        mode = "EXECUTE" if mode == "EXECUTE" else "CHAT"
        goal = str(parsed.get("goal", "")).strip() or "Solve the user request."
        success_criteria = parsed.get("success_criteria", [])
        constraints = parsed.get("constraints", [])
        if not isinstance(success_criteria, list):
            success_criteria = []
        if not isinstance(constraints, list):
            constraints = []

        raw_plan = parsed.get("plan", [])
        plan: List[Dict[str, Any]] = []
        if isinstance(raw_plan, list):
            for row in raw_plan[:8]:
                if not isinstance(row, dict):
                    continue
                step = str(row.get("step", "")).strip() or "Execute next action"
                tool = str(row.get("tool", "")).strip()
                arguments = ensure_dict(row.get("arguments", {}))
                if tool and tool not in self.tools.tools:
                    tool = ""
                plan.append({"step": step, "tool": tool, "arguments": arguments})

        clarification_question = str(parsed.get("clarification_question", "")).strip()
        final_reply = str(parsed.get("final_reply", "")).strip()
        if mode == "EXECUTE" and not plan:
            mode = "CHAT"
            clarification_question = clarification_question or "Please provide one concrete execution target so I can plan and run tools."

        return {
            "mode": mode,
            "goal": goal,
            "success_criteria": [str(x).strip() for x in success_criteria if str(x).strip()][:6],
            "constraints": [str(x).strip() for x in constraints if str(x).strip()][:6],
            "clarification_question": clarification_question,
            "final_reply": final_reply,
            "plan": plan,
        }

    def _resolve_from_outputs(self, pointer: str, tool_outputs: Dict[str, Dict[str, Any]]) -> Any:
        if not pointer:
            return None
        # pointer format: "tool_name.field.subfield"
        parts = pointer.split(".")
        if not parts:
            return None
        base = tool_outputs.get(parts[0], {})
        cur: Any = base
        for key in parts[1:]:
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                return None
        return cur

    def _resolve_arguments(self, arguments: Dict[str, Any], task: str, tool_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for key, value in arguments.items():
            if key.endswith("_from"):
                continue
            if isinstance(value, str) and value.endswith(".text") and value.split(".")[0] in tool_outputs:
                resolved[key] = self._resolve_from_outputs(value, tool_outputs)
                continue
            if isinstance(value, str) and "." in value and value.split(".")[0] in tool_outputs:
                maybe = self._resolve_from_outputs(value, tool_outputs)
                resolved[key] = maybe
                continue
            resolved[key] = value

        # Backward-compatible aliases for planners that use *_from keys.
        for key, value in arguments.items():
            if not key.endswith("_from"):
                continue
            dest_key = key[: -len("_from")]
            if isinstance(value, str):
                resolved[dest_key] = self._resolve_from_outputs(value, tool_outputs)

        for req_key in ["task", "query"]:
            if req_key in resolved and resolved[req_key] in {None, ""}:
                resolved[req_key] = task
        return resolved

    def decide_next_action(self, loop_state: Dict[str, Any]) -> Dict[str, Any]:
        task = loop_state.get("task", "")
        cursor = int(loop_state.get("plan_cursor", 0))
        plan = loop_state.get("plan", [])
        tool_outputs = loop_state.get("tool_outputs", {})
        observations = loop_state.get("observations", [])

        if self.client is None:
            if cursor >= len(plan):
                return {
                    "decision": "FINAL",
                    "step": "Finalize answer",
                    "tool_name": "",
                    "arguments": {},
                    "final_answer": "",
                }
            step = plan[cursor]
            tool_name = str(step.get("tool", "")).strip()
            if not tool_name:
                return {
                    "decision": "FINAL",
                    "step": step.get("step", "Finalize answer"),
                    "tool_name": "",
                    "arguments": {},
                    "final_answer": "",
                }
            arguments = self._resolve_arguments(ensure_dict(step.get("arguments", {})), task=task, tool_outputs=tool_outputs)
            return {
                "decision": "TOOL",
                "step": step.get("step", "Run tool"),
                "tool_name": tool_name,
                "arguments": arguments,
                "final_answer": "",
            }

        system_prompt = (
            "You are the Planner+Executor controller in an agent loop. "
            "Given goal, plan, and observations, decide ONE next action. "
            "Return strict JSON only."
        )
        payload = {
            "task": task,
            "goal": loop_state.get("goal", ""),
            "success_criteria": loop_state.get("success_criteria", []),
            "constraints": loop_state.get("constraints", []),
            "plan": plan,
            "plan_cursor": cursor,
            "observations": observations[-6:],
            "tool_outputs_preview": {
                name: to_json_text(val, max_chars=800) for name, val in list(tool_outputs.items())[-6:]
            },
            "tool_catalog": self.tools.catalog_for_planner(),
            "output_schema": {
                "decision": "TOOL | FINAL | CLARIFY",
                "step": "string",
                "tool_name": "string",
                "arguments": "dict",
                "final_answer": "string",
                "clarification_question": "string",
            },
            "rules": [
                "Prefer finishing early if success criteria are met.",
                "When decision=TOOL, tool_name must exist in tool_catalog and arguments must be executable.",
                "Never invent tools.",
                "Avoid repetitive failing calls: if blocked, use CLARIFY.",
            ],
        }
        parsed = self._llm_json(system_prompt, payload)
        if not isinstance(parsed, dict):
            loop_state["plan_cursor"] = cursor + 1
            return self.decide_next_action(loop_state)

        decision = str(parsed.get("decision", "FINAL")).upper()
        if decision not in {"TOOL", "FINAL", "CLARIFY"}:
            decision = "FINAL"
        tool_name = str(parsed.get("tool_name", "")).strip()
        arguments = ensure_dict(parsed.get("arguments", {}))
        final_answer = str(parsed.get("final_answer", "")).strip()
        clarification_question = str(parsed.get("clarification_question", "")).strip()
        step = str(parsed.get("step", "")).strip() or "Execute next action"
        if decision == "TOOL" and tool_name not in self.tools.tools:
            decision = "CLARIFY"
            clarification_question = clarification_question or "I need a valid tool/action target before continuing."
        return {
            "decision": decision,
            "step": step,
            "tool_name": tool_name,
            "arguments": arguments,
            "final_answer": final_answer,
            "clarification_question": clarification_question,
        }

    def _compose_final_answer(self, loop_state: Dict[str, Any]) -> str:
        observations = loop_state.get("observations", [])
        if not observations:
            return "No execution output was produced."

        if self.client is not None:
            system_prompt = (
                "You are the final response composer for a tool-using agent. "
                "Summarize completed work clearly and factually from observations. "
                "Return strict JSON with key 'final_answer'."
            )
            payload = {
                "task": loop_state.get("task", ""),
                "goal": loop_state.get("goal", ""),
                "observations": observations[-10:],
                "output_schema": {"final_answer": "string"},
            }
            parsed = self._llm_json(system_prompt, payload)
            if isinstance(parsed, dict):
                answer = str(parsed.get("final_answer", "")).strip()
                if answer:
                    return answer

        last_ok = None
        for row in reversed(observations):
            if row.get("ok"):
                last_ok = row
                break
        if not last_ok:
            return "Execution finished with errors. Please refine the goal or provide missing inputs."
        result = last_ok.get("result", {})
        if isinstance(result, dict):
            if "summary" in result:
                return str(result.get("summary", "")).strip() or to_json_text(result, max_chars=800)
            if "report_markdown" in result:
                return str(result.get("report_markdown", "")).strip()
            if "snippets" in result:
                snippets = result.get("snippets", [])
                if isinstance(snippets, list) and snippets:
                    lines = []
                    for idx, row in enumerate(snippets[:5], start=1):
                        lines.append(f"{idx}. {row.get('snippet', '')} (source: {row.get('source', '')})")
                    return "\n".join(lines)
        return to_json_text(result, max_chars=1200)

    def _compact_tool_result(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return {"value": truncate_text(result, max_chars=600)}
        out = dict(result)
        text_value = out.get("text")
        if isinstance(text_value, str):
            out["text_preview"] = truncate_text(text_value, max_chars=500)
            out["text_chars"] = len(text_value)
            out.pop("text", None)
        return out

    def run(self, task: str, trace_id: str, recent_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        understanding = self.understand_goal(task, recent_messages)
        mode = understanding.get("mode", "CHAT")
        plan = understanding.get("plan", [])

        if mode != "EXECUTE":
            reply = understanding.get("final_reply") or understanding.get("clarification_question") or DEFAULT_CLARIFICATION_REPLY
            return {
                "intent": "GENERAL_CHAT",
                "reply": reply,
                "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                "planner_mode": mode,
                "goal": understanding.get("goal", ""),
                "trace_id": trace_id,
            }

        loop_state: Dict[str, Any] = {
            "trace_id": trace_id,
            "task": task,
            "goal": understanding.get("goal", ""),
            "success_criteria": understanding.get("success_criteria", []),
            "constraints": understanding.get("constraints", []),
            "plan": plan,
            "plan_cursor": 0,
            "tool_outputs": {},
            "observations": [],
            "executed_tools": [],
            "status": "running",
        }

        max_iterations = 12
        repeated_calls: Dict[str, int] = {}
        for iteration in range(1, max_iterations + 1):
            decision = self.decide_next_action(loop_state)
            loop_state["last_decision"] = decision

            if decision.get("decision") == "FINAL":
                loop_state["status"] = "completed"
                answer = decision.get("final_answer") or self._compose_final_answer(loop_state)
                return {
                    "intent": "DYNAMIC_EXECUTION",
                    "status": "completed",
                    "goal": loop_state["goal"],
                    "plan": [row.get("step", "") for row in loop_state.get("plan", [])],
                    "executed_tools": loop_state["executed_tools"],
                    "observations": loop_state["observations"][-8:],
                    "result_summary": answer,
                    "trace_id": trace_id,
                }

            if decision.get("decision") == "CLARIFY":
                loop_state["status"] = "clarification_needed"
                question = decision.get("clarification_question") or "Please share one missing constraint so I can continue."
                return {
                    "intent": "GENERAL_CHAT",
                    "reply": question,
                    "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                    "planner_mode": "CLARIFY",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                }

            if decision.get("decision") != "TOOL":
                loop_state["status"] = "completed"
                return {
                    "intent": "GENERAL_CHAT",
                    "reply": DEFAULT_CLARIFICATION_REPLY,
                    "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                    "planner_mode": "UNKNOWN",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                }

            tool_name = str(decision.get("tool_name", "")).strip()
            raw_args = ensure_dict(decision.get("arguments", {}))
            args = self._resolve_arguments(raw_args, task=task, tool_outputs=loop_state["tool_outputs"])
            if tool_name not in self.tools.tools:
                obs = {
                    "iteration": iteration,
                    "step": decision.get("step", "Run tool"),
                    "tool": tool_name,
                    "arguments": args,
                    "ok": False,
                    "error": f"tool not found: {tool_name}",
                    "result": {},
                }
                loop_state["observations"].append(obs)
                continue

            call_sig = f"{tool_name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
            repeated_calls[call_sig] = repeated_calls.get(call_sig, 0) + 1
            if repeated_calls[call_sig] > 2:
                loop_state["status"] = "clarification_needed"
                return {
                    "intent": "GENERAL_CHAT",
                    "reply": "I am repeating the same failing action. Please provide an additional constraint or expected output format.",
                    "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                    "planner_mode": "REPEAT_GUARD",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                }

            try:
                result = self.tools.execute(tool_name, **args)
                ok = not (isinstance(result, dict) and result.get("ok") is False)
                error = result.get("error") if isinstance(result, dict) else None
            except Exception as e:
                result = {"ok": False, "error": str(e)}
                ok = False
                error = str(e)

            loop_state["tool_outputs"][tool_name] = result if isinstance(result, dict) else {"result": result}
            loop_state["executed_tools"].append(tool_name)
            compact_result = self._compact_tool_result(result)
            loop_state["observations"].append(
                {
                    "iteration": iteration,
                    "step": decision.get("step", "Run tool"),
                    "tool": tool_name,
                    "arguments": args,
                    "ok": ok,
                    "error": error,
                    "result": compact_result,
                    "result_preview": to_json_text(compact_result, max_chars=900),
                }
            )
            loop_state["plan_cursor"] = int(loop_state.get("plan_cursor", 0)) + 1

        loop_state["status"] = "max_iterations"
        return {
            "intent": "DYNAMIC_EXECUTION",
            "status": "max_iterations_reached",
            "goal": loop_state["goal"],
            "plan": [row.get("step", "") for row in loop_state.get("plan", [])],
            "executed_tools": loop_state["executed_tools"],
            "observations": loop_state["observations"][-8:],
            "result_summary": self._compose_final_answer(loop_state),
            "trace_id": trace_id,
        }


class IntentRouterAgent(BaseAgent):
    def __init__(self, understanding_engine: RequestUnderstandingEngine):
        super().__init__("intent_router")
        self.understanding_engine = understanding_engine

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"]["task"]
        understanding = self.understanding_engine.understand(task, history=state.get("recent_messages", []))
        intent = understanding.get("intent", "GENERAL_CHAT")
        mode = understanding.get("mode", "ANSWER_ONLY")
        needs_knowledge = bool(understanding.get("needs_knowledge", False))
        confidence = float(understanding.get("confidence", 0.5))
        requirements = normalize_requirements(understanding.get("requirements"), task)
        task_domain = understanding.get("task_domain", "general")
        supported_execution = bool(understanding.get("supported_execution", True))
        file_path = understanding.get("file_path")

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

        if intent == "DOC_SUMMARY":
            receiver = "doc_summarizer"
            route_mode = "direct_summary"
            return_to = "user"
        elif intent == "KNOWLEDGE_LOOKUP":
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
                "file_path": file_path,
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


class DocSummaryAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("doc_summarizer")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        task = message["content"].get("task", "")
        file_path = message["content"].get("file_path")
        if not file_path:
            return {
                "sender": self.name,
                "receiver": "user",
                "type": "final_result",
                "priority": 50,
                "content": {
                    "intent": "DOC_SUMMARY",
                    "ok": False,
                    "error": "file_path is required for DOC_SUMMARY.",
                    "reply": "Please provide a .docx file path.",
                    "examples": [
                        '/file summarize /absolute/path/to/file.docx',
                        '{"intent":"DOC_SUMMARY","mode":"EXECUTE","file_path":"/absolute/path/to/file.docx"}',
                    ],
                },
                "metadata": {"trace_id": state["trace_id"]},
            }

        read_result = self.tools.execute("read_document_file", path=file_path)
        if not read_result.get("ok"):
            return {
                "sender": self.name,
                "receiver": "user",
                "type": "final_result",
                "priority": 60,
                "content": {
                    "intent": "DOC_SUMMARY",
                    "ok": False,
                    "error": read_result.get("error", "failed to read file"),
                    "source_path": file_path,
                },
                "metadata": {"trace_id": state["trace_id"]},
            }

        summary_result = self.tools.execute("summarize_text", text=read_result.get("text", ""), max_sentences=6)
        state["doc_summary"] = {
            "source_path": read_result.get("path", file_path),
            "word_count": read_result.get("word_count", 0),
            "summary": summary_result.get("summary", ""),
        }

        return {
            "sender": self.name,
            "receiver": "user",
            "type": "final_result",
            "priority": 70,
            "content": {
                "intent": "DOC_SUMMARY",
                "ok": True,
                "task": task,
                "source_path": read_result.get("path", file_path),
                "word_count": read_result.get("word_count", 0),
                "summary": summary_result.get("summary", ""),
                "highlights": summary_result.get("highlights", []),
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
    workspace_path = Path(workspace).expanduser().resolve()

    def kb_search(query: str, top_k: int = 5) -> Dict[str, Any]:
        hits = memory.search("knowledge", query=query, top_k=top_k)
        snippets = []
        for row in hits:
            text = row["text"][:280].replace("\n", " ")
            snippets.append({"doc_id": row["doc_id"], "source": row["source"], "snippet": text})
        return {"snippets": snippets}

    def kb_add_document(path: str, title: Optional[str] = None) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = workspace_path / p
        p = p.resolve()
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"file not found: {p}"}
        if p.suffix.lower() not in {".md", ".txt"}:
            return {"ok": False, "error": "only .md/.txt are supported for kb_add_document"}
        text = p.read_text(encoding="utf-8", errors="ignore")
        doc_id = f"doc_ext_{now_ms()}"
        memory.put_knowledge_doc(doc_id=doc_id, title=title or p.name, text=text, source=str(p))
        return {"ok": True, "doc_id": doc_id, "source": str(p), "chars": len(text)}

    def list_workspace_files(pattern: str = "*", recursive: bool = True, limit: int = 200) -> Dict[str, Any]:
        paths: List[Path]
        if recursive:
            paths = list(workspace_path.rglob(pattern))
        else:
            paths = list(workspace_path.glob(pattern))
        files = []
        for p in paths:
            if p.is_file():
                files.append(str(p.resolve()))
        files = sorted(files)[: max(1, min(limit, 1000))]
        return {"workspace": str(workspace_path), "count": len(files), "files": files}

    def search_workspace_text(query: str, top_k: int = 20) -> Dict[str, Any]:
        if not query.strip():
            return {"matches": [], "count": 0}
        rg = subprocess.run(
            ["rg", "-n", "--no-heading", query, str(workspace_path)],
            capture_output=True,
            text=True,
        )
        lines = []
        if rg.returncode in {0, 1}:
            for raw in (rg.stdout or "").splitlines():
                if raw.strip():
                    lines.append(raw.strip())
        rows = []
        for line in lines[: max(1, min(top_k, 200))]:
            parts = line.split(":", 2)
            if len(parts) == 3:
                rows.append({"path": parts[0], "line": parts[1], "text": parts[2]})
            else:
                rows.append({"raw": line})
        return {"count": len(rows), "matches": rows}

    def read_text_file(path: str, max_chars: int = 20000) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = workspace_path / p
        p = p.resolve()
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"file not found: {p}"}
        text = p.read_text(encoding="utf-8", errors="ignore")
        text = text[: max(500, min(max_chars, 200000))]
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        return {"ok": True, "path": str(p), "text": text, "word_count": len(words)}

    def read_json_file(path: str) -> Dict[str, Any]:
        row = read_text_file(path=path, max_chars=500000)
        if not row.get("ok"):
            return row
        try:
            parsed = json.loads(row["text"])
        except Exception as e:
            return {"ok": False, "error": f"failed to parse json: {e}", "path": row.get("path")}
        return {"ok": True, "path": row.get("path"), "json": parsed}

    def read_csv_preview(path: str, max_rows: int = 20) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = workspace_path / p
        p = p.resolve()
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"file not found: {p}"}
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for idx, row in enumerate(reader):
                if idx >= max(1, min(max_rows, 500)):
                    break
                rows.append(dict(row))
        return {"ok": True, "path": str(p), "headers": headers, "rows": rows, "row_count": len(rows)}

    def profile_csv_columns(path: str, max_rows: int = 500) -> Dict[str, Any]:
        preview = read_csv_preview(path=path, max_rows=max_rows)
        if not preview.get("ok"):
            return preview
        headers = preview.get("headers", [])
        rows = preview.get("rows", [])
        stats = []
        for h in headers:
            values = [r.get(h, "") for r in rows]
            non_empty = [v for v in values if str(v).strip() != ""]
            numeric = 0
            for v in non_empty:
                try:
                    float(str(v))
                    numeric += 1
                except Exception:
                    pass
            stats.append(
                {
                    "column": h,
                    "non_empty": len(non_empty),
                    "non_empty_ratio": round(len(non_empty) / max(1, len(rows)), 4),
                    "numeric_ratio": round(numeric / max(1, len(non_empty)), 4),
                    "sample_values": [str(v) for v in non_empty[:5]],
                }
            )
        return {"ok": True, "path": preview.get("path"), "rows_analyzed": len(rows), "column_stats": stats}

    def write_text_file(path: str, text: str, overwrite: bool = True) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = workspace_path / p
        p = p.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not overwrite:
            return {"ok": False, "error": f"file already exists: {p}"}
        p.write_text(text or "", encoding="utf-8")
        return {"ok": True, "path": str(p), "chars": len(text or "")}

    def list_available_tools() -> Dict[str, Any]:
        return {"tools": tools.list_tools()}

    def read_document_file(path: str) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            return {"ok": False, "error": f"file not found: {file_path}"}

        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            try:
                with zipfile.ZipFile(file_path, "r") as zf:
                    xml_parts: List[str] = []
                    for name in zf.namelist():
                        if name.startswith("word/") and name.endswith(".xml"):
                            xml_parts.append(zf.read(name).decode("utf-8", errors="ignore"))
                if not xml_parts:
                    return {"ok": False, "error": "no readable XML parts found in .docx"}
                xml_text = "\n".join(xml_parts)
                paragraphs = re.findall(r"<w:p[^>]*>(.*?)</w:p>", xml_text, flags=re.DOTALL)
                extracted_lines: List[str] = []
                for para in paragraphs:
                    pieces = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, flags=re.DOTALL)
                    if pieces:
                        extracted_lines.append(unescape("".join(pieces)).strip())
                text = "\n".join([line for line in extracted_lines if line])
                words = [w for w in re.split(r"\s+", text.strip()) if w]
                return {"ok": True, "path": str(file_path), "text": text, "word_count": len(words)}
            except Exception as e:
                return {"ok": False, "error": f"failed to parse docx: {e}"}

        if suffix == ".pdf":
            # Try pypdf first, then pdftotext CLI as fallback.
            try:
                from pypdf import PdfReader  # type: ignore

                logging.getLogger("pypdf").setLevel(logging.ERROR)
                reader = PdfReader(str(file_path))
                pages: List[str] = []
                for page in reader.pages:
                    pages.append((page.extract_text() or "").strip())
                text = "\n".join([p for p in pages if p])
                words = [w for w in re.split(r"\s+", text.strip()) if w]
                return {
                    "ok": True,
                    "path": str(file_path),
                    "text": text,
                    "word_count": len(words),
                    "page_count": len(reader.pages),
                }
            except Exception:
                pass

            try:
                proc = subprocess.run(
                    ["pdftotext", str(file_path), "-"],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    text = proc.stdout or ""
                    words = [w for w in re.split(r"\s+", text.strip()) if w]
                    return {"ok": True, "path": str(file_path), "text": text, "word_count": len(words)}
            except Exception:
                pass

            return {
                "ok": False,
                "error": "failed to parse pdf. install pypdf (pip install pypdf) or pdftotext.",
            }

        if suffix in {".txt", ".md"}:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                words = [w for w in re.split(r"\s+", text.strip()) if w]
                return {"ok": True, "path": str(file_path), "text": text, "word_count": len(words)}
            except Exception as e:
                return {"ok": False, "error": f"failed to read text file: {e}"}

        if suffix == ".doc":
            return {"ok": False, "error": "legacy .doc is not supported yet. Please convert to .docx."}
        return {"ok": False, "error": f"unsupported file type: {suffix}"}

    def summarize_text(text: str, max_sentences: int = 6) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {"summary": "", "highlights": []}

        # Simple local summarizer: take leading informative sentences.
        chunks = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", raw) if s.strip()]
        highlights = chunks[: max(1, min(max_sentences, 12))]
        summary = " ".join(highlights)
        return {"summary": summary, "highlights": highlights}

    def extract_key_points(text: str, max_points: int = 8) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {"key_points": []}
        chunks = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", raw) if s.strip()]
        points = []
        for row in chunks:
            if len(row) >= 12:
                points.append(row)
            if len(points) >= max(1, min(max_points, 20)):
                break
        return {"key_points": points}

    def process_data(task: str, mode_hint: Optional[str] = None) -> Dict[str, Any]:
        mode = mode_hint if mode_hint in {"classification", "regression"} else "classification"
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

    def model_suggest(
        task: str,
        data_profile: Optional[Dict[str, Any]] = None,
        model_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode = (data_profile or {}).get("mode", "classification")
        if mode == "classification":
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
        extra_notes: Optional[List[str]] = None,
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
        if extra_notes:
            lines.append("")
            lines.append("## Notes")
            for note in extra_notes:
                lines.append(f"- {note}")
        report_md = "\n".join(lines)
        return {"report_markdown": report_md, "best_model": best["model"], "metrics": evaluation_result.get("all_metrics", [])}

    def skill_install_from_git(repo_url: str, ref: Optional[str] = None, alias: Optional[str] = None) -> Dict[str, Any]:
        return skill_store.install_from_git(repo_url=repo_url, ref=ref, alias=alias)

    def skill_list_installed() -> Dict[str, Any]:
        return {"skills": skill_store.list_skills()}

    tools.register(
        ToolSpec(
            name="list_available_tools",
            func=list_available_tools,
            description="List all registered tools with schemas and examples.",
            input_schema={},
            permission="tool_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="list_workspace_files",
            func=list_workspace_files,
            description="List files under workspace by glob pattern.",
            input_schema={"pattern": "string, default *", "recursive": "bool", "limit": "int"},
            examples=["list_workspace_files(pattern='*.pdf')", "list_workspace_files(pattern='**/*.md', recursive=True)"],
            permission="file_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="search_workspace_text",
            func=search_workspace_text,
            description="Search text in workspace files using ripgrep.",
            input_schema={"query": "string", "top_k": "int"},
            examples=["search_workspace_text(query='memory', top_k=10)"],
            permission="file_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="read_text_file",
            func=read_text_file,
            description="Read a text/markdown/source file.",
            input_schema={"path": "string", "max_chars": "int"},
            permission="file_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="read_json_file",
            func=read_json_file,
            description="Read and parse a JSON file.",
            input_schema={"path": "string"},
            permission="file_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="read_csv_preview",
            func=read_csv_preview,
            description="Preview CSV headers and rows.",
            input_schema={"path": "string", "max_rows": "int"},
            permission="file_read",
            owner_agent="data_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="profile_csv_columns",
            func=profile_csv_columns,
            description="Infer basic column stats for CSV data.",
            input_schema={"path": "string", "max_rows": "int"},
            permission="data_exec",
            owner_agent="data_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="write_text_file",
            func=write_text_file,
            description="Write text content into a file in workspace.",
            input_schema={"path": "string", "text": "string", "overwrite": "bool"},
            permission="file_write",
            owner_agent="reporter",
        )
    )
    tools.register(
        ToolSpec(
            name="kb_search",
            func=kb_search,
            description="Retrieve relevant snippets from in-memory knowledge base.",
            input_schema={"query": "string", "top_k": "int"},
            permission="kb_read",
            owner_agent="kb_retriever",
        )
    )
    tools.register(
        ToolSpec(
            name="kb_add_document",
            func=kb_add_document,
            description="Ingest a local .md/.txt file into knowledge memory.",
            input_schema={"path": "string", "title": "string|null"},
            permission="kb_write",
            owner_agent="kb_retriever",
        )
    )
    tools.register(
        ToolSpec(
            name="read_document_file",
            func=read_document_file,
            description="Read .docx/.pdf/.txt/.md files and extract text.",
            input_schema={"path": "string"},
            permission="file_read",
            owner_agent="doc_summarizer",
        )
    )
    tools.register(
        ToolSpec(
            name="read_word_file",
            func=read_document_file,
            description="Backward-compatible alias of read_document_file.",
            input_schema={"path": "string"},
            permission="file_read",
            owner_agent="doc_summarizer",
        )
    )
    tools.register(
        ToolSpec(
            name="summarize_text",
            func=summarize_text,
            description="Summarize plain text into concise sentences.",
            input_schema={"text": "string", "max_sentences": "int"},
            permission="nlp_local",
            owner_agent="doc_summarizer",
        )
    )
    tools.register(
        ToolSpec(
            name="extract_key_points",
            func=extract_key_points,
            description="Extract key points from text.",
            input_schema={"text": "string", "max_points": "int"},
            permission="nlp_local",
            owner_agent="doc_summarizer",
        )
    )
    tools.register(
        ToolSpec(
            name="process_data",
            func=process_data,
            description="Create a data preprocessing plan.",
            input_schema={"task": "string", "mode_hint": "classification|regression|null"},
            permission="data_exec",
            owner_agent="data_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="feature_plan",
            func=feature_plan,
            description="Suggest feature engineering ideas.",
            input_schema={"task": "string", "data_profile": "dict|null"},
            permission="data_exec",
            owner_agent="data_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="model_suggest",
            func=model_suggest,
            description="Recommend model candidates for a task.",
            input_schema={"task": "string", "data_profile": "dict|null", "model_hint": "string|null"},
            permission="ml_plan",
            owner_agent="model_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="tune_models",
            func=tune_models,
            description="Suggest tuning gains and recommended models.",
            input_schema={"task": "string", "models": "list[string]"},
            permission="ml_train",
            owner_agent="trainer",
        )
    )
    tools.register(
        ToolSpec(
            name="train_models",
            func=train_models,
            description="Run local simulated training and produce metrics.",
            input_schema={"task": "string", "models": "list[string]"},
            permission="ml_train",
            owner_agent="trainer",
        )
    )
    tools.register(
        ToolSpec(
            name="evaluate_models",
            func=evaluate_models,
            description="Evaluate model runs and pick best candidate.",
            input_schema={"task": "string", "train_result": "dict"},
            permission="ml_eval",
            owner_agent="evaluator",
        )
    )
    tools.register(
        ToolSpec(
            name="error_analyze",
            func=error_analyze,
            description="Perform simple error analysis from metrics.",
            input_schema={"task": "string", "evaluation_result": "dict"},
            permission="ml_eval",
            owner_agent="evaluator",
        )
    )
    tools.register(
        ToolSpec(
            name="generate_report",
            func=generate_report,
            description="Generate markdown report from evaluation artifacts.",
            input_schema={"task": "string", "evaluation_result": "dict", "executed_steps": "list[string]|null"},
            permission="report_write",
            owner_agent="reporter",
        )
    )
    tools.register(
        ToolSpec(
            name="skill_install_from_git",
            func=skill_install_from_git,
            description="Download or update a skill from a git repository.",
            input_schema={"repo_url": "string", "ref": "string|null", "alias": "string|null"},
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
            description="List skills installed in local .skills directory.",
            input_schema={},
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
        self.workspace = str(Path(workspace).expanduser().resolve())
        self.memory = MemoryStore()
        loaded = load_markdown_as_knowledge(self.memory, self.workspace)
        self.memory.put("workflow", "bootstrap", {"knowledge_docs_loaded": loaded})
        router_model = os.getenv("ROUTER_MODEL", "gpt-4o")
        self.understanding_engine = RequestUnderstandingEngine(model=router_model, workspace=self.workspace)
        self.tools = build_tool_registry(self.memory, workspace=self.workspace)
        self.agents: Dict[str, BaseAgent] = {
            "intent_router": IntentRouterAgent(self.understanding_engine),
            "kb_retriever": KBRetrieverAgent(self.tools),
            "doc_summarizer": DocSummaryAgent(self.tools),
            "planner": PlannerAgent(),
            "workflow_controller": WorkflowControllerAgent(),
            "data_agent": DataAgent(self.tools),
            "model_agent": ModelAgent(self.tools),
            "trainer": TrainerAgent(self.tools),
            "evaluator": EvaluatorAgent(self.tools),
            "reporter": ReporterAgent(self.tools),
        }
        self.supervisor = Supervisor(self.agents)
        orchestrator_model = os.getenv("ORCHESTRATOR_MODEL", router_model)
        self.dynamic_orchestrator = DynamicLoopOrchestrator(
            model=orchestrator_model,
            tools=self.tools,
            memory=self.memory,
            workspace=self.workspace,
        )

    def run(self, user_text: str, session_id: str = "default") -> Tuple[Message, State]:
        if parse_bool(os.getenv("USE_LEGACY_ROUTER", "0"), default=False):
            return self._run_legacy(user_text=user_text, session_id=session_id)
        return self._run_dynamic(user_text=user_text, session_id=session_id)

    def _run_dynamic(self, user_text: str, session_id: str = "default") -> Tuple[Message, State]:
        trace_id = f"trace_{now_ms()}"
        started_at = now_ms()
        self.memory.append_session_message(session_id, "user", user_text)
        recent_messages = self.memory.get_session_messages(session_id, limit=12)

        loop_result = self.dynamic_orchestrator.run(
            task=user_text,
            trace_id=trace_id,
            recent_messages=recent_messages,
        )
        message: Message = {
            "sender": "dynamic_orchestrator",
            "receiver": "user",
            "type": "final_result",
            "priority": 80,
            "content": loop_result,
            "metadata": {"trace_id": trace_id},
        }
        state: State = {
            "trace_id": trace_id,
            "session_id": session_id,
            "recent_messages": recent_messages,
            "messages": [message],
            "loop_result": loop_result,
            "execution_metadata": {"started_at_ms": started_at, "ended_at_ms": now_ms()},
        }
        self.memory.put("workflow", trace_id, {"state": state, "final_message": message})
        self.memory.append_session_message(session_id, "assistant_structured", to_json_text(loop_result, max_chars=2000))
        return message, state

    def _run_legacy(self, user_text: str, session_id: str = "default") -> Tuple[Message, State]:
        trace_id = f"trace_{now_ms()}"
        self.memory.append_session_message(session_id, "user", user_text)
        recent_messages = self.memory.get_session_messages(session_id, limit=10)
        state: State = {
            "trace_id": trace_id,
            "session_id": session_id,
            "recent_messages": recent_messages,
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
            "doc_summary": None,
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
