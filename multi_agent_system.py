from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import csv
import hashlib
import heapq
import json
import logging
import math
import os
import re
import subprocess
import time
import urllib.request
import urllib.parse
import urllib.error
import zipfile
from html import unescape
from pathlib import Path
import xml.etree.ElementTree as ET

from src.policy.execution import ExecutionPolicy
from src.skills.store import SkillStore
from src.tools.registry import ToolRegistry, ToolSpec


Message = Dict[str, Any]
State = Dict[str, Any]


def now_ms() -> int:
    return int(time.time() * 1000)


def resolve_runtime_dir(workspace: Path, preferred: str, legacy: str) -> Path:
    preferred_path = workspace / preferred
    legacy_path = workspace / legacy
    if preferred_path.exists():
        return preferred_path
    if legacy_path.exists():
        return legacy_path
    return preferred_path


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
    "Run CODE_TASK to inspect/update project files and run tests in workspace",
    "Use network_http_request to call an API endpoint",
    "Use sqlite_query on a local .db file for data inspection",
    "Use browser_open_page to inspect page state and links",
    "/file summarize /path/to/file.docx",
    "/file summarize /path/to/file.pdf",
    'Use explicit JSON command: {"intent":"ML_WORKFLOW","mode":"EXECUTE","requirements":{...}}',
]

UNSUPPORTED_EXECUTION_REPLY = (
    "This request is outside the currently executable pipeline. "
    "Current executable scope: KNOWLEDGE_LOOKUP, DOC_SUMMARY, CODE_TASK, tabular ML workflow, network/API calls, sqlite queries, and lightweight browser actions."
)


def default_general_chat_payload() -> Dict[str, Any]:
    return {
        "reply": DEFAULT_CLARIFICATION_REPLY,
        "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
    }


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_path_candidates(text: str, extensions: List[str]) -> List[str]:
    if not text:
        return []
    ext_group = "|".join([re.escape(ext.lstrip(".")) for ext in extensions if ext])
    if not ext_group:
        return []

    candidates: List[str] = []

    # Quoted paths, can include spaces.
    quoted = re.findall(rf"[\"']([^\"']+\.(?:{ext_group}))[\"']", text, flags=re.IGNORECASE)
    candidates.extend(quoted)

    # Absolute paths, allow spaces, stop at extension.
    abs_paths = re.findall(rf"(/[^\"']+\.(?:{ext_group}))", text, flags=re.IGNORECASE)
    candidates.extend(abs_paths)

    # Simple relative tokens without spaces.
    rel_tokens = re.findall(rf"(?:^|\s)([^\"'\s]+\.(?:{ext_group}))(?=$|\s)", text, flags=re.IGNORECASE)
    candidates.extend(rel_tokens)

    cleaned: List[str] = []
    for raw in candidates:
        token = str(raw).strip()
        token = token.strip("()[]{}<>")
        token = token.rstrip(".,;:!?")
        if token:
            cleaned.append(token)
    return _dedupe_keep_order(cleaned)


def _resolve_existing_path(candidate: str, workspace: Optional[str] = None) -> Optional[str]:
    token = (candidate or "").strip()
    if not token:
        return None
    p = Path(token).expanduser()
    if not p.is_absolute() and workspace:
        p = Path(workspace).expanduser() / p
    try:
        p = p.resolve()
    except Exception:
        return None
    if p.exists() and p.is_file():
        return str(p)
    return None


def resolve_document_path_from_text(task: str, workspace: Optional[str] = None) -> Optional[str]:
    text = task.strip()
    if not text:
        return None

    for candidate in _extract_path_candidates(text, extensions=["pdf", "docx", "txt", "md"]):
        resolved = _resolve_existing_path(candidate, workspace=workspace)
        if resolved:
            return resolved

    if not workspace:
        return None

    ws = Path(workspace).expanduser().resolve()
    if not ws.exists():
        return None

    candidates = (
        list(ws.rglob("*.pdf"))
        + list(ws.rglob("*.docx"))
        + list(ws.rglob("*.txt"))
        + list(ws.rglob("*.md"))
    )
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


def resolve_spreadsheet_path_from_text(task: str, workspace: Optional[str] = None) -> Optional[str]:
    text = task.strip()
    if not text:
        return None

    for candidate in _extract_path_candidates(text, extensions=["csv", "xlsx"]):
        resolved = _resolve_existing_path(candidate, workspace=workspace)
        if resolved:
            return resolved

    if not workspace:
        return None

    ws = Path(workspace).expanduser().resolve()
    if not ws.exists():
        return None

    candidates = list(ws.rglob("*.csv")) + list(ws.rglob("*.xlsx"))
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


def resolve_code_path_from_text(task: str, workspace: Optional[str] = None) -> Optional[str]:
    text = task.strip()
    if not text:
        return None

    ext_pattern = r"(?:py|js|ts|tsx|jsx|java|go|rs|cpp|c|h|hpp|cs|rb|php|sh|yaml|yml|json|toml|ini|cfg)"

    abs_match = re.search(rf"(/[^\"'\s]+?\.{ext_pattern})", text, flags=re.IGNORECASE)
    if abs_match:
        p = Path(abs_match.group(1)).expanduser().resolve()
        if p.exists() and p.is_file():
            return str(p)

    quoted = re.findall(rf"[\"']([^\"']+\.{ext_pattern})[\"']", text, flags=re.IGNORECASE)
    for q in quoted:
        p = Path(q).expanduser()
        if not p.is_absolute() and workspace:
            p = Path(workspace) / p
        p = p.resolve()
        if p.exists() and p.is_file():
            return str(p)

    rel_tokens = re.findall(rf"([A-Za-z0-9_\-./]+\.{ext_pattern})", text, flags=re.IGNORECASE)
    for token in rel_tokens:
        p = Path(token).expanduser()
        if not p.is_absolute() and workspace:
            p = Path(workspace) / p
        p = p.resolve()
        if p.exists() and p.is_file():
            return str(p)
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
    if intent not in {"GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW", "DOC_SUMMARY", "CODE_TASK"}:
        return None

    mode_raw = str(parsed.get("mode", "EXECUTE")).upper()
    mode = "EXECUTE" if mode_raw in {"EXECUTE", "RUN", "WORKFLOW"} else "ANSWER_ONLY"
    needs_knowledge = parse_bool(parsed.get("needs_knowledge"), default=False)
    requirements: Dict[str, Any] = {}
    if intent == "ML_WORKFLOW":
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
        "code_path": str(parsed.get("code_path", "")).strip() or None,
        "command": str(parsed.get("command", "")).strip() or None,
        "find_text": parsed.get("find_text"),
        "replace_text": parsed.get("replace_text"),
        "count": parsed.get("count", 1),
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
            "(3) DOC_SUMMARY for reading and summarizing .docx/.pdf files by file path, "
            "(4) spreadsheet preview/profile for .csv/.xlsx files, "
            "(5) CODE_TASK for code analysis/editing/testing in workspace via tools. "
            "Computer vision/object detection is not executable in current pipeline."
        )
        user_prompt = {
            "task": task,
            "recent_history": history or [],
            "allowed_intents": ["GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW", "DOC_SUMMARY", "CODE_TASK"],
            "requirement_fields": REQUIREMENT_BOOL_FIELDS + ["model_hint"],
            "instructions": [
                "For casual chat or capability questions, use GENERAL_CHAT.",
                "For historical/doc retrieval requests, use KNOWLEDGE_LOOKUP.",
                "For data/model/training/evaluation/report work, use ML_WORKFLOW.",
                "For requests to summarize/read a Word/PDF file, use DOC_SUMMARY and extract file_path when present.",
                "For CSV/XLSX inspection or profiling requests, prefer ML_WORKFLOW with EXECUTE and include file path.",
                "For codebase analysis/edit/update/test tasks, use CODE_TASK.",
                "Distinguish answer-only vs execution: capability inquiry should be answer-only.",
                "Do not claim unsupported execution capabilities.",
                "Set needs_knowledge=true when ML task should reference knowledge/history/policy/docs.",
                "If uncertain, use GENERAL_CHAT and ask a focused clarification.",
                "Do not hallucinate constraints.",
            ],
            "output_schema_hint": {
                "intent": "GENERAL_CHAT | KNOWLEDGE_LOOKUP | ML_WORKFLOW | DOC_SUMMARY | CODE_TASK",
                "mode": "ANSWER_ONLY | EXECUTE",
                "confidence": "0.0-1.0",
                "needs_knowledge": "bool",
                "task_domain": "tabular_ml | knowledge | doc | code | cv | nlp | general",
                "supported_execution": "bool",
                "file_path": "string|null",
                "code_path": "string|null",
                "command": "string|null",
                "find_text": "string|null",
                "replace_text": "string|null",
                "count": "int",
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
        if intent not in {"GENERAL_CHAT", "KNOWLEDGE_LOOKUP", "ML_WORKFLOW", "DOC_SUMMARY", "CODE_TASK"}:
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
        code_path = str(parsed.get("code_path", "")).strip() or None
        command = str(parsed.get("command", "")).strip() or None
        find_text_raw = parsed.get("find_text")
        replace_text_raw = parsed.get("replace_text")
        find_text = find_text_raw if isinstance(find_text_raw, str) and find_text_raw != "" else None
        replace_text = replace_text_raw if isinstance(replace_text_raw, str) else None
        try:
            replace_count = int(parsed.get("count", 1))
        except Exception:
            replace_count = 1
        replace_count = max(1, min(replace_count, 1000))
        task_domain = str(parsed.get("task_domain", "general")).lower().strip()
        if task_domain not in {"tabular_ml", "knowledge", "cv", "nlp", "doc", "code", "general"}:
            task_domain = "general"
        supported_execution = parse_bool(
            parsed.get("supported_execution"),
            default=task_domain in {"tabular_ml", "knowledge", "doc", "code", "general"},
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
            "code_path": code_path,
            "command": command,
            "find_text": find_text,
            "replace_text": replace_text,
            "count": replace_count,
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

        inferred_code_path = resolve_code_path_from_text(task, workspace=self.workspace)
        if inferred_code_path:
            return {
                "intent": "CODE_TASK",
                "mode": "EXECUTE",
                "confidence": 0.82,
                "needs_knowledge": False,
                "task_domain": "code",
                "supported_execution": True,
                "file_path": None,
                "code_path": inferred_code_path,
                "command": None,
                "find_text": None,
                "replace_text": None,
                "count": 1,
                "reply": "",
                "suggestions": [],
                "requirements": {},
                "source": f"{source}:code_path_resolver",
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
    - experience: reusable solved-problem summaries and skill hints
    - profile: user preferences
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {
            "session": {},
            "workflow": {},
            "knowledge": {},
            "experience": {},
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

    def append_workflow_event(self, trace_id: str, event: Dict[str, Any]) -> None:
        key = f"events::{trace_id}"
        row = self._data["workflow"].setdefault(key, {"events": []})
        events = row.setdefault("events", [])
        events.append(event)
        if len(events) > 200:
            row["events"] = events[-200:]

    def get_workflow_events(self, trace_id: str, limit: int = 40) -> List[Dict[str, Any]]:
        row = self._data["workflow"].get(f"events::{trace_id}", {"events": []})
        events = row.get("events", [])
        if not isinstance(events, list):
            return []
        return events[-max(1, limit):]

    def put_knowledge_doc(
        self,
        doc_id: str,
        title: str,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        row = {
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "source": source,
            "updated_at_ms": now_ms(),
        }
        if isinstance(metadata, dict):
            row["metadata"] = dict(metadata)
        self._data["knowledge"][doc_id] = row

    def get_knowledge_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self._data["knowledge"].get(doc_id)
        if isinstance(row, dict):
            return dict(row)
        return None

    def list_knowledge_sources(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = list(self._data["knowledge"].values())
        rows.sort(key=lambda x: x.get("updated_at_ms", 0), reverse=True)
        out = []
        for row in rows[: max(1, min(limit, 1000))]:
            out.append(
                {
                    "doc_id": row.get("doc_id"),
                    "title": row.get("title"),
                    "source": row.get("source"),
                    "updated_at_ms": row.get("updated_at_ms"),
                    "metadata": row.get("metadata", {}),
                }
            )
        return out

    def put_experience(self, exp_id: str, entry: Dict[str, Any]) -> None:
        row = dict(entry)
        row.setdefault("experience_id", exp_id)
        row.setdefault("created_at_ms", now_ms())
        row["updated_at_ms"] = now_ms()
        self._data["experience"][exp_id] = row

    def recent_experiences(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = list(self._data["experience"].values())
        rows.sort(key=lambda x: x.get("created_at_ms", 0), reverse=True)
        return rows[: max(1, min(limit, 100))]

    def all_experiences(self) -> List[Dict[str, Any]]:
        rows = list(self._data["experience"].values())
        rows.sort(key=lambda x: x.get("created_at_ms", 0))
        return rows

    def search_experiences(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = (query or "").lower().strip()
        if not q:
            return self.recent_experiences(limit=top_k)
        terms = [t for t in re.split(r"[\s,.;:!?，。；：！？、/\\-]+", q) if len(t) >= 2]
        if not terms:
            terms = [q]
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for row in self._data["experience"].values():
            text = "\n".join(
                [
                    str(row.get("problem", "")),
                    str(row.get("goal", "")),
                    str(row.get("summary", "")),
                    str(row.get("skill_candidate", {}).get("name", "")),
                    " ".join([str(x) for x in row.get("executed_tools", [])]),
                ]
            ).lower()
            score = sum(text.count(term) for term in terms)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[: max(1, min(top_k, 50))]]

    def search(self, namespace: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self._must_namespace(namespace)
        if namespace != "knowledge":
            return []
        q = (query or "").lower().strip()
        if not q:
            return []
        terms = [t for t in re.split(r"[\s,.;:!?，。；：！？、/\\-]+", q) if len(t) >= 2]
        if not terms:
            terms = [q]
        rows = list(self._data["knowledge"].values())
        if not rows:
            return []

        # BM25 document statistics
        doc_tokens: List[List[str]] = []
        doc_tf: List[Dict[str, int]] = []
        doc_lengths: List[int] = []
        df: Dict[str, int] = {}
        for row in rows:
            text = f"{row.get('title', '')}\n{row.get('text', '')}".lower()
            tokens = [t for t in re.split(r"[\s,.;:!?，。；：！？、/\\-]+", text) if t]
            if not tokens:
                tokens = ["_empty_"]
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token in tf.keys():
                df[token] = df.get(token, 0) + 1
            doc_tokens.append(tokens)
            doc_tf.append(tf)
            doc_lengths.append(len(tokens))

        num_docs = len(rows)
        avgdl = sum(doc_lengths) / max(1, len(doc_lengths))
        k1 = 1.5
        b = 0.75

        # Grep-style recall over source files (fast lexical exact-match signal).
        grep_hits_by_source: Dict[str, int] = {}
        source_files = []
        for row in rows:
            source = str(row.get("source", ""))
            if source and os.path.isfile(source):
                source_files.append(source)
        # De-duplicate while preserving order.
        source_files = list(dict.fromkeys(source_files))
        if source_files:
            grep_terms = [q] + [t for t in terms if t != q][:4]
            for term in grep_terms:
                try:
                    proc = subprocess.run(
                        ["rg", "-n", "-i", "--no-heading", "-F", term, *source_files],
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode not in {0, 1}:
                        continue
                    for line in (proc.stdout or "").splitlines():
                        parts = line.split(":", 2)
                        if len(parts) < 3:
                            continue
                        src = parts[0]
                        grep_hits_by_source[src] = grep_hits_by_source.get(src, 0) + 1
                except Exception:
                    # Keep retrieval resilient when rg is unavailable.
                    break

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            tf = doc_tf[idx]
            dl = doc_lengths[idx]
            bm25_score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                n = df.get(term, 0)
                if f <= 0 or n <= 0:
                    continue
                idf = math.log(1.0 + (num_docs - n + 0.5) / (n + 0.5))
                bm25_score += idf * ((f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(1e-9, avgdl))))

            source = str(row.get("source", ""))
            grep_hits = grep_hits_by_source.get(source, 0)
            title_text = str(row.get("title", "")).lower()
            title_hits = sum(title_text.count(term) for term in terms)

            final_score = bm25_score + 0.35 * grep_hits + 0.15 * title_hits
            if final_score > 0:
                scored.append((final_score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[: max(1, top_k)]]

    def _must_namespace(self, namespace: str) -> None:
        if namespace not in self._data:
            raise ValueError(f"unknown namespace: {namespace}")


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


class ExperienceAgent(BaseAgent):
    """
    Captures solved-problem experience and proposes reusable skill candidates.
    Persists entries under `skills/experience_catalog.json`.
    """

    def __init__(
        self,
        memory: MemoryStore,
        workspace: str,
        model: str = "gpt-4o",
        runtime_tool_names: Optional[List[str]] = None,
    ):
        super().__init__("experience_agent")
        self.memory = memory
        self.workspace = Path(workspace).expanduser().resolve()
        self.model = model
        self.runtime_tool_names = set([str(x).strip() for x in (runtime_tool_names or []) if str(x).strip()])
        self.catalog_path = self.workspace / "skills" / "experience_catalog.json"
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None
        self._load_catalog()

    def handle(self, message: Message, state: State) -> Message:
        payload = message.get("content", {})
        entry = self.summarize_and_store(
            session_id=str(payload.get("session_id", state.get("session_id", "default"))),
            trace_id=str(payload.get("trace_id", state.get("trace_id", f"trace_{now_ms()}"))),
            task=str(payload.get("task", "")),
            response=ensure_dict(payload.get("response", {})),
            loop_state=ensure_dict(payload.get("loop_state", {})),
        )
        return {
            "sender": self.name,
            "receiver": payload.get("return_to", "user"),
            "type": "experience_logged",
            "priority": 55,
            "content": {"experience": entry},
            "metadata": {"trace_id": state.get("trace_id", "")},
        }

    def _load_catalog(self) -> None:
        if not self.catalog_path.exists():
            return
        try:
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except Exception:
            return
        entries = data.get("experiences", []) if isinstance(data, dict) else []
        if not isinstance(entries, list):
            return
        for row in entries:
            if not isinstance(row, dict):
                continue
            exp_id = str(row.get("experience_id", "")).strip()
            if not exp_id:
                continue
            self.memory.put_experience(exp_id, row)

    def _save_catalog(self) -> None:
        payload = {
            "updated_at_ms": now_ms(),
            "experiences": self.memory.all_experiences()[-500:],
        }
        self.catalog_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sanitize_skill_name(self, raw: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(raw or "").strip().lower())
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = f"skill_{now_ms()}"
        return name[:80]

    def _upsert_local_skill_manifest(self, skill_name: str, skill_path: Path) -> None:
        manifest_path = self.workspace / "skills" / "skills_manifest.json"
        payload: Dict[str, Any] = {"skills": []}
        if manifest_path.exists():
            try:
                parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict) and isinstance(parsed.get("skills"), list):
                    payload = parsed
            except Exception:
                payload = {"skills": []}
        rows = [row for row in payload.get("skills", []) if isinstance(row, dict) and row.get("name") != skill_name]
        rows.append(
            {
                "name": skill_name,
                "repo_url": "local_distilled",
                "path": str(skill_path),
                "ref": "local",
                "commit": "",
                "installed_at_ms": now_ms(),
            }
        )
        payload["skills"] = rows
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _normalize_tool_chain(self, tool_chain: Any) -> List[str]:
        if not isinstance(tool_chain, list):
            return []
        out: List[str] = []
        for raw in tool_chain:
            name = str(raw).strip()
            if not name:
                continue
            if self.runtime_tool_names and name not in self.runtime_tool_names:
                continue
            out.append(name)
        return out[:12]

    def _tool_chain_signature(self, tool_chain: List[str]) -> str:
        return " -> ".join([str(x).strip() for x in tool_chain if str(x).strip()])

    def _extract_tool_chain_from_skill_md(self, skill_md_path: Path) -> List[str]:
        if not skill_md_path.exists() or not skill_md_path.is_file():
            return []
        try:
            raw = skill_md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        in_chain = False
        out: List[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("## tool chain"):
                in_chain = True
                continue
            if stripped.startswith("## ") and in_chain:
                break
            if not in_chain:
                continue
            m = re.search(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", stripped)
            if not m:
                continue
            name = m.group(1).strip()
            if not name:
                continue
            if self.runtime_tool_names and name not in self.runtime_tool_names:
                continue
            out.append(name)
        return out[:12]

    def _existing_skill_signatures(self) -> set[str]:
        signatures: set[str] = set()
        manifest_path = self.workspace / "skills" / "skills_manifest.json"
        if not manifest_path.exists():
            return signatures
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return signatures
        rows = data.get("skills", []) if isinstance(data, dict) else []
        if not isinstance(rows, list):
            return signatures
        for row in rows:
            if not isinstance(row, dict):
                continue
            path = str(row.get("path", "")).strip()
            if not path:
                continue
            chain = self._extract_tool_chain_from_skill_md(Path(path) / "SKILL.md")
            sig = self._tool_chain_signature(chain)
            if sig:
                signatures.add(sig)
        return signatures

    def _existing_experience_signatures(self) -> set[str]:
        signatures: set[str] = set()
        for row in self.memory.all_experiences():
            if not isinstance(row, dict):
                continue
            candidate = ensure_dict(row.get("skill_candidate", {}))
            chain = self._normalize_tool_chain(candidate.get("tool_chain", []))
            if not chain:
                chain = self._normalize_tool_chain(row.get("executed_tools", []))
            sig = self._tool_chain_signature(chain)
            if sig:
                signatures.add(sig)
        return signatures

    def _should_distill_experience(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        intent = str(entry.get("intent", "")).upper().strip()
        status = str(entry.get("status", "")).lower().strip()
        executed_tools = self._normalize_tool_chain(entry.get("executed_tools", []))
        failed = entry.get("what_failed", [])
        failed_count = len(failed) if isinstance(failed, list) else 0
        candidate = ensure_dict(entry.get("skill_candidate", {}))
        tool_chain = self._normalize_tool_chain(candidate.get("tool_chain", []))
        if not tool_chain:
            tool_chain = executed_tools

        if intent == "GENERAL_CHAT":
            return {"allow": False, "reason": "intent_is_general_chat"}
        if status in {"clarification_needed", "clarify", "chat", "unknown"}:
            return {"allow": False, "reason": f"status_not_distillable:{status}"}
        if not tool_chain:
            return {"allow": False, "reason": "empty_tool_chain"}
        if len(executed_tools) < 2 and intent not in {"DOC_SUMMARY", "KNOWLEDGE_LOOKUP", "DYNAMIC_EXECUTION"}:
            return {"allow": False, "reason": "insufficient_tool_depth"}
        failure_rate = failed_count / max(1, len(executed_tools))
        if failure_rate > 0.5:
            return {"allow": False, "reason": f"high_failure_rate:{round(failure_rate, 3)}"}

        signature = self._tool_chain_signature(tool_chain)
        if not signature:
            return {"allow": False, "reason": "empty_signature"}

        existing_signatures = self._existing_skill_signatures() | self._existing_experience_signatures()
        if signature in existing_signatures:
            return {"allow": False, "reason": "duplicate_tool_chain_signature"}

        chain_set = set(tool_chain)
        for sig in existing_signatures:
            sig_set = set([x.strip() for x in sig.split("->") if x.strip()])
            if not sig_set:
                continue
            overlap = len(chain_set & sig_set) / max(1, len(chain_set | sig_set))
            if overlap >= 0.8:
                return {"allow": False, "reason": f"near_duplicate_overlap:{round(overlap, 3)}"}

        return {
            "allow": True,
            "reason": "pass",
            "tool_chain_signature": signature,
            "failure_rate": round(failure_rate, 4),
        }

    def _distill_experience_to_skill(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidate = ensure_dict(entry.get("skill_candidate", {}))
        raw_name = candidate.get("name") or f"skill_{entry.get('intent', 'general')}"
        skill_name = self._sanitize_skill_name(str(raw_name))
        distilled_root = self.workspace / "skills" / "distilled"
        skill_path = distilled_root / skill_name
        skill_path.mkdir(parents=True, exist_ok=True)
        skill_md_path = skill_path / "SKILL.md"

        tool_chain = candidate.get("tool_chain", [])
        if not isinstance(tool_chain, list):
            tool_chain = []
        playbook_steps = entry.get("playbook_steps", [])
        if not isinstance(playbook_steps, list):
            playbook_steps = []
        what_failed = entry.get("what_failed", [])
        if not isinstance(what_failed, list):
            what_failed = []

        lines = [
            f"# {skill_name}",
            "",
            "## Purpose",
            str(candidate.get("description", "Distilled skill from solved runtime experience.")).strip(),
            "",
            "## When To Use",
            str(candidate.get("when_to_use", "Use when the new task has similar objective and constraints.")).strip(),
            "",
            "## Workflow",
        ]
        def _format_playbook_line(raw: Any) -> str:
            text = str(raw).strip()
            if text.startswith("{") and "step" in text and "tool" in text:
                step_m = re.search(r"'step'\s*:\s*'([^']+)'", text)
                tool_m = re.search(r"'tool'\s*:\s*'([^']+)'", text)
                if step_m and tool_m:
                    return f"{step_m.group(1)} (tool: {tool_m.group(1)})"
            return text

        if playbook_steps:
            for idx, step in enumerate(playbook_steps[:12], start=1):
                lines.append(f"{idx}. {_format_playbook_line(step)}")
        elif tool_chain:
            for idx, tool in enumerate(tool_chain[:12], start=1):
                lines.append(f"{idx}. Execute `{str(tool).strip()}`")
        else:
            lines.append("1. Clarify goal and constraints.")
            lines.append("2. Execute tools incrementally with verification.")
            lines.append("3. Summarize outputs and next actions.")

        lines.extend(["", "## Common Failure Patterns"])
        if what_failed:
            for row in what_failed[:8]:
                lines.append(f"- {str(row).strip()}")
        else:
            lines.append("- No notable failure pattern captured yet.")

        lines.extend(["", "## Tool Chain"])
        if tool_chain:
            for tool in tool_chain[:12]:
                lines.append(f"- `{str(tool).strip()}`")
        else:
            lines.append("- (none)")

        lines.extend(
            [
                "",
                "## Provenance",
                f"- experience_id: `{entry.get('experience_id', '')}`",
                f"- trace_id: `{entry.get('trace_id', '')}`",
                f"- intent: `{entry.get('intent', '')}`",
                f"- created_at_ms: `{entry.get('created_at_ms', 0)}`",
                "",
            ]
        )
        skill_md_path.write_text("\n".join(lines), encoding="utf-8")
        self._upsert_local_skill_manifest(skill_name=skill_name, skill_path=skill_path)
        return {
            "name": skill_name,
            "path": str(skill_path),
            "skill_md": str(skill_md_path),
        }

    def _llm_experience_summary(
        self,
        task: str,
        response: Dict[str, Any],
        loop_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None
        system_prompt = (
            "You are an experience mining module. "
            "Convert one solved run into reusable engineering experience. "
            "Return strict JSON."
        )
        payload = {
            "task": task,
            "response": {
                "intent": response.get("intent"),
                "status": response.get("status"),
                "summary": truncate_text(response.get("result_summary") or response.get("reply", ""), max_chars=1200),
                "executed_tools": response.get("executed_tools", []),
            },
            "observations": loop_state.get("observations", [])[-8:],
            "output_schema": {
                "problem": "string",
                "summary": "string",
                "what_worked": ["string"],
                "what_failed": ["string"],
                "playbook_steps": ["string"],
                "skill_candidate": {
                    "name": "string",
                    "description": "string",
                    "when_to_use": "string",
                    "tool_chain": ["string"],
                },
            },
        }
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

    def summarize_and_store(
        self,
        session_id: str,
        trace_id: str,
        task: str,
        response: Dict[str, Any],
        loop_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        loop_state = loop_state or {}
        executed_tools = response.get("executed_tools", []) or loop_state.get("executed_tools", [])
        if not isinstance(executed_tools, list):
            executed_tools = []
        observations = loop_state.get("observations", []) if isinstance(loop_state.get("observations"), list) else []
        failures = [str(row.get("error", "")) for row in observations if isinstance(row, dict) and not row.get("ok", True)]
        failures = [x for x in failures if x][:6]
        plan = response.get("plan", []) or loop_state.get("plan", [])
        if not isinstance(plan, list):
            plan = []

        llm_row = self._llm_experience_summary(task=task, response=response, loop_state=loop_state) or {}
        summary = str(llm_row.get("summary", "")).strip() or truncate_text(
            response.get("result_summary") or response.get("reply", "No summary."),
            max_chars=1200,
        )
        what_worked = llm_row.get("what_worked", [])
        if not isinstance(what_worked, list) or not what_worked:
            what_worked = [f"Executed tool: {tool}" for tool in executed_tools[:8]]

        what_failed = llm_row.get("what_failed", [])
        if not isinstance(what_failed, list) or not what_failed:
            what_failed = failures or []

        playbook_steps = llm_row.get("playbook_steps", [])
        if not isinstance(playbook_steps, list) or not playbook_steps:
            playbook_steps = [str(x) for x in plan[:10] if str(x).strip()]

        skill_candidate = llm_row.get("skill_candidate", {})
        if not isinstance(skill_candidate, dict) or not skill_candidate:
            top_tool = executed_tools[0] if executed_tools else "general_workflow"
            skill_candidate = {
                "name": f"skill_{top_tool}".lower(),
                "description": "Reusable workflow learned from solved user requests.",
                "when_to_use": "Use when a new task has similar goal and tool pattern.",
                "tool_chain": executed_tools[:8],
            }

        exp_id = f"exp_{now_ms()}"
        entry = {
            "experience_id": exp_id,
            "session_id": session_id,
            "trace_id": trace_id,
            "problem": str(llm_row.get("problem", "")).strip() or task,
            "goal": response.get("goal", ""),
            "intent": response.get("intent", "GENERAL_CHAT"),
            "status": response.get("status", response.get("planner_mode", "unknown")),
            "summary": summary,
            "what_worked": [str(x).strip() for x in what_worked if str(x).strip()][:10],
            "what_failed": [str(x).strip() for x in what_failed if str(x).strip()][:10],
            "playbook_steps": [str(x).strip() for x in playbook_steps if str(x).strip()][:12],
            "executed_tools": [str(x) for x in executed_tools][:12],
            "skill_candidate": {
                "name": str(skill_candidate.get("name", "skill_general")).strip() or "skill_general",
                "description": str(skill_candidate.get("description", "Reusable skill candidate")).strip(),
                "when_to_use": str(skill_candidate.get("when_to_use", "When similar requests appear")).strip(),
                "tool_chain": [str(x) for x in skill_candidate.get("tool_chain", executed_tools)][:12]
                if isinstance(skill_candidate.get("tool_chain", executed_tools), list)
                else [str(x) for x in executed_tools][:12],
            },
            "created_at_ms": now_ms(),
        }
        self.memory.put_experience(exp_id, entry)
        self._save_catalog()
        distill_decision = self._should_distill_experience(entry)
        entry["distill_decision"] = distill_decision
        self.memory.put_experience(exp_id, entry)
        self._save_catalog()
        try:
            if distill_decision.get("allow", False):
                distilled = self._distill_experience_to_skill(entry)
                if distilled:
                    entry["distilled_skill"] = distilled
                    self.memory.put_experience(exp_id, entry)
                    self._save_catalog()
        except Exception:
            pass
        return entry


class DynamicLoopOrchestrator:
    """
    Goal -> Understand -> Plan -> Tool Calling -> Observation -> Memory/State -> Continue/Finish
    """

    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        memory: MemoryStore,
        workspace: str,
        experience_agent: Optional[ExperienceAgent] = None,
    ):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.experience_agent = experience_agent
        self.policy = ExecutionPolicy()
        self.workspace = Path(workspace).expanduser().resolve()
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None
        self.enable_verifier = parse_bool(os.getenv("ENABLE_VERIFIER", "1"), default=True)

    def available(self) -> bool:
        return self.client is not None

    def _extract_lookup_token(self, task: str) -> Optional[str]:
        text = (task or "").strip()
        if not text:
            return None
        uuid_match = re.search(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", text)
        if uuid_match:
            return uuid_match.group(0)
        hex_match = re.search(r"\b[0-9a-fA-F]{16,}\b", text)
        if hex_match:
            return hex_match.group(0)
        quoted = re.findall(r"[\"']([^\"']{8,})[\"']", text)
        for q in quoted:
            if any(ch.isdigit() for ch in q):
                return q.strip()
        return None

    def _rule_based_preplan(self, task: str) -> Optional[Dict[str, Any]]:
        text = (task or "").strip()
        if not text:
            return None
        low = text.lower()
        if (
            ("logistic regression" in low or "logisticregression" in low)
            and any(k in low for k in ["download", "dataset", "data", "train", "model"])
        ):
            return {
                "mode": "EXECUTE",
                "goal": "Prepare a dataset and run logistic regression with executable outputs.",
                "success_criteria": ["Return dataset path, training metrics, and reproducible artifacts."],
                "constraints": [],
                "clarification_question": "",
                "final_reply": "",
                "plan": [
                    {
                        "step": "Prepare dataset and run logistic regression demo",
                        "tool": "prepare_logistic_regression_demo",
                        "arguments": {"task": task, "output_path": "data/logreg_dataset.csv"},
                    }
                ],
            }

        token = self._extract_lookup_token(text)
        asks_lookup = any(k in low for k in ["find", "search", "locate", "lookup", "where", "contains", "key", "id"])
        mentions_json = "json" in low
        if not ((mentions_json and asks_lookup) or (token and asks_lookup)):
            return None

        query = token or text
        return {
            "mode": "EXECUTE",
            "goal": "Locate target key/value in workspace JSON files.",
            "success_criteria": ["Return matched files and lines, or clearly state no match found."],
            "constraints": [],
            "clarification_question": "",
            "final_reply": "",
            "plan": [
                {
                    "step": "List JSON files in workspace",
                    "tool": "list_workspace_files",
                    "arguments": {"pattern": "*.json", "recursive": True, "limit": 500},
                },
                {
                    "step": "Search key across workspace text",
                    "tool": "search_workspace_text",
                    "arguments": {"query": query, "top_k": 80},
                },
            ],
        }

    def _get_installed_skills(self) -> List[str]:
        if "skill_list_installed" not in self.tools.tools:
            return []
        try:
            row = self.tools.execute("skill_list_installed")
            skills = row.get("skills", [])
            if isinstance(skills, list):
                descriptions: List[str] = []
                for skill in skills:
                    if not isinstance(skill, dict):
                        continue
                    name = str(skill.get("name", "unknown_skill")).strip() or "unknown_skill"
                    repo = str(skill.get("repo_url", "")).strip()
                    ref = str(skill.get("ref", "")).strip()
                    path = str(skill.get("path", "")).strip()
                    desc = f"{name}: installed skill"
                    if repo:
                        desc += f" from {repo}"
                    if ref:
                        desc += f" (ref={ref})"
                    skill_md = ""
                    if path:
                        p = Path(path) / "SKILL.md"
                        if p.exists() and p.is_file():
                            try:
                                raw = p.read_text(encoding="utf-8", errors="ignore")
                                for line in raw.splitlines():
                                    line = line.strip()
                                    if not line:
                                        continue
                                    if line.startswith("#"):
                                        continue
                                    skill_md = line
                                    break
                            except Exception:
                                pass
                    if skill_md:
                        desc += f". {truncate_text(skill_md, max_chars=160)}"
                    descriptions.append(desc)
                return descriptions[:20]
        except Exception:
            return []
        return []

    def _record_event(self, trace_id: str, phase: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "ts_ms": now_ms(),
            "phase": phase,
            "payload": payload,
        }
        self.memory.append_workflow_event(trace_id, event)
        return event

    def _capture_experience(
        self,
        session_id: str,
        trace_id: str,
        task: str,
        response: Dict[str, Any],
        loop_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.experience_agent is None:
            return None
        try:
            return self.experience_agent.summarize_and_store(
                session_id=session_id,
                trace_id=trace_id,
                task=task,
                response=response,
                loop_state=loop_state,
            )
        except Exception:
            return None

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

    def _json_schema_from_input_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for key, raw_type in (schema or {}).items():
            t = str(raw_type).lower()
            prop: Dict[str, Any] = {"description": str(raw_type)}
            if "int" in t:
                prop["type"] = "integer"
            elif "bool" in t:
                prop["type"] = "boolean"
            elif "list" in t:
                prop["type"] = "array"
                prop["items"] = {"type": "string"}
            elif "dict" in t or "json" in t or "object" in t:
                prop["type"] = "object"
                prop["additionalProperties"] = True
            else:
                prop["type"] = "string"
            if "null" in t:
                prop = {"anyOf": [prop, {"type": "null"}]}
            properties[str(key)] = prop
            if "default" not in t and "null" not in t:
                required.append(str(key))
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": True,
        }

    def _openai_tool_definitions(self) -> List[Dict[str, Any]]:
        defs = []
        for spec in self.tools.tools.values():
            defs.append(
                {
                    "type": "function",
                    "name": spec.name,
                    "description": spec.description or spec.name,
                    "parameters": self._json_schema_from_input_schema(spec.input_schema),
                }
            )
        return defs

    def _extract_response_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return calls
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")
            if item_type not in {"function_call", "tool_call"}:
                continue
            name = getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("name")
            args = getattr(item, "arguments", None)
            if args is None and isinstance(item, dict):
                args = item.get("arguments", "{}")
            call_id = getattr(item, "call_id", None)
            if call_id is None and isinstance(item, dict):
                call_id = item.get("call_id") or item.get("id")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(args, str):
                try:
                    args = json.dumps(args or {}, ensure_ascii=False)
                except Exception:
                    args = "{}"
            calls.append(
                {
                    "name": name.strip(),
                    "arguments": args,
                    "call_id": str(call_id or f"call_{now_ms()}"),
                }
            )
        return calls

    def _run_model_tool_loop(
        self,
        task: str,
        trace_id: str,
        session_id: str,
        recent_messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None

        # Canonical agent loop:
        # messages -> model -> tool_use? -> execute tools -> append tool_result -> loop
        system_prompt = (
            "You are the runtime controller in a tool-using agent loop. "
            "Follow this pattern exactly: understand goal -> decide next action -> call tools if needed -> observe outputs -> continue or stop. "
            "You decide when to call tools and when to finish. "
            "When information is missing, ask one concise clarification question instead of guessing. "
            "Do not fabricate tool outputs. Use only available tools."
        )
        recent_context = []
        for row in recent_messages[-8:]:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "user")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = str(row.get("content", "")).strip()
            if not text:
                continue
            recent_context.append({"role": role, "content": truncate_text(text, max_chars=1000)})

        user_payload = {
            "task": task,
            "workspace_root": str(self.workspace),
            "recent_context": recent_context,
            "instruction": (
                "If tools are needed, call them. If not, answer directly. "
                "If blocked by missing inputs, ask a focused question."
            ),
        }

        tool_defs = self._openai_tool_definitions()
        observations: List[Dict[str, Any]] = []
        executed_tools: List[str] = []
        loop_state: Dict[str, Any] = {
            "task": task,
            "goal": task,
            "observations": observations,
            "executed_tools": executed_tools,
            "plan": [],
        }
        max_iters = 24
        repeated_calls: Dict[str, int] = {}

        self._record_event(trace_id, "tool_loop_start", {"task_preview": truncate_text(task, max_chars=220)})

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                tools=tool_defs,
                temperature=0,
            )
        except Exception:
            return None

        for iteration in range(1, max_iters + 1):
            calls = self._extract_response_tool_calls(response)
            if not calls:
                final_text = (getattr(response, "output_text", "") or "").strip()
                if not final_text:
                    final_text = self._compose_final_answer(loop_state)
                normalized = final_text.lower()
                looks_like_clarify = (
                    "?" in final_text
                    and any(
                        marker in normalized
                        for marker in [
                            "please provide",
                            "could you",
                            "can you",
                            "which",
                            "what",
                            "path",
                            "file",
                            "dataset",
                        ]
                    )
                )
                if looks_like_clarify and not executed_tools:
                    self._record_event(
                        trace_id,
                        "finish",
                        {"status": "clarification_needed", "iteration": iteration, "via": "model_text"},
                    )
                    out = {
                        "intent": "GENERAL_CHAT",
                        "reply": final_text,
                        "suggestions": [],
                        "planner_mode": "MODEL_CLARIFY",
                        "goal": task,
                        "trace_id": trace_id,
                        "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                    }
                else:
                    self._record_event(trace_id, "finish", {"status": "completed", "iteration": iteration, "via": "model_text"})
                    out = {
                        "intent": "DYNAMIC_EXECUTION",
                        "status": "completed",
                        "goal": task,
                        "plan": [],
                        "executed_tools": executed_tools,
                        "observations": observations[-8:],
                        "result_summary": final_text,
                        "trace_id": trace_id,
                        "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                    }

                exp = self._capture_experience(
                    session_id=session_id,
                    trace_id=trace_id,
                    task=task,
                    response=out,
                    loop_state=loop_state,
                )
                if exp and out.get("intent") == "DYNAMIC_EXECUTION":
                    out["experience_id"] = exp.get("experience_id")
                    out["skill_candidate"] = exp.get("skill_candidate", {})
                return out

            outputs_for_model: List[Dict[str, Any]] = []
            for call in calls:
                tool_name = str(call.get("name", "")).strip()
                args = ensure_dict(call.get("arguments", "{}"))
                call_sig = f"{tool_name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                repeated_calls[call_sig] = repeated_calls.get(call_sig, 0) + 1

                self._record_event(
                    trace_id,
                    "tool_call",
                    {
                        "iteration": iteration,
                        "tool": tool_name,
                        "arguments_preview": to_json_text(args, max_chars=350),
                        "repeat_count": repeated_calls[call_sig],
                    },
                )

                if repeated_calls[call_sig] > 3:
                    result = {
                        "ok": False,
                        "error": "repeat_guard: same tool call repeated too many times; ask user for clarification.",
                    }
                    ok = False
                    err = result["error"]
                else:
                    spec = self.tools.get_spec(tool_name)
                    if spec is None:
                        result = {"ok": False, "error": f"tool not found: {tool_name}"}
                        ok = False
                        err = result["error"]
                    else:
                        policy_result = self.policy.evaluate(tool_name=tool_name, spec=spec, args=args, task=task)
                        if not policy_result.get("allow", False):
                            result = {"ok": False, "error": policy_result.get("reason", "blocked by policy")}
                            ok = False
                            err = result["error"]
                        else:
                            try:
                                result = self.tools.execute(tool_name, **args)
                                ok = not (isinstance(result, dict) and result.get("ok") is False)
                                err = result.get("error") if isinstance(result, dict) else None
                            except Exception as e:
                                result = {"ok": False, "error": str(e)}
                                ok = False
                                err = str(e)

                executed_tools.append(tool_name)
                compact = self._compact_tool_result(result)
                obs = {
                    "iteration": iteration,
                    "step": f"tool_use:{tool_name}",
                    "tool": tool_name,
                    "arguments": args,
                    "ok": ok,
                    "error": err,
                    "result": compact,
                    "result_preview": to_json_text(compact, max_chars=900),
                }
                observations.append(obs)
                self._record_event(
                    trace_id,
                    "observation",
                    {
                        "iteration": iteration,
                        "tool": tool_name,
                        "ok": ok,
                        "error": err,
                        "result_preview": to_json_text(compact, max_chars=350),
                    },
                )
                outputs_for_model.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(call.get("call_id", f"call_{now_ms()}")),
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

            try:
                response = self.client.responses.create(
                    model=self.model,
                    previous_response_id=getattr(response, "id", None),
                    input=outputs_for_model,
                    tools=tool_defs,
                    temperature=0,
                )
            except Exception as e:
                observations.append(
                    {
                        "iteration": iteration,
                        "step": "model_followup",
                        "tool": "model",
                        "arguments": {},
                        "ok": False,
                        "error": f"model follow-up failed: {e}",
                        "result": {},
                    }
                )
                break

        self._record_event(trace_id, "finish", {"status": "max_iterations"})
        fallback = {
            "intent": "DYNAMIC_EXECUTION",
            "status": "max_iterations_reached",
            "goal": task,
            "plan": [],
            "executed_tools": executed_tools,
            "observations": observations[-8:],
            "result_summary": self._compose_final_answer(loop_state),
            "trace_id": trace_id,
            "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
        }
        exp = self._capture_experience(
            session_id=session_id,
            trace_id=trace_id,
            task=task,
            response=fallback,
            loop_state=loop_state,
        )
        if exp:
            fallback["experience_id"] = exp.get("experience_id")
            fallback["skill_candidate"] = exp.get("skill_candidate", {})
        return fallback

    def _fallback_understand_goal(self, task: str) -> Dict[str, Any]:
        task_text = (task or "").strip()
        low = task_text.lower()
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
            if intent == "CODE_TASK":
                code_path = explicit.get("code_path")
                command = explicit.get("command")
                find_text = explicit.get("find_text")
                replace_text = explicit.get("replace_text")
                replace_count = explicit.get("count", 1)
                plan = []
                if code_path:
                    plan.append({"step": "Read target code file", "tool": "read_code_file", "arguments": {"path": code_path}})
                if code_path and isinstance(find_text, str) and isinstance(replace_text, str):
                    plan.append(
                        {
                            "step": "Apply deterministic text replacement",
                            "tool": "replace_text_in_file",
                            "arguments": {
                                "path": code_path,
                                "find_text": find_text,
                                "replace_text": replace_text,
                                "count": replace_count,
                            },
                        }
                    )
                if command:
                    plan.append(
                        {
                            "step": "Run workspace command",
                            "tool": "run_shell_command",
                            "arguments": {"command": command, "timeout_s": 90},
                        }
                    )
                if not plan:
                    plan.append({"step": "Search related code context", "tool": "search_workspace_text", "arguments": {"query": task, "top_k": 20}})
                return {
                    "mode": "EXECUTE",
                    "goal": "Execute code-focused task in workspace and return actionable output.",
                    "success_criteria": ["Provide concrete findings or run result for the code task."],
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

        doc_candidates = _extract_path_candidates(task_text, extensions=["pdf", "docx", "txt", "md"])
        if doc_candidates:
            return {
                "mode": "CHAT",
                "goal": "Request corrected document path.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": (
                    f"I found a document path-like input (`{doc_candidates[0]}`), but it is not accessible in the current workspace. "
                    "Please confirm the exact readable path."
                ),
                "plan": [],
            }

        inferred_spreadsheet_path = resolve_spreadsheet_path_from_text(task, workspace=str(self.workspace))
        if inferred_spreadsheet_path:
            return {
                "mode": "EXECUTE",
                "goal": f"Read and analyze tabular data from {inferred_spreadsheet_path}.",
                "success_criteria": ["Provide dataset structure and quick data profile."],
                "constraints": [],
                "clarification_question": "",
                "final_reply": "",
                "plan": [
                    {
                        "step": "Write and run Python analyzer for tabular file",
                        "tool": "analyze_tabular_with_python",
                        "arguments": {"path": inferred_spreadsheet_path, "max_rows": 200},
                    },
                ],
            }

        sheet_candidates = _extract_path_candidates(task_text, extensions=["csv", "xlsx"])
        if sheet_candidates:
            return {
                "mode": "CHAT",
                "goal": "Request corrected tabular file path.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": (
                    f"I found a tabular path-like input (`{sheet_candidates[0]}`), but that file is not accessible right now. "
                    "Please confirm the path or place the file under the workspace."
                ),
                "plan": [],
            }

        inferred_code_path = resolve_code_path_from_text(task, workspace=str(self.workspace))
        if inferred_code_path:
            return {
                "mode": "EXECUTE",
                "goal": f"Analyze and update code task on {inferred_code_path} in workspace.",
                "success_criteria": ["Provide code-focused findings or action result."],
                "constraints": [],
                "clarification_question": "",
                "final_reply": "",
                "plan": [
                    {"step": "Read code file", "tool": "read_code_file", "arguments": {"path": inferred_code_path}},
                ],
            }

        if ("walk through" in low or "walkthrough" in low or "list files" in low) and "workspace" in low:
            return {
                "mode": "EXECUTE",
                "goal": "Inspect workspace structure and present an overview.",
                "success_criteria": ["Return a concise list of files/directories to orient the user."],
                "constraints": [],
                "clarification_question": "",
                "final_reply": "",
                "plan": [
                    {
                        "step": "List workspace files recursively",
                        "tool": "list_workspace_files",
                        "arguments": {"pattern": "*", "recursive": True, "limit": 200},
                    }
                ],
            }

        if "what models" in low and ("provide" in low or "available" in low):
            return {
                "mode": "CHAT",
                "goal": "Explain currently available modeling capabilities.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": (
                    "Current built-in workflow supports tabular ML tasks: data prep, model suggestion, tuning, training, evaluation, and reporting. "
                    "Typical model candidates include Logistic Regression, Random Forest, and XGBoost-style baselines depending on available tooling."
                ),
                "plan": [],
            }

        if "object detection" in low:
            return {
                "mode": "CHAT",
                "goal": "Set accurate expectation for computer-vision capability.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": (
                    "Object detection is not a built-in executable pipeline here. "
                    "I can still scaffold code and a training/evaluation plan (for example with YOLO/Faster R-CNN) if you provide dataset format and runtime constraints."
                ),
                "plan": [],
            }

        if ("preprocess" in low or "data preprocessing" in low) and "?" in task_text:
            return {
                "mode": "CHAT",
                "goal": "Collect minimum inputs to run data preprocessing.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": (
                    "Yes. Share the dataset path, data format (CSV/XLSX/JSON), and target outcome, and I will run preprocessing plus a quick quality profile."
                ),
                "plan": [],
            }

        if "tool" not in low and any(token in low for token in ["word file", "docx", "pdf", "document"]) and any(
            token in low for token in ["summar", "summary", "read", "analy", "walk through"]
        ):
            return {
                "mode": "CHAT",
                "goal": "Collect readable document path before summarization.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": (
                    "Yes. Share the readable document path (.docx/.pdf/.txt/.md), and I will summarize it with key points and highlights."
                ),
                "plan": [],
            }

        if "tool" in low and ("pdf" in low or "word" in low or "docx" in low):
            return {
                "mode": "CHAT",
                "goal": "Explain document-reading tools.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": (
                    "For document reading, the core path is `read_document_file` (DOCX/PDF/TXT/MD) followed by `summarize_text`. "
                    "If you send a readable file path, I can run the flow directly."
                ),
                "plan": [],
            }

        if low in {"hi", "hello", "hey"}:
            return {
                "mode": "CHAT",
                "goal": "Acknowledge greeting and prompt for a concrete task.",
                "success_criteria": [],
                "constraints": [],
                "clarification_question": "",
                "final_reply": (
                    "Ready. Tell me the concrete task and expected output, and include a file path if you want tool execution."
                ),
                "plan": [],
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
                "I can work with knowledge lookup, document reading, CSV/XLSX data profiling, code analysis/edit/test actions, ML workflow blocks, report generation, and skill installation. "
                "Tell me the concrete task and any file/data path."
            ),
            "plan": [],
        }

    def understand_goal(self, task: str, recent_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        rule_plan = self._rule_based_preplan(task)
        if rule_plan:
            return rule_plan

        skill_descriptions = self._get_installed_skills()
        recent_experiences = self.memory.recent_experiences(limit=5)
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
            "skill_descriptions": skill_descriptions,
            "recent_experiences": recent_experiences,
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
                "Use skill_descriptions as capability hints only; do not invent unavailable tools.",
                "If skill_descriptions are relevant, reflect that in plan steps.",
                "If recent_experiences are relevant, reuse successful tool chains and avoid known failure patterns.",
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

    def _build_step_contracts(
        self,
        plan: List[Dict[str, Any]],
        goal: str,
        constraints: List[str],
        success_criteria: List[str],
    ) -> List[Dict[str, Any]]:
        contracts: List[Dict[str, Any]] = []
        for idx, step in enumerate(plan):
            tool = str(step.get("tool", "")).strip()
            step_name = str(step.get("step", f"step_{idx+1}")).strip() or f"step_{idx+1}"
            args = ensure_dict(step.get("arguments", {}))
            preconditions = ["tool must exist"] if tool else ["step can be solved by reasoning/finalization"]
            for key, value in args.items():
                if isinstance(value, str) and (value.endswith(".text") or "." in value):
                    preconditions.append(f"argument `{key}` must resolve from previous tool outputs")
            postconditions = []
            if tool:
                postconditions.extend(["tool execution succeeds", "tool output is non-empty"])
            if idx == len(plan) - 1 and success_criteria:
                postconditions.append("final answer should satisfy success criteria")
            contracts.append(
                {
                    "step_index": idx,
                    "step": step_name,
                    "tool": tool,
                    "goal": goal,
                    "constraints": [str(x) for x in constraints][:8],
                    "success_criteria": [str(x) for x in success_criteria][:8],
                    "preconditions": preconditions[:8],
                    "postconditions": postconditions[:8],
                    "max_retry": 2,
                }
            )
        return contracts

    def _constraint_agent(self, loop_state: Dict[str, Any]) -> Dict[str, Any]:
        contracts = self._build_step_contracts(
            plan=loop_state.get("plan", []),
            goal=str(loop_state.get("goal", "")),
            constraints=loop_state.get("constraints", []),
            success_criteria=loop_state.get("success_criteria", []),
        )
        return {
            "contracts": contracts,
            "global_constraints": [str(x) for x in loop_state.get("constraints", [])][:8],
        }

    def _check_contract_preconditions(self, loop_state: Dict[str, Any], decision: Dict[str, Any]) -> Tuple[bool, str]:
        cursor = int(loop_state.get("plan_cursor", 0))
        contracts = loop_state.get("step_contracts", [])
        if not isinstance(contracts, list) or cursor >= len(contracts):
            return True, "no_contract"
        contract = ensure_dict(contracts[cursor])
        tool_name = str(decision.get("tool_name", "")).strip()
        expected = str(contract.get("tool", "")).strip()
        if expected and tool_name and expected != tool_name:
            return False, f"selector chose `{tool_name}` but contract expects `{expected}`"
        args = ensure_dict(decision.get("arguments", {}))
        for key, value in args.items():
            if isinstance(value, str) and "." in value and value.split(".")[0] not in loop_state.get("tool_outputs", {}):
                return False, f"argument `{key}` points to unresolved output `{value}`"
        return True, "preconditions_ok"

    def _verifier_agent(
        self,
        loop_state: Dict[str, Any],
        decision: Dict[str, Any],
        ok: bool,
        result: Dict[str, Any],
        error: str,
    ) -> Dict[str, Any]:
        cursor = int(loop_state.get("plan_cursor", 0))
        contracts = loop_state.get("step_contracts", [])
        contract = ensure_dict(contracts[cursor]) if isinstance(contracts, list) and cursor < len(contracts) else {}
        checks: List[str] = []
        passed = bool(ok)
        if not ok:
            checks.append("tool execution failed")
        if isinstance(result, dict) and ok:
            def _has_signal(v: Any) -> bool:
                if v is None:
                    return False
                if isinstance(v, str):
                    return bool(v.strip())
                if isinstance(v, (list, dict, tuple, set)):
                    return len(v) > 0
                return True

            non_empty = any(_has_signal(result.get(k)) for k in result.keys())
            if not non_empty:
                passed = False
                checks.append("tool output empty")
        if contract.get("postconditions") and not passed:
            checks.append("postconditions not met")
        root_cause = self._classify_tool_failure(
            tool_name=str(decision.get("tool_name", "")),
            args=ensure_dict(decision.get("arguments", {})),
            error=str(error or ""),
        )
        if passed and not checks:
            checks.append("all checks passed")
        fix = ""
        if not passed:
            fix = "replan_with_recovery"
            if root_cause == "missing_input":
                fix = "request_or_discover_input_path"
            elif root_cause == "policy_block":
                fix = "request_approval_or_reduce_risk"
            elif root_cause == "timeout":
                fix = "retry_with_higher_timeout"
        return {
            "pass": passed,
            "checks": checks,
            "root_cause": root_cause,
            "fix_suggestion": fix,
            "contract_step": contract.get("step", ""),
        }

    def _selector_agent(self, loop_state: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.decide_next_action(loop_state)
        decision = str(raw.get("decision", "FINAL")).upper()
        if decision not in {"TOOL", "FINAL", "CLARIFY"}:
            decision = "FINAL"
        # Dynamic selection among reason/tool/code: when shell/code activity exists, bias to code execution.
        tool_name = str(raw.get("tool_name", "")).strip()
        if decision == "TOOL":
            if tool_name == "run_shell_command":
                raw["selector_mode"] = "CODE"
            elif tool_name:
                raw["selector_mode"] = "TOOL"
            else:
                raw["selector_mode"] = "REASON"
        else:
            raw["selector_mode"] = "REASON" if decision == "FINAL" else "CLARIFY"
        raw["decision"] = decision
        return raw

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

    def _extract_path_argument(self, args: Dict[str, Any]) -> Optional[str]:
        for key in ["path", "file_path", "source_path"]:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _classify_tool_failure(self, tool_name: str, args: Dict[str, Any], error: str) -> str:
        err = (error or "").strip().lower()
        if not err:
            return "unknown"
        if "tool not found" in err:
            return "tool_missing"
        if "blocked by policy" in err or "blocked by safety rule" in err:
            return "policy_block"
        if "file not found" in err or "no such file" in err:
            return "missing_input"
        if "timed out" in err or "timeout" in err:
            return "timeout"
        if "failed to parse" in err or "unsupported" in err or "not a zip file" in err:
            return "parse_error"
        if tool_name == "run_shell_command" and ("command not found" in err or "not found" in err):
            return "command_missing"
        if "json" in err:
            return "json_error"
        return "unknown"

    def _inject_recovery_steps(
        self,
        loop_state: Dict[str, Any],
        steps: List[Dict[str, Any]],
        reason: str,
        iteration: int,
    ) -> None:
        if not steps:
            return
        plan = list(loop_state.get("plan", []))
        cursor = int(loop_state.get("plan_cursor", 0))
        insertion_index = min(len(plan), cursor + 1)
        normalized: List[Dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool", "")).strip()
            if not tool or tool not in self.tools.tools:
                continue
            normalized.append(
                {
                    "step": str(step.get("step", "Recovery action")).strip() or "Recovery action",
                    "tool": tool,
                    "arguments": ensure_dict(step.get("arguments", {})),
                }
            )
        if not normalized:
            return
        plan[insertion_index:insertion_index] = normalized
        loop_state["plan"] = plan[:48]
        reflections = loop_state.setdefault("reflections", [])
        reflections.append(
            {
                "iteration": iteration,
                "reason": reason,
                "root_cause": reason.split(":", 1)[0] if isinstance(reason, str) else "recovery",
                "decision_quality": "adjusted",
                "fix_applied": "insert_recovery_steps",
                "outcome": "replan",
                "inserted_steps": [row.get("step", "") for row in normalized],
            }
        )

    def _build_recovery_steps(
        self,
        task: str,
        tool_name: str,
        args: Dict[str, Any],
        category: str,
        error: str,
    ) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        path = self._extract_path_argument(args)
        token = self._extract_lookup_token(task)
        path_obj: Optional[Path] = None
        if path:
            try:
                p = Path(path).expanduser()
                if not p.is_absolute():
                    p = (self.workspace / p).resolve()
                else:
                    p = p.resolve()
                path_obj = p
            except Exception:
                path_obj = None

        if category == "missing_input":
            pattern = "*"
            if path_obj is not None and path_obj.suffix:
                pattern = f"*{path_obj.suffix.lower()}"
            steps.append(
                {
                    "step": "Discover candidate files in workspace",
                    "tool": "list_workspace_files",
                    "arguments": {"pattern": pattern, "recursive": True, "limit": 300},
                }
            )
            if token:
                steps.append(
                    {
                        "step": "Search target token across workspace text",
                        "tool": "search_workspace_text",
                        "arguments": {"query": token, "top_k": 80},
                    }
                )
            return steps

        if category == "parse_error":
            if path_obj is not None and path_obj.suffix.lower() in {".xlsx", ".csv"} and tool_name != "analyze_tabular_with_python":
                steps.append(
                    {
                        "step": "Run robust Python tabular analyzer",
                        "tool": "analyze_tabular_with_python",
                        "arguments": {"path": str(path_obj), "max_rows": 300},
                    }
                )
                return steps
            if path_obj is not None and path_obj.suffix.lower() == ".json":
                if token:
                    steps.append(
                        {
                            "step": "Search token in JSON/text content",
                            "tool": "search_workspace_text",
                            "arguments": {"query": token, "top_k": 80},
                        }
                    )
                else:
                    steps.append(
                        {
                            "step": "Read raw text file for manual parse fallback",
                            "tool": "read_text_file",
                            "arguments": {"path": str(path_obj), "max_chars": 50000},
                        }
                    )
                return steps

        if category == "json_error":
            if token:
                steps.append(
                    {
                        "step": "Search target token across workspace",
                        "tool": "search_workspace_text",
                        "arguments": {"query": token, "top_k": 80},
                    }
                )
            return steps

        if category == "timeout" and tool_name == "run_shell_command":
            command = str(args.get("command", "")).strip()
            if command:
                timeout_s = int(args.get("timeout_s", 90) or 90)
                steps.append(
                    {
                        "step": "Retry shell command with larger timeout",
                        "tool": "run_shell_command",
                        "arguments": {"command": command, "timeout_s": min(300, max(120, timeout_s * 2))},
                    }
                )
            return steps

        if category == "command_missing" and tool_name == "run_shell_command":
            command = str(args.get("command", "")).strip()
            if command.startswith("pip "):
                steps.append(
                    {
                        "step": "Retry package command through python module",
                        "tool": "run_shell_command",
                        "arguments": {"command": command.replace("pip ", "python3 -m pip ", 1), "timeout_s": 120},
                    }
                )
            return steps

        if token and any(k in (task or "").lower() for k in ["find", "search", "json", "key", "id"]):
            steps.append(
                {
                    "step": "Search target token across workspace",
                    "tool": "search_workspace_text",
                    "arguments": {"query": token, "top_k": 80},
                }
            )
        return steps

    def _reflect_and_recover(
        self,
        loop_state: Dict[str, Any],
        iteration: int,
        task: str,
        tool_name: str,
        args: Dict[str, Any],
        ok: bool,
        error: str,
    ) -> Dict[str, Any]:
        if ok:
            return {"action": "continue", "reason": "ok"}

        category = self._classify_tool_failure(tool_name=tool_name, args=args, error=error)
        stats = loop_state.setdefault("recovery_stats", {})
        key = f"{tool_name}:{category}"
        attempts = int(stats.get(key, 0)) + 1
        stats[key] = attempts

        if category == "policy_block":
            return {
                "action": "clarify",
                "reason": category,
                "reply": "This action is blocked by policy. Approve the risky action or choose a lower-risk alternative.",
            }

        if attempts > 3:
            return {
                "action": "clarify",
                "reason": category,
                "reply": (
                    f"Recovery attempts exhausted for `{tool_name}` ({category}). "
                    "Please provide one additional constraint (path, expected output, or allowed tool)."
                ),
            }

        steps = self._build_recovery_steps(task=task, tool_name=tool_name, args=args, category=category, error=error)
        if steps:
            self._inject_recovery_steps(
                loop_state=loop_state,
                steps=steps,
                reason=f"{category}:{truncate_text(error, max_chars=120)}",
                iteration=iteration,
            )
            return {"action": "replan", "reason": category, "inserted": len(steps)}

        return {"action": "continue", "reason": category}

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
            last_err = None
            for row in reversed(observations):
                err = str(row.get("error", "")).strip()
                if err:
                    last_err = err
                    break
            if last_err:
                return f"Execution finished with errors. Last error: {last_err}"
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

    def run(
        self,
        task: str,
        trace_id: str,
        recent_messages: List[Dict[str, Any]],
        session_id: str = "default",
    ) -> Dict[str, Any]:
        model_loop_result = self._run_model_tool_loop(
            task=task,
            trace_id=trace_id,
            session_id=session_id,
            recent_messages=recent_messages,
        )
        if model_loop_result is not None:
            return model_loop_result

        understanding = self.understand_goal(task, recent_messages)
        start_event = self._record_event(
            trace_id,
            "understand",
            {
                "mode": understanding.get("mode", "CHAT"),
                "goal": understanding.get("goal", ""),
                "plan_steps": len(understanding.get("plan", [])),
            },
        )
        mode = understanding.get("mode", "CHAT")
        plan = understanding.get("plan", [])

        if mode != "EXECUTE":
            reply = understanding.get("final_reply") or understanding.get("clarification_question") or DEFAULT_CLARIFICATION_REPLY
            self._record_event(trace_id, "finish", {"status": "chat", "reason": "mode_not_execute"})
            return {
                "intent": "GENERAL_CHAT",
                "reply": reply,
                "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                "planner_mode": mode,
                "goal": understanding.get("goal", ""),
                "trace_id": trace_id,
                "event_log_tail": self.memory.get_workflow_events(trace_id, limit=12),
            }

        optimized_plan: List[Dict[str, Any]] = []
        if isinstance(plan, list):
            for idx, step in enumerate(plan[:16]):
                row = ensure_dict(step)
                tool = str(row.get("tool", "")).strip()
                if tool and tool not in self.tools.tools:
                    tool = ""
                optimized_plan.append(
                    {
                        "step": str(row.get("step", f"Step {idx+1}")).strip() or f"Step {idx+1}",
                        "tool": tool,
                        "arguments": ensure_dict(row.get("arguments", {})),
                    }
                )
        if not optimized_plan:
            optimized_plan = [{"step": "Search workspace context", "tool": "search_workspace_text", "arguments": {"query": task, "top_k": 20}}]

        loop_state: Dict[str, Any] = {
            "trace_id": trace_id,
            "task": task,
            "goal": understanding.get("goal", ""),
            "success_criteria": understanding.get("success_criteria", []),
            "constraints": understanding.get("constraints", []),
            "plan": optimized_plan,
            "plan_cursor": 0,
            "tool_outputs": {},
            "observations": [],
            "executed_tools": [],
            "status": "running",
            "events": [start_event],
            "reflections": [],
            "recovery_stats": {},
            "step_contracts": [],
            "verifier_notes": [],
        }
        constraint_row = self._constraint_agent(loop_state)
        loop_state["step_contracts"] = constraint_row.get("contracts", [])

        max_iterations = 12
        repeated_calls: Dict[str, int] = {}
        for iteration in range(1, max_iterations + 1):
            decision = self._selector_agent(loop_state)
            loop_state["last_decision"] = decision
            decision_event = self._record_event(
                trace_id,
                "decision",
                {
                    "iteration": iteration,
                    "decision": decision.get("decision", "UNKNOWN"),
                    "tool_name": decision.get("tool_name", ""),
                    "step": decision.get("step", ""),
                    "selector_mode": decision.get("selector_mode", ""),
                },
            )
            loop_state["events"].append(decision_event)

            if decision.get("decision") == "FINAL":
                loop_state["status"] = "completed"
                answer = decision.get("final_answer") or self._compose_final_answer(loop_state)
                self._record_event(trace_id, "finish", {"status": "completed", "iteration": iteration})
                response = {
                    "intent": "DYNAMIC_EXECUTION",
                    "status": "completed",
                    "goal": loop_state["goal"],
                    "plan": [row.get("step", "") for row in loop_state.get("plan", [])],
                    "executed_tools": loop_state["executed_tools"],
                    "observations": loop_state["observations"][-8:],
                    "reflections": loop_state.get("reflections", [])[-8:],
                    "verifier_notes": loop_state.get("verifier_notes", [])[-8:],
                    "result_summary": answer,
                    "trace_id": trace_id,
                    "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                }
                exp = self._capture_experience(
                    session_id=session_id,
                    trace_id=trace_id,
                    task=task,
                    response=response,
                    loop_state=loop_state,
                )
                if exp:
                    response["experience_id"] = exp.get("experience_id")
                    response["skill_candidate"] = exp.get("skill_candidate", {})
                return response

            if decision.get("decision") == "CLARIFY":
                loop_state["status"] = "clarification_needed"
                question = decision.get("clarification_question") or "Please share one missing constraint so I can continue."
                self._record_event(trace_id, "finish", {"status": "clarify", "iteration": iteration})
                response = {
                    "intent": "GENERAL_CHAT",
                    "reply": question,
                    "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                    "planner_mode": "CLARIFY",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                    "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                }
                self._capture_experience(
                    session_id=session_id,
                    trace_id=trace_id,
                    task=task,
                    response=response,
                    loop_state=loop_state,
                )
                return response

            if decision.get("decision") != "TOOL":
                loop_state["status"] = "completed"
                self._record_event(trace_id, "finish", {"status": "unknown_decision", "iteration": iteration})
                response = {
                    "intent": "GENERAL_CHAT",
                    "reply": DEFAULT_CLARIFICATION_REPLY,
                    "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                    "planner_mode": "UNKNOWN",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                    "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                }
                self._capture_experience(
                    session_id=session_id,
                    trace_id=trace_id,
                    task=task,
                    response=response,
                    loop_state=loop_state,
                )
                return response

            tool_name = str(decision.get("tool_name", "")).strip()
            raw_args = ensure_dict(decision.get("arguments", {}))
            args = self._resolve_arguments(raw_args, task=task, tool_outputs=loop_state["tool_outputs"])
            decision["arguments"] = args
            pre_ok, pre_reason = self._check_contract_preconditions(loop_state=loop_state, decision=decision)
            if not pre_ok:
                loop_state["reflections"].append(
                    {
                        "iteration": iteration,
                        "reason": pre_reason,
                        "root_cause": "contract_precondition_failed",
                        "decision_quality": "invalid",
                        "fix_applied": "clarify_or_replan",
                        "outcome": "blocked_before_tool",
                    }
                )
                recovery = self._reflect_and_recover(
                    loop_state=loop_state,
                    iteration=iteration,
                    task=task,
                    tool_name=tool_name or "unknown_tool",
                    args=args,
                    ok=False,
                    error=pre_reason,
                )
                if recovery.get("action") == "replan":
                    loop_state["plan_cursor"] = int(loop_state.get("plan_cursor", 0)) + 1
                    continue
                loop_state["status"] = "clarification_needed"
                self._record_event(trace_id, "finish", {"status": "precondition_block", "iteration": iteration})
                response = {
                    "intent": "GENERAL_CHAT",
                    "reply": f"Step precondition failed: {pre_reason}. Please provide a missing constraint.",
                    "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                    "planner_mode": "PRECONDITION_BLOCK",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                    "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                }
                self._capture_experience(
                    session_id=session_id,
                    trace_id=trace_id,
                    task=task,
                    response=response,
                    loop_state=loop_state,
                )
                return response
            spec = self.tools.get_spec(tool_name)
            if spec is None:
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
                loop_state["events"].append(
                    self._record_event(
                        trace_id,
                        "observation",
                        {"iteration": iteration, "tool": tool_name, "ok": False, "error": obs["error"]},
                    )
                )
                continue

            policy_result = self.policy.evaluate(tool_name=tool_name, spec=spec, args=args, task=task)
            if not policy_result.get("allow", False):
                obs = {
                    "iteration": iteration,
                    "step": decision.get("step", "Run tool"),
                    "tool": tool_name,
                    "arguments": args,
                    "ok": False,
                    "error": policy_result.get("reason", "blocked by policy"),
                    "risk": policy_result.get("risk", "unknown"),
                    "result": {"ok": False, "error": policy_result.get("reason", "blocked by policy")},
                }
                loop_state["observations"].append(obs)
                loop_state["events"].append(
                    self._record_event(
                        trace_id,
                        "policy_block",
                        {
                            "iteration": iteration,
                            "tool": tool_name,
                            "risk": policy_result.get("risk", "unknown"),
                            "reason": policy_result.get("reason", ""),
                        },
                    )
                )
                response = {
                    "intent": "GENERAL_CHAT",
                    "reply": policy_result.get("reason", "Blocked by policy."),
                    "suggestions": [
                        "Provide explicit approval if you want to run this high-risk action.",
                        "Or ask me to continue with a lower-risk plan.",
                    ],
                    "planner_mode": "POLICY_BLOCK",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                    "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                }
                self._capture_experience(
                    session_id=session_id,
                    trace_id=trace_id,
                    task=task,
                    response=response,
                    loop_state=loop_state,
                )
                return response

            call_sig = f"{tool_name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
            repeated_calls[call_sig] = repeated_calls.get(call_sig, 0) + 1
            if repeated_calls[call_sig] > 2:
                last_error = ""
                for row in reversed(loop_state.get("observations", [])):
                    if str(row.get("tool", "")) == tool_name and str(row.get("error", "")).strip():
                        last_error = str(row.get("error", "")).strip()
                        break
                recovery = self._reflect_and_recover(
                    loop_state=loop_state,
                    iteration=iteration,
                    task=task,
                    tool_name=tool_name,
                    args=args,
                    ok=False,
                    error=last_error or "repeated_failing_call",
                )
                loop_state["events"].append(
                    self._record_event(
                        trace_id,
                        "reflect",
                        {
                            "iteration": iteration,
                            "tool": tool_name,
                            "reason": recovery.get("reason", "repeat_guard"),
                            "action": recovery.get("action", "continue"),
                        },
                    )
                )
                if recovery.get("action") == "replan":
                    loop_state["plan_cursor"] = int(loop_state.get("plan_cursor", 0)) + 1
                    continue
                loop_state["status"] = "clarification_needed"
                self._record_event(trace_id, "finish", {"status": "repeat_guard", "iteration": iteration, "tool": tool_name})
                reply = str(recovery.get("reply", "")).strip() or (
                    "I am repeating the same failing action. Please provide one more constraint so I can recover."
                )
                response = {
                    "intent": "GENERAL_CHAT",
                    "reply": reply,
                    "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                    "planner_mode": "REPEAT_GUARD",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                    "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                }
                self._capture_experience(
                    session_id=session_id,
                    trace_id=trace_id,
                    task=task,
                    response=response,
                    loop_state=loop_state,
                )
                return response

            loop_state["events"].append(
                self._record_event(
                    trace_id,
                    "tool_call",
                    {
                        "iteration": iteration,
                        "tool": tool_name,
                        "risk": policy_result.get("risk", "unknown"),
                        "arguments_preview": to_json_text(args, max_chars=350),
                    },
                )
            )
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
                    "risk": policy_result.get("risk", "unknown"),
                    "result": compact_result,
                    "result_preview": to_json_text(compact_result, max_chars=900),
                }
            )
            loop_state["events"].append(
                self._record_event(
                    trace_id,
                    "observation",
                    {
                        "iteration": iteration,
                        "tool": tool_name,
                        "ok": ok,
                        "error": error,
                        "result_preview": to_json_text(compact_result, max_chars=350),
                    },
                )
            )
            if self.enable_verifier:
                verifier = self._verifier_agent(
                    loop_state=loop_state,
                    decision=decision,
                    ok=ok,
                    result=result if isinstance(result, dict) else {"value": result},
                    error=str(error or ""),
                )
                loop_state.setdefault("verifier_notes", []).append(
                    {
                        "iteration": iteration,
                        "tool": tool_name,
                        "pass": verifier.get("pass", False),
                        "root_cause": verifier.get("root_cause", ""),
                        "checks": verifier.get("checks", []),
                        "fix_suggestion": verifier.get("fix_suggestion", ""),
                    }
                )
                loop_state["events"].append(
                    self._record_event(
                        trace_id,
                        "verify",
                        {
                            "iteration": iteration,
                            "tool": tool_name,
                            "pass": verifier.get("pass", False),
                            "root_cause": verifier.get("root_cause", ""),
                            "fix_suggestion": verifier.get("fix_suggestion", ""),
                        },
                    )
                )
                if not verifier.get("pass", False):
                    loop_state["reflections"].append(
                        {
                            "iteration": iteration,
                            "reason": "verifier_failed",
                            "root_cause": verifier.get("root_cause", ""),
                            "decision_quality": "weak",
                            "fix_applied": verifier.get("fix_suggestion", ""),
                            "outcome": "needs_recovery",
                        }
                    )
                verifier_pass = bool(verifier.get("pass", False))
                verifier_root = str(verifier.get("root_cause", ""))
            else:
                verifier_pass = bool(ok)
                verifier_root = ""
                loop_state.setdefault("verifier_notes", []).append(
                    {
                        "iteration": iteration,
                        "tool": tool_name,
                        "pass": verifier_pass,
                        "root_cause": "",
                        "checks": ["verifier disabled"],
                        "fix_suggestion": "",
                    }
                )
                loop_state["events"].append(
                    self._record_event(
                        trace_id,
                        "verify",
                        {
                            "iteration": iteration,
                            "tool": tool_name,
                            "pass": verifier_pass,
                            "root_cause": "",
                            "fix_suggestion": "verifier_disabled",
                        },
                    )
                )
            recovery = self._reflect_and_recover(
                loop_state=loop_state,
                iteration=iteration,
                task=task,
                tool_name=tool_name,
                args=args,
                ok=bool(ok and verifier_pass),
                error=str(error or verifier_root),
            )
            loop_state["events"].append(
                self._record_event(
                    trace_id,
                    "reflect",
                    {
                        "iteration": iteration,
                        "tool": tool_name,
                        "ok": ok,
                        "reason": recovery.get("reason", ""),
                        "action": recovery.get("action", "continue"),
                    },
                )
            )
            if recovery.get("action") == "clarify":
                loop_state["status"] = "clarification_needed"
                self._record_event(trace_id, "finish", {"status": "clarify_after_reflect", "iteration": iteration})
                response = {
                    "intent": "GENERAL_CHAT",
                    "reply": str(recovery.get("reply", "")).strip()
                    or "Please provide one additional constraint so I can continue.",
                    "suggestions": DEFAULT_CLARIFICATION_SUGGESTIONS,
                    "planner_mode": "REFLECT_CLARIFY",
                    "goal": loop_state["goal"],
                    "trace_id": trace_id,
                    "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
                }
                self._capture_experience(
                    session_id=session_id,
                    trace_id=trace_id,
                    task=task,
                    response=response,
                    loop_state=loop_state,
                )
                return response

            loop_state["plan_cursor"] = int(loop_state.get("plan_cursor", 0)) + 1

        loop_state["status"] = "max_iterations"
        self._record_event(trace_id, "finish", {"status": "max_iterations"})
        response = {
            "intent": "DYNAMIC_EXECUTION",
            "status": "max_iterations_reached",
            "goal": loop_state["goal"],
            "plan": [row.get("step", "") for row in loop_state.get("plan", [])],
            "executed_tools": loop_state["executed_tools"],
            "observations": loop_state["observations"][-8:],
            "reflections": loop_state.get("reflections", [])[-8:],
            "verifier_notes": loop_state.get("verifier_notes", [])[-8:],
            "result_summary": self._compose_final_answer(loop_state),
            "trace_id": trace_id,
            "event_log_tail": self.memory.get_workflow_events(trace_id, limit=20),
        }
        exp = self._capture_experience(
            session_id=session_id,
            trace_id=trace_id,
            task=task,
            response=response,
            loop_state=loop_state,
        )
        if exp:
            response["experience_id"] = exp.get("experience_id")
            response["skill_candidate"] = exp.get("skill_candidate", {})
        return response


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
        code_path = understanding.get("code_path")
        command = understanding.get("command")
        find_text = understanding.get("find_text")
        replace_text = understanding.get("replace_text")
        replace_count = understanding.get("count", 1)

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
        elif intent == "CODE_TASK" and needs_knowledge:
            receiver = "kb_retriever"
            route_mode = "enrich_workflow"
            return_to = "planner"
        elif intent == "CODE_TASK":
            receiver = "planner"
            route_mode = "direct_code"
            return_to = "planner"
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
        state["code_path"] = code_path
        state["command"] = command
        state["find_text"] = find_text
        state["replace_text"] = replace_text
        state["count"] = replace_count
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
                "code_path": code_path,
                "command": command,
                "find_text": find_text,
                "replace_text": replace_text,
                "count": replace_count,
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
        file_path = message["content"].get("file_path")
        code_path = message["content"].get("code_path")
        command = message["content"].get("command")
        find_text = message["content"].get("find_text")
        replace_text = message["content"].get("replace_text")
        replace_count = message["content"].get("count", 1)
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
                    "file_path": file_path,
                    "code_path": code_path,
                    "command": command,
                    "find_text": find_text,
                    "replace_text": replace_text,
                    "count": replace_count,
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
        intent = str(message["content"].get("intent", state.get("intent", "ML_WORKFLOW"))).upper()
        knowledge_context = message["content"].get("knowledge_context", state.get("knowledge_context", []))
        requirements: Dict[str, Any] = {}
        code_path = message["content"].get("code_path", state.get("code_path"))
        command = message["content"].get("command", state.get("command"))
        find_text = message["content"].get("find_text", state.get("find_text"))
        replace_text = message["content"].get("replace_text", state.get("replace_text"))
        replace_count = message["content"].get("count", state.get("count", 1))
        if intent == "CODE_TASK":
            steps: List[Dict[str, Any]] = []
            if code_path:
                steps.append(
                    {
                        "kind": "code_read",
                        "agent": "code_agent",
                        "name": "Read target code file",
                        "code_path": code_path,
                    }
                )
            if code_path and isinstance(find_text, str) and isinstance(replace_text, str):
                steps.append(
                    {
                        "kind": "code_patch",
                        "agent": "code_agent",
                        "name": "Apply deterministic code patch",
                        "code_path": code_path,
                        "find_text": find_text,
                        "replace_text": replace_text,
                        "count": replace_count,
                    }
                )
            if command:
                steps.append(
                    {
                        "kind": "code_execute",
                        "agent": "code_agent",
                        "name": "Run workspace command",
                        "command": command,
                    }
                )
            if not steps:
                steps.append(
                    {
                        "kind": "code_task",
                        "agent": "code_agent",
                        "name": "Inspect codebase for requested task",
                    }
                )
            state["requirements"] = {}
        else:
            incoming_requirements = message["content"].get("requirements")
            requirements = normalize_requirements(incoming_requirements, task)
            steps = build_dynamic_workflow_steps(requirements)
            state["requirements"] = requirements

        state["workflow_intent"] = intent
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
                "intent": intent,
                "task": task,
                "plan": state["plan"],
                "steps": steps,
                "requirements": requirements,
                "knowledge_context": knowledge_context,
                "code_path": code_path,
                "command": command,
                "find_text": find_text,
                "replace_text": replace_text,
                "count": replace_count,
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
            state["workflow_intent"] = message["content"].get("intent", state.get("intent", "ML_WORKFLOW"))
            state["workflow_task"] = message["content"]["task"]
            state["code_path"] = message["content"].get("code_path", state.get("code_path"))
            state["command"] = message["content"].get("command", state.get("command"))
            state["find_text"] = message["content"].get("find_text", state.get("find_text"))
            state["replace_text"] = message["content"].get("replace_text", state.get("replace_text"))
            state["count"] = message["content"].get("count", state.get("count", 1))
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
        workflow_intent = str(state.get("workflow_intent", state.get("intent", "ML_WORKFLOW"))).upper()

        if cursor >= len(steps):
            artifacts = state.get("artifacts", {})
            if workflow_intent == "CODE_TASK":
                command_result = artifacts.get("command_result", {})
                code_context = artifacts.get("code_context", {})
                code_search = artifacts.get("code_search", {})
                patch_result = artifacts.get("patch_result", {})
                summary_parts = ["Code task completed."]
                if isinstance(command_result, dict):
                    rc = command_result.get("return_code")
                    if rc is not None:
                        summary_parts.append(f"Command return code: {rc}.")
                if isinstance(patch_result, dict) and patch_result.get("ok"):
                    summary_parts.append(f"Applied replacements: {patch_result.get('replacements', 0)}.")
                if isinstance(code_context, dict) and code_context.get("path"):
                    summary_parts.append(f"Code file analyzed: {code_context.get('path')}.")
                if isinstance(code_search, dict) and code_search.get("count"):
                    summary_parts.append(f"Code search matches: {code_search.get('count')}.")
                return {
                    "sender": self.name,
                    "receiver": "user",
                    "type": "final_result",
                    "priority": 80,
                    "content": {
                        "intent": "CODE_TASK",
                        "summary": " ".join(summary_parts),
                        "executed_steps": state.get("executed_steps", []),
                        "code_path": state.get("code_path"),
                        "command": state.get("command"),
                        "find_text": state.get("find_text"),
                        "replace_text": state.get("replace_text"),
                        "count": state.get("count", 1),
                        "command_result": command_result,
                        "patch_result": patch_result,
                        "code_context": code_context,
                        "code_search": code_search,
                    },
                    "metadata": {"trace_id": state["trace_id"]},
                }

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
                    "intent": workflow_intent if workflow_intent else "ML_WORKFLOW",
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
                "intent": workflow_intent,
                "step": next_step,
                "requirements": state.get("requirements", {}),
                "knowledge_context": state.get("knowledge_context", []),
                "code_path": state.get("code_path"),
                "command": state.get("command"),
                "find_text": state.get("find_text"),
                "replace_text": state.get("replace_text"),
                "count": state.get("count", 1),
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


class CodeAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__("code_agent")
        self.tools = tools

    def handle(self, message: Message, state: State) -> Message:
        step = message["content"].get("step", {"kind": "code_task"})
        step_kind = str(step.get("kind", "code_task"))
        task = message["content"]["task"]
        artifacts = message["content"].get("artifacts", {})
        code_path = step.get("code_path") or message["content"].get("code_path")
        command = step.get("command") or message["content"].get("command")
        updates: Dict[str, Any] = {}

        if step_kind == "code_read":
            if code_path:
                updates["code_context"] = self.tools.execute("read_code_file", path=code_path, max_chars=50000)
            else:
                updates["code_search"] = self.tools.execute("search_workspace_text", query=task, top_k=20)
        elif step_kind == "code_execute":
            if command:
                timeout_s = int(step.get("timeout_s", 90) or 90)
                updates["command_result"] = self.tools.execute(
                    "run_shell_command",
                    command=command,
                    timeout_s=timeout_s,
                )
            else:
                updates["command_result"] = {"ok": False, "error": "command is required for code_execute"}
        else:
            # Generic code task: optionally read code, apply deterministic patch, then run command.
            if code_path:
                updates["code_context"] = self.tools.execute("read_code_file", path=code_path, max_chars=50000)

            find_text = step.get("find_text")
            replace_text = step.get("replace_text")
            if code_path and isinstance(find_text, str) and isinstance(replace_text, str):
                updates["patch_result"] = self.tools.execute(
                    "replace_text_in_file",
                    path=code_path,
                    find_text=find_text,
                    replace_text=replace_text,
                    count=int(step.get("count", 1) or 1),
                )

            if command:
                updates["command_result"] = self.tools.execute(
                    "run_shell_command",
                    command=command,
                    timeout_s=int(step.get("timeout_s", 90) or 90),
                )

            if not updates:
                updates["code_search"] = self.tools.execute("search_workspace_text", query=task, top_k=20)

        if "command_result" in updates:
            state["command_result"] = updates["command_result"]

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
    workspace_path = Path(workspace).expanduser().resolve()
    skill_store = SkillStore(root=str(workspace_path / "skills"))

    def resolve_workspace_file(path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = workspace_path / p
        return p.resolve()

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

    def network_http_request(
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, Any]] = None,
        body: Optional[str] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout_s: int = 20,
        approved: bool = False,
    ) -> Dict[str, Any]:
        if not url or not str(url).strip():
            return {"ok": False, "error": "url is required"}
        method_u = str(method or "GET").upper().strip()
        if method_u not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            return {"ok": False, "error": f"unsupported method: {method_u}"}

        req_headers: Dict[str, str] = {}
        if isinstance(headers, dict):
            for k, v in headers.items():
                req_headers[str(k)] = str(v)

        data_bytes: Optional[bytes] = None
        if isinstance(json_body, dict):
            payload = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            data_bytes = payload
            req_headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data_bytes = body.encode("utf-8")

        req = urllib.request.Request(url=str(url).strip(), method=method_u, data=data_bytes, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=max(2, min(int(timeout_s or 20), 120))) as resp:
                raw = resp.read(2_000_000)
                content_type = str(resp.headers.get("Content-Type", ""))
                text = raw.decode("utf-8", errors="ignore")
                out_headers = {}
                for key in ["Content-Type", "Content-Length", "Date", "Server"]:
                    val = resp.headers.get(key)
                    if val is not None:
                        out_headers[key] = str(val)
                return {
                    "ok": True,
                    "url": str(url).strip(),
                    "method": method_u,
                    "status_code": int(getattr(resp, "status", 200)),
                    "content_type": content_type,
                    "headers": out_headers,
                    "body_text": truncate_text(text, max_chars=12000),
                    "body_chars": len(text),
                    "approved": bool(approved),
                }
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body_text = ""
            return {
                "ok": False,
                "url": str(url).strip(),
                "method": method_u,
                "status_code": int(e.code),
                "error": f"http error: {e}",
                "body_text": truncate_text(body_text, max_chars=6000),
                "approved": bool(approved),
            }
        except Exception as e:
            return {"ok": False, "url": str(url).strip(), "method": method_u, "error": f"request failed: {e}", "approved": bool(approved)}

    def network_download_file(
        url: str,
        output_path: str,
        timeout_s: int = 40,
        overwrite: bool = True,
        approved: bool = False,
    ) -> Dict[str, Any]:
        if not url or not str(url).strip():
            return {"ok": False, "error": "url is required"}
        if not output_path or not str(output_path).strip():
            return {"ok": False, "error": "output_path is required"}
        out = resolve_workspace_file(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not overwrite:
            return {"ok": False, "error": f"output file exists: {out}"}

        req = urllib.request.Request(url=str(url).strip(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=max(2, min(int(timeout_s or 40), 180))) as resp:
                data = resp.read(25_000_000)
                out.write_bytes(data)
                return {
                    "ok": True,
                    "url": str(url).strip(),
                    "output_path": str(out),
                    "bytes": len(data),
                    "status_code": int(getattr(resp, "status", 200)),
                    "approved": bool(approved),
                }
        except Exception as e:
            return {"ok": False, "url": str(url).strip(), "output_path": str(out), "error": f"download failed: {e}", "approved": bool(approved)}

    def _parse_sql_params(params: Any) -> Tuple[Any, Optional[str]]:
        if params is None:
            return (), None
        if isinstance(params, (list, tuple)):
            return tuple(params), None
        if isinstance(params, dict):
            return dict(params), None
        if isinstance(params, str):
            text = params.strip()
            if not text:
                return (), None
            try:
                parsed = json.loads(text)
            except Exception as e:
                return (), f"failed to parse params json: {e}"
            if isinstance(parsed, list):
                return tuple(parsed), None
            if isinstance(parsed, dict):
                return dict(parsed), None
            return (), "params json must be list or dict"
        return (), "unsupported params type"

    def sqlite_query(
        db_path: str,
        query: str,
        params: Any = None,
        limit: int = 200,
        approved: bool = False,
    ) -> Dict[str, Any]:
        import sqlite3

        if not db_path:
            return {"ok": False, "error": "db_path is required"}
        if not query or not str(query).strip():
            return {"ok": False, "error": "query is required"}
        query_text = str(query).strip()
        if not query_text.lower().startswith("select"):
            return {"ok": False, "error": "sqlite_query only supports SELECT statements"}

        p = Path(db_path).expanduser()
        if not p.is_absolute():
            p = workspace_path / p
        p = p.resolve()
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"db file not found: {p}"}
        sql_params, err = _parse_sql_params(params)
        if err:
            return {"ok": False, "error": err}
        q = query_text.rstrip(" ;")
        q = f"{q} LIMIT {max(1, min(int(limit or 200), 5000))}"
        try:
            conn = sqlite3.connect(str(p))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(q, sql_params)
            rows = cur.fetchall()
            columns = [str(x[0]) for x in (cur.description or [])]
            data_rows = []
            for row in rows:
                item = {}
                for col in columns:
                    val = row[col]
                    if isinstance(val, (bytes, bytearray)):
                        item[col] = f"<bytes:{len(val)}>"
                    else:
                        item[col] = val
                data_rows.append(item)
            conn.close()
            return {
                "ok": True,
                "db_path": str(p),
                "query": q,
                "row_count": len(data_rows),
                "columns": columns,
                "rows": data_rows,
                "approved": bool(approved),
            }
        except Exception as e:
            return {"ok": False, "db_path": str(p), "error": f"sqlite query failed: {e}", "approved": bool(approved)}

    def sqlite_execute(
        db_path: str,
        statement: str,
        params: Any = None,
        approved: bool = False,
    ) -> Dict[str, Any]:
        import sqlite3

        if not db_path:
            return {"ok": False, "error": "db_path is required"}
        if not statement or not str(statement).strip():
            return {"ok": False, "error": "statement is required"}
        stmt = str(statement).strip()
        if stmt.lower().startswith("select"):
            return {"ok": False, "error": "use sqlite_query for SELECT statements"}

        p = Path(db_path).expanduser()
        if not p.is_absolute():
            p = workspace_path / p
        p = p.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        sql_params, err = _parse_sql_params(params)
        if err:
            return {"ok": False, "error": err}
        try:
            conn = sqlite3.connect(str(p))
            cur = conn.cursor()
            cur.execute(stmt, sql_params)
            conn.commit()
            rowcount = int(cur.rowcount if cur.rowcount is not None else -1)
            lastrowid = int(cur.lastrowid if cur.lastrowid is not None else 0)
            conn.close()
            return {
                "ok": True,
                "db_path": str(p),
                "statement": stmt,
                "rowcount": rowcount,
                "lastrowid": lastrowid,
                "approved": bool(approved),
            }
        except Exception as e:
            return {"ok": False, "db_path": str(p), "error": f"sqlite execute failed: {e}", "approved": bool(approved)}

    browser_state: Dict[str, Any] = {
        "url": "",
        "title": "",
        "html_preview": "",
        "text_preview": "",
        "links": [],
        "updated_at_ms": 0,
    }

    def _extract_links_from_html(html: str, base_url: str) -> List[Dict[str, str]]:
        links: List[Dict[str, str]] = []
        for m in re.finditer(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", html, flags=re.IGNORECASE | re.DOTALL):
            href = (m.group(1) or "").strip()
            txt = re.sub(r"<[^>]+>", " ", m.group(2) or "")
            txt = re.sub(r"\s+", " ", txt).strip()
            if not href:
                continue
            abs_url = urllib.parse.urljoin(base_url, href)
            links.append({"text": txt[:160], "href": abs_url})
            if len(links) >= 80:
                break
        return links

    def browser_open_page(url: str, timeout_s: int = 20, approved: bool = False) -> Dict[str, Any]:
        row = network_http_request(
            url=url,
            method="GET",
            headers={"User-Agent": "multiagents-browser/1.0"},
            timeout_s=timeout_s,
            approved=approved,
        )
        if not row.get("ok"):
            return row
        html = str(row.get("body_text", ""))
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        links = _extract_links_from_html(html, base_url=str(row.get("url", url)))
        browser_state.update(
            {
                "url": str(row.get("url", url)),
                "title": title,
                "html_preview": truncate_text(html, max_chars=7000),
                "text_preview": truncate_text(text, max_chars=3500),
                "links": links[:40],
                "updated_at_ms": now_ms(),
            }
        )
        return {
            "ok": True,
            "url": browser_state["url"],
            "title": browser_state["title"],
            "text_preview": browser_state["text_preview"],
            "links": browser_state["links"][:20],
            "link_count": len(browser_state["links"]),
            "approved": bool(approved),
        }

    def browser_click_link(link_text: str = "", link_index: int = 1, approved: bool = False) -> Dict[str, Any]:
        links = browser_state.get("links", [])
        if not isinstance(links, list) or not links:
            return {"ok": False, "error": "browser state has no links; run browser_open_page first"}
        picked = None
        if isinstance(link_text, str) and link_text.strip():
            q = link_text.strip().lower()
            for link in links:
                txt = str(link.get("text", "")).lower()
                href = str(link.get("href", "")).lower()
                if q in txt or q in href:
                    picked = link
                    break
        if picked is None:
            idx = max(1, int(link_index or 1))
            idx = min(idx, len(links))
            picked = links[idx - 1]
        href = str(picked.get("href", "")).strip()
        if not href:
            return {"ok": False, "error": "selected link has empty href"}
        row = browser_open_page(url=href, timeout_s=20, approved=approved)
        if not row.get("ok"):
            return row
        row["clicked_link"] = picked
        return row

    def browser_get_state() -> Dict[str, Any]:
        return {
            "ok": True,
            "state": {
                "url": browser_state.get("url", ""),
                "title": browser_state.get("title", ""),
                "text_preview": browser_state.get("text_preview", ""),
                "links": browser_state.get("links", [])[:20],
                "updated_at_ms": browser_state.get("updated_at_ms", 0),
            },
        }

    def browser_find_text(pattern: str, max_matches: int = 20) -> Dict[str, Any]:
        if not pattern or not str(pattern).strip():
            return {"ok": False, "error": "pattern is required"}
        text = str(browser_state.get("text_preview", ""))
        if not text:
            return {"ok": False, "error": "browser state is empty; run browser_open_page first"}
        q = str(pattern).strip().lower()
        matches = []
        low = text.lower()
        start = 0
        while len(matches) < max(1, min(int(max_matches or 20), 200)):
            idx = low.find(q, start)
            if idx < 0:
                break
            left = max(0, idx - 80)
            right = min(len(text), idx + len(q) + 80)
            matches.append({"offset": idx, "snippet": text[left:right]})
            start = idx + len(q)
        return {"ok": True, "pattern": pattern, "count": len(matches), "matches": matches}

    def observe_browser_state() -> Dict[str, Any]:
        return browser_get_state()

    def observe_git_diff(pathspec: str = "", max_chars: int = 20000) -> Dict[str, Any]:
        cmd = ["git", "-C", str(workspace_path), "diff"]
        if isinstance(pathspec, str) and pathspec.strip():
            cmd.extend(["--", pathspec.strip()])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip(), "command": " ".join(cmd)}
            names = subprocess.run(
                ["git", "-C", str(workspace_path), "diff", "--name-only"],
                capture_output=True,
                text=True,
            )
            files = [ln.strip() for ln in (names.stdout or "").splitlines() if ln.strip()]
            return {
                "ok": True,
                "pathspec": pathspec,
                "changed_files": files,
                "diff_text": truncate_text(proc.stdout or "", max_chars=max(1000, min(max_chars, 120000))),
            }
        except Exception as e:
            return {"ok": False, "error": f"git diff observation failed: {e}"}

    def observe_error_logs(pattern: str = "error|exception|traceback", top_k: int = 80) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                ["rg", "-n", "--no-heading", "-i", "-g", "*.log", pattern, str(workspace_path)],
                capture_output=True,
                text=True,
            )
            lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            rows = []
            for line in lines[: max(1, min(int(top_k or 80), 500))]:
                parts = line.split(":", 2)
                if len(parts) == 3:
                    rows.append({"path": parts[0], "line": parts[1], "text": parts[2]})
                else:
                    rows.append({"raw": line})
            return {"ok": True, "pattern": pattern, "count": len(rows), "matches": rows}
        except Exception as e:
            return {"ok": False, "error": f"log observation failed: {e}", "pattern": pattern}

    def observe_recent_events(trace_id: str = "", limit: int = 30) -> Dict[str, Any]:
        if isinstance(trace_id, str) and trace_id.strip():
            rows = memory.get_workflow_events(trace_id.strip(), limit=max(1, min(int(limit or 30), 200)))
            return {"ok": True, "trace_id": trace_id.strip(), "events": rows}
        events = []
        workflow_rows = memory._data.get("workflow", {})
        for key, value in workflow_rows.items():
            if not isinstance(key, str) or not key.startswith("events::"):
                continue
            tr = key.split("events::", 1)[-1]
            if isinstance(value, dict):
                evs = value.get("events", [])
                if isinstance(evs, list) and evs:
                    events.append({"trace_id": tr, "latest_event": evs[-1], "event_count": len(evs)})
        events.sort(key=lambda x: x.get("latest_event", {}).get("ts_ms", 0), reverse=True)
        return {"ok": True, "traces": events[: max(1, min(int(limit or 30), 200))]}

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

    def knowledge_add_reference(
        path: str,
        title: Optional[str] = None,
        category: str = "reference",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not path or not str(path).strip():
            return {"ok": False, "error": "path is required"}
        p = resolve_workspace_file(path)
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"file not found: {p}"}
        suffix = p.suffix.lower()
        text = ""
        if suffix in {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".csv", ".sql"}:
            row = read_text_file(path=str(p), max_chars=400000)
            if not row.get("ok"):
                return row
            text = str(row.get("text", ""))
        elif suffix in {".pdf", ".docx"}:
            row = read_document_file(path=str(p))
            if not row.get("ok"):
                return row
            text = str(row.get("text", ""))
        else:
            return {"ok": False, "error": f"unsupported reference file type: {suffix}"}
        doc_id = f"doc_ref_{now_ms()}"
        safe_tags = []
        if isinstance(tags, list):
            safe_tags = [str(x).strip() for x in tags if str(x).strip()][:20]
        metadata = {"category": str(category or "reference").strip() or "reference", "tags": safe_tags}
        memory.put_knowledge_doc(
            doc_id=doc_id,
            title=title or p.name,
            text=text,
            source=str(p),
            metadata=metadata,
        )
        return {"ok": True, "doc_id": doc_id, "source": str(p), "chars": len(text), "metadata": metadata}

    def knowledge_ingest_workspace_docs(
        pattern: str = "*.md",
        recursive: bool = True,
        limit: int = 200,
        category: str = "reference",
    ) -> Dict[str, Any]:
        files = list_workspace_files(pattern=pattern, recursive=recursive, limit=limit).get("files", [])
        if not isinstance(files, list):
            return {"ok": False, "error": "failed to list files"}
        ingested = []
        errors = []
        for file_path in files[: max(1, min(int(limit or 200), 1000))]:
            row = knowledge_add_reference(path=str(file_path), title=None, category=category, tags=[])
            if row.get("ok"):
                ingested.append({"doc_id": row.get("doc_id"), "source": row.get("source"), "chars": row.get("chars", 0)})
            else:
                errors.append({"source": str(file_path), "error": row.get("error", "ingest failed")})
        return {"ok": True, "count": len(ingested), "ingested": ingested[:200], "errors": errors[:50]}

    def knowledge_list_sources(limit: int = 100) -> Dict[str, Any]:
        return {"ok": True, "sources": memory.list_knowledge_sources(limit=max(1, min(int(limit or 100), 1000)))}

    def knowledge_get_doc(doc_id: str, max_chars: int = 30000) -> Dict[str, Any]:
        if not doc_id or not str(doc_id).strip():
            return {"ok": False, "error": "doc_id is required"}
        row = memory.get_knowledge_doc(str(doc_id).strip())
        if not row:
            return {"ok": False, "error": f"doc not found: {doc_id}"}
        text = str(row.get("text", ""))
        return {
            "ok": True,
            "doc_id": row.get("doc_id"),
            "title": row.get("title"),
            "source": row.get("source"),
            "metadata": row.get("metadata", {}),
            "text": truncate_text(text, max_chars=max(1000, min(int(max_chars or 30000), 400000))),
        }

    def read_csv_preview(path: str, max_rows: int = 20) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        p = resolve_workspace_file(path)
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

    def _xlsx_col_idx(cell_ref: str, fallback_idx: int) -> int:
        m = re.match(r"([A-Z]+)", (cell_ref or "").upper())
        if not m:
            return fallback_idx
        letters = m.group(1)
        value = 0
        for ch in letters:
            value = value * 26 + (ord(ch) - ord("A") + 1)
        return max(0, value - 1)

    def _xlsx_shared_strings(zf: zipfile.ZipFile) -> List[str]:
        if "xl/sharedStrings.xml" not in zf.namelist():
            return []
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        try:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        except Exception:
            return []
        out: List[str] = []
        for si in root.findall("m:si", ns):
            texts = []
            for t in si.findall(".//m:t", ns):
                texts.append(t.text or "")
            out.append("".join(texts))
        return out

    def _xlsx_sheet_target(zf: zipfile.ZipFile, sheet_name: Optional[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
        ns_wb = {
            "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        ns_rel = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}
        try:
            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        except Exception:
            return None, None, []

        id_to_target: Dict[str, str] = {}
        for rel in rels.findall("p:Relationship", ns_rel):
            rid = rel.attrib.get("Id", "")
            target = rel.attrib.get("Target", "")
            if rid and target:
                id_to_target[rid] = target

        sheets: List[Tuple[str, str]] = []
        for s in wb.findall("m:sheets/m:sheet", ns_wb):
            name = s.attrib.get("name", "")
            rid = s.attrib.get(f"{{{ns_wb['r']}}}id", "")
            if name and rid:
                sheets.append((name, rid))

        sheet_names = [s[0] for s in sheets]
        if not sheets:
            return None, None, sheet_names

        chosen_name, chosen_rid = sheets[0]
        if sheet_name:
            for name, rid in sheets:
                if name.lower() == sheet_name.lower():
                    chosen_name, chosen_rid = name, rid
                    break
        target = id_to_target.get(chosen_rid)
        if not target:
            return None, chosen_name, sheet_names
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        return target, chosen_name, sheet_names

    def _read_xlsx_preview(path: Path, max_rows: int = 20, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        try:
            with zipfile.ZipFile(path, "r") as zf:
                sheet_target, active_sheet, all_sheets = _xlsx_sheet_target(zf, sheet_name)
                if not sheet_target or sheet_target not in zf.namelist():
                    return {"ok": False, "error": "failed to locate worksheet in .xlsx", "path": str(path)}
                shared_strings = _xlsx_shared_strings(zf)
                ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                root = ET.fromstring(zf.read(sheet_target))
        except Exception as e:
            return {"ok": False, "error": f"failed to parse xlsx: {e}", "path": str(path)}

        parsed_rows: List[Dict[int, Any]] = []
        for row_elem in root.findall(".//m:sheetData/m:row", ns):
            row_data: Dict[int, Any] = {}
            fallback_idx = 0
            for cell in row_elem.findall("m:c", ns):
                col_idx = _xlsx_col_idx(cell.attrib.get("r", ""), fallback_idx)
                fallback_idx = col_idx + 1
                cell_type = cell.attrib.get("t", "")
                value: Any = ""

                inline_text = []
                for t_elem in cell.findall("m:is/m:t", ns):
                    inline_text.append(t_elem.text or "")
                if inline_text:
                    value = "".join(inline_text)
                else:
                    v_elem = cell.find("m:v", ns)
                    raw = (v_elem.text or "") if v_elem is not None else ""
                    if cell_type == "s":
                        try:
                            i = int(raw)
                            value = shared_strings[i] if 0 <= i < len(shared_strings) else raw
                        except Exception:
                            value = raw
                    elif cell_type == "b":
                        value = raw == "1"
                    else:
                        value = raw
                row_data[col_idx] = value
            parsed_rows.append(row_data)
            if len(parsed_rows) >= max(2, min(max_rows + 1, 1001)):
                break

        if not parsed_rows:
            return {
                "ok": True,
                "path": str(path),
                "sheet_name": active_sheet or "",
                "sheet_names": all_sheets,
                "headers": [],
                "rows": [],
                "row_count": 0,
            }

        max_col = 0
        for row in parsed_rows:
            if row:
                max_col = max(max_col, max(row.keys()))

        raw_headers = []
        header_row = parsed_rows[0]
        for idx in range(max_col + 1):
            cell_val = str(header_row.get(idx, "")).strip()
            raw_headers.append(cell_val if cell_val else f"column_{idx+1}")

        dedup_headers: List[str] = []
        seen: Dict[str, int] = {}
        for h in raw_headers:
            base = h
            if base not in seen:
                seen[base] = 1
                dedup_headers.append(base)
            else:
                seen[base] += 1
                dedup_headers.append(f"{base}_{seen[base]}")

        data_rows: List[Dict[str, Any]] = []
        for row in parsed_rows[1:]:
            mapped: Dict[str, Any] = {}
            for idx, h in enumerate(dedup_headers):
                mapped[h] = row.get(idx, "")
            data_rows.append(mapped)
            if len(data_rows) >= max(1, min(max_rows, 1000)):
                break

        return {
            "ok": True,
            "path": str(path),
            "sheet_name": active_sheet or "",
            "sheet_names": all_sheets,
            "headers": dedup_headers,
            "rows": data_rows,
            "row_count": len(data_rows),
        }

    def read_spreadsheet_preview(path: str, max_rows: int = 20, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        p = resolve_workspace_file(path)
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"file not found: {p}"}
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return read_csv_preview(path=str(p), max_rows=max_rows)
        if suffix == ".xlsx":
            return _read_xlsx_preview(path=p, max_rows=max_rows, sheet_name=sheet_name)
        return {"ok": False, "error": f"unsupported spreadsheet file type: {suffix}", "path": str(p)}

    def profile_tabular_columns(path: str, max_rows: int = 500, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        preview = read_spreadsheet_preview(path=path, max_rows=max_rows, sheet_name=sheet_name)
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

    def profile_csv_columns(path: str, max_rows: int = 500) -> Dict[str, Any]:
        # Backward-compatible alias.
        return profile_tabular_columns(path=path, max_rows=max_rows)

    def analyze_tabular_with_python(path: str, max_rows: int = 200, timeout_s: int = 120) -> Dict[str, Any]:
        """
        Codex-style execution helper:
        1) write a temporary Python analyzer script
        2) run it in workspace
        3) return parsed JSON result
        """
        if not path:
            return {"ok": False, "error": "path is required"}
        p = resolve_workspace_file(path)
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"file not found: {p}"}

        tmp_dir = resolve_runtime_dir(workspace_path, "tmp", ".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        script_path = tmp_dir / f"tabular_analyzer_{now_ms()}.py"

        script_code = r'''
import csv
import json
import pathlib
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

def safe_float(v):
    try:
        return float(str(v))
    except Exception:
        return None

def profile_rows(headers, rows):
    stats = []
    for h in headers:
        values = [r.get(h, "") for r in rows]
        non_empty = [v for v in values if str(v).strip() != ""]
        numeric = 0
        for v in non_empty:
            if safe_float(v) is not None:
                numeric += 1
        stats.append(
            {
                "column": h,
                "non_empty": len(non_empty),
                "non_empty_ratio": round(len(non_empty) / max(1, len(rows)), 4),
                "numeric_ratio": round(numeric / max(1, len(non_empty)), 4),
                "sample_values": [str(v) for v in non_empty[:5]],
            }
        )
    return stats

def parse_text_table(path, max_rows):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return {"headers": [], "rows": [], "row_count": 0, "detected_format": "empty_text"}

    sample = "\n".join(lines[:50])
    delim_candidates = [",", "\t", ";", "|"]
    counts = {d: sample.count(d) for d in delim_candidates}
    delimiter = max(counts, key=counts.get)
    if counts.get(delimiter, 0) == 0:
        # Not a real table, return text preview as single-column rows.
        headers = ["text_line"]
        rows = [{"text_line": ln} for ln in lines[:max_rows]]
        return {"headers": headers, "rows": rows, "row_count": len(rows), "detected_format": "plain_text"}

    reader = csv.reader(lines, delimiter=delimiter)
    parsed = []
    for idx, row in enumerate(reader):
        if idx >= max_rows + 1:
            break
        parsed.append(row)

    if not parsed:
        return {"headers": [], "rows": [], "row_count": 0, "detected_format": "delimited_text"}

    raw_headers = [str(x).strip() if str(x).strip() else f"column_{i+1}" for i, x in enumerate(parsed[0])]
    headers = []
    seen = {}
    for h in raw_headers:
        if h not in seen:
            seen[h] = 1
            headers.append(h)
        else:
            seen[h] += 1
            headers.append(f"{h}_{seen[h]}")

    out_rows = []
    for row in parsed[1:]:
        mapped = {}
        for i, h in enumerate(headers):
            mapped[h] = row[i] if i < len(row) else ""
        out_rows.append(mapped)

    return {"headers": headers, "rows": out_rows, "row_count": len(out_rows), "detected_format": "delimited_text"}

def xlsx_col_idx(cell_ref, fallback_idx):
    m = re.match(r"([A-Z]+)", (cell_ref or "").upper())
    if not m:
        return fallback_idx
    letters = m.group(1)
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return max(0, value - 1)

def parse_xlsx(path, max_rows):
    ns_wb = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    ns_rel = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path, "r") as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        id_to_target = {}
        for rel in rels.findall("p:Relationship", ns_rel):
            rid = rel.attrib.get("Id", "")
            target = rel.attrib.get("Target", "")
            if rid and target:
                id_to_target[rid] = target

        sheets = []
        for s in wb.findall("m:sheets/m:sheet", ns_wb):
            name = s.attrib.get("name", "")
            rid = s.attrib.get(f"{{{ns_wb['r']}}}id", "")
            if name and rid:
                sheets.append((name, rid))
        if not sheets:
            return {"headers": [], "rows": [], "row_count": 0, "detected_format": "xlsx_empty", "sheet_name": "", "sheet_names": []}

        sheet_name, rid = sheets[0]
        target = id_to_target.get(rid, "").lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        root = ET.fromstring(zf.read(target))

        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.findall("m:si", ns):
                texts = []
                for t in si.findall(".//m:t", ns):
                    texts.append(t.text or "")
                shared_strings.append("".join(texts))

        parsed_rows = []
        for row_elem in root.findall(".//m:sheetData/m:row", ns):
            row_data = {}
            fallback_idx = 0
            for cell in row_elem.findall("m:c", ns):
                col_idx = xlsx_col_idx(cell.attrib.get("r", ""), fallback_idx)
                fallback_idx = col_idx + 1
                cell_type = cell.attrib.get("t", "")

                inline_text = []
                for t_elem in cell.findall("m:is/m:t", ns):
                    inline_text.append(t_elem.text or "")
                if inline_text:
                    value = "".join(inline_text)
                else:
                    v_elem = cell.find("m:v", ns)
                    raw = (v_elem.text or "") if v_elem is not None else ""
                    if cell_type == "s":
                        try:
                            i = int(raw)
                            value = shared_strings[i] if 0 <= i < len(shared_strings) else raw
                        except Exception:
                            value = raw
                    elif cell_type == "b":
                        value = raw == "1"
                    else:
                        value = raw
                row_data[col_idx] = value
            parsed_rows.append(row_data)
            if len(parsed_rows) >= max_rows + 1:
                break

    if not parsed_rows:
        return {"headers": [], "rows": [], "row_count": 0, "detected_format": "xlsx", "sheet_name": sheet_name, "sheet_names": [s[0] for s in sheets]}

    max_col = 0
    for row in parsed_rows:
        if row:
            max_col = max(max_col, max(row.keys()))

    raw_headers = []
    header_row = parsed_rows[0]
    for idx in range(max_col + 1):
        val = str(header_row.get(idx, "")).strip()
        raw_headers.append(val if val else f"column_{idx+1}")

    headers = []
    seen = {}
    for h in raw_headers:
        if h not in seen:
            seen[h] = 1
            headers.append(h)
        else:
            seen[h] += 1
            headers.append(f"{h}_{seen[h]}")

    rows = []
    for row in parsed_rows[1:]:
        mapped = {}
        for idx, h in enumerate(headers):
            mapped[h] = row.get(idx, "")
        rows.append(mapped)

    return {
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
        "detected_format": "xlsx",
        "sheet_name": sheet_name,
        "sheet_names": [s[0] for s in sheets],
    }

def main():
    path = pathlib.Path(sys.argv[1]).expanduser()
    max_rows = max(1, min(int(sys.argv[2]) if len(sys.argv) > 2 else 200, 2000))

    if not path.exists() or not path.is_file():
        print(json.dumps({"ok": False, "error": f"file not found: {path}"}, ensure_ascii=False))
        return

    suffix = path.suffix.lower()
    payload = {"ok": True, "path": str(path), "suffix": suffix}

    try:
        if suffix == ".xlsx" and zipfile.is_zipfile(path):
            parsed = parse_xlsx(path, max_rows=max_rows)
        else:
            parsed = parse_text_table(path, max_rows=max_rows)
        payload.update(parsed)
        payload["column_stats"] = profile_rows(parsed.get("headers", []), parsed.get("rows", []))
        payload["summary"] = (
            f"Analyzed {path.name}: format={payload.get('detected_format')}, "
            f"rows={payload.get('row_count', 0)}, columns={len(payload.get('headers', []))}."
        )
    except Exception as e:
        payload = {"ok": False, "path": str(path), "error": f"analysis failed: {e}"}

    print(json.dumps(payload, ensure_ascii=False))

if __name__ == "__main__":
    main()
'''.lstrip()

        try:
            script_path.write_text(script_code, encoding="utf-8")
        except Exception as e:
            return {"ok": False, "error": f"failed to write analyzer script: {e}"}

        try:
            proc = subprocess.run(
                ["python3", str(script_path), str(p), str(max_rows)],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=max(10, min(int(timeout_s or 120), 300)),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "python analyzer timed out", "script_path": str(script_path), "path": str(p)}
        except Exception as e:
            return {"ok": False, "error": f"failed to run analyzer script: {e}", "script_path": str(script_path), "path": str(p)}

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        parsed: Optional[Dict[str, Any]] = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            except Exception:
                continue

        if parsed is None:
            return {
                "ok": False,
                "error": "analyzer script did not return valid JSON",
                "path": str(p),
                "script_path": str(script_path),
                "return_code": proc.returncode,
                "stdout": truncate_text(stdout, max_chars=4000),
                "stderr": truncate_text(stderr, max_chars=4000),
            }

        parsed["script_path"] = str(script_path)
        parsed["return_code"] = proc.returncode
        parsed["stderr"] = truncate_text(stderr, max_chars=2000)
        if proc.returncode != 0 and parsed.get("ok") is not False:
            parsed["ok"] = False
            parsed["error"] = f"python analyzer exited with code {proc.returncode}"
        return parsed

    def prepare_logistic_regression_demo(
        task: str,
        output_path: str = "data/logreg_dataset.csv",
        max_rows: int = 5000,
    ) -> Dict[str, Any]:
        """
        Robust end-to-end demo:
        1) Try downloading a public binary-classification dataset
        2) If download fails, generate synthetic binary data
        3) Train logistic regression with pure numpy
        4) Save dataset + training report
        """

        import numpy as np
        import warnings

        def sigmoid(x: Any) -> Any:
            arr = np.array(x, dtype=np.float64)
            arr = np.clip(arr, -50.0, 50.0)
            return 1.0 / (1.0 + np.exp(-arr))

        def compute_metrics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
            y_t = np.array(y_true, dtype=np.int64)
            y_p = np.array(y_pred, dtype=np.int64)
            tp = int(np.sum((y_t == 1) & (y_p == 1)))
            fp = int(np.sum((y_t == 0) & (y_p == 1)))
            tn = int(np.sum((y_t == 0) & (y_p == 0)))
            fn = int(np.sum((y_t == 1) & (y_p == 0)))
            accuracy = float((tp + tn) / max(1, len(y_t)))
            precision = float(tp / max(1, tp + fp))
            recall = float(tp / max(1, tp + fn))
            f1 = float(2 * precision * recall / max(1e-12, precision + recall))
            return {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            }

        def train_logreg_numpy(x: Any, y: Any, lr: float = 0.1, epochs: int = 1200) -> Dict[str, Any]:
            x_arr = np.nan_to_num(np.array(x, dtype=np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
            y_arr = np.array(y, dtype=np.float64).reshape(-1, 1)
            n, d = x_arr.shape
            mean = x_arr.mean(axis=0, keepdims=True)
            std = x_arr.std(axis=0, keepdims=True)
            std[std < 1e-9] = 1.0
            x_norm = (x_arr - mean) / std
            x_bias = np.concatenate([np.ones((n, 1), dtype=np.float64), x_norm], axis=1)
            w = np.zeros((d + 1, 1), dtype=np.float64)
            losses: List[float] = []

            for _ in range(epochs):
                with np.errstate(all="ignore"):
                    probs = sigmoid(x_bias @ w)
                    grad = (x_bias.T @ (probs - y_arr)) / n
                w = w - lr * grad
                w = np.nan_to_num(w, nan=0.0, posinf=50.0, neginf=-50.0)
                loss = -np.mean(y_arr * np.log(probs + 1e-12) + (1 - y_arr) * np.log(1 - probs + 1e-12))
                if not np.isfinite(loss):
                    loss = float("nan")
                losses.append(float(loss))

            clean_losses = [x for x in losses if isinstance(x, float) and math.isfinite(x)]
            return {
                "weights": w.reshape(-1).tolist(),
                "mean": mean.reshape(-1).tolist(),
                "std": std.reshape(-1).tolist(),
                "final_loss": round(float(clean_losses[-1]), 6) if clean_losses else None,
                "loss_head": [round(float(x), 6) for x in clean_losses[:5]],
                "loss_tail": [round(float(x), 6) for x in clean_losses[-5:]],
            }

        def predict_with_model(model: Dict[str, Any], x: Any) -> Any:
            import numpy as np

            x_arr = np.nan_to_num(np.array(x, dtype=np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
            mean = np.array(model["mean"], dtype=np.float64).reshape(1, -1)
            std = np.array(model["std"], dtype=np.float64).reshape(1, -1)
            std[std < 1e-9] = 1.0
            x_norm = (x_arr - mean) / std
            x_bias = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm], axis=1)
            w = np.array(model["weights"], dtype=np.float64).reshape(-1, 1)
            with np.errstate(all="ignore"):
                probs = sigmoid(x_bias @ w).reshape(-1)
            preds = (probs >= 0.5).astype(np.int64)
            return probs.tolist(), preds.tolist()

        def try_download_binary_dataset(max_samples: int) -> Tuple[Optional[List[str]], Optional[List[List[float]]], Optional[List[int]], str]:
            # Pima Indians Diabetes dataset (last column is binary target)
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            try:
                with urllib.request.urlopen(url, timeout=12) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
            except Exception as e:
                return None, None, None, f"download_failed: {e}"

            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if not lines:
                return None, None, None, "download_failed: empty dataset"

            features: List[List[float]] = []
            labels: List[int] = []
            for ln in lines[: max(50, min(max_samples, 20000))]:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    vals = [float(x) for x in parts]
                except Exception:
                    continue
                x = vals[:-1]
                y = int(round(vals[-1]))
                if y not in {0, 1}:
                    continue
                features.append(x)
                labels.append(y)

            if len(features) < 50:
                return None, None, None, "download_failed: insufficient valid rows"
            headers = [f"x{i+1}" for i in range(len(features[0]))]
            return headers, features, labels, "downloaded"

        def generate_synthetic_binary_dataset(max_samples: int) -> Tuple[List[str], List[List[float]], List[int], str]:
            import numpy as np

            n = max(500, min(max_samples, 5000))
            d = 8
            rng = np.random.default_rng(42)
            x = rng.normal(0.0, 1.0, size=(n, d))
            w_true = rng.normal(0.0, 1.0, size=(d,))
            noise = rng.normal(0.0, 0.5, size=(n,))
            with np.errstate(all="ignore"):
                logits = x @ w_true + noise
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            probs = sigmoid(logits)
            y = (probs >= 0.5).astype(np.int64)
            headers = [f"x{i+1}" for i in range(d)]
            return headers, x.tolist(), y.tolist(), "generated"

        output = resolve_workspace_file(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        report_path = output.with_suffix(".report.md")

        headers = None
        x_rows = None
        y_rows = None
        source = "generated"
        source_note = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            headers, x_rows, y_rows, source = try_download_binary_dataset(max_rows)
            if headers is None or x_rows is None or y_rows is None:
                source_note = source
                headers, x_rows, y_rows, source = generate_synthetic_binary_dataset(max_rows)

        n = len(x_rows)
        if n < 50:
            return {"ok": False, "error": "failed to prepare enough rows for logistic regression"}

        split_idx = max(1, int(n * 0.8))
        x_train = x_rows[:split_idx]
        y_train = y_rows[:split_idx]
        x_test = x_rows[split_idx:]
        y_test = y_rows[split_idx:]
        if len(x_test) < 20:
            x_train = x_rows[:-20]
            y_train = y_rows[:-20]
            x_test = x_rows[-20:]
            y_test = y_rows[-20:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model = train_logreg_numpy(x_train, y_train, lr=0.08, epochs=1000)
            probs, preds = predict_with_model(model, x_test)
        metrics = compute_metrics(y_test, preds)

        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers + ["label"])
            for x_row, y_val in zip(x_rows, y_rows):
                writer.writerow([*x_row, int(y_val)])

        report_lines = [
            "# Logistic Regression Demo Report",
            "",
            f"- Task: {task}",
            f"- Dataset source: {source}",
            f"- Dataset path: {output}",
            f"- Rows: {len(x_rows)}",
            f"- Features: {len(headers)}",
            f"- Train rows: {len(x_train)}",
            f"- Test rows: {len(x_test)}",
            "",
            "## Metrics",
            f"- Accuracy: {metrics['accuracy']}",
            f"- Precision: {metrics['precision']}",
            f"- Recall: {metrics['recall']}",
            f"- F1: {metrics['f1']}",
            "",
            "## Training",
            f"- Final loss: {model.get('final_loss')}",
            f"- Loss head: {model.get('loss_head')}",
            f"- Loss tail: {model.get('loss_tail')}",
            "",
        ]
        if source_note:
            report_lines.append("## Notes")
            report_lines.append(f"- Download fallback reason: {source_note}")
            report_lines.append("")
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        preview_rows: List[Dict[str, Any]] = []
        for i, x_row in enumerate(x_rows[:5]):
            item = {headers[j]: round(float(x_row[j]), 6) for j in range(len(headers))}
            item["label"] = int(y_rows[i])
            preview_rows.append(item)

        return {
            "ok": True,
            "summary": (
                f"Prepared dataset ({source}) and trained logistic regression. "
                f"F1={metrics['f1']}, accuracy={metrics['accuracy']}. "
                f"Dataset saved to {output}."
            ),
            "dataset_source": source,
            "dataset_path": str(output),
            "report_path": str(report_path),
            "rows": len(x_rows),
            "feature_count": len(headers),
            "metrics": metrics,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "sample_rows": preview_rows,
        }

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

    def read_code_file(path: str, max_chars: int = 50000) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        p = resolve_workspace_file(path)
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"file not found: {p}"}
        text = p.read_text(encoding="utf-8", errors="ignore")
        text = text[: max(500, min(max_chars, 300000))]
        return {
            "ok": True,
            "path": str(p),
            "text": text,
            "line_count": text.count("\n") + 1 if text else 0,
            "suffix": p.suffix.lower(),
        }

    def read_code_span(path: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
        row = read_code_file(path=path, max_chars=300000)
        if not row.get("ok"):
            return row
        text = str(row.get("text", ""))
        lines = text.splitlines()
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = max(start, int(end_line or start))
        start = min(start, max(1, total))
        end = min(end, max(1, total))
        selected = lines[start - 1 : end]
        numbered = []
        for idx, line in enumerate(selected, start=start):
            numbered.append(f"{idx}: {line}")
        return {
            "ok": True,
            "path": row.get("path"),
            "start_line": start,
            "end_line": end,
            "line_count": total,
            "content": "\n".join(numbered),
        }

    def replace_text_in_file(path: str, find_text: str, replace_text: str, count: int = 1) -> Dict[str, Any]:
        if not path:
            return {"ok": False, "error": "path is required"}
        if find_text is None or find_text == "":
            return {"ok": False, "error": "find_text is required"}
        p = resolve_workspace_file(path)
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"file not found: {p}"}
        original = p.read_text(encoding="utf-8", errors="ignore")
        replace_limit = -1 if int(count) <= 0 else int(count)
        replaced = original.replace(find_text, replace_text, replace_limit)
        if replaced == original:
            return {"ok": False, "error": "find_text not found", "path": str(p), "replacements": 0}
        num_replacements = original.count(find_text) if replace_limit == -1 else min(original.count(find_text), replace_limit)
        p.write_text(replaced, encoding="utf-8")
        return {"ok": True, "path": str(p), "replacements": num_replacements}

    def run_shell_command(command: str, timeout_s: int = 90, approved: bool = False) -> Dict[str, Any]:
        if not command or not str(command).strip():
            return {"ok": False, "error": "command is required"}
        cmd = str(command).strip()
        if re.match(r"^pip(?:\d+(?:\.\d+)*)?\s+", cmd):
            cmd = re.sub(r"^pip(?:\d+(?:\.\d+)*)?\s+", "python3 -m pip ", cmd, count=1)
        blocked_patterns = [
            r"\brm\s+-rf\s+/",
            r"\bsudo\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bmkfs\b",
            r"\bdd\s+if=",
            r"\bgit\s+reset\s+--hard\b",
            r":\(\)\s*\{",
        ]
        for pat in blocked_patterns:
            if re.search(pat, cmd, flags=re.IGNORECASE):
                return {"ok": False, "error": f"command blocked by safety rule: {pat}"}

        try:
            proc = subprocess.run(
                ["/bin/zsh", "-lc", cmd],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=max(1, min(int(timeout_s or 90), 300)),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "command timed out"}
        except Exception as e:
            return {"ok": False, "error": f"failed to run command: {e}"}

        stdout = truncate_text(proc.stdout or "", max_chars=12000)
        stderr = truncate_text(proc.stderr or "", max_chars=12000)
        ok = proc.returncode == 0
        return {
            "ok": ok,
            "command": cmd,
            "cwd": str(workspace_path),
            "return_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "approved": bool(approved),
        }

    def list_available_tools() -> Dict[str, Any]:
        return {"tools": tools.list_tools()}

    def experience_recent(limit: int = 10) -> Dict[str, Any]:
        return {"experiences": memory.recent_experiences(limit=limit)}

    def experience_search(query: str, top_k: int = 5) -> Dict[str, Any]:
        return {"experiences": memory.search_experiences(query=query, top_k=top_k)}

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
            name="experience_recent",
            func=experience_recent,
            description="List recent solved-problem experiences.",
            input_schema={"limit": "int"},
            permission="kb_read",
            owner_agent="experience_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="experience_search",
            func=experience_search,
            description="Search reusable experiences by semantic-like term matching.",
            input_schema={"query": "string", "top_k": "int"},
            permission="kb_read",
            owner_agent="experience_agent",
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
            name="network_http_request",
            func=network_http_request,
            description="Send HTTP request and return status/headers/body preview.",
            input_schema={
                "url": "string",
                "method": "GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS",
                "headers": "dict|null",
                "body": "string|null",
                "json_body": "dict|null",
                "timeout_s": "int",
                "approved": "bool",
            },
            permission="network_read",
            owner_agent="supervisor",
            timeout_s=120,
        )
    )
    tools.register(
        ToolSpec(
            name="network_download_file",
            func=network_download_file,
            description="Download a URL into workspace file path.",
            input_schema={"url": "string", "output_path": "string", "timeout_s": "int", "overwrite": "bool", "approved": "bool"},
            permission="network_read",
            owner_agent="supervisor",
            timeout_s=180,
        )
    )
    tools.register(
        ToolSpec(
            name="sqlite_query",
            func=sqlite_query,
            description="Execute a SELECT query against a local sqlite database.",
            input_schema={"db_path": "string", "query": "string", "params": "list|dict|string|null", "limit": "int", "approved": "bool"},
            permission="db_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="sqlite_execute",
            func=sqlite_execute,
            description="Execute non-SELECT SQL statements against local sqlite database.",
            input_schema={"db_path": "string", "statement": "string", "params": "list|dict|string|null", "approved": "bool"},
            permission="db_write",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="browser_open_page",
            func=browser_open_page,
            description="Open a web page (lightweight browser) and store page state.",
            input_schema={"url": "string", "timeout_s": "int", "approved": "bool"},
            permission="browser_read",
            owner_agent="supervisor",
            timeout_s=120,
        )
    )
    tools.register(
        ToolSpec(
            name="browser_click_link",
            func=browser_click_link,
            description="Navigate by clicking a link from current browser state (by text or index).",
            input_schema={"link_text": "string", "link_index": "int", "approved": "bool"},
            permission="browser_action",
            owner_agent="supervisor",
            timeout_s=120,
        )
    )
    tools.register(
        ToolSpec(
            name="browser_get_state",
            func=browser_get_state,
            description="Get current browser state snapshot.",
            input_schema={},
            permission="browser_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="browser_find_text",
            func=browser_find_text,
            description="Find text pattern in current browser state preview.",
            input_schema={"pattern": "string", "max_matches": "int"},
            permission="browser_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="observe_git_diff",
            func=observe_git_diff,
            description="Observe current git diff and changed files.",
            input_schema={"pathspec": "string", "max_chars": "int"},
            permission="observe_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="observe_error_logs",
            func=observe_error_logs,
            description="Observe error-like entries from .log files in workspace.",
            input_schema={"pattern": "string", "top_k": "int"},
            permission="observe_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="observe_recent_events",
            func=observe_recent_events,
            description="Observe recent workflow events by trace id or across traces.",
            input_schema={"trace_id": "string", "limit": "int"},
            permission="observe_read",
            owner_agent="supervisor",
        )
    )
    tools.register(
        ToolSpec(
            name="observe_browser_state",
            func=observe_browser_state,
            description="Observe current browser state (alias of browser_get_state).",
            input_schema={},
            permission="observe_read",
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
            description="Preview CSV headers and rows (legacy alias; for CSV/XLSX use read_spreadsheet_preview).",
            input_schema={"path": "string", "max_rows": "int"},
            permission="file_read",
            owner_agent="data_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="read_spreadsheet_preview",
            func=read_spreadsheet_preview,
            description="Preview tabular data from .csv or .xlsx with headers and sample rows.",
            input_schema={"path": "string", "max_rows": "int", "sheet_name": "string|null"},
            permission="file_read",
            owner_agent="data_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="profile_csv_columns",
            func=profile_csv_columns,
            description="Infer basic column stats for CSV data (legacy alias).",
            input_schema={"path": "string", "max_rows": "int"},
            permission="data_exec",
            owner_agent="data_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="profile_tabular_columns",
            func=profile_tabular_columns,
            description="Infer basic column stats for .csv or .xlsx data.",
            input_schema={"path": "string", "max_rows": "int", "sheet_name": "string|null"},
            permission="data_exec",
            owner_agent="data_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="analyze_tabular_with_python",
            func=analyze_tabular_with_python,
            description="Write a temporary Python analyzer script, execute it, and return tabular analysis for .csv/.xlsx/text files.",
            input_schema={"path": "string", "max_rows": "int", "timeout_s": "int"},
            permission="code_exec",
            owner_agent="code_agent",
            timeout_s=240,
            retry=0,
        )
    )
    tools.register(
        ToolSpec(
            name="prepare_logistic_regression_demo",
            func=prepare_logistic_regression_demo,
            description="Prepare a binary dataset (download with fallback to synthetic), train logistic regression with numpy, and save artifacts.",
            input_schema={"task": "string", "output_path": "string", "max_rows": "int"},
            permission="ml_train",
            owner_agent="trainer",
            timeout_s=240,
            retry=0,
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
            name="read_code_file",
            func=read_code_file,
            description="Read source code/config file in workspace.",
            input_schema={"path": "string", "max_chars": "int"},
            permission="file_read",
            owner_agent="code_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="read_code_span",
            func=read_code_span,
            description="Read a code file with line numbers for a specific line range.",
            input_schema={"path": "string", "start_line": "int", "end_line": "int"},
            examples=["read_code_span(path='multi_agent_system.py', start_line=1, end_line=120)"],
            permission="file_read",
            owner_agent="code_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="replace_text_in_file",
            func=replace_text_in_file,
            description="Replace text in a workspace file for deterministic code edits.",
            input_schema={"path": "string", "find_text": "string", "replace_text": "string", "count": "int"},
            permission="file_write",
            owner_agent="code_agent",
        )
    )
    tools.register(
        ToolSpec(
            name="run_shell_command",
            func=run_shell_command,
            description="Run a shell command in workspace root and return stdout/stderr.",
            input_schema={"command": "string", "timeout_s": "int", "approved": "bool"},
            examples=["run_shell_command(command='pytest -q', timeout_s=120)", "run_shell_command(command='python3 terminal_chat.py --help')"],
            permission="code_exec",
            owner_agent="code_agent",
            timeout_s=180,
            retry=0,
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
            name="knowledge_add_reference",
            func=knowledge_add_reference,
            description="Ingest product/domain/API/style reference file into knowledge memory with metadata.",
            input_schema={"path": "string", "title": "string|null", "category": "string", "tags": "list[string]|null"},
            permission="kb_write",
            owner_agent="kb_retriever",
        )
    )
    tools.register(
        ToolSpec(
            name="knowledge_ingest_workspace_docs",
            func=knowledge_ingest_workspace_docs,
            description="Bulk ingest workspace docs into knowledge memory.",
            input_schema={"pattern": "string", "recursive": "bool", "limit": "int", "category": "string"},
            permission="kb_write",
            owner_agent="kb_retriever",
        )
    )
    tools.register(
        ToolSpec(
            name="knowledge_list_sources",
            func=knowledge_list_sources,
            description="List knowledge sources and metadata.",
            input_schema={"limit": "int"},
            permission="kb_read",
            owner_agent="kb_retriever",
        )
    )
    tools.register(
        ToolSpec(
            name="knowledge_get_doc",
            func=knowledge_get_doc,
            description="Get one knowledge document by id.",
            input_schema={"doc_id": "string", "max_chars": "int"},
            permission="kb_read",
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
            description="List skills installed in local skills directory.",
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
        experience_model = os.getenv("EXPERIENCE_MODEL", router_model)
        self.understanding_engine = RequestUnderstandingEngine(model=router_model, workspace=self.workspace)
        self.tools = build_tool_registry(self.memory, workspace=self.workspace)
        self.experience_agent = ExperienceAgent(
            memory=self.memory,
            workspace=self.workspace,
            model=experience_model,
            runtime_tool_names=list(self.tools.tools.keys()),
        )
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
            "code_agent": CodeAgent(self.tools),
            "experience_agent": self.experience_agent,
        }
        self.supervisor = Supervisor(self.agents)
        orchestrator_model = os.getenv("ORCHESTRATOR_MODEL", router_model)
        self.dynamic_orchestrator = DynamicLoopOrchestrator(
            model=orchestrator_model,
            tools=self.tools,
            memory=self.memory,
            workspace=self.workspace,
            experience_agent=self.experience_agent,
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
            session_id=session_id,
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
            "workflow_intent": "",
            "code_path": None,
            "command": None,
            "find_text": None,
            "replace_text": None,
            "count": 1,
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
