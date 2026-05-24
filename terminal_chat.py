from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from multi_agent_system import MultiAgentSystem


def try_import_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except Exception:
        return None


class ChatResponder:
    def __init__(self, model: str):
        self.model = model
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        OpenAI = try_import_openai()
        if api_key and OpenAI is not None:
            self.client = OpenAI(api_key=api_key)

    def available(self) -> bool:
        return self.client is not None

    def render(self, user_input: str, result: Dict[str, Any], trace: str) -> str:
        if result.get("intent") == "GENERAL_CHAT":
            return self._fallback(result, trace)

        if not self.client:
            return self._fallback(result, trace)

        system_prompt = (
            "You are an assistant for a multi-agent ML platform. "
            "Respond in concise English. "
            "For dynamic execution results, clearly summarize goal, executed tools, and outcome."
        )
        user_prompt = {
            "user_input": user_input,
            "structured_result": result,
            "trace_id": trace,
        }
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
            )
            text = getattr(resp, "output_text", None)
            if text:
                return text.strip()
        except Exception as e:
            return f"{self._fallback(result, trace)}\n\n(LLM call failed; fallback used. Error: {e})"
        return self._fallback(result, trace)

    def _fallback(self, result: Dict[str, Any], trace: str) -> str:
        intent = result.get("intent", "UNKNOWN")
        if intent == "GENERAL_CHAT":
            return result.get("reply", "Please provide more details.")
        if intent in {"KNOWLEDGE_LOOKUP", "KB_QUERY"}:
            rows = result.get("snippets", [])
            if not rows:
                return f"[{trace}] No matching knowledge snippets found."
            lines = [f"[{trace}] Knowledge snippets:"]
            for i, row in enumerate(rows, start=1):
                lines.append(f"{i}. {row.get('snippet', '')} (source: {row.get('source', '')})")
            return "\n".join(lines)
        if intent in {"ML_WORKFLOW", "DS_PIPELINE"}:
            executed_steps = result.get("executed_steps", [])
            lines = [
                f"[{trace}] ML workflow finished. Best model: {result.get('best_model', 'N/A')}",
            ]
            if executed_steps:
                lines.append("Executed steps: " + " -> ".join(executed_steps))
            report = result.get("report", "")
            if report:
                lines.append(report)
            sources = result.get("knowledge_sources", [])
            if sources:
                lines.append("")
                lines.append("Knowledge sources used:")
                for src in sources:
                    lines.append(f"- {src}")
            return (
                "\n".join(lines)
            )
        if intent == "CODE_TASK":
            lines = [f"[{trace}] Code task finished."]
            summary = result.get("summary", "")
            if summary:
                lines.append(summary)
            executed_steps = result.get("executed_steps", [])
            if executed_steps:
                lines.append("Executed steps: " + " -> ".join(executed_steps))
            code_path = result.get("code_path")
            if code_path:
                lines.append(f"Code path: {code_path}")
            command = result.get("command")
            if command:
                lines.append(f"Command: {command}")
            command_result = result.get("command_result", {})
            if isinstance(command_result, dict) and command_result:
                lines.append(f"Return code: {command_result.get('return_code', 'N/A')}")
                stdout = (command_result.get("stdout") or "").strip()
                stderr = (command_result.get("stderr") or "").strip()
                if stdout:
                    lines.append("")
                    lines.append("stdout:")
                    lines.append(stdout)
                if stderr:
                    lines.append("")
                    lines.append("stderr:")
                    lines.append(stderr)
            return "\n".join(lines)
        if intent == "DYNAMIC_EXECUTION":
            lines = [
                f"[{trace}] Dynamic loop finished ({result.get('status', 'completed')}).",
            ]
            goal = result.get("goal")
            if goal:
                lines.append(f"Goal: {goal}")
            plan = result.get("plan", [])
            if plan:
                lines.append("Plan:")
                for i, step in enumerate(plan, start=1):
                    lines.append(f"{i}. {step}")
            executed_tools = result.get("executed_tools", [])
            if executed_tools:
                lines.append("Executed tools: " + " -> ".join(executed_tools))
            summary = result.get("result_summary", "")
            if summary:
                lines.append("")
                lines.append("Result:")
                lines.append(summary)
            exp_id = result.get("experience_id")
            if exp_id:
                lines.append("")
                lines.append(f"Experience logged: {exp_id}")
                skill = result.get("skill_candidate", {})
                if isinstance(skill, dict) and skill.get("name"):
                    lines.append(f"Skill candidate: {skill.get('name')}")
            observations = result.get("observations", [])
            if observations:
                lines.append("")
                lines.append("Observations:")
                for i, row in enumerate(observations[-4:], start=1):
                    tool = row.get("tool", "unknown_tool")
                    ok = row.get("ok", True)
                    if ok:
                        lines.append(f"{i}. {tool}: ok")
                    else:
                        lines.append(f"{i}. {tool}: failed ({row.get('error', 'unknown error')})")
            reflections = result.get("reflections", [])
            if reflections:
                lines.append("")
                lines.append("Reflections:")
                for i, row in enumerate(reflections[-3:], start=1):
                    reason = row.get("reason", "n/a")
                    inserted = row.get("inserted_steps", [])
                    if isinstance(inserted, list) and inserted:
                        lines.append(f"{i}. {reason} -> {' | '.join([str(x) for x in inserted])}")
                    else:
                        lines.append(f"{i}. {reason}")
            events = result.get("event_log_tail", [])
            if events:
                phases = [str(e.get("phase", "")) for e in events if isinstance(e, dict) and e.get("phase")]
                if phases:
                    lines.append("")
                    lines.append("Event phases: " + " -> ".join(phases[-8:]))
            return "\n".join(lines)
        if intent == "DOC_SUMMARY":
            if not result.get("ok", True):
                return f"[{trace}] DOC_SUMMARY failed: {result.get('error', 'unknown error')}"
            lines = [
                f"[{trace}] Document summary complete.",
                f"Source: {result.get('source_path', 'N/A')}",
                f"Word count: {result.get('word_count', 0)}",
                "",
                "Summary:",
                result.get("summary", ""),
            ]
            highlights = result.get("highlights", [])
            if highlights:
                lines.append("")
                lines.append("Highlights:")
                for i, row in enumerate(highlights, start=1):
                    lines.append(f"{i}. {row}")
            return "\n".join(lines)
        return f"[{trace}] {json.dumps(result, ensure_ascii=False, indent=2)}"


def print_help() -> None:
    print("Commands:")
    print("  /exit        Exit")
    print("  /tools       List registered tools")
    print("  /skills list")
    print("  /skills install <repo_url> [alias] [ref]")
    print("  /file summarize <path_to_docx_or_pdf>")
    print("  /raw on|off  Toggle raw structured output")
    print("  /help        Show help")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chat with multi-agent system")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--session-id", default="terminal", help="session id")
    parser.add_argument("--workspace", default=".", help="knowledge scan folder")
    args = parser.parse_args()

    app = MultiAgentSystem(workspace=args.workspace)
    responder = ChatResponder(model=args.model)
    show_raw = False

    print("Multi-Agent Terminal started. Type /help for commands.")
    if not responder.available():
        print("Note: OpenAI client or OPENAI_API_KEY not detected; using local fallback responses.")

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye.")
            break
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/tools":
            tools = app.tools.list_tools()
            print(json.dumps(tools, ensure_ascii=False, indent=2))
            continue
        if user_input.startswith("/skills"):
            parts = user_input.split()
            if len(parts) >= 2 and parts[1] == "list":
                result = app.tools.execute("skill_list_installed")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif len(parts) >= 3 and parts[1] == "install":
                repo_url = parts[2]
                alias = parts[3] if len(parts) >= 4 else None
                ref = parts[4] if len(parts) >= 5 else None
                result = app.tools.execute("skill_install_from_git", repo_url=repo_url, alias=alias, ref=ref)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("Usage: /skills list | /skills install <repo_url> [alias] [ref]")
            continue
        if user_input.startswith("/file"):
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3 and parts[1] == "summarize":
                file_path = parts[2].strip()
                explicit = json.dumps(
                    {
                        "intent": "DOC_SUMMARY",
                        "mode": "EXECUTE",
                        "file_path": file_path,
                    },
                    ensure_ascii=False,
                )
                final_message, state = app.run(explicit, session_id=args.session_id)
                trace = state["trace_id"]
                content = final_message.get("content", {})
                if show_raw:
                    print(json.dumps(content, ensure_ascii=False, indent=2))
                else:
                    text = responder.render(user_input=user_input, result=content, trace=trace)
                    print(f"Agent> {text}")
            else:
                print("Usage: /file summarize <path_to_docx_or_pdf>")
            continue
        if user_input.startswith("/raw"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                show_raw = parts[1] == "on"
                print(f"raw mode: {parts[1]}")
            else:
                print("Usage: /raw on|off")
            continue

        final_message, state = app.run(user_input, session_id=args.session_id)
        trace = state["trace_id"]
        content = final_message.get("content", {})
        if show_raw:
            print(json.dumps(content, ensure_ascii=False, indent=2))
            continue
        text = responder.render(user_input=user_input, result=content, trace=trace)
        print(f"Agent> {text}")


if __name__ == "__main__":
    main()
