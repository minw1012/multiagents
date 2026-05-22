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
            "Respond in concise Chinese. If KB query, summarize findings with sources. "
            "If DS pipeline, summarize data/model/evaluation/report clearly."
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
            return f"{self._fallback(result, trace)}\n\n(LLM调用失败，已降级。错误: {e})"
        return self._fallback(result, trace)

    def _fallback(self, result: Dict[str, Any], trace: str) -> str:
        intent = result.get("intent", "UNKNOWN")
        if intent == "GENERAL_CHAT":
            lines = [result.get("reply", "Please provide more details.")]
            suggestions = result.get("suggestions", [])
            if suggestions:
                lines.append("")
                lines.append("Examples:")
                for i, text in enumerate(suggestions, start=1):
                    lines.append(f"{i}. {text}")
            return "\n".join(lines)
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
        return f"[{trace}] {json.dumps(result, ensure_ascii=False, indent=2)}"


def print_help() -> None:
    print("命令:")
    print("  /exit        退出")
    print("  /tools       查看已注册工具")
    print("  /skills list")
    print("  /skills install <repo_url> [alias] [ref]")
    print("  /raw on|off  是否显示结构化结果")
    print("  /help        显示帮助")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chat with multi-agent system")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--session-id", default="terminal", help="session id")
    parser.add_argument("--workspace", default=".", help="knowledge scan folder")
    args = parser.parse_args()

    app = MultiAgentSystem(workspace=args.workspace)
    responder = ChatResponder(model=args.model)
    show_raw = False

    print("Multi-Agent Terminal 已启动。输入 /help 查看命令。")
    if not responder.available():
        print("提示: 未检测到可用 OpenAI 客户端或 OPENAI_API_KEY，当前使用本地降级回复。")

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
                print("用法: /skills list | /skills install <repo_url> [alias] [ref]")
            continue
        if user_input.startswith("/raw"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                show_raw = parts[1] == "on"
                print(f"raw mode: {parts[1]}")
            else:
                print("用法: /raw on|off")
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
