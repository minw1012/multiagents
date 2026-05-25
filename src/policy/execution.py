import ipaddress
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Tuple

from src.policy.risk import TOOL_RISK_BY_PERMISSION


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


class ExecutionPolicy:
    """
    Codex-inspired safety gate:
    - categorize tool calls by risk
    - enforce explicit approval for high-risk actions
    - return machine-readable decision for orchestrator
    """

    def __init__(self):
        self._risk_by_permission = dict(TOOL_RISK_BY_PERMISSION)
        self._trusted_domains = {
            "raw.githubusercontent.com",
            "github.com",
            "api.github.com",
            "localhost",
            "127.0.0.1",
        }
        env_domains = os.getenv("TRUSTED_NETWORK_DOMAINS", "").strip()
        if env_domains:
            for item in env_domains.split(","):
                dom = item.strip().lower()
                if dom:
                    self._trusted_domains.add(dom)

    def risk_level(self, permission: str) -> str:
        return self._risk_by_permission.get(permission, "medium")

    def _host_is_private_or_local(self, host: str) -> bool:
        host_l = (host or "").strip().lower()
        if not host_l:
            return True
        if host_l in {"localhost", "127.0.0.1", "::1"}:
            return True
        try:
            ip = ipaddress.ip_address(host_l)
            return bool(ip.is_private or ip.is_loopback or ip.is_link_local)
        except Exception:
            return False

    def _check_network_boundary(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        url = str(args.get("url", "")).strip()
        if not url:
            return True, "no_url"
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            return False, f"invalid URL for `{tool_name}`."
        scheme = (parsed.scheme or "").lower()
        host = (parsed.hostname or "").lower()
        if scheme not in {"http", "https"}:
            return False, f"Blocked `{tool_name}`: only http/https URLs are allowed."
        if scheme == "http" and host not in {"localhost", "127.0.0.1"}:
            return False, f"Blocked `{tool_name}`: non-local HTTP is not allowed; use HTTPS."
        if self._host_is_private_or_local(host) and host not in {"localhost", "127.0.0.1"}:
            return False, f"Blocked `{tool_name}`: private network host is outside trust boundary."
        if host and host not in self._trusted_domains:
            approved = parse_bool(args.get("approved"), default=False)
            if not approved:
                return (
                    False,
                    f"Host `{host}` is outside trusted domains. Add `approved=true` to allow this network access.",
                )
        return True, "network_allowed"

    def _check_filesystem_boundary(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        db_path = str(args.get("db_path", "")).strip()
        if not db_path:
            return True, "no_db_path"
        p = Path(db_path).expanduser()
        if p.is_absolute():
            blocked_prefixes = ["/etc", "/bin", "/sbin", "/usr/bin", "/System", "/private/etc"]
            if any(str(p).startswith(pref) for pref in blocked_prefixes):
                return False, f"Blocked `{tool_name}`: database path is outside allowed trust boundary."
        return True, "fs_allowed"

    def evaluate(self, tool_name: str, spec: Any, args: Dict[str, Any], task: str) -> Dict[str, Any]:
        risk = self.risk_level(spec.permission)
        approved = parse_bool(args.get("approved"), default=False)
        task_l = task.lower()

        if spec.permission in {"network_read", "network_write"}:
            ok, reason = self._check_network_boundary(tool_name=tool_name, args=args)
            if not ok:
                return {"allow": False, "risk": risk, "reason": reason}
        if spec.permission in {"db_read", "db_write"}:
            ok, reason = self._check_filesystem_boundary(tool_name=tool_name, args=args)
            if not ok:
                return {"allow": False, "risk": risk, "reason": reason}

        if not approved and risk == "high":
            approved = ("approve" in task_l and tool_name.lower() in task_l) or ("approved" in task_l)

        if risk == "high" and not approved:
            return {
                "allow": False,
                "risk": risk,
                "reason": (
                    f"Tool `{tool_name}` is high risk ({spec.permission}). "
                    "Add `approved=true` for this tool call or explicitly approve in your request."
                ),
            }

        return {
            "allow": True,
            "risk": risk,
            "reason": "allowed",
        }
