from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.policy.risk import TOOL_RISK_BY_PERMISSION

ToolFunc = Callable[..., Dict[str, Any]]


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

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        return self.tools.get(name)

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
                    "risk_level": TOOL_RISK_BY_PERMISSION.get(spec.permission, "medium"),
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
                    "risk_level": TOOL_RISK_BY_PERMISSION.get(spec.permission, "medium"),
                }
            )
        return rows
