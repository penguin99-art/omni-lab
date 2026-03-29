"""ToolRegistry: auto-generates OpenAI function-calling schemas from decorated Python functions."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Callable, get_type_hints

log = logging.getLogger(__name__)

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(description: str):
    """Decorator: register a plain function as an LLM-callable tool."""
    def decorator(func: Callable) -> Callable:
        func._tool_description = description  # type: ignore[attr-defined]
        return func
    return decorator


class ToolRegistry:

    def __init__(self, tools: list[Callable] | None = None) -> None:
        self._tools: dict[str, Callable] = {}
        for t in (tools or []):
            self.register(t)

    def register(self, func: Callable) -> None:
        self._tools[func.__name__] = func

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def definitions(self) -> list[dict]:
        """Generate OpenAI function-calling compatible tool definitions."""
        defs: list[dict] = []
        for name, func in self._tools.items():
            desc = getattr(func, "_tool_description", "") or func.__doc__ or name
            try:
                hints = get_type_hints(func)
            except Exception:
                hints = {}
            sig = inspect.signature(func)
            properties: dict[str, dict] = {}
            required: list[str] = []
            for pname, param in sig.parameters.items():
                ptype = hints.get(pname, str)
                json_type = _TYPE_MAP.get(ptype, "string")
                properties[pname] = {"type": json_type, "description": pname}
                if param.default is inspect.Parameter.empty:
                    required.append(pname)

            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return defs

    async def execute(self, name: str, args: dict) -> str:
        func = self._tools.get(name)
        if not func:
            return f"未知工具: {name}"
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = await asyncio.get_event_loop().run_in_executor(None, lambda: func(**args))
            return str(result)
        except Exception as e:
            log.exception("Tool %s execution failed", name)
            return f"工具执行失败: {e}"
