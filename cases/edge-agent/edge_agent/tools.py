"""Tool system inspired by claude-code's Tool.ts / tools.ts.

Key patterns adopted:
  - Tool as a rich protocol (not just a function): validation, permissions,
    concurrency flags, result size limits
  - tool() decorator preserved for backward compat, but now produces Tool objects
  - ToolPool with assembly, deny-rules, and parallel orchestration
  - partition_tool_calls for safe concurrent execution
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, get_type_hints

log = logging.getLogger(__name__)

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

MAX_RESULT_CHARS = 30_000


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    output: str = ""
    error: str = ""
    is_error: bool = False


# ---------------------------------------------------------------------------
# Tool protocol (inspired by claude-code Tool.ts)
# ---------------------------------------------------------------------------

class Tool(ABC):
    """Rich tool contract.  Every tool is an instance of this class."""

    name: str
    description: str

    @abstractmethod
    def parameters_schema(self) -> dict:
        """Return JSON Schema for the tool parameters."""
        ...

    async def validate_input(self, args: dict) -> str | None:
        """Return an error string if args are invalid, else None."""
        return None

    def is_read_only(self, args: dict | None = None) -> bool:
        return False

    def is_parallel_safe(self, args: dict | None = None) -> bool:
        return self.is_read_only(args)

    def is_enabled(self) -> bool:
        return True

    @abstractmethod
    async def execute(self, args: dict) -> ToolResult:
        ...

    def openai_schema(self) -> dict:
        """OpenAI function-calling compatible schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


# ---------------------------------------------------------------------------
# FunctionTool: wraps a decorated Python function as a Tool
# ---------------------------------------------------------------------------

class FunctionTool(Tool):
    """Wraps a plain @tool-decorated function into the Tool protocol."""

    def __init__(
        self,
        func: Callable,
        description: str = "",
        read_only: bool = False,
        parallel_safe: bool | None = None,
    ) -> None:
        self._func = func
        self.name = func.__name__
        self.description = description or getattr(func, "_tool_description", "") or func.__doc__ or self.name
        self._read_only = read_only
        self._parallel_safe = parallel_safe if parallel_safe is not None else read_only
        self._schema: dict | None = None

    def parameters_schema(self) -> dict:
        if self._schema is not None:
            return self._schema
        try:
            hints = get_type_hints(self._func)
        except Exception:
            hints = {}
        sig = inspect.signature(self._func)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            ptype = hints.get(pname, str)
            json_type = _TYPE_MAP.get(ptype, "string")
            properties[pname] = {"type": json_type, "description": pname}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        self._schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        return self._schema

    def is_read_only(self, args: dict | None = None) -> bool:
        return self._read_only

    def is_parallel_safe(self, args: dict | None = None) -> bool:
        return self._parallel_safe

    async def execute(self, args: dict) -> ToolResult:
        try:
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**args)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self._func(**args))
            text = str(result)
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + f"\n... (truncated, {len(str(result))} chars total)"
            return ToolResult(output=text)
        except Exception as e:
            log.exception("Tool %s execution failed", self.name)
            return ToolResult(error=f"工具执行失败: {e}", is_error=True)


# ---------------------------------------------------------------------------
# @tool decorator (backward compatible, now produces FunctionTool metadata)
# ---------------------------------------------------------------------------

def tool(description: str, *, read_only: bool = False, parallel_safe: bool | None = None):
    """Decorator: mark a function as an LLM-callable tool."""
    def decorator(func: Callable) -> Callable:
        func._tool_description = description
        func._tool_read_only = read_only
        func._tool_parallel_safe = parallel_safe
        return func
    return decorator


# ---------------------------------------------------------------------------
# ToolPool: assembly + orchestration (inspired by assembleToolPool / runTools)
# ---------------------------------------------------------------------------

class ToolPool:
    """Manages tools: schema generation, execution, and parallel orchestration."""

    def __init__(self, tools: list[Callable | Tool] | None = None, deny: list[str] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._deny = set(deny or [])
        for t in (tools or []):
            self.add(t)

    def add(self, t: Callable | Tool) -> None:
        if isinstance(t, Tool):
            tool_obj = t
        else:
            tool_obj = FunctionTool(
                t,
                read_only=getattr(t, "_tool_read_only", False),
                parallel_safe=getattr(t, "_tool_parallel_safe", None),
            )
        if tool_obj.name not in self._deny:
            self._tools[tool_obj.name] = tool_obj

    def register(self, t: Callable | Tool) -> None:
        """Backward-compatible alias used by existing tests and callers."""
        self.add(t)

    @property
    def names(self) -> list[str]:
        return [n for n, t in self._tools.items() if t.is_enabled()]

    def definitions(self) -> list[dict]:
        return [t.openai_schema() for t in self._tools.values() if t.is_enabled()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def execute(self, name: str, args: dict) -> str:
        """Execute a single tool by name. Returns result string."""
        t = self._tools.get(name)
        if not t:
            return f"未知工具: {name}"
        if not t.is_enabled():
            return f"工具已禁用: {name}"

        validation_error = await t.validate_input(args)
        if validation_error:
            return f"参数验证失败: {validation_error}"

        result = await t.execute(args)
        return result.error if result.is_error else result.output

    async def execute_batch(
        self,
        calls: list[dict],
        *,
        max_concurrency: int = 5,
    ) -> list[dict]:
        """Execute tool calls with smart batching.

        Consecutive read-only + parallel-safe tools run concurrently;
        mutating tools break the batch and run serially.
        Inspired by claude-code's partitionToolCalls + runToolsConcurrently.
        """
        batches = self._partition(calls)
        all_results: list[dict] = []

        for batch in batches:
            if len(batch) == 1:
                r = await self._run_one(batch[0])
                all_results.append(r)
            else:
                sem = asyncio.Semaphore(max_concurrency)
                async def run_with_sem(call):
                    async with sem:
                        return await self._run_one(call)
                results = await asyncio.gather(*(run_with_sem(c) for c in batch))
                all_results.extend(results)

        return all_results

    def _partition(self, calls: list[dict]) -> list[list[dict]]:
        """Partition tool calls into batches: parallel-safe groups vs serial."""
        batches: list[list[dict]] = []
        current_parallel: list[dict] = []

        for call in calls:
            fn_name = call.get("name", "")
            fn_args = call.get("args", {})
            t = self._tools.get(fn_name)

            if t and t.is_parallel_safe(fn_args):
                current_parallel.append(call)
            else:
                if current_parallel:
                    batches.append(current_parallel)
                    current_parallel = []
                batches.append([call])

        if current_parallel:
            batches.append(current_parallel)

        return batches if batches else [[c] for c in calls]

    async def _run_one(self, call: dict) -> dict:
        name = call.get("name", "")
        args = call.get("args", {})
        result_str = await self.execute(name, args)
        return {"tool": name, "args": args, "result": result_str[:2000]}


# ---------------------------------------------------------------------------
# Backward compat: ToolRegistry alias
# ---------------------------------------------------------------------------

ToolRegistry = ToolPool
