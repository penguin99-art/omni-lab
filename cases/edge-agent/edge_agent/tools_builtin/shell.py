"""Shell tool with safety filtering."""

from __future__ import annotations

import subprocess

from edge_agent.tools import tool

BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    ":(){ :|:& };:",
    "> /dev/sd",
    "chmod -R 777 /",
    "mv /* ",
]


@tool("Execute a shell command with safety guards")
def shell(command: str) -> str:
    cmd_lower = command.lower().strip()
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return f"Blocked: dangerous pattern '{pattern}'"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if not output:
            output = f"(exit code: {result.returncode})"
        if len(output) > 10_000:
            output = output[:10_000] + "\n... (truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out (30s limit)"
    except Exception as e:
        return f"Execution failed: {e}"
