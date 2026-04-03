"""Tool definitions and execution for the workstation agent."""

import subprocess
from pathlib import Path
from datetime import date

BLOCKED_PATTERNS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=",
    "shutdown", "reboot", ":(){ :|:& };:",
    "> /dev/sd", "chmod -R 777 /",
]

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a shell command on this workstation. Use for: listing files, checking status, running scripts, git operations, system info, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file on this workstation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent dirs if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "Save an important fact to persistent memory. Use when the user mentions something worth remembering across sessions: preferences, decisions, project context, people, deadlines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_today",
            "description": "Update today's plan/journal. Use to add tasks, mark progress, or append notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to append to today's file"},
                },
                "required": ["content"],
            },
        },
    },
]

MAX_OUTPUT = 8000


def execute(name: str, args: dict, *, memory_file: Path, today_file: Path) -> str:
    """Execute a tool by name and return the result string."""
    try:
        if name == "shell":
            cmd = args["command"]
            if any(p in cmd.lower() for p in BLOCKED_PATTERNS):
                return "Blocked: dangerous command"
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
            )
            out = (r.stdout + r.stderr).strip()
            return out[:MAX_OUTPUT] if out else f"(exit {r.returncode})"

        if name == "read_file":
            p = Path(args["path"]).expanduser()
            if not p.exists():
                return f"File not found: {p}"
            return p.read_text(encoding="utf-8")[:MAX_OUTPUT]

        if name == "write_file":
            p = Path(args["path"]).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"], encoding="utf-8")
            return f"Written: {p}"

        if name == "memory_save":
            with open(memory_file, "a", encoding="utf-8") as f:
                f.write(f"\n- [{date.today().isoformat()}] {args['fact']}")
            return f"Remembered: {args['fact']}"

        if name == "update_today":
            with open(today_file, "a", encoding="utf-8") as f:
                f.write(f"\n{args['content']}")
            return "Updated today's file."

        return f"Unknown tool: {name}"
    except subprocess.TimeoutExpired:
        return "Command timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"
