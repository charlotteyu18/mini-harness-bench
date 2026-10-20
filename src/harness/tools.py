"""Toolbelt: the tools the agent can call, sandboxed to a task workdir."""

import subprocess
from pathlib import Path


class Toolbelt:
    def __init__(self, workdir: Path):
        self.workdir = workdir.resolve()

    # -- path safety -----------------------------------------------------
    def _resolve(self, path: str) -> Path:
        p = (self.workdir / path).resolve()
        if p != self.workdir and self.workdir not in p.parents:
            raise ValueError(f"path escapes workdir: {path}")
        return p

    # -- tools -----------------------------------------------------------
    def read_file(self, path: str) -> str:
        return self._resolve(path).read_text()

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {path} ({len(content)} chars)"

    def run_shell(self, command: str, timeout: int = 60) -> str:
        try:
            r = subprocess.run(
                command, shell=True, cwd=self.workdir,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {timeout}s: {command}"
        out = (r.stdout + r.stderr).strip()
        if len(out) > 4000:
            out = out[:4000] + "\n...[truncated]"
        return f"exit={r.returncode}\n{out}"

    def grep(self, pattern: str, path: str = ".") -> str:
        r = subprocess.run(
            ["grep", "-rn", pattern, str(self._resolve(path))],
            capture_output=True, text=True, timeout=15,
        )
        out = r.stdout.strip()[:3000]
        return out or "(no matches)"

    # -- dispatch --------------------------------------------------------
    def execute(self, name: str, arguments: dict) -> str:
        try:
            if name == "read_file":
                return self.read_file(arguments["path"])
            if name == "write_file":
                return self.write_file(arguments["path"], arguments["content"])
            if name == "run_shell":
                return self.run_shell(
                    arguments["command"], int(arguments.get("timeout", 60))
                )
            if name == "grep":
                return self.grep(arguments["pattern"], arguments.get("path", "."))
            return f"unknown tool: {name}"
        except Exception as e:  # never let a tool crash the loop
            return f"TOOL ERROR: {type(e).__name__}: {e}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the task workdir.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a file in the task workdir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the task workdir (e.g. pytest).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files under path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
]
