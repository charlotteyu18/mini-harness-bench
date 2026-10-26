"""Toolbelt: the tools the agent can call, sandboxed to a task workdir.

Two guardrail layers (Phase 4):
  1. Path jail: read/write/grep can never escape the workdir (_resolve).
  2. Shell screening: run_shell commands matching the denylist below are
     refused before execution, and in "readonly" permission mode write_file
     and run_shell are refused entirely. A refusal is returned to the agent
     as an ordinary tool result (so the model can adapt) and flagged
     guardrail_blocked in the trajectory, so interventions are countable.

Honest limit: this is a denylist safety net, not a sandbox. The shell
still runs as the user, and a sufficiently creative command can dodge a
pattern list. A true sandbox would need containers or a syscall filter.
"""

import re
import subprocess
from pathlib import Path

# (pattern, reason) — each pattern is a destructive shape, not a word list.
SHELL_DENYLIST: list[tuple[str, str]] = [
    (r"\brm\s+-(?=[a-z]*r)(?=[a-z]*f)[a-z]+\b",
     "recursive + force delete"),
    (r"\bsudo\b", "privilege escalation"),
    (r"\|\s*(sh|bash|zsh|dash)\b", "piping downloaded content into a shell"),
    (r"\bmkfs\b", "filesystem formatting"),
    (r"\bdd\b.*\bof=/dev/", "raw device write"),
    (r">\s*/dev/(sd|hd|nvme|vd)[a-z0-9]*", "redirect into a disk device"),
]
SHELL_DENYLIST_COMPILED = [(re.compile(p), reason) for p, reason in SHELL_DENYLIST]


class Toolbelt:
    def __init__(self, workdir: Path, permissions: str = "full"):
        # permissions: "full" (default) | "readonly" (read + grep only)
        self.workdir = workdir.resolve()
        self.permissions = permissions

    # -- guardrails ------------------------------------------------------
    def _shell_refusal(self, command: str) -> str | None:
        """Reason string if the command must not run, else None."""
        for pattern, reason in SHELL_DENYLIST_COMPILED:
            if pattern.search(command):
                return f"denylist match ({reason})"
        return None

    def _refused(self, reason: str) -> tuple[str, bool]:
        return f"[guardrail blocked: {reason}]", True

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
    # Returns (result, blocked): blocked is True when a guardrail refused
    # the call before execution. The agent records the flag so blocks are
    # countable in telemetry; the model sees the refusal as a tool result.
    def execute(self, name: str, arguments: dict) -> tuple[str, bool]:
        try:
            if name == "write_file" and self.permissions == "readonly":
                return self._refused("readonly mode: writes disabled")
            if name == "run_shell" and self.permissions == "readonly":
                return self._refused("readonly mode: shell disabled")
            if name == "read_file":
                return self.read_file(arguments["path"]), False
            if name == "write_file":
                return self.write_file(arguments["path"], arguments["content"]), False
            if name == "run_shell":
                command = arguments["command"]
                reason = self._shell_refusal(command)
                if reason:
                    return self._refused(reason)
                return self.run_shell(
                    command, int(arguments.get("timeout", 60))
                ), False
            if name == "grep":
                return self.grep(arguments["pattern"], arguments.get("path", ".")), False
            return f"unknown tool: {name}", False
        except Exception as e:  # never let a tool crash the loop
            return f"TOOL ERROR: {type(e).__name__}: {e}", False


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
