"""The agent loop: system prompt -> LLM -> tools -> observe -> repeat.

Two harness knobs are exposed for experiments:
  max_iterations:   cap on the plan/act/observe loop
  context_strategy: "full" keeps all history;
                    "sliding_window" trims old tool outputs to bound tokens
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .llm import LLMClient
from .tools import TOOL_SCHEMAS, Toolbelt

SYSTEM_PROMPT = """You are a coding agent working in a task directory.
You have tools: read_file, write_file, run_shell, grep.
Think step by step, use tools to inspect and change code, and VERIFY with
run_shell (e.g. run the tests) before finishing.
When the task is complete and verified, reply with exactly: DONE
Do not ask questions. Keep tool use focused and minimal."""


@dataclass
class HarnessConfig:
    name: str = "default"
    max_iterations: int = 12
    context_strategy: str = "full"  # "full" | "sliding_window"
    model: str | None = None
    window_messages: int = 8  # kept messages for sliding_window


class Agent:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.llm = LLMClient(model=config.model)

    def run(self, task_prompt: str, workdir: Path, traj_path: Path) -> dict:
        belt = Toolbelt(workdir)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]
        steps: list[dict] = []
        done = False

        for i in range(self.config.max_iterations):
            text, tool_calls, metrics = self.llm.chat(messages, TOOL_SCHEMAS)
            step = {
                "step": i,
                "llm_metrics": metrics.to_dict(),
                "assistant_text": text,
                "tool_calls": [
                    {"name": tc["name"], "arguments": tc["arguments"]}
                    for tc in tool_calls
                ],
                "tool_results": [],
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": text or "",
                    # the API requires tool messages to follow an assistant
                    # message that declares the tool_calls
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ]
                    or None,
                }
            )
            if not tool_calls:
                done = "DONE" in text.upper()
                steps.append(step)
                break
            for tc in tool_calls:
                result = belt.execute(tc["name"], tc["arguments"])
                step["tool_results"].append(
                    {"tool": tc["name"], "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            steps.append(step)
            self._apply_context_strategy(messages)

        traj_path.parent.mkdir(parents=True, exist_ok=True)
        with traj_path.open("w") as f:
            for s in steps:
                f.write(json.dumps(s) + "\n")
        return {"steps": steps, "done": done, "trajectory": str(traj_path)}

    def _apply_context_strategy(self, messages: list[dict]) -> None:
        if self.config.context_strategy != "sliding_window":
            return
        # keep system + original task prompt + the most recent window
        keep = 2 + self.config.window_messages
        if len(messages) > keep:
            dropped = len(messages) - keep
            del messages[2 : 2 + dropped]
            messages.insert(
                2,
                {
                    "role": "system",
                    "content": f"[harness: trimmed {dropped} older messages "
                    f"to bound context]",
                },
            )
