"""The agent loop: system prompt -> LLM -> tools -> observe -> repeat.

Two harness knobs are exposed for experiments:
  max_iterations:   cap on the plan/act/observe loop
  context_strategy: "full" keeps all history;
                    "sliding_window" trims old tool outputs to bound tokens
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .llm import CallMetrics, LLMClient, LLMError
from .mock import MockLLMClient
from .tools import TOOL_SCHEMAS, Toolbelt

SYSTEM_PROMPT_V1 = """You are a coding agent working in a task directory.
You have tools: read_file, write_file, run_shell, grep.
Think step by step, use tools to inspect and change code, and VERIFY with
run_shell (e.g. run the tests) before finishing.
When the task is complete and verified, reply with exactly: DONE
Do not ask questions. Keep tool use focused and minimal."""

# v2: hypothesis-first variant. The A/B dimension for the prompt experiment
# (Virio JD: "A/B test prompt variations, measure their impact").
SYSTEM_PROMPT_V2 = """You are a coding agent working in a task directory.
You have tools: read_file, write_file, run_shell, grep.

Work in this order:
1. READ the code and the tests first. State your fix hypothesis in ONE
   sentence before touching anything.
2. Make the SMALLEST change that fixes the bug.
3. RE-RUN the tests after every edit and read the output.
4. Only reply exactly DONE after you have seen the tests pass yourself.

Do not ask questions. Keep tool use focused and minimal."""

SYSTEM_PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2}

# Backwards-compatible alias: SYSTEM_PROMPT == v1.
SYSTEM_PROMPT = SYSTEM_PROMPT_V1


@dataclass
class HarnessConfig:
    name: str = "default"
    max_iterations: int = 12
    context_strategy: str = "full"  # "full" | "sliding_window"
    model: str | None = None
    window_messages: int = 8  # kept messages for sliding_window
    system_prompt: str = "v1"  # "v1" | "v2" — the prompt A/B dimension
    # reliability knobs (AI Fabrik JD: retry, timeout, graceful degradation)
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3
    llm_retry_backoff_s: float = 1.0


class Agent:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.system_prompt = SYSTEM_PROMPTS.get(config.system_prompt, SYSTEM_PROMPT_V1)
        if (config.model or "").lower() == "mock":
            # offline mode: scripted plans, no API key needed
            self.llm = MockLLMClient(model=config.model)
        else:
            self.llm = LLMClient(
                model=config.model,
                timeout_s=config.llm_timeout_s,
                max_retries=config.llm_max_retries,
                retry_backoff_s=config.llm_retry_backoff_s,
            )

    def run(self, task_prompt: str, workdir: Path, traj_path: Path) -> dict:
        belt = Toolbelt(workdir)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        steps: list[dict] = []
        done = False
        stop_reason = "max_iterations"

        for i in range(self.config.max_iterations):
            try:
                text, tool_calls, metrics = self.llm.chat(messages, TOOL_SCHEMAS)
            except LLMError as e:
                # graceful degradation: record the failure, keep the
                # trajectory, stop instead of crashing the whole eval.
                steps.append(
                    {
                        "step": i,
                        "llm_metrics": CallMetrics().to_dict(),
                        "assistant_text": "",
                        "tool_calls": [],
                        "tool_results": [],
                        "error": f"LLMError: {e}",
                    }
                )
                stop_reason = "llm_error"
                break
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
                stop_reason = "done" if done else "gave_up"
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
        return {
            "steps": steps,
            "done": done,
            "stop_reason": stop_reason,
            "trajectory": str(traj_path),
        }

    def _apply_context_strategy(self, messages: list[dict]) -> None:
        if self.config.context_strategy != "sliding_window":
            return
        # keep system + original task prompt + the most recent window
        if len(messages) <= 2 + self.config.window_messages:
            return
        cut = len(messages) - self.config.window_messages
        # Never split an assistant(tool_calls) -> tool(result) group:
        # if the window would start with tool results, drop them too.
        # The API rejects tool messages whose declaring assistant
        # message was trimmed away ("messages with role 'tool' must
        # be a response to a preceeding message with 'tool_calls'").
        while cut < len(messages) and messages[cut]["role"] == "tool":
            cut += 1
        dropped = cut - 2
        del messages[2:cut]
        messages.insert(
            2,
            {
                "role": "system",
                "content": f"[harness: trimmed {dropped} older messages "
                f"to bound context]",
            },
        )
