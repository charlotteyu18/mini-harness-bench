"""The agent loop: system prompt -> LLM -> tools -> observe -> repeat.

Three harness knobs are exposed for experiments:
  max_iterations:   cap on the plan/act/observe loop
  context_strategy: "full" keeps all history;
                    "sliding_window" trims old tool outputs to bound tokens
  mode:             "react" acts turn by turn;
                    "plan_then_act" makes one tool-less planning call first
                    and executes against the plan
  token_budget:     optional per-task cap on total (prompt + completion)
                    tokens; the agent gets a wrap-up warning at 80% and is
                    stopped when the budget is spent
"""

import json
import time
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

# Planning prompt for plan-then-act mode: a separate, tool-less call whose
# only job is to produce the plan that the acting loop then executes.
PLANNING_SYSTEM_PROMPT = """You are planning a coding task in a task directory.
Output a short numbered plan (at most 6 steps): what to read first, what to
change, and how to verify the fix. Be specific to the task. Output ONLY the
plan — no code, no tool calls."""

# Backwards-compatible alias: SYSTEM_PROMPT == v1.
SYSTEM_PROMPT = SYSTEM_PROMPT_V1


@dataclass
class HarnessConfig:
    name: str = "default"
    max_iterations: int = 12
    context_strategy: str = "full"  # "full" | "sliding_window" | "retrieved"
    mode: str = "react"  # "react" | "plan_then_act" — the strategy A/B dimension
    token_budget: int | None = None  # per-task token cap; None = unlimited
    permissions: str = "full"  # "full" | "readonly" — guardrail mode
    cheap_model: str | None = None  # tiering: start here, escalate to `model`
    tier_escalate_after_steps: int = 6  # escalate when step index reaches this
    retrieval_top_k: int = 3  # retrieved strategy: files injected up front
    retrieval_max_chars: int = 4000  # per-file cap on injected content
    retrieval_order: str = "score"  # "score" | "path" — path order is cache-friendly
    parallel_tools: bool = False  # run independent calls in a turn concurrently
    model: str | None = None
    window_messages: int = 8  # kept messages for sliding_window
    system_prompt: str = "v1"  # "v1" | "v2" — the prompt A/B dimension
    # reliability knobs (AI Fabrik JD: retry, timeout, graceful degradation)
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3
    llm_retry_backoff_s: float = 1.0


def _parallel_safe(tool_calls: list[dict]) -> bool:
    """True when a turn's calls may run concurrently without changing
    the outcome. Reads (read_file, grep) are pure. Writes are safe only
    when they target distinct files that no call in the same turn reads.
    Shell commands are never parallelized: they can interact through
    the filesystem in ways the harness cannot see.
    """
    if len(tool_calls) < 2:
        return False
    writes: set[str] = set()
    reads: set[str] = set()
    for tc in tool_calls:
        name, args = tc["name"], tc["arguments"]
        if name == "run_shell":
            return False
        if name == "write_file":
            path = args.get("path", "")
            if path in writes:
                return False
            writes.add(path)
        elif name == "read_file":
            reads.add(args.get("path", ""))
        elif name != "grep":
            return False  # unknown tool: stay sequential
    return not (writes & reads)


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
        # Model tiering: one client, model switched between calls. The
        # strong model is whatever the client resolved to (config model,
        # MODEL env, or the default); tiering starts on the cheap model.
        self._strong_model = self.llm.model
        self._tier_escalated = False
        if config.cheap_model:
            self.llm.model = config.cheap_model

    def run(self, task_prompt: str, workdir: Path, traj_path: Path) -> dict:
        belt = Toolbelt(workdir, permissions=self.config.permissions)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        # RAG-style file context: retrieve the most relevant files up
        # front and inject them into the task message, so the agent
        # starts with the modules it needs instead of discovering them
        # one read_file call at a time. (Mock-visible marker: the mock
        # client shortens its plan when this block is present, the way
        # a real model with the context in hand skips those reads.)
        retrieved_files: list[str] = []
        if self.config.context_strategy == "retrieved":
            from .retrieval import retrieve_files

            retrieved = retrieve_files(
                workdir,
                task_prompt,
                top_k=self.config.retrieval_top_k,
                max_chars=self.config.retrieval_max_chars,
                order=self.config.retrieval_order,
            )
            retrieved_files = [r["path"] for r in retrieved]
            if retrieved:
                block = "\n\n## Retrieved context\nThe following files look most relevant to the task:\n"
                for r in retrieved:
                    block += f"\n### {r['path']}\n```\n{r['content']}\n```\n"
                messages[1]["content"] = task_prompt + block
        steps: list[dict] = []
        done = False
        stop_reason = "max_iterations"
        spent_tokens = 0
        budget_warned = False
        tool_error_seen = False

        if self.config.mode == "plan_then_act":
            # messages[1] already carries the retrieved block when the
            # retrieved strategy is on, so the planner sees the same
            # context the actor will.
            plan_step = self._make_plan(messages[1]["content"])
            steps.append(plan_step)
            pm = plan_step["llm_metrics"]
            spent_tokens += pm.get("prompt_tokens", 0) + pm.get("completion_tokens", 0)
            if plan_step.get("plan_text"):
                messages.append(
                    {"role": "assistant", "content": f"Plan:\n{plan_step['plan_text']}"}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Execute your plan step by step, adapting "
                        "if the code disagrees with it. Verify with the tests "
                        "before replying DONE.",
                    }
                )

        for i in range(self.config.max_iterations):
            if (
                self.config.token_budget is not None
                and spent_tokens >= self.config.token_budget
            ):
                # hard stop BEFORE the next call: overshoot is bounded
                # by the single call that crossed the line.
                stop_reason = "budget_exceeded"
                break
            if (
                self.config.cheap_model
                and not self._tier_escalated
                and (
                    i >= self.config.tier_escalate_after_steps
                    or tool_error_seen
                )
            ):
                # One-way escalation: cheap model struggled (error or
                # long horizon) — switch to the strong model and stay.
                self.llm.model = self._strong_model
                self._tier_escalated = True
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
            spent_tokens += metrics.prompt_tokens + metrics.completion_tokens
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
            tool_start = time.monotonic()
            if self.config.parallel_tools and _parallel_safe(tool_calls):
                # Independent calls run concurrently; executor.map
                # preserves input order, so the trajectory stays
                # deterministic and tool results still line up with
                # their declaring calls.
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=len(tool_calls)) as ex:
                    results = list(
                        ex.map(
                            lambda tc: belt.execute(tc["name"], tc["arguments"]),
                            tool_calls,
                        )
                    )
            else:
                results = [
                    belt.execute(tc["name"], tc["arguments"]) for tc in tool_calls
                ]
            step["tool_ms"] = round((time.monotonic() - tool_start) * 1000, 1)
            for tc, (result, blocked) in zip(tool_calls, results):
                entry = {"tool": tc["name"], "result": result}
                if blocked:
                    entry["guardrail_blocked"] = True
                if result.startswith("TOOL ERROR"):
                    tool_error_seen = True
                step["tool_results"].append(entry)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            steps.append(step)
            if (
                self.config.token_budget is not None
                and not budget_warned
                and spent_tokens >= 0.8 * self.config.token_budget
            ):
                # soft warning once, at 80%: give the model a chance to
                # land the plane before the hard stop.
                budget_warned = True
                messages.append(
                    {
                        "role": "user",
                        "content": f"[harness: {spent_tokens} of "
                        f"{self.config.token_budget} budget tokens used "
                        f"— wrap up: finish and verify now]",
                    }
                )
            self._apply_context_strategy(messages)

        traj_path.parent.mkdir(parents=True, exist_ok=True)
        with traj_path.open("w") as f:
            for s in steps:
                f.write(json.dumps(s) + "\n")
        return {
            "steps": steps,
            "done": done,
            "stop_reason": stop_reason,
            "tokens_spent": spent_tokens,
            "trajectory": str(traj_path),
            "retrieved_files": retrieved_files,
        }

    def _make_plan(self, task_prompt: str) -> dict:
        """One tool-less planning call. Its metrics ride along in the
        trajectory (flagged plan=True) so cost comparisons stay honest.
        On LLMError, degrade to plain ReAct instead of failing the task.
        Note: with sliding_window the plan sits at index 2 and can be
        trimmed on long tasks — pinning it is future work."""
        plan_messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]
        try:
            text, _tool_calls, metrics = self.llm.chat(plan_messages, None)
        except LLMError as e:
            return {
                "step": -1,
                "plan": True,
                "llm_metrics": CallMetrics().to_dict(),
                "assistant_text": "",
                "tool_calls": [],
                "tool_results": [],
                "error": f"planning failed, continuing without plan: {e}",
            }
        return {
            "step": -1,
            "plan": True,
            "plan_text": text,
            "llm_metrics": metrics.to_dict(),
            "assistant_text": text,
            "tool_calls": [],
            "tool_results": [],
        }

    def _apply_context_strategy(self, messages: list[dict]) -> None:
        # "full" and "retrieved" keep everything: retrieved manages
        # context by injecting relevant files up front, not by trimming.
        if self.config.context_strategy != "sliding_window":
            return
        # Drop any previous trim notice first: it lives at the end now
        # (see below), outside the trimmed range, so without this it
        # would accumulate one copy per trim.
        messages[:] = [
            m
            for m in messages
            if not (
                m["role"] == "system"
                and m["content"].startswith("[harness: trimmed")
            )
        ]
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
        # Append the notice at the END, not after the task prompt: a
        # varying notice early in the list would poison prefix-cache
        # matching for everything after it (its count changes every
        # trim); at the end it only costs itself.
        messages.append(
            {
                "role": "system",
                "content": f"[harness: trimmed {dropped} older messages "
                f"to bound context]",
            },
        )
