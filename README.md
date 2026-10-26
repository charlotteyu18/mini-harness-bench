# mini-harness-bench

A from-scratch coding agent harness with built-in inference telemetry and a
tiny eval suite — built to study how harness design decisions affect both task
success and inference cost.

Motivation: I build production AI agents by day; this is my lab for going one
layer deeper into the harness itself — the loop, the tools, the context
management — and measuring what each choice costs in latency and tokens.

## What it does

- **Minimal ReAct harness** (`src/harness/`): system prompt, tool-calling loop
  (`read_file`, `write_file`, `run_shell`, `grep`), sandboxed per-task workdir,
  trajectory logging to JSONL.
- **Inference telemetry** (`src/harness/llm.py`, `telemetry.py`): per-call
  **TTFT**, **call latency**, **effective throughput (tok/s)** and token usage,
  measured over a streaming OpenAI-compatible API. Only honest, client-side
  measurable metrics are reported — per-token timings derived from chunk
  arrival times are deliberately excluded (the API batches many tokens per
  chunk, so those numbers measure network batching, not model speed).
- **Reliability** (`src/harness/llm.py`, `agent.py`): retries with exponential
  backoff + jitter on transient API failures, request timeouts, a circuit
  breaker after repeated consecutive failures, graceful degradation
  (failed runs record a `stop_reason` instead of crashing the eval), and
  optional per-task token budgets (wrap-up warning at 80%, hard stop at
  the cap).
- **Guardrails** (`src/harness/tools.py`): file tools are jailed to the task
  workdir; shell commands matching a destructive denylist (`rm -rf`,
  `sudo`, piping downloads into a shell, `mkfs`, raw device writes) are
  refused before execution, and a `readonly` permission mode disables
  writes and shell entirely. Refusals reach the agent as tool results and
  are counted in telemetry as `guardrail_blocks`.
- **Eval runner** (`src/eval/runner.py`): runs each task in an isolated temp
  dir, grades with the task's `verify` command, records pass/fail + metrics.
- **LLM-as-judge** (`src/harness/judge.py`): with `--judge`, each finished
  trajectory is also scored 0–5 on a rubric (solution correctness, tool
  efficiency, verification discipline, recovery) — pytest says *whether* it
  worked, the judge says *how well*.
- **Config experiments** (`configs/`): compare harness knobs head-to-head —
  e.g. `full` history vs `sliding_window` context trimming, and pure ReAct
  vs plan-then-act (`configs/plan.json`: one tool-less planning call, then
  execution against the plan; planning failures degrade back to ReAct).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY (any OpenAI-compatible endpoint)

# run the sample tasks with the baseline harness
python -m src.eval.runner --config configs/serial.json

# run again with sliding-window context, then compare
python -m src.eval.runner --config configs/sliding.json --out results/results_sliding.jsonl
cat results/results.jsonl results/results_sliding.jsonl > results/all.jsonl
python -m src.eval.report --results results/all.jsonl
```

### Offline mock mode (no API key)

`MODEL=mock` swaps the OpenAI client for a scripted one
(`src/harness/mock.py`) that streams canned tool-call plans for the sample
tasks through the exact same response-parsing and telemetry code path as the
real client. Use it to smoke-test the harness, the eval runner, and the
report without spending tokens:

```bash
MODEL=mock python -m src.eval.runner --config configs/serial.json --out results/mock_serial.jsonl
MODEL=mock python -m src.eval.runner --config configs/sliding.json --out results/mock_sliding.jsonl
python -m src.eval.report --results results/mock_serial.jsonl
```

The mock is deterministic — every sample task passes — so a green mock run
tells you the loop, tool sandbox, verify grading, and trajectory logging all
work; swap in your key (`MODEL=gpt-4o-mini`) when you want real model
behavior.

## Example results

Full-suite live runs (gpt-4o-mini, prompt v1, 8 tasks including one
multi-file task; judge = gpt-4o-mini, rubric average /5). Full-context ran
once (2026-10-20); sliding-window ran twice (2026-10-20 and 2026-10-21):

| config | tasks | pass rate | avg steps | avg TTFT (ms) | avg call latency (s) | effective throughput (tok/s) | avg tokens/task | judge avg |
|---|---|---|---|---|---|---|---|---|
| serial-full-context (run 1) | 8 | 88% | 5.5 | 935.1 | 1.28 | 29.7 | 4090 | 4.5 |
| serial-sliding-window (run 1) | 8 | 100% | 5.2 | 921.5 | 1.19 | 29.9 | 3150 | 4.8 |
| serial-sliding-window (run 2) | 8 | 100% | 5.9 | 892.1 | 1.27 | 27.5 | 3765 | 4.4 |

Reproduce with:

```bash
python -m src.eval.runner --tasks tasks/extended_tasks.json --config configs/serial.json --judge --out results/live-serial.jsonl
python -m src.eval.runner --tasks tasks/extended_tasks.json --config configs/sliding.json --judge --out results/live-sliding.jsonl
cat results/live-serial.jsonl results/live-sliding.jsonl > results/live-all.jsonl
python -m src.eval.report --results results/live-all.jsonl
```

Reading of these runs: sliding-window passed 8/8 in both runs and used
fewer tokens per task than full-context both times — but the margin moved
(3150 vs 4090 tokens in run 1, 3765 vs 4090 in run 2), and per-task
behavior swung widely. `count-errors` is the poster child: full-context
looped to the iteration cap and failed it (12 steps, 14.8k tokens, judge
1.5/5); sliding run 1 solved it cleanly (10 steps, judge 4.5); sliding
run 2 thrashed to the cap (12 steps, judge 1.0) yet still passed pytest —
pass/fail and trajectory quality are different axes, which is exactly why
the judge scores sit next to the test results. With n=8 and this much
run-to-run variance, treat every delta here as directional, not proven;
repeats and a bigger suite are needed before claiming a real tradeoff.

This eval also caught a real harness bug: the first sliding-window run
crashed mid-suite with an OpenAI 400 — trimming was slicing an
assistant(tool_calls) → tool(result) group in half and orphaning the tool
messages. Fixed (trimming now never splits a tool group) and the API-error
path hardened so a non-retryable error degrades to a per-task `llm_error`
instead of killing the whole run; the table above is the post-fix rerun.

## Metrics glossary

- **TTFT** — time to first chunk: how fast the model starts responding each turn.
- **Call latency** — full round-trip wall time per LLM call.
- **Effective throughput** — completion tokens / total call seconds: the speed
  the harness actually experienced (not derived from per-chunk arrival times).
- **Tokens/task** — the real cost driver for agentic workloads: every extra
  tool round-trip multiplies prompt tokens.
- **guardrail_blocks** — tool calls intercepted by guardrails (denylist or
  readonly mode) in a task.
- **stop_reason** — how a run ended: `done` | `max_iterations` | `llm_error` |
  `budget_exceeded` | `gave_up`.

## Roadmap

- [x] Honest telemetry (TTFT, call latency, effective throughput — no chunk-arrival artifacts)
- [x] Reliability: retries with backoff, timeouts, circuit breaker, graceful degradation
- [x] LLM-as-judge scoring with rubrics, alongside deterministic pytest grading
- [x] More tasks: multi-file refactors, longer horizons (8 total)
- [x] Prompt versioning: prompt v1/v2 as an A/B config dimension
- [x] Plan-then-act vs pure ReAct: plan-then-act implemented as a config dimension (live A/B numbers to come)
- [x] Token budgets per task (80% wrap-up warning, hard stop at the cap; `configs/budget.json`)
- [ ] Model tiering (cheap vs strong model routing)
- [x] Guardrails: shell denylist + readonly permission mode, blocks counted in telemetry
- [ ] CI: run the eval suite on every push via GitHub Actions
- [ ] Parallel subagent config (fan-out independent tool calls, measure latency win vs token cost)
- [ ] KV-cache-friendly prompt layout experiments
- [ ] Trajectory visualizer
