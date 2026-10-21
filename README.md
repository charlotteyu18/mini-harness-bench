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
  breaker after repeated consecutive failures, and graceful degradation
  (failed runs record a `stop_reason` instead of crashing the eval).
- **Eval runner** (`src/eval/runner.py`): runs each task in an isolated temp
  dir, grades with the task's `verify` command, records pass/fail + metrics.
- **Config experiments** (`configs/`): compare harness knobs head-to-head —
  e.g. `full` history vs `sliding_window` context trimming.

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

`python -m src.eval.report --results results/all.jsonl` prints a comparison
table like this:

| config | tasks | pass rate | avg steps | avg TTFT (ms) | avg call latency (s) | effective throughput (tok/s) | avg tokens/task |
|---|---|---|---|---|---|---|---|

Early runs (gpt-4o-mini, 3/3 pass on both configs) show a latency-vs-tokens
tradeoff between full-history and sliding-window context — but with n=3 tasks
the run-to-run variance is large, so treat single-run deltas as directional,
not conclusive. Growing the task suite is on the roadmap for exactly this
reason. Run it yourself; your mileage will vary.

## Metrics glossary

- **TTFT** — time to first chunk: how fast the model starts responding each turn.
- **Call latency** — full round-trip wall time per LLM call.
- **Effective throughput** — completion tokens / total call seconds: the speed
  the harness actually experienced (not derived from per-chunk arrival times).
- **Tokens/task** — the real cost driver for agentic workloads: every extra
  tool round-trip multiplies prompt tokens.
- **stop_reason** — how a run ended: `done` | `max_iterations` | `llm_error` |
  `gave_up`.

## Roadmap

- [x] Honest telemetry (TTFT, call latency, effective throughput — no chunk-arrival artifacts)
- [x] Reliability: retries with backoff, timeouts, circuit breaker, graceful degradation
- [ ] LLM-as-judge scoring with rubrics, alongside deterministic pytest grading
- [ ] More tasks: multi-file refactors, longer horizons (8–10 total)
- [ ] Prompt versioning: prompt v1/v2 as an A/B config dimension
- [ ] Plan-then-act vs pure ReAct experiment
- [ ] Token budgets per task; model tiering (cheap vs strong model routing)
- [ ] Guardrails: tool permission boundaries, destructive-command interception
- [ ] CI: run the eval suite on every push via GitHub Actions
- [ ] Parallel subagent config (fan-out independent tool calls, measure latency win vs token cost)
- [ ] KV-cache-friendly prompt layout experiments
- [ ] Trajectory visualizer
