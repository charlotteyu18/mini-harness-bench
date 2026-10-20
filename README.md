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
  **TTFT**, **ITL (p50/p95)**, **throughput (tok/s)** and token usage, measured
  over a streaming OpenAI-compatible API.
- **Eval runner** (`src/eval/runner.py`): runs each task in an isolated temp
  dir, grades with the task's `verify` command, records pass/fail + metrics.
- **Config experiments** (`configs/`): compare harness knobs head-to-head —
  e.g. `full` history vs `sliding_window` context trimming.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY (any OpenAI-compatible endpoint)

# run the sample tasks with the baseline harness
python src/eval/runner.py --config configs/serial.json

# run again with sliding-window context, then compare
python src/eval/runner.py --config configs/sliding.json --out results/results_sliding.json
cat results/results.jsonl results/results_sliding.json > results/all.jsonl
python src/eval/report.py --results results/all.jsonl
```

## Example results

| config | tasks | pass rate | avg steps | avg TTFT (ms) | median ITL (ms) | avg throughput (tok/s) | avg tokens/task |
|---|---|---|---|---|---|---|---|
| serial-full-context | 3 | 100% | 5.0 | 412 | 18.4 | 54.2 | 8200 |
| serial-sliding-window | 3 | 100% | 5.3 | 398 | 17.9 | 55.1 | 6100 |

*(numbers from a sample run with gpt-4o-mini — run it yourself, your mileage will vary)*

Takeaway pattern: sliding-window trimming cuts ~25% of tokens per task with no
pass-rate drop on short tasks — the interesting question is where it starts to
hurt on long-horizon ones.

## Metrics glossary

- **TTFT** — time to first token: how fast the model starts responding each turn.
- **ITL** — inter-token latency: streaming smoothness within a turn.
- **Throughput** — completion tokens per generation-second.
- **Tokens/task** — the real cost driver for agentic workloads: every extra
  tool round-trip multiplies prompt tokens.

## Roadmap

- [ ] Parallel subagent config (fan-out independent tool calls, measure latency win vs token cost)
- [ ] More tasks: multi-file refactors, longer horizons
- [ ] KV-cache-friendly prompt layout experiments
- [ ] Trajectory visualizer
