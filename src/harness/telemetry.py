"""Aggregate per-step telemetry into trajectory / run level summaries."""

import statistics


def summarize_trajectory(steps: list[dict]) -> dict:
    ttfts, latencies, tps, prompt_toks, comp_toks = [], [], [], 0, 0
    tool_calls = 0
    guardrail_blocks = 0
    errors = 0
    for s in steps:
        m = s["llm_metrics"]
        if s.get("error"):
            errors += 1
            continue  # error steps carry zeroed metrics; don't skew averages
        if m.get("ttft_ms"):
            ttfts.append(m["ttft_ms"])
        if m.get("total_ms"):
            latencies.append(m["total_ms"] / 1000)
        if m.get("throughput_tps"):
            tps.append(m["throughput_tps"])
        prompt_toks += m.get("prompt_tokens", 0)
        comp_toks += m.get("completion_tokens", 0)
        tool_calls += len(s["tool_calls"])
        guardrail_blocks += sum(
            1 for r in s["tool_results"] if r.get("guardrail_blocked")
        )
    return {
        "steps": len(steps),
        "tool_calls": tool_calls,
        "guardrail_blocks": guardrail_blocks,
        "llm_errors": errors,
        "avg_ttft_ms": round(statistics.mean(ttfts), 1) if ttfts else 0,
        "avg_call_latency_s": round(statistics.mean(latencies), 2) if latencies else 0,
        "avg_throughput_tps": round(statistics.mean(tps), 1) if tps else 0,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "total_tokens": prompt_toks + comp_toks,
    }


def summarize_results(results: list[dict]) -> dict:
    """Group per-task results by config name into a comparison table."""
    by_config: dict[str, list[dict]] = {}
    for r in results:
        by_config.setdefault(r["config"], []).append(r)
    table = []
    for config, rs in by_config.items():
        passed = sum(1 for r in rs if r["passed"])
        table.append(
            {
                "config": config,
                "tasks": len(rs),
                "pass_rate": round(passed / len(rs), 3),
                "avg_steps": round(statistics.mean([r["metrics"]["steps"] for r in rs]), 1),
                "avg_ttft_ms": round(statistics.mean([r["metrics"]["avg_ttft_ms"] for r in rs]), 1),
                "avg_call_latency_s": round(statistics.mean([r["metrics"]["avg_call_latency_s"] for r in rs]), 2),
                "avg_throughput_tps": round(statistics.mean([r["metrics"]["avg_throughput_tps"] for r in rs]), 1),
                "avg_tokens_per_task": round(statistics.mean([r["metrics"]["total_tokens"] for r in rs])),
            }
        )
    return {"table": table}


def to_markdown(summary: dict) -> str:
    lines = [
        "| config | tasks | pass rate | avg steps | avg TTFT (ms) | avg call latency (s) | effective throughput (tok/s) | avg tokens/task |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["table"]:
        lines.append(
            f"| {row['config']} | {row['tasks']} | {row['pass_rate']:.0%} "
            f"| {row['avg_steps']} | {row['avg_ttft_ms']} | {row['avg_call_latency_s']} "
            f"| {row['avg_throughput_tps']} | {row['avg_tokens_per_task']} |"
        )
    return "\n".join(lines)
