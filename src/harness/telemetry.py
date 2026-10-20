"""Aggregate per-step telemetry into trajectory / run level summaries."""

import statistics


def summarize_trajectory(steps: list[dict]) -> dict:
    ttfts, itls, tps, prompt_toks, comp_toks = [], [], [], 0, 0
    tool_calls = 0
    for s in steps:
        m = s["llm_metrics"]
        ttfts.append(m["ttft_ms"])
        itls.append(m["itl_ms_p50"])
        if m["throughput_tps"]:
            tps.append(m["throughput_tps"])
        prompt_toks += m["prompt_tokens"]
        comp_toks += m["completion_tokens"]
        tool_calls += len(s["tool_calls"])
    return {
        "steps": len(steps),
        "tool_calls": tool_calls,
        "avg_ttft_ms": round(statistics.mean(ttfts), 1) if ttfts else 0,
        "median_itl_ms": round(statistics.median(itls), 2) if itls else 0,
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
                "median_itl_ms": round(statistics.mean([r["metrics"]["median_itl_ms"] for r in rs]), 2),
                "avg_throughput_tps": round(statistics.mean([r["metrics"]["avg_throughput_tps"] for r in rs]), 1),
                "avg_tokens_per_task": round(statistics.mean([r["metrics"]["total_tokens"] for r in rs])),
            }
        )
    return {"table": table}


def to_markdown(summary: dict) -> str:
    lines = [
        "| config | tasks | pass rate | avg steps | avg TTFT (ms) | median ITL (ms) | avg throughput (tok/s) | avg tokens/task |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["table"]:
        lines.append(
            f"| {row['config']} | {row['tasks']} | {row['pass_rate']:.0%} "
            f"| {row['avg_steps']} | {row['avg_ttft_ms']} | {row['median_itl_ms']} "
            f"| {row['avg_throughput_tps']} | {row['avg_tokens_per_task']} |"
        )
    return "\n".join(lines)
