"""Render a trajectory JSONL file as a self-contained HTML page.

Usage:
    python -m src.eval.visualize --trajectory trajectories/fib-off-by-one.jsonl
    python -m src.eval.visualize --trajectory trajectories/fib-off-by-one.jsonl --out /tmp/fib.html

The runner writes one trajectory per task to
trajectories/<config-name>/<task-id>.jsonl. The page shows the run
summary (steps, tokens, TTFT, guardrail blocks, models used, stop reason)
and every step: assistant text, tool calls with arguments, and tool
results tucked into <details> so long outputs stay scannable. No
JavaScript, no external assets — the file works opened straight from
disk, which also makes it safe to share.
"""

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.telemetry import summarize_trajectory  # noqa: E402

MAX_RESULT_CHARS = 2000
MAX_TEXT_CHARS = 4000


def _trunc(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} more chars]"


def _badges(step: dict) -> str:
    badges = []
    if step.get("plan"):
        badges.append('<span class="badge plan">plan</span>')
    if step.get("error"):
        badges.append('<span class="badge error">error</span>')
    if any(r.get("guardrail_blocked") for r in step.get("tool_results", [])):
        badges.append('<span class="badge blocked">guardrail blocked</span>')
    return " ".join(badges)


def _render_step(step: dict) -> str:
    m = step.get("llm_metrics", {})
    metrics = (
        f"model {html.escape(str(m.get('model', '')) or '—')} · "
        f"TTFT {m.get('ttft_ms', 0):.0f}ms · "
        f"latency {m.get('total_ms', 0):.0f}ms · "
        f"tokens {m.get('prompt_tokens', 0)}+{m.get('completion_tokens', 0)}"
    )
    if step.get("tool_ms") is not None:
        metrics += f" · tools {step['tool_ms']:.0f}ms"
    parts = [
        f"<section><h2>Step {step.get('step')} {_badges(step)}</h2>",
        f"<p class='metrics'>{metrics}</p>",
    ]
    if step.get("assistant_text"):
        parts.append(
            f"<pre>{html.escape(_trunc(step['assistant_text'], MAX_TEXT_CHARS))}</pre>"
        )
    if step.get("error"):
        parts.append(f"<p class='err'>{html.escape(step['error'])}</p>")
    for tc, tr in zip(step.get("tool_calls", []), step.get("tool_results", [])):
        blocked = " 🔒" if tr.get("guardrail_blocked") else ""
        parts.append(
            f"<details><summary>{html.escape(tc['name'])}{blocked} "
            f"<code>{html.escape(json.dumps(tc.get('arguments', {})))}</code></summary>"
            f"<pre>{html.escape(_trunc(str(tr.get('result', '')), MAX_RESULT_CHARS))}</pre>"
            f"</details>"
        )
    parts.append("</section>")
    return "\n".join(parts)


def render(steps: list[dict], title: str) -> str:
    summary = summarize_trajectory(steps)
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
    )
    body = "\n".join(_render_step(s) for s in steps)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#222}}
section{{border-top:1px solid #ddd;padding:1em 0}}
pre{{background:#f6f6f6;padding:0.8em;overflow-x:auto;white-space:pre-wrap}}
.metrics{{color:#555}} .err{{color:#a00}}
.badge{{font-size:0.8em;padding:0.1em 0.5em;border-radius:1em;color:#fff}}
.plan{{background:#3a7}} .error{{background:#a33}} .blocked{{background:#c80}}
details{{margin:0.5em 0}} summary{{cursor:pointer}}
table{{border-collapse:collapse}} td{{border:1px solid #ddd;padding:0.3em 0.8em}}
</style></head>
<body><h1>{html.escape(title)}</h1>
<table>{rows}</table>
{body}
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a trajectory JSONL as HTML.")
    ap.add_argument("--trajectory", required=True, help="trajectory .jsonl file")
    ap.add_argument("--out", help="output .html file (default: same name, .html)")
    args = ap.parse_args()
    traj = Path(args.trajectory)
    if not traj.exists():
        ap.error(f"no such file: {traj}")
    steps = [
        json.loads(line) for line in traj.read_text().splitlines() if line.strip()
    ]
    out = Path(args.out) if args.out else traj.with_suffix(".html")
    out.write_text(render(steps, traj.stem))
    print(f"wrote {out} ({len(steps)} steps)")


if __name__ == "__main__":
    main()
