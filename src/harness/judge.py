"""LLM-as-judge: rubric scoring of an agent trajectory, alongside pytest pass/fail.

Deterministic tests (pytest) say *whether* the task was solved; the judge
says *how well* it was solved. Both scores are stored side by side in the
results file so the eval matrix can compare harness designs on quality, not
just pass/fail. This is the dimension the harness-engineer JDs ask for by
name (Virio: "LLM-as-judge evals"; AI Fabrik: "grading rubrics").

Usage: grade() is called once per task after the verify step, only when the
runner is invoked with --judge. It never raises: any failure (no API key,
bad response, unparseable JSON) returns {"error": ...} so a judge hiccup
can never crash an eval run.
"""

import json
import os

RUBRIC = {
    "solution_correctness": (
        "Did the trajectory actually fix/implement what the task asked, "
        "with tests run and passing before declaring done? (0 = no evidence "
        "of a working solution, 5 = clear fix + verified passing tests)"
    ),
    "tool_efficiency": (
        "Focused, purposeful tool use vs. flailing: minimal redundant reads, "
        "no repeated identical calls, no wandering. (0 = chaotic, 5 = tight)"
    ),
    "verification_discipline": (
        "Ran the verification command (tests) after the change and read the "
        "result before finishing. (0 = never verified, 5 = verified each change)"
    ),
    "recovery": (
        "Handled failures and tool errors sensibly (retried, adjusted, "
        "read error output) rather than ignoring them or looping. "
        "(0 = ignored/crashed, 5 = clean recovery; 5 also if nothing failed)"
    ),
}

JUDGE_SYSTEM = """You are grading a coding agent's trajectory on a 0-5 rubric.
Be strict and evidence-based: quote nothing, but score only what the steps show.

Rubric:
""" + "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items()) + """

Return ONLY a JSON object, no markdown fences, no commentary:
{"scores": {"solution_correctness": <0-5>, "tool_efficiency": <0-5>,
"verification_discipline": <0-5>, "recovery": <0-5>},
"rationale": "<=3 sentences explaining the scores"}"""


def _format_steps(steps: list[dict], max_chars: int = 6000) -> str:
    """Compact a trajectory for the judge: one line per step."""
    lines = []
    for s in steps:
        idx = s.get("step", "?")
        if s.get("error"):
            lines.append(f"step {idx}: ERROR {s['error']}")
            continue
        text = (s.get("assistant_text") or "").strip().replace("\n", " ")
        calls = ", ".join(
            f"{c['name']}({str(c['arguments'])[:80]})"
            for c in s.get("tool_calls", [])
        )
        results = " | ".join(
            f"{r['tool']}->{str(r['result'])[:120].replace(chr(10), ' ')}"
            for r in s.get("tool_results", [])
        )
        lines.append(f"step {idx}: {text[:160]} [{calls}] {results[:160]}")
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[:max_chars] + "\n...[truncated]"


def parse_judge_json(raw: str) -> dict | None:
    """Extract the scores object from a judge reply; None if unparseable."""
    text = raw.strip()
    if text.startswith("```"):
        # tolerate fenced output even though the prompt forbids it
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    scores = obj.get("scores")
    if not isinstance(scores, dict):
        return None
    cleaned = {}
    for dim in RUBRIC:
        try:
            cleaned[dim] = max(0, min(5, float(scores.get(dim, 0))))
        except (TypeError, ValueError):
            cleaned[dim] = 0.0
    return {"scores": cleaned, "rationale": str(obj.get("rationale", ""))[:500]}


def grade(
    task_prompt: str,
    steps: list[dict],
    verify_output: str,
    model: str | None = None,
) -> dict:
    """Score one finished task. Never raises; returns {"error": ...} on failure."""
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "no OPENAI_API_KEY; judge skipped"}
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60,
        )
        resp = client.chat.completions.create(
            model=model or os.getenv("MODEL_JUDGE", os.getenv("MODEL", "gpt-4o-mini")),
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"TASK:\n{task_prompt}\n\n"
                        f"VERIFY (pytest) OUTPUT:\n{verify_output[-1500:]}\n\n"
                        f"TRAJECTORY:\n{_format_steps(steps)}"
                    ),
                },
            ],
            temperature=0,
        )
        parsed = parse_judge_json(resp.choices[0].message.content or "")
        if parsed is None:
            return {"error": "judge returned unparseable JSON"}
        parsed["model"] = resp.model if hasattr(resp, "model") else None
        return parsed
    except Exception as e:  # judge must never crash the eval
        return {"error": f"{type(e).__name__}: {e}"}
