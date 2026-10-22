"""Eval runner: set up each task in an isolated workdir, run the agent,
grade with the task's verify command, record pass/fail + telemetry."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dotenv import load_dotenv  # noqa: E402

from harness.agent import Agent, HarnessConfig  # noqa: E402
from harness.judge import grade  # noqa: E402
from harness.telemetry import summarize_trajectory  # noqa: E402


def run_task(
    task: dict, config: HarnessConfig, traj_dir: Path, judge: bool = False
) -> dict:
    with tempfile.TemporaryDirectory(prefix="task_") as td:
        workdir = Path(td)
        for rel, content in task["files"].items():
            p = workdir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

        agent = Agent(config)
        outcome = agent.run(
            task["prompt"], workdir, traj_dir / f"{task['id']}.jsonl"
        )
        try:
            vr = subprocess.run(
                task["verify"], shell=True, cwd=workdir,
                capture_output=True, text=True, timeout=180,
            )
            passed = vr.returncode == 0
            verify_out = (vr.stdout + vr.stderr).strip()[-1500:]
        except subprocess.TimeoutExpired:
            passed, verify_out = False, "verify TIMEOUT"

        metrics = summarize_trajectory(outcome["steps"])
        result = {
            "task_id": task["id"],
            "config": config.name,
            "passed": passed,
            "agent_done": outcome["done"],
            "stop_reason": outcome["stop_reason"],
            "verify_output": verify_out,
            "metrics": metrics,
        }
        if judge:
            # LLM-as-judge sits alongside pytest: quality score vs. pass/fail.
            result["judge"] = grade(
                task["prompt"], outcome["steps"], verify_out
            )
        return result


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/sample_tasks.json")
    ap.add_argument("--config", default="configs/serial.json")
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument(
        "--judge", action="store_true",
        help="score each trajectory with the LLM-as-judge rubric "
             "(needs OPENAI_API_KEY; never fails the run)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    tasks = json.loads((root / args.tasks).read_text())
    cfg = json.loads((root / args.config).read_text())
    config = HarnessConfig(
        name=cfg["name"],
        max_iterations=cfg.get("max_iterations", 12),
        context_strategy=cfg.get("context_strategy", "full"),
        mode=cfg.get("mode", "react"),
        # config file wins; MODEL env var (e.g. MODEL=mock) is the override
        model=cfg.get("model") or os.getenv("MODEL"),
        window_messages=cfg.get("window_messages", 8),
        system_prompt=cfg.get("system_prompt", "v1"),
        llm_timeout_s=cfg.get("llm_timeout_s", 120.0),
        llm_max_retries=cfg.get("llm_max_retries", 3),
        llm_retry_backoff_s=cfg.get("llm_retry_backoff_s", 1.0),
    )
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj_dir = root / "trajectories" / config.name

    print(f"config={config.name} strategy={config.context_strategy} "
          f"mode={config.mode} "
          f"prompt={config.system_prompt} tasks={len(tasks)} "
          f"judge={args.judge}")
    with out_path.open("w") as f:
        for t in tasks:
            r = run_task(t, config, traj_dir, judge=args.judge)
            f.write(json.dumps(r) + "\n")
            f.flush()
            mark = "PASS" if r["passed"] else "FAIL"
            j = r.get("judge") or {}
            jstr = ""
            if j.get("scores"):
                avg = sum(j["scores"].values()) / len(j["scores"])
                jstr = f" judge={avg:.1f}/5"
            elif j.get("error"):
                jstr = f" judge=ERR({j['error'][:40]})"
            print(f"[{mark}] {t['id']} "
                  f"steps={r['metrics']['steps']} "
                  f"tokens={r['metrics']['total_tokens']} "
                  f"ttft={r['metrics']['avg_ttft_ms']}ms{jstr}")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
