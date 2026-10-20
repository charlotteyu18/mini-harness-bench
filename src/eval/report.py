"""Print a markdown comparison table across configs from results.jsonl."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness.telemetry import summarize_results, to_markdown  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.jsonl")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    rows = [
        json.loads(line)
        for line in (root / args.results).read_text().splitlines()
        if line.strip()
    ]
    print(to_markdown(summarize_results(rows)))


if __name__ == "__main__":
    main()
