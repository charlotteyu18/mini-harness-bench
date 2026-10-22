"""Offline mock LLM: scripted tool-call plans, streamed through the same
chunk-aggregation code path as the real OpenAI client.

Lets you run the whole pipeline (agent loop -> tools -> verify -> report)
end to end without an API key:

    MODEL=mock python -m src.eval.runner --config configs/serial.json

The mock picks a plan by matching the task prompt against the known sample
tasks, then streams one scripted "assistant turn" per chat() call: text in a
few chunks, tool calls as streaming deltas (exercising the tool-call
buffering in llm.py), and a usage payload on the final chunk.
"""

import json
import time
from types import SimpleNamespace

from .llm import LLMClient

# -- fixed file contents the mock "writes" -----------------------------------

FIXED_FIB = """def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""

FIXED_STATS = """def mean(xs):
    if not xs:
        return 0.0
    return sum(xs) / len(xs)
"""

FIXED_SLUGIFY = """import re


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
"""

FIXED_RETRY = '''def backoff(attempt, base=1.0):
    """Delay in seconds before retry number `attempt`.
    attempt counts from 0: backoff(0) == base.
    """
    return base * 2 ** attempt
'''

FIXED_RANKING = '''def topk(items, k):
    """items: list of (name, score). Return the top-k names by
    score descending. Ties keep input order; k larger than the
    list returns everything; k == 0 returns [].
    """
    ranked = sorted(items, key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:k]]
'''

FIXED_MONEY = '''def to_cents(price_str):
    """'4.50' -> 450, '4.5' -> 450, '10' -> 1000."""
    dollars, dot, cents = price_str.partition(".")
    if not dot:
        return int(dollars) * 100
    cents = (cents + "00")[:2]  # zero-pad: "4.5" -> "50"
    return int(dollars) * 100 + int(cents)
'''

FIXED_LOGS = '''import re


def count_errors(lines):
    """Count lines reporting an error: the whole word "error",
    any case. "error-free" must NOT count.
    """
    return sum(1 for line in lines if re.search(r"(?<![\\w-])error(?![\\w-])", line, re.IGNORECASE))
'''

FIXED_CLI = '''import sys


def greet(name):
    return f"hello {name}"


def add(a, b):
    return str(int(a) + int(b))


COMMANDS = {"greet": greet, "add": add}


def main(argv):
    cmd = COMMANDS[argv[1]]
    return cmd(*argv[2:])


if __name__ == "__main__":
    print(main(sys.argv))
'''

PYTEST = "python -m pytest"


def _turn(text, *tools):
    return {
        "text": text,
        "tools": [
            {"name": name, "arguments": args} for name, args in tools
        ],
    }


PLANS = {
    "fib.py": [
        _turn(
            "I'll read fib.py to locate the bug.",
            ("read_file", {"path": "fib.py"}),
        ),
        _turn(
            "Found it: the loop ends with the right values but returns b "
            "instead of a. Fixing the return value.",
            ("write_file", {"path": "fib.py", "content": FIXED_FIB}),
        ),
        _turn(
            "Verifying with the test suite.",
            ("run_shell", {"command": f"{PYTEST} test_fib.py -q", "timeout": 60}),
        ),
        _turn("Tests pass. DONE"),
    ],
    "stats.py": [
        _turn(
            "I'll read stats.py to see the crash.",
            ("read_file", {"path": "stats.py"}),
        ),
        _turn(
            "mean() divides by len(xs) with no empty-list guard. "
            "Adding one that returns 0.0.",
            ("write_file", {"path": "stats.py", "content": FIXED_STATS}),
        ),
        _turn(
            "Verifying with the test suite.",
            ("run_shell", {"command": f"{PYTEST} test_stats.py -q", "timeout": 60}),
        ),
        _turn("Tests pass. DONE"),
    ],
    "slugify.py": [
        _turn(
            "I'll read slugify.py and its test to learn the expected behavior.",
            ("read_file", {"path": "slugify.py"}),
            ("read_file", {"path": "test_slugify.py"}),
        ),
        _turn(
            "The tests want lowercase, non-alphanumeric runs collapsed to a "
            "single hyphen, and no leading/trailing hyphens. Implementing.",
            ("write_file", {"path": "slugify.py", "content": FIXED_SLUGIFY}),
        ),
        _turn(
            "Verifying with the test suite.",
            ("run_shell", {"command": f"{PYTEST} test_slugify.py -q", "timeout": 60}),
        ),
        _turn("Tests pass. DONE"),
    ],
    "retry.py": [
        _turn(
            "I'll read retry.py and its test to see the expected curve.",
            ("read_file", {"path": "retry.py"}),
            ("read_file", {"path": "test_retry.py"}),
        ),
        _turn(
            "The test wants doubling delays: base * 2**attempt, not "
            "base * attempt**2. Fixing the formula.",
            ("write_file", {"path": "retry.py", "content": FIXED_RETRY}),
        ),
        _turn(
            "Verifying with the test suite.",
            ("run_shell", {"command": f"{PYTEST} test_retry.py -q", "timeout": 60}),
        ),
        _turn("Tests pass. DONE"),
    ],
    "ranking.py": [
        _turn(
            "I'll read ranking.py and its test to learn the expected behavior.",
            ("read_file", {"path": "ranking.py"}),
            ("read_file", {"path": "test_ranking.py"}),
        ),
        _turn(
            "Sort by score descending (stable for ties), take top k names. "
            "Implementing.",
            ("write_file", {"path": "ranking.py", "content": FIXED_RANKING}),
        ),
        _turn(
            "Verifying with the test suite.",
            ("run_shell", {"command": f"{PYTEST} test_ranking.py -q", "timeout": 60}),
        ),
        _turn("Tests pass. DONE"),
    ],
    "money.py": [
        _turn(
            "Multi-file bug: orders.py calls money.to_cents(). Reading all three.",
            ("read_file", {"path": "money.py"}),
            ("read_file", {"path": "orders.py"}),
            ("read_file", {"path": "test_orders.py"}),
        ),
        _turn(
            "The bug is in money.py: cents are not zero-padded, so '4.5' "
            "becomes 405 instead of 450. Padding to two digits.",
            ("write_file", {"path": "money.py", "content": FIXED_MONEY}),
        ),
        _turn(
            "Verifying with the test suite.",
            ("run_shell", {"command": f"{PYTEST} test_orders.py -q", "timeout": 60}),
        ),
        _turn("Tests pass. DONE"),
    ],
    "logs.py": [
        _turn(
            "I'll read logs.py and its test to see the counting rule.",
            ("read_file", {"path": "logs.py"}),
            ("read_file", {"path": "test_logs.py"}),
        ),
        _turn(
            "Substring match over-counts 'error-free'. Using a whole-word "
            "case-insensitive regex instead.",
            ("write_file", {"path": "logs.py", "content": FIXED_LOGS}),
        ),
        _turn(
            "Verifying with the test suite.",
            ("run_shell", {"command": f"{PYTEST} test_logs.py -q", "timeout": 60}),
        ),
        _turn("Tests pass. DONE"),
    ],
    "cli.py": [
        _turn(
            "I'll read cli.py and its test to see the dispatch bug.",
            ("read_file", {"path": "cli.py"}),
            ("read_file", {"path": "test_cli.py"}),
        ),
        _turn(
            "main() passes argv[1:] to the command, which includes the command "
            "name itself. Slicing from argv[2:] instead.",
            ("write_file", {"path": "cli.py", "content": FIXED_CLI}),
        ),
        _turn(
            "Verifying with the test suite.",
            ("run_shell", {"command": f"{PYTEST} test_cli.py -q", "timeout": 60}),
        ),
        _turn("Tests pass. DONE"),
    ],
}


def plan_for(prompt: str) -> list[dict]:
    """Pick the scripted plan whose key file is mentioned in the prompt."""
    for key, plan in PLANS.items():
        if key in prompt:
            return plan
    # Unknown task: do one harmless read, then stop. The verify step will
    # decide pass/fail; the loop itself still exercises the full path.
    return [_turn("I don't have a plan for this task.", ("grep", {"pattern": "TODO"})), _turn("DONE")]


class MockLLMClient(LLMClient):
    """Drop-in for LLMClient that needs no API key.

    Overrides only the transport (_create_stream); the response parsing,
    tool-call buffering and telemetry measurement in chat() run unchanged.
    """

    def __init__(self, model: str | None = None):
        self.model = model or "mock"
        self._plan: list[dict] | None = None
        self._turn_idx = 0
        # reliability knobs inherited from LLMClient.chat(); mock transport
        # never fails, so retries/backoff are inert here.
        self.max_retries = 0
        self.retry_backoff_s = 0.0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_cooldown_s = 60.0
        self._consec_failures = 0
        self._circuit_open_until = 0.0

    def _create_stream(self, messages, tools):
        if self._plan is None:
            prompt = messages[1]["content"] if len(messages) > 1 else ""
            self._plan = plan_for(prompt)
        if tools is None:
            # Planning call (plan-then-act mode): a text-only turn that
            # does NOT consume a scripted plan turn.
            return self._stream_turn(
                _turn(
                    "Plan: 1. Read the code and the tests. 2. Make the "
                    "smallest fix. 3. Run the tests and confirm they pass."
                )
            )
        turn = self._plan[min(self._turn_idx, len(self._plan) - 1)]
        self._turn_idx += 1
        return self._stream_turn(turn)

    # -- fake streaming transport ------------------------------------------
    def _stream_turn(self, turn: dict):
        """Yield fake OpenAI-style chunks for one assistant turn."""
        # text in a few chunks so TTFT/ITL are measured
        text = turn["text"]
        for i in range(0, len(text), 14):
            yield self._chunk(content=text[i : i + 14])
            time.sleep(0.002)
        for i, tc in enumerate(turn["tools"]):
            call_id = f"mock_call_{self._turn_idx}_{i}"
            args_json = json.dumps(tc["arguments"])
            # name arrives first, arguments stream in fragments
            yield self._chunk(
                tool_calls=[
                    self._tool_delta(
                        i, call_id=call_id, name=tc["name"], args=""
                    )
                ]
            )
            time.sleep(0.002)
            for j in range(0, len(args_json), 20):
                yield self._chunk(
                    tool_calls=[
                        self._tool_delta(i, args=args_json[j : j + 20])
                    ]
                )
                time.sleep(0.001)
        # usage payload on the trailing chunk, like the real API
        yield self._chunk(
            usage=SimpleNamespace(
                prompt_tokens=120 + 40 * self._turn_idx,
                completion_tokens=25 + 10 * self._turn_idx,
            )
        )

    @staticmethod
    def _chunk(content: str = "", tool_calls: list | None = None, usage=None):
        delta = SimpleNamespace(content=content, tool_calls=tool_calls or [])
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta)], usage=usage
        )

    @staticmethod
    def _tool_delta(index: int, call_id: str = "", name: str = "", args: str = ""):
        fn = SimpleNamespace(name=name or None, arguments=args or None)
        return SimpleNamespace(
            index=index, id=call_id or None, function=fn
        )
