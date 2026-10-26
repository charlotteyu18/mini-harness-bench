"""OpenAI-compatible chat client with streaming + honest inference telemetry.

Honest metrics only:
  TTFT (time to first chunk), total call latency, token usage (from the
  API's usage payload), and effective throughput
  (completion_tokens / total call seconds).

Deliberately NOT reported: per-token ITL / throughput derived from
client-side chunk arrival times. The API delivers many tokens per chunk in
bursts, so chunk-arrival deltas measure network batching, not model speed
(the old code reported 1500+ tok/s on gpt-4o-mini — a measurement artifact).

Reliability: transient API failures are retried with exponential backoff +
jitter; a circuit breaker trips after repeated consecutive failures;
non-retryable errors (e.g. bad API key) fail fast via LLMError.
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# Errors worth retrying. Auth / bad-request errors are NOT here on purpose:
# retrying them burns time and never succeeds.
RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class LLMError(RuntimeError):
    """Raised when the model call cannot be completed (retries exhausted,
    circuit breaker open, or non-retryable API error)."""


@dataclass
class CallMetrics:
    ttft_ms: float = 0.0          # time to first chunk (honest)
    total_ms: float = 0.0         # full round-trip wall time (honest)
    prompt_tokens: int = 0        # from API usage payload
    completion_tokens: int = 0    # from API usage payload
    throughput_tps: float = 0.0   # effective: completion_tokens / total seconds
    model: str = ""               # model used for this call (tiering)

    def to_dict(self):
        return asdict(self)


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_s: float = 60.0,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=timeout_s,
        )
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown_s = circuit_breaker_cooldown_s
        self._consec_failures = 0
        self._circuit_open_until = 0.0

    # -- public API ------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Returns (assistant_text, tool_calls, CallMetrics).

        tool_calls: list of {"id", "name", "arguments"} (arguments already parsed).
        Raises LLMError when the call cannot be completed.
        """
        now = time.monotonic()
        if now < self._circuit_open_until:
            raise LLMError(
                f"circuit breaker open ({self._consec_failures} consecutive "
                f"failures); cooling down"
            )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                text, tool_calls, metrics = self._chat_once(messages, tools)
            except RETRYABLE_ERRORS as e:
                last_exc = e
                if attempt < self.max_retries:
                    delay = self.retry_backoff_s * (2**attempt)
                    time.sleep(delay + random.uniform(0, 0.5))
                    continue
                break
            except Exception as e:
                # Non-retryable (400 bad request, auth, ...): retrying
                # cannot help. Break so the tail below wraps it in
                # LLMError and the agent degrades gracefully instead of
                # the raw exception crashing the whole eval.
                last_exc = e
                break
            else:
                self._consec_failures = 0
                # record the model actually used: tiering switches it
                # between calls, so the metric must come from call time.
                metrics.model = self.model
                return text, tool_calls, metrics
        self._consec_failures += 1
        if self._consec_failures >= self.circuit_breaker_threshold:
            self._circuit_open_until = (
                time.monotonic() + self.circuit_breaker_cooldown_s
            )
        raise LLMError(
            f"LLM call failed after {self.max_retries + 1} attempt(s): {last_exc}"
        )

    # -- internals -------------------------------------------------------
    def _chat_once(self, messages, tools):
        start = time.perf_counter()
        stream = self._create_stream(messages, tools)

        first_token_at: float | None = None
        content_parts: list[str] = []
        tool_buf: dict[int, dict] = {}
        usage = None

        for chunk in stream:
            now = time.perf_counter()
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = delta.content or ""
            delta_tcs = delta.tool_calls or []
            if (piece or delta_tcs) and first_token_at is None:
                first_token_at = now
            if piece:
                content_parts.append(piece)
            for tc in delta_tcs:
                buf = tool_buf.setdefault(
                    tc.index, {"id": "", "name": "", "args": ""}
                )
                if tc.id:
                    buf["id"] += tc.id
                if tc.function:
                    if tc.function.name:
                        buf["name"] += tc.function.name
                    if tc.function.arguments:
                        buf["args"] += tc.function.arguments

        end = time.perf_counter()
        tool_calls = []
        for buf in tool_buf.values():
            try:
                args = json.loads(buf["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_parse_error": buf["args"]}
            tool_calls.append(
                {"id": buf["id"], "name": buf["name"], "arguments": args}
            )

        metrics = self._measure(start, end, first_token_at, usage)
        return "".join(content_parts), tool_calls, metrics

    def _create_stream(self, messages, tools):
        kwargs: dict = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
        try:
            return self.client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
        except Exception:
            # some OpenAI-compatible servers reject stream_options
            return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _measure(start, end, first_token_at, usage):
        total_s = end - start
        m = CallMetrics(total_ms=total_s * 1000)
        if first_token_at is not None:
            m.ttft_ms = (first_token_at - start) * 1000
        if usage is not None:
            m.prompt_tokens = usage.prompt_tokens or 0
            m.completion_tokens = usage.completion_tokens or 0
        if total_s > 0 and m.completion_tokens:
            # effective throughput: what the harness actually experienced.
            # NOT derived from per-chunk arrival times (see module docstring).
            m.throughput_tps = m.completion_tokens / total_s
        return m
