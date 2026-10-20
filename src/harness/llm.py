"""OpenAI-compatible chat client with streaming + inference telemetry.

Measures the metrics NVIDIA's harness team cares about:
  TTFT (time to first token), ITL (inter-token latency),
  throughput (tokens/sec) and token usage per call.
"""

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

from openai import OpenAI


@dataclass
class CallMetrics:
    ttft_ms: float = 0.0          # time to first token
    itl_ms_p50: float = 0.0       # median inter-token latency
    itl_ms_p95: float = 0.0       # p95 inter-token latency
    total_ms: float = 0.0         # full round-trip wall time
    prompt_tokens: int = 0
    completion_tokens: int = 0
    throughput_tps: float = 0.0   # completion tokens / generation seconds

    def to_dict(self):
        return asdict(self)


class LLMClient:
    def __init__(self, model: str | None = None):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Returns (assistant_text, tool_calls, CallMetrics).

        tool_calls: list of {"id", "name", "arguments"} (arguments already parsed).
        """
        start = time.perf_counter()
        stream = self._create_stream(messages, tools)

        first_token_at: float | None = None
        token_times: list[float] = []
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
            if piece or delta_tcs:  # any activity counts as token arrival
                if first_token_at is None:
                    first_token_at = now
                token_times.append(now)
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

        metrics = self._measure(
            start, end, first_token_at, token_times, usage
        )
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
    def _measure(start, end, first_token_at, token_times, usage):
        m = CallMetrics(total_ms=(end - start) * 1000)
        if first_token_at is not None:
            m.ttft_ms = (first_token_at - start) * 1000
        if len(token_times) >= 2:
            itls = [
                (b - a) * 1000 for a, b in zip(token_times, token_times[1:])
            ]
            m.itl_ms_p50 = statistics.median(itls)
            m.itl_ms_p95 = (
                statistics.quantiles(itls, n=100)[94] if len(itls) >= 2 else itls[0]
            )
            gen_s = token_times[-1] - token_times[0]
        else:
            gen_s = 0.0
        if usage is not None:
            m.prompt_tokens = usage.prompt_tokens or 0
            m.completion_tokens = usage.completion_tokens or 0
        if gen_s > 0 and m.completion_tokens:
            m.throughput_tps = m.completion_tokens / gen_s
        return m
