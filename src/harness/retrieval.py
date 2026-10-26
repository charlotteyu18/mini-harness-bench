"""Keyword retrieval for RAG-style file context.

Deliberately simple and deterministic: tokenize the task prompt, score each
source file by how many *distinct* query tokens it contains, return the
top-k. No embeddings, no API calls — the point is to study what retrieved
context does to the loop (fewer tool round-trips, less wandering through
irrelevant files), not to build a production retriever.

Test files are excluded: they assert behavior rather than provide
implementation context, and a test file trivially matches the task's own
keywords (function names, input shapes), which would crowd out the modules
the agent actually needs to read.
"""

import re
from pathlib import Path

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "with", "on",
    "is", "are", "be", "by", "at", "as", "it", "this", "that", "from",
    "into", "use", "using", "you", "your", "when", "then", "do", "does",
    "not", "no", "if", "than", "so", "we", "our",
}

MAX_FILE_BYTES = 200_000  # skip anything bigger; not worth indexing


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.split(r"[^a-z0-9_]+", text.lower())
        if tok and tok not in STOPWORDS
    }


def retrieve_files(
    workdir: Path,
    query: str,
    top_k: int = 3,
    max_chars: int = 4000,
) -> list[dict]:
    """Return up to top_k files as {path, content}, best score first.

    Score = number of distinct query tokens present in the file. Ties
    break alphabetically by path, so results are deterministic.
    """
    query_tokens = _tokens(query)
    scored: list[tuple[int, str, str]] = []
    for p in sorted(workdir.rglob("*.py")):
        rel = str(p.relative_to(workdir))
        if p.name.startswith("test_"):
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            content = p.read_text(errors="ignore")
        except OSError:
            continue
        if not content.strip():
            continue
        score = len(query_tokens & _tokens(content))
        if score:
            scored.append((score, rel, content[:max_chars]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"path": rel, "content": content}
        for _, rel, content in scored[:top_k]
    ]
