from __future__ import annotations

import re
import sys
import time
from typing import List, Optional

try:
    import ollama
except ImportError:
    sys.exit("Missing dependency 'ollama'. Run: pip install ollama")

from . import config


def banner(text: str) -> None:
    print(f"\n{'─' * 70}\n{text}\n{'─' * 70}")


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window character chunker with sentence-boundary snapping.
    Used for prose (txt/pdf/docx/image descriptions) — NOT for code, see chunk_code()."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        window = text[start:end]
        last_period = window.rfind(". ")
        if last_period > size * 0.5 and end < n:
            end = start + last_period + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


_CODE_BOUNDARY_PATTERN = re.compile(
    r"^\s*(def |class |async def |function |const \w+\s*=\s*\(|export |public |private |protected |"
    r"@\w+|//.*$|#.*$)",
)


def chunk_code(text: str, max_lines: int = 60) -> List[str]:
    """Code-aware chunker: splits on blank-line-preceded function/class boundaries
    so a chunk doesn't cut a function in half. Falls back to a flat line-window
    if the file has no recognisable boundaries (e.g. dense one-liners, minified code).
    Preserves original formatting/indentation (no whitespace collapsing, unlike chunk_text)."""
    lines = text.splitlines()
    if not lines:
        return []

    boundaries = [0]
    for i, line in enumerate(lines):
        if i > 0 and _CODE_BOUNDARY_PATTERN.match(line) and lines[i - 1].strip() == "":
            boundaries.append(i)
    boundaries.append(len(lines))

    raw_chunks: List[str] = []
    for start_i, end_i in zip(boundaries[:-1], boundaries[1:]):
        block = "\n".join(lines[start_i:end_i]).strip()
        if block:
            raw_chunks.append(block)

    merged: List[str] = []
    buffer = ""
    for block in raw_chunks:
        candidate = (buffer + "\n\n" + block).strip() if buffer else block
        if len(candidate.splitlines()) <= max_lines:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            block_lines = block.splitlines()
            for j in range(0, len(block_lines), max_lines):
                merged.append("\n".join(block_lines[j:j + max_lines]))
            buffer = ""
    if buffer:
        merged.append(buffer)

    return merged if merged else [text.strip()] if text.strip() else []


def safe_ollama_chat(messages: List[dict], model: str = None, **kwargs) -> str:
    """Wrapper around ollama.chat with basic retry, returns the text content."""
    model = model or config.CHAT_MODEL
    last_err = None
    for _attempt in range(config.OLLAMA_TIMEOUT_RETRIES + 1):
        try:
            resp = ollama.chat(model=model, messages=messages, **kwargs)
            return resp["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(
        f"Ollama chat call failed after retries (model={model}). "
        f"Is 'ollama serve' running and is the model pulled? "
        f"Underlying error: {last_err}"
    )


def safe_ollama_embed(text: str, model: str = None) -> List[float]:
    model = model or config.EMBED_MODEL
    last_err = None
    for _attempt in range(config.OLLAMA_TIMEOUT_RETRIES + 1):
        try:
            resp = ollama.embeddings(model=model, prompt=text)
            return resp["embedding"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(
        f"Ollama embedding call failed after retries (model={model}). "
        f"Is 'ollama pull {model}' done? Underlying error: {last_err}"
    )


def llm(prompt: str, system: Optional[str] = None, model: str = None) -> str:
    """Convenience: single-turn prompt -> text."""
    model = model or config.CHAT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return safe_ollama_chat(messages, model=model)
