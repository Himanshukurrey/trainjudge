"""Small text helpers shared across modules."""

from __future__ import annotations

import re

# Reasoning models (Qwen3 and others) may emit <think>…</think> before the answer;
# an unterminated block runs to the end of the output.
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """The output with any think blocks removed, trimmed."""
    return THINK_RE.sub("", text).strip()


def format_duration(seconds: float | None) -> str:
    """A short human duration: 42s, 3m 07s, 1h 05m."""
    if seconds is None:
        return "?"
    whole = round(seconds)
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
