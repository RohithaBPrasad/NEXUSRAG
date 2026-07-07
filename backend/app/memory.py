from __future__ import annotations

from typing import List, Tuple


class Memory:
    """Plain rolling window of (user, assistant) turns. Kept simple and
    inspectable on purpose — no hidden state, no summarisation magic
    unless you call summarize() yourself."""

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.turns: List[Tuple[str, str]] = []

    def add(self, user_msg: str, assistant_msg: str) -> None:
        self.turns.append((user_msg, assistant_msg))
        self.turns = self.turns[-self.max_turns:]

    def as_text(self) -> str:
        if not self.turns:
            return "(no previous conversation)"
        lines = []
        for u, a in self.turns:
            lines.append(f"User: {u}\nAssistant: {a}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns = []
