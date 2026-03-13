# memory.py – Tracks conversation history so Mona remembers context.

from collections import deque
from config import MAX_HISTORY_TURNS


class ConversationMemory:
    """
    Stores the last N turns of conversation.
    Each turn is a dict: {"role": "user"|"assistant", "content": str}
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        # deque auto-drops oldest pair when full
        self._history: deque = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        print("🧹 Memory cleared.")
