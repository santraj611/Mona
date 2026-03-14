# Mona - Local Personal Assistant
# Copyright (C) 2025
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


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
