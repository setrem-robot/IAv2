"""Memoria de conversa.

Mantem uma janela deslizante das ultimas mensagens. Modelos pequenos (como o
llama3.2:1b) degradam rapido com contexto longo, entao o limite e baixo de proposito.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable

from roboteye.llm.base import ChatMessage


class ConversationMemory:
    """Historico limitado de mensagens, seguro entre threads."""

    def __init__(self, system_prompt: str, *, max_messages: int = 8) -> None:
        self._system = ChatMessage(role="system", content=system_prompt)
        self._history: deque[ChatMessage] = deque(maxlen=max_messages)
        self._lock = threading.Lock()

    def add_user(self, text: str) -> None:
        self._append(ChatMessage(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self._append(ChatMessage(role="assistant", content=text))

    def _append(self, message: ChatMessage) -> None:
        if not message.content.strip():
            return
        with self._lock:
            self._history.append(message)

    def build_prompt(self) -> list[ChatMessage]:
        """Mensagens a enviar ao modelo: prompt de sistema + historico."""
        with self._lock:
            return [self._system, *self._history]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    def replace_system_prompt(self, prompt: str) -> None:
        with self._lock:
            self._system = ChatMessage(role="system", content=prompt)

    def __len__(self) -> int:
        with self._lock:
            return len(self._history)

    def __iter__(self) -> Iterable[ChatMessage]:
        with self._lock:
            return iter(list(self._history))
