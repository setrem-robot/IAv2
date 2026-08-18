"""Construcao do cliente de LLM a partir da configuracao."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roboteye.llm.base import LLMClient
from roboteye.llm.echo import EchoClient
from roboteye.llm.ollama import OllamaClient

if TYPE_CHECKING:
    from roboteye.config import LLMSettings


def create_llm_client(settings: LLMSettings) -> LLMClient:
    """Instancia o cliente pedido na configuracao."""
    match settings.backend:
        case "ollama":
            return OllamaClient(settings)
        case "echo":
            return EchoClient()
        case other:  # pragma: no cover - config.py ja valida
            raise ValueError(f"backend de LLM desconhecido: {other!r}")
