"""Testes do cliente Ollama, com transporte HTTP simulado."""

from __future__ import annotations

import json

import httpx
import pytest

from roboteye.config import LLMSettings
from roboteye.llm.base import ChatMessage, LLMError, collect
from roboteye.llm.ollama import OllamaClient

MENSAGENS = [ChatMessage(role="user", content="olá")]


def cliente_com(handler) -> OllamaClient:
    """Cria um cliente cujo transporte é controlado pelo teste."""
    client = OllamaClient(LLMSettings(host="http://fake:11434", model="teste"))
    client._client = httpx.Client(
        base_url="http://fake:11434",
        transport=httpx.MockTransport(handler),
    )
    return client


def ndjson(*eventos: dict) -> bytes:
    return "\n".join(json.dumps(evento) for evento in eventos).encode()


class TestStreamReply:
    def test_junta_os_pedacos_da_resposta(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ndjson(
                    {"message": {"content": "Olá"}, "done": False},
                    {"message": {"content": ", humano"}, "done": False},
                    {"message": {"content": "."}, "done": True},
                ),
            )

        client = cliente_com(handler)
        try:
            assert collect(client.stream_reply(MENSAGENS)) == "Olá, humano."
        finally:
            client.close()

    def test_para_no_done(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ndjson(
                    {"message": {"content": "fim"}, "done": True},
                    {"message": {"content": "ignorado"}, "done": False},
                ),
            )

        client = cliente_com(handler)
        try:
            assert collect(client.stream_reply(MENSAGENS)) == "fim"
        finally:
            client.close()

    def test_linhas_invalidas_sao_ignoradas(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            content = b"nao e json\n" + ndjson({"message": {"content": "ok"}, "done": True})
            return httpx.Response(200, content=content)

        client = cliente_com(handler)
        try:
            assert collect(client.stream_reply(MENSAGENS)) == "ok"
        finally:
            client.close()

    def test_erro_no_corpo_vira_excecao(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=ndjson({"error": "modelo não carregado"}))

        client = cliente_com(handler)
        try:
            with pytest.raises(LLMError, match="modelo não carregado"):
                collect(client.stream_reply(MENSAGENS))
        finally:
            client.close()

    def test_modelo_ausente_explica_como_resolver(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model not found"})

        client = cliente_com(handler)
        try:
            with pytest.raises(LLMError, match="ollama pull"):
                collect(client.stream_reply(MENSAGENS))
        finally:
            client.close()

    def test_falha_de_conexao_explica_o_host(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("recusada", request=request)

        client = cliente_com(handler)
        try:
            with pytest.raises(LLMError, match="http://fake:11434"):
                collect(client.stream_reply(MENSAGENS))
        finally:
            client.close()

    def test_envia_o_modelo_configurado(self) -> None:
        capturado: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            capturado.update(json.loads(request.content))
            return httpx.Response(200, content=ndjson({"message": {"content": "x"}, "done": True}))

        client = cliente_com(handler)
        try:
            collect(client.stream_reply(MENSAGENS))
        finally:
            client.close()

        assert capturado["model"] == "teste"
        assert capturado["stream"] is True


class TestDisponibilidade:
    def test_servidor_no_ar(self) -> None:
        client = cliente_com(lambda _: httpx.Response(200, json={"models": []}))
        try:
            assert client.is_available()
        finally:
            client.close()

    def test_servidor_fora_do_ar(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("recusada", request=request)

        client = cliente_com(handler)
        try:
            assert not client.is_available()
        finally:
            client.close()

    def test_lista_modelos(self) -> None:
        payload = {"models": [{"name": "llama3.2:1b"}, {"name": "qwen2.5:3b"}]}
        client = cliente_com(lambda _: httpx.Response(200, json=payload))
        try:
            assert client.list_models() == ["llama3.2:1b", "qwen2.5:3b"]
        finally:
            client.close()
