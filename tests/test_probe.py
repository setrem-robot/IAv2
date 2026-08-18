"""Testes da sonda que descobre o que a máquina da IA tem."""

from __future__ import annotations

import httpx
import pytest

from roboteye.llm.probe import explain, normalize_host, probe_ollama


class TestNormalizeHost:
    @pytest.mark.parametrize(
        ("digitado", "esperado"),
        [
            ("192.168.1.50", "http://192.168.1.50:11434"),
            ("192.168.1.50:11434", "http://192.168.1.50:11434"),
            ("http://192.168.1.50:11434/", "http://192.168.1.50:11434"),
            ("localhost:1234", "http://localhost:1234"),
            ("https://ia.local:443", "https://ia.local:443"),
        ],
    )
    def test_completa_o_que_a_pessoa_nao_digita(self, digitado: str, esperado: str) -> None:
        assert normalize_host(digitado) == esperado

    def test_vazio_continua_vazio(self) -> None:
        assert normalize_host("   ") == ""


class _RespostaFake:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class TestProbeOllama:
    def test_lista_os_modelos_em_ordem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pedidos: list[str] = []

        def fake_get(url: str, **_: object) -> _RespostaFake:
            pedidos.append(url)
            return _RespostaFake({"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:1b"}]})

        monkeypatch.setattr(httpx, "get", fake_get)
        resultado = probe_ollama("192.168.1.50")

        assert pedidos == ["http://192.168.1.50:11434/api/tags"]
        assert resultado.ok
        assert resultado.models == ("llama3.2:1b", "qwen3:8b")

    def test_falha_de_rede_vira_explicacao(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_get(*_: object, **__: object) -> _RespostaFake:
            raise httpx.ConnectError("recusado")

        monkeypatch.setattr(httpx, "get", fake_get)
        resultado = probe_ollama("192.168.1.50")

        assert not resultado.ok
        assert "conexao recusada" in resultado.error
        # Mesmo falhando, devolve o endereço já normalizado — é o que a página e
        # o assistente mostram de volta a quem digitou.
        assert resultado.host == "http://192.168.1.50:11434"

    def test_endereco_vazio_nao_bate_na_rede(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(*_: object, **__: object) -> None:
            raise AssertionError("nao deveria ter tentado a rede")

        monkeypatch.setattr(httpx, "get", explode)
        assert probe_ollama("").error == "informe o endereco"


class TestExplain:
    @pytest.mark.parametrize(
        ("erro", "trecho"),
        [
            (httpx.ConnectTimeout("estourou"), "tempo limite"),
            (httpx.ConnectError("recusado"), "conexao recusada"),
            (RuntimeError("Name or service not known"), "nao resolvido"),
        ],
    )
    def test_traduz_para_algo_acionavel(self, erro: Exception, trecho: str) -> None:
        assert trecho in explain(erro)
