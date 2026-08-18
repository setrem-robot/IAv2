"""Testes da seleção de backends e da personalidade."""

from __future__ import annotations

import pytest

from roboteye.config import LLMSettings, VoiceSettings
from roboteye.llm.echo import EchoClient
from roboteye.llm.factory import create_llm_client
from roboteye.llm.ollama import OllamaClient
from roboteye.llm.persona import PersonaStore
from roboteye.speech.factory import create_tts_engine
from roboteye.speech.null_engine import NullEngine
from roboteye.speech.piper_engine import PiperEngine
from roboteye.speech.player import NullSink, create_audio_sink


class TestLLMFactory:
    def test_backend_ollama(self) -> None:
        assert isinstance(create_llm_client(LLMSettings(backend="ollama")), OllamaClient)

    def test_backend_echo(self) -> None:
        assert isinstance(create_llm_client(LLMSettings(backend="echo")), EchoClient)

    def test_backend_desconhecido(self) -> None:
        with pytest.raises(ValueError, match="desconhecido"):
            create_llm_client(LLMSettings(backend="inexistente"))


class TestTTSFactory:
    @pytest.mark.parametrize(
        ("backend", "esperado"),
        [("piper", PiperEngine), ("null", NullEngine)],
    )
    def test_backends(self, backend: str, esperado: type) -> None:
        assert isinstance(create_tts_engine(VoiceSettings(backend=backend)), esperado)

    def test_backend_desconhecido(self) -> None:
        with pytest.raises(ValueError, match="desconhecido"):
            create_tts_engine(VoiceSettings(backend="inexistente"))

    def test_motores_nao_carregam_modelo_na_criacao(self, tmp_path) -> None:
        # Criar o motor com um caminho inexistente não pode falhar: só o warm_up falha.
        create_tts_engine(VoiceSettings(backend="piper", model_path=tmp_path / "nao-existe.onnx"))


class TestAudioSinkFactory:
    def test_backend_null_usa_saida_muda(self) -> None:
        assert isinstance(create_audio_sink(VoiceSettings(backend="null")), NullSink)

    def test_saida_muda_aceita_tudo(self) -> None:
        sink = NullSink()
        sink.start(None)  # type: ignore[arg-type]
        sink.write(b"\x00\x00")
        sink.stop()
        sink.close()


class TestEchoClient:
    def test_devolve_uma_fala_pronta(self) -> None:
        client = EchoClient(seed=1)
        resposta = "".join(client.stream_reply([])).strip()
        assert resposta

    def test_e_deterministico_com_semente(self) -> None:
        primeira = "".join(EchoClient(seed=7).stream_reply([]))
        segunda = "".join(EchoClient(seed=7).stream_reply([]))
        assert primeira == segunda

    def test_esta_sempre_disponivel(self) -> None:
        assert EchoClient().is_available()


class TestNullEngine:
    def test_nao_produz_audio(self) -> None:
        assert list(NullEngine().synthesize("qualquer coisa")) == []


class TestPersona:
    def _persona(self, tmp_path, language: str = "en"):
        return PersonaStore(tmp_path, "atlas").load(language)

    def test_prompt_menciona_o_idioma_por_extenso(self, tmp_path) -> None:
        assert "English" in self._persona(tmp_path, "en").system_prompt()
        assert "Brazilian Portuguese" in self._persona(tmp_path, "pt").system_prompt()

    def test_idioma_desconhecido_entra_como_veio(self, tmp_path) -> None:
        assert "xx" in self._persona(tmp_path, "xx").system_prompt()

    def test_proibe_markdown_e_emoji(self, tmp_path) -> None:
        prompt = self._persona(tmp_path).system_prompt().lower()
        assert "markdown" in prompt
        assert "emoji" in prompt

    def test_usa_a_persona_embutida_quando_nao_ha_arquivo(self, tmp_path) -> None:
        assert "Atlas" in self._persona(tmp_path).system_prompt()
