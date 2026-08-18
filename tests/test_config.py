"""Testes da leitura de configuração."""

from __future__ import annotations

import pytest

from roboteye.config import (
    ConfigError,
    FaceSettings,
    LLMSettings,
    Settings,
    VoiceSettings,
    parse_color,
)


class TestParseColor:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("#04C9FD", (4, 201, 253)),
            ("04C9FD", (4, 201, 253)),
            ("255,180,37", (255, 180, 37)),
            ("  #000000  ", (0, 0, 0)),
        ],
    )
    def test_formatos_aceitos(self, entrada: str, esperado: tuple[int, int, int]) -> None:
        assert parse_color(entrada) == esperado

    @pytest.mark.parametrize("entrada", ["#GGG", "1,2", "300,0,0", ""])
    def test_valores_invalidos(self, entrada: str) -> None:
        with pytest.raises(ConfigError):
            parse_color(entrada)


class TestLLMSettings:
    def test_usa_padroes_sem_variaveis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ROBOTEYE_LLM_MODEL", raising=False)
        assert LLMSettings.from_env().model == "llama3.2:1b"

    def test_le_do_ambiente(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_LLM_MODEL", "qwen2.5:3b")
        monkeypatch.setenv("ROBOTEYE_OLLAMA_HOST", "http://10.0.0.5:11434/")
        settings = LLMSettings.from_env()
        assert settings.model == "qwen2.5:3b"
        assert settings.host == "http://10.0.0.5:11434"  # barra final removida

    def test_backend_invalido_e_rejeitado(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_LLM_BACKEND", "gpt")
        with pytest.raises(ConfigError, match="invalido"):
            LLMSettings.from_env()


class TestVoiceSettings:
    def test_config_derivada_do_modelo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE_MODEL", "/tmp/voz/personalizada.onnx")
        monkeypatch.delenv("ROBOTEYE_VOICE_CONFIG", raising=False)
        settings = VoiceSettings.from_env()
        assert settings.resolved_config_path().name == "personalizada.onnx.json"

    def test_config_explicita_tem_prioridade(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE_MODEL", "/tmp/voz/personalizada.onnx")
        monkeypatch.setenv("ROBOTEYE_VOICE_CONFIG", "/tmp/outro.json")
        assert VoiceSettings.from_env().resolved_config_path().name == "outro.json"

    def test_caminho_relativo_resolve_a_partir_da_raiz(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE_MODEL", "models/x/y.onnx")
        assert VoiceSettings.from_env().model_path.is_absolute()


class TestFaceSettings:
    @pytest.mark.parametrize(
        ("valor", "esperado"),
        [("true", True), ("1", True), ("yes", True), ("false", False), ("off", False)],
    )
    def test_booleanos(self, monkeypatch: pytest.MonkeyPatch, valor: str, esperado: bool) -> None:
        monkeypatch.setenv("ROBOTEYE_FACE_FULLSCREEN", valor)
        assert FaceSettings.from_env().fullscreen is esperado

    def test_booleano_invalido(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_FACE_ENABLED", "talvez")
        with pytest.raises(ConfigError, match="booleano"):
            FaceSettings.from_env()

    def test_dimensao_minima(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_FACE_WIDTH", "10")
        with pytest.raises(ConfigError, match=">="):
            FaceSettings.from_env()


class TestSettings:
    def test_monta_todas_as_secoes(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_LOG_LEVEL", "debug")
        settings = Settings.from_env(env_file=tmp_path / "inexistente.env")
        assert settings.log_level == "DEBUG"
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.voice, VoiceSettings)
        assert isinstance(settings.face, FaceSettings)
