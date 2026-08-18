"""Testes do catálogo de vozes e da troca de voz."""

from __future__ import annotations

from pathlib import Path

import pytest

from roboteye import voice_catalog
from roboteye.config import (
    ConfigError,
    Settings,
    VoiceSettings,
    model_path_for_voice,
)


class TestReservaConfigurada:
    """`ROBOTEYE_VOICE_FALLBACK` aceita "auto", "off" ou o nome de uma voz."""

    def _voz(self, **kwargs) -> VoiceSettings:
        return VoiceSettings(voice="francisca", **kwargs)

    def test_auto_deixa_o_catalogo_escolher(self) -> None:
        assert self._voz(fallback="auto").fallback_voice() in {"dora", "dii"}

    def test_off_desliga(self) -> None:
        assert self._voz(fallback="off").fallback_voice() is None
        assert self._voz(fallback="false").fallback_voice() is None

    def test_nome_explicito_manda(self) -> None:
        """A heurística de hardware acerta na maioria, não em todo caso.

        Um mini-PC ARM potente ou um Pi só para a face aguentam a voz pesada;
        fixar o nome é a saída para quando o palpite não serve.
        """
        assert self._voz(fallback="dii").fallback_voice() == "dii"

    def test_nome_invalido_falha_na_configuracao(self) -> None:
        """Melhor quebrar ao subir do que no meio de uma fala, sem rede."""
        with pytest.raises(ConfigError, match="desconhecida"):
            self._voz(fallback="nao-existe").fallback_voice()

    def test_voz_offline_nao_ganha_reserva(self) -> None:
        assert VoiceSettings(voice="dii", fallback="auto").fallback_voice() is None


class TestCatalogo:
    def test_a_voz_padrao_existe(self) -> None:
        assert voice_catalog.get(voice_catalog.DEFAULT_VOICE) is not None

    def test_tem_voz_em_portugues(self) -> None:
        vozes_pt = [v for v in voice_catalog.CATALOG.values() if v.language == "pt"]
        assert vozes_pt, "o catálogo deveria oferecer ao menos uma voz em português"

    @pytest.mark.parametrize("key", voice_catalog.names())
    def test_toda_voz_esta_completa(self, key: str) -> None:
        spec = voice_catalog.get(key)
        assert spec is not None
        assert spec.description
        assert spec.language
        assert spec.engine in {"piper", "kokoro", "edge"}

        if spec.engine == "piper":
            # No Piper cada voz tem seu par (.onnx, .onnx.json).
            assert spec.model_url.endswith(".onnx")
            assert spec.config_url.endswith(".onnx.json")
            assert spec.speaker is None
        elif spec.engine == "kokoro":
            # No Kokoro o pacote é compartilhado e a voz é escolhida por nome.
            assert spec.model_url.endswith(".onnx")
            assert spec.config_url.endswith(".bin")
            assert spec.speaker
        else:
            # Na nuvem não há arquivo nenhum, só o nome da voz a pedir.
            assert not spec.model_url
            assert not spec.config_url
            assert spec.speaker

    def test_tem_voz_em_cada_motor(self) -> None:
        motores = {v.engine for v in voice_catalog.CATALOG.values()}
        assert motores == {"piper", "kokoro", "edge"}

    @pytest.mark.parametrize("key", voice_catalog.names())
    def test_so_vozes_online_precisam_de_reserva(self, key: str) -> None:
        """Uma voz offline não precisa de plano B; uma online precisa, e no idioma dela."""
        spec = voice_catalog.get(key)
        assert spec is not None
        reserva = voice_catalog.fallback_for(key)

        if spec.engine != "edge":
            assert reserva is None
            return

        assert reserva is not None, f"{key} roda na nuvem e ficaria muda sem internet"
        alvo = voice_catalog.get(reserva)
        assert alvo is not None
        assert alvo.engine != "edge", "a reserva de uma voz online tem de rodar offline"
        assert alvo.language == spec.language, "cair para outro idioma não ajuda ninguém"

    def test_a_reserva_e_leve_em_maquina_modesta(self) -> None:
        """Num Raspberry Pi, cair numa voz pesada troca um problema por outro.

        As vozes Kokoro soam melhor, mas são 325 MB e sintetizam a ~0,4x do
        tempo real. Num Pi isso transformaria "sem internet" em "fala
        arrastada" — pior que o problema original. Por isso o hardware modesto
        cai nas vozes Piper, oito vezes mais rápidas.
        """
        pesada = voice_catalog.fallback_for("francisca", light=False)
        leve = voice_catalog.fallback_for("francisca", light=True)

        assert pesada != leve
        assert voice_catalog.CATALOG[pesada].engine == "kokoro"
        assert voice_catalog.CATALOG[leve].engine == "piper"
        assert voice_catalog.CATALOG[leve].language == "pt", "a reserva tem de falar a língua"

    def test_vozes_online_nao_tem_o_que_baixar(self) -> None:
        assert not voice_catalog.needs_download("thalita")
        assert voice_catalog.needs_download("dora")

    def test_vozes_kokoro_compartilham_os_arquivos(self, tmp_path: Path) -> None:
        """Baixar a segunda voz Kokoro não deve baixar nada de novo."""
        dora = voice_catalog.CATALOG["dora"].target_paths(tmp_path)
        heart = voice_catalog.CATALOG["heart"].target_paths(tmp_path)
        assert dora == heart

    def test_vozes_piper_nao_compartilham_arquivos(self, tmp_path: Path) -> None:
        dii = voice_catalog.CATALOG["dii"].target_paths(tmp_path)
        faber = voice_catalog.CATALOG["faber"].target_paths(tmp_path)
        assert dii != faber

    def test_busca_ignora_caixa_e_espacos(self) -> None:
        assert voice_catalog.get("  DII ") is voice_catalog.get("dii")

    def test_voz_desconhecida_devolve_none(self) -> None:
        assert voice_catalog.get("inexistente") is None

    def test_idioma_de_voz_desconhecida_cai_no_padrao(self) -> None:
        assert voice_catalog.language_of("inexistente", default="xx") == "xx"

    def test_caminhos_derivam_do_nome(self, tmp_path: Path) -> None:
        modelo, config = voice_catalog.CATALOG["dii"].target_paths(tmp_path)
        assert modelo == tmp_path / "dii" / "dii.onnx"
        assert config == tmp_path / "dii" / "dii.onnx.json"


class TestSelecaoDeVoz:
    def test_nome_da_voz_resolve_o_caminho(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE", "dii")
        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)

        settings = VoiceSettings.from_env()

        assert settings.voice == "dii"
        assert settings.model_path.name == "dii.onnx"
        assert settings.language == "pt"

    def test_caminho_explicito_tem_precedencia(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE", "dii")
        monkeypatch.setenv("ROBOTEYE_VOICE_MODEL", "/tmp/outro/modelo.onnx")

        assert VoiceSettings.from_env().model_path.name == "modelo.onnx"

    def test_voz_desconhecida_lista_as_opcoes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE", "inexistente")
        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)

        with pytest.raises(ConfigError, match="dii"):
            VoiceSettings.from_env()

    def test_nome_da_voz_ignora_caixa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE", "DII")
        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)

        assert VoiceSettings.from_env().voice == "dii"

    def test_model_path_for_voice_rejeita_desconhecida(self) -> None:
        with pytest.raises(ConfigError, match="desconhecida"):
            model_path_for_voice("inexistente")


class TestSelecaoDeMotor:
    """Cada voz sabe em qual motor roda; `auto` respeita isso."""

    def _voice(self, monkeypatch: pytest.MonkeyPatch, **env: str) -> VoiceSettings:
        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)
        for chave, valor in env.items():
            monkeypatch.setenv(f"ROBOTEYE_{chave}", valor)
        return VoiceSettings.from_env()

    def test_auto_usa_o_motor_da_voz_piper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._voice(monkeypatch, VOICE="dii", TTS_BACKEND="auto").engine == "piper"

    def test_auto_usa_o_motor_da_voz_kokoro(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._voice(monkeypatch, VOICE="dora", TTS_BACKEND="auto").engine == "kokoro"

    def test_backend_explicito_tem_a_palavra_final(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regressão: com TTS_BACKEND=piper no .env, uma voz Kokoro era carregada
        # pelo motor errado e quebrava ao ler o .onnx.
        assert self._voice(monkeypatch, VOICE="dora", TTS_BACKEND="null").engine == "null"

    def test_voz_kokoro_traz_o_nome_do_falante(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._voice(monkeypatch, VOICE="dora").speaker == "pf_dora"

    def test_voz_piper_nao_tem_falante(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._voice(monkeypatch, VOICE="dii").speaker is None

    def test_falante_pode_ser_sobrescrito(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._voice(monkeypatch, VOICE="dora", VOICE_SPEAKER="pm_santa")
        assert settings.speaker == "pm_santa"

    def test_config_do_kokoro_aponta_para_o_pacote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._voice(monkeypatch, VOICE="dora")
        assert settings.resolved_config_path().name == "voices.bin"

    def test_config_do_piper_aponta_para_o_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._voice(monkeypatch, VOICE="dii")
        assert settings.resolved_config_path().name == "dii.onnx.json"


class TestFabricaDeMotores:
    def test_cria_o_motor_certo_para_cada_voz(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from roboteye.speech.factory import create_tts_engine

        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)
        monkeypatch.setenv("ROBOTEYE_TTS_BACKEND", "auto")

        monkeypatch.setenv("ROBOTEYE_VOICE", "dii")
        assert create_tts_engine(VoiceSettings.from_env()).name == "piper"

        monkeypatch.setenv("ROBOTEYE_VOICE", "dora")
        assert create_tts_engine(VoiceSettings.from_env()).name == "kokoro"


class TestIdiomaDerivadoDaVoz:
    """De nada adianta uma voz brasileira se o modelo responde em inglês."""

    def _settings(self, tmp_path: Path) -> Settings:
        return Settings.from_env(env_file=tmp_path / "inexistente.env")

    def test_voz_brasileira_faz_responder_em_portugues(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE", "dii")
        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)
        monkeypatch.delenv("ROBOTEYE_REPLY_LANGUAGE", raising=False)

        assert self._settings(tmp_path).llm.reply_language == "pt"

    def test_voz_inglesa_faz_responder_em_ingles(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE", "lessac")
        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)
        monkeypatch.delenv("ROBOTEYE_REPLY_LANGUAGE", raising=False)

        assert self._settings(tmp_path).llm.reply_language == "en"

    def test_escolha_explicita_do_usuario_tem_a_palavra_final(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE", "dii")
        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)
        monkeypatch.setenv("ROBOTEYE_REPLY_LANGUAGE", "en")

        assert self._settings(tmp_path).llm.reply_language == "en"


class TestPersonalidadePorIdioma:
    def test_prompt_pede_portugues_do_brasil(self, tmp_path: Path) -> None:
        from roboteye.llm.persona import PersonaStore

        prompt = PersonaStore(tmp_path, "atlas").load("pt").system_prompt()
        assert "Brazilian Portuguese" in prompt
