"""Testes da linha de comando e do diagnóstico."""

from __future__ import annotations

import pytest

from roboteye import voice_catalog
from roboteye.cli import build_parser, main
from roboteye.config import FaceSettings, LLMSettings, Settings, VoiceSettings
from roboteye.diagnostics import Status, run_diagnostics


@pytest.fixture(autouse=True)
def ambiente_limpo(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Isola os testes do `.env` e da configuração real da máquina."""
    monkeypatch.setenv("ROBOTEYE_LLM_BACKEND", "echo")
    monkeypatch.setenv("ROBOTEYE_TTS_BACKEND", "null")
    monkeypatch.setenv("ROBOTEYE_FACE_ENABLED", "false")
    monkeypatch.setenv("ROBOTEYE_VOICE_MODEL", str(tmp_path / "voz.onnx"))


class TestParser:
    @pytest.mark.parametrize(
        "argumentos",
        [
            ["run"],
            ["chat"],
            ["face"],
            ["say", "texto"],
            ["doctor"],
            ["setup"],
            ["setup", "--no-llm", "--non-interactive"],
            ["models"],
            ["voice", "list"],
            ["voice", "download"],
        ],
    )
    def test_subcomandos_disponiveis(self, argumentos: list[str]) -> None:
        assert build_parser().parse_args(argumentos).handler is not None

    def test_voice_exige_um_subcomando(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["voice"])

    def test_say_junta_as_palavras(self) -> None:
        args = build_parser().parse_args(["say", "olá", "mundo"])
        assert args.text == ["olá", "mundo"]

    def test_fullscreen_e_opcional(self) -> None:
        assert build_parser().parse_args(["run"]).fullscreen is False
        assert build_parser().parse_args(["run", "--fullscreen"]).fullscreen is True

    def test_voice_download_tem_padrao(self) -> None:
        padrao = build_parser().parse_args(["voice", "download"]).key
        assert padrao == voice_catalog.DEFAULT_VOICE

    @pytest.mark.parametrize("comando", [["run"], ["chat"], ["say", "oi"]])
    def test_voice_pode_ser_trocada_na_linha_de_comando(self, comando: list[str]) -> None:
        assert build_parser().parse_args([*comando, "--voice", "dii"]).voice == "dii"

    def test_comando_invalido_encerra(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["inexistente"])


class TestComandos:
    def test_voice_list_lista_o_catalogo(self, capsys) -> None:
        assert main(["voice", "list"]) == 0
        assert voice_catalog.DEFAULT_VOICE in capsys.readouterr().out

    def test_voice_download_desconhecida_falha(self, capsys) -> None:
        assert main(["voice", "download", "inexistente"]) == 1

    def test_doctor_roda_sem_explodir(self) -> None:
        # Com backends echo/null e modelo inexistente, deve reportar falha, não travar.
        assert main(["doctor"]) in (0, 1)

    def test_say_com_tts_mudo_avisa_que_nao_gerou_audio(self, tmp_path, capsys) -> None:
        destino = tmp_path / "saida.wav"
        with pytest.raises(RuntimeError, match="nenhum audio"):
            main(["say", "olá", "--output", str(destino)])

    def test_configuracao_invalida_retorna_erro(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("ROBOTEYE_FACE_WIDTH", "isso-nao-e-numero")
        assert main(["doctor"]) == 1
        assert "configuracao" in capsys.readouterr().err

    def test_voice_inexistente_na_linha_de_comando_retorna_erro(self, capsys) -> None:
        assert main(["say", "oi", "--voice", "inexistente"]) == 1
        assert "inexistente" in capsys.readouterr().err

    def test_models_lista_o_que_a_maquina_da_ia_tem(self, monkeypatch, capsys) -> None:
        from roboteye.llm.probe import ProbeResult

        monkeypatch.setattr(
            "roboteye.llm.probe.probe_ollama",
            lambda *_, **__: ProbeResult(
                ok=True, host="http://ia:11434", latency_ms=9, models=("a:1b", "b:8b")
            ),
        )
        assert main(["models"]) == 0
        assert "a:1b" in capsys.readouterr().out

    def test_models_com_maquina_fora_do_ar_retorna_erro(self, monkeypatch, capsys) -> None:
        from roboteye.llm.probe import ProbeResult

        monkeypatch.setattr(
            "roboteye.llm.probe.probe_ollama",
            lambda *_, **__: ProbeResult(ok=False, host="http://ia:11434", error="recusado"),
        )
        assert main(["models"]) == 1
        assert "recusado" in capsys.readouterr().err

    def test_setup_nao_interativo_grava_o_env(self, tmp_path, monkeypatch, capsys) -> None:
        from roboteye.web import envfile

        env = tmp_path / ".env"
        env.write_text("# meu comentario\nROBOTEYE_VOICE=francisca\n", encoding="utf-8")
        monkeypatch.setattr(
            "roboteye.voices.download_voice", lambda *_, **__: tmp_path / "voz.onnx"
        )

        codigo = main(
            [
                "--env-file",
                str(env),
                "setup",
                "--non-interactive",
                "--no-llm",
                "--voice",
                "dii",
                "--skip-download",
            ]
        )

        assert codigo == 0
        valores = envfile.read(env)
        assert valores["ROBOTEYE_VOICE"] == "dii"
        assert valores["ROBOTEYE_LLM_BACKEND"] == "echo"
        assert "# meu comentario" in env.read_text(encoding="utf-8")

    def test_voice_list_marca_a_voz_em_uso(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("ROBOTEYE_VOICE", "dii")
        monkeypatch.delenv("ROBOTEYE_VOICE_MODEL", raising=False)

        assert main(["voice", "list"]) == 0

        linha_marcada = next(
            linha for linha in capsys.readouterr().out.splitlines() if linha.startswith(" *")
        )
        assert "dii" in linha_marcada


class TestDiagnostics:
    def _settings(self, **kwargs) -> Settings:
        base = {
            "llm": LLMSettings(backend="echo"),
            "voice": VoiceSettings(backend="null"),
            "face": FaceSettings(),
        }
        return Settings(**{**base, **kwargs})

    def test_relatorio_tem_uma_linha_por_verificacao(self) -> None:
        relatorio = run_diagnostics(self._settings())
        assert len(relatorio.checks) >= 4
        assert "Diagnostico do RobotEye" in relatorio.render()

    def test_backend_mudo_gera_aviso_e_nao_falha(self) -> None:
        relatorio = run_diagnostics(self._settings())
        voz = [c for c in relatorio.checks if c.name == "motor de voz"]
        assert voz and voz[0].status is Status.WARN
        assert relatorio.ok

    def test_modelo_ausente_e_falha_com_dica(self, tmp_path) -> None:
        settings = self._settings(
            voice=VoiceSettings(backend="piper", model_path=tmp_path / "nao-existe.onnx")
        )
        relatorio = run_diagnostics(settings)

        modelo = [c for c in relatorio.checks if c.name == "modelo de voz"]
        if modelo:  # só aparece se o piper estiver instalado
            assert modelo[0].status is Status.FAIL
            assert "voice download" in modelo[0].hint
            assert not relatorio.ok
