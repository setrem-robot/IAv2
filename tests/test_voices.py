"""Testes do catálogo de vozes (sem baixar nada da rede)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from roboteye import voices
from roboteye.voice_catalog import DEFAULT_VOICE
from roboteye.voices import CATALOG, VoiceDownloadError, download_voice


class TestCatalogo:
    def test_a_voz_padrao_existe(self) -> None:
        assert DEFAULT_VOICE in CATALOG

    def test_toda_voz_local_tem_modelo_e_configuracao(self) -> None:
        for spec in CATALOG.values():
            assert spec.description
            if not spec.model_url:
                continue  # voz da nuvem: não há arquivo para baixar
            assert spec.model_url.endswith(".onnx")
            assert spec.config_url  # .onnx.json no Piper, pacote de vozes no Kokoro

    def test_baixar_voz_online_explica_que_nao_ha_arquivo(self, tmp_path: Path) -> None:
        with pytest.raises(VoiceDownloadError, match="nuvem"):
            download_voice("thalita", models_dir=tmp_path)

    def test_caminhos_de_destino(self, tmp_path: Path) -> None:
        modelo, config = CATALOG["dii"].target_paths(tmp_path)
        assert modelo == tmp_path / "dii" / "dii.onnx"
        assert config == tmp_path / "dii" / "dii.onnx.json"


class TestDownload:
    def test_voz_desconhecida_lista_as_opcoes(self, tmp_path: Path) -> None:
        with pytest.raises(VoiceDownloadError, match="dii"):
            download_voice("inexistente", models_dir=tmp_path)

    def test_nao_rebaixa_o_que_ja_existe(self, tmp_path: Path, monkeypatch) -> None:
        modelo, config = CATALOG["dii"].target_paths(tmp_path)
        modelo.parent.mkdir(parents=True)
        modelo.write_bytes(b"modelo")
        config.write_text("{}")

        def falhar(*_: object, **__: object) -> None:
            raise AssertionError("não deveria baixar nada")

        monkeypatch.setattr(voices, "_download", falhar)

        assert download_voice("dii", models_dir=tmp_path) == modelo

    def test_baixa_modelo_e_configuracao(self, tmp_path: Path, monkeypatch) -> None:
        baixados: list[str] = []

        def fake_download(url: str, destino: Path, _progress) -> None:
            baixados.append(destino.name)
            destino.write_bytes(b"conteudo")

        monkeypatch.setattr(voices, "_download", fake_download)

        caminho = download_voice("dii", models_dir=tmp_path)

        assert sorted(baixados) == ["dii.onnx", "dii.onnx.json"]
        assert caminho.is_file()

    def test_erro_de_rede_vira_excecao_do_dominio(self, tmp_path: Path, monkeypatch) -> None:
        def explode(*_: object, **__: object):
            raise httpx.ConnectError("sem rede")

        monkeypatch.setattr(httpx, "stream", explode)

        with pytest.raises(VoiceDownloadError, match="falha ao baixar"):
            download_voice("dii", models_dir=tmp_path)

    def test_arquivo_parcial_e_removido_apos_falha(self, tmp_path: Path, monkeypatch) -> None:
        def explode(*_: object, **__: object):
            raise httpx.ConnectError("sem rede")

        monkeypatch.setattr(httpx, "stream", explode)

        with pytest.raises(VoiceDownloadError):
            download_voice("dii", models_dir=tmp_path)

        assert list((tmp_path / "dii").glob("*.part")) == []


class TestProgresso:
    def test_barra_nao_quebra_sem_tamanho_total(self, capsys) -> None:
        voices.console_progress("modelo.onnx", 1_048_576, None)
        assert "1.0 MB" in capsys.readouterr().out

    def test_barra_mostra_porcentagem(self, capsys) -> None:
        voices.console_progress("modelo.onnx", 50, 100)
        assert "50%" in capsys.readouterr().out
