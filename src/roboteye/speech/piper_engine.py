"""Motor de TTS local baseado no Piper (ONNX).

E o motor padrao: roda inteiramente offline e sintetiza muito mais rapido que o
tempo real mesmo em CPU modesta, o que elimina a latencia de rede da API remota.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboteye.logging_setup import get_logger
from roboteye.speech.base import AudioFormat, SpeechChunk, SpeechError

if TYPE_CHECKING:
    from roboteye.config import VoiceSettings

logger = get_logger(__name__)

_INSTALL_HINT = (
    'Piper nao esta instalado. Rode: pip install -e ".[tts]" (ou pip install piper-tts sounddevice)'
)


class PiperEngine:
    """Sintetiza voz com um modelo Piper local."""

    name = "piper"

    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings
        self._voice: Any | None = None
        self._syn_config: Any | None = None

    # -- ciclo de vida -----------------------------------------------------
    def warm_up(self) -> None:
        """Carrega o modelo ONNX (~2 s). Chamado no arranque para nao pagar isso na 1a fala."""
        if self._voice is not None:
            return

        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise SpeechError(_INSTALL_HINT) from exc

        model_path = self._settings.model_path
        config_path = self._settings.resolved_config_path()
        _ensure_model_files(model_path, config_path)

        logger.info("carregando voz: %s", model_path.name)
        try:
            self._voice = PiperVoice.load(model_path, config_path=config_path)
        except Exception as exc:
            raise SpeechError(f"falha ao carregar o modelo de voz {model_path}: {exc}") from exc

        self._syn_config = SynthesisConfig(
            length_scale=self._settings.length_scale,
            noise_scale=self._settings.noise_scale,
            noise_w_scale=self._settings.noise_w,
        )
        logger.debug("voz carregada")

    def close(self) -> None:
        self._voice = None
        self._syn_config = None

    # -- sintese -----------------------------------------------------------
    def synthesize(self, text: str) -> Iterator[SpeechChunk]:
        if not text.strip():
            return

        self.warm_up()
        assert self._voice is not None  # garantido por warm_up

        try:
            for chunk in self._voice.synthesize(text, syn_config=self._syn_config):
                yield SpeechChunk(
                    audio=chunk.audio_int16_bytes,
                    format=AudioFormat(
                        sample_rate=chunk.sample_rate,
                        channels=chunk.sample_channels,
                        sample_width=chunk.sample_width,
                    ),
                )
        except Exception as exc:
            raise SpeechError(f"falha na sintese: {exc}") from exc


def _ensure_model_files(model_path: Path, config_path: Path) -> None:
    if not model_path.is_file():
        raise SpeechError(
            f"modelo de voz nao encontrado em {model_path}. Baixe com: roboteye voice download"
        )
    if not config_path.is_file():
        raise SpeechError(
            f"configuracao do modelo nao encontrada em {config_path}. "
            "O Piper precisa do arquivo .onnx.json ao lado do modelo."
        )
