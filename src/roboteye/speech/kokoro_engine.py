"""Motor de TTS local baseado no Kokoro (ONNX).

Um unico modelo de 325 MB que carrega dezenas de vozes, a 24 kHz. A voz sai
audivelmente melhor que a do Piper — em troca de umas oito vezes mais CPU
(RTF ~0,4 contra ~0,05). Numa maquina de mesa a diferenca nem aparece; num
Raspberry Pi, aparece bastante, e por isso o Piper continua sendo o padrao.

Usa `onnxruntime`, a mesma base do Piper: nao arrasta PyTorch para o projeto.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from roboteye.logging_setup import get_logger
from roboteye.speech.base import AudioFormat, SpeechChunk, SpeechError

if TYPE_CHECKING:
    from roboteye.config import VoiceSettings

logger = get_logger(__name__)

_INSTALL_HINT = 'Kokoro nao esta instalado. Rode: pip install -e ".[kokoro]"'

#: O Kokoro trabalha com codigos de idioma proprios.
_LANGUAGES = {
    "pt": "pt-br",
    "en": "en-us",
    "es": "es",
    "fr": "fr-fr",
    "it": "it",
    "ja": "ja",
    "zh": "cmn",
    "hi": "hi",
}

DEFAULT_SPEAKER = "pf_dora"


class KokoroEngine:
    """Sintetiza voz com o modelo Kokoro local."""

    name = "kokoro"

    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings
        self._kokoro: Any | None = None
        self._speaker = settings.speaker or DEFAULT_SPEAKER
        self._language = _LANGUAGES.get(settings.language, "en-us")

    # -- ciclo de vida -----------------------------------------------------
    def warm_up(self) -> None:
        """Carrega o modelo (~1,5 s). Feito no arranque para nao pagar na 1a fala."""
        if self._kokoro is not None:
            return

        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise SpeechError(_INSTALL_HINT) from exc

        model_path = self._settings.model_path
        voices_path = self._settings.resolved_config_path()
        _ensure_files(model_path, voices_path)

        logger.info("carregando voz kokoro: %s", self._speaker)
        try:
            self._kokoro = Kokoro(str(model_path), str(voices_path))
        except Exception as exc:
            raise SpeechError(f"falha ao carregar o modelo Kokoro {model_path}: {exc}") from exc

        available = set(self._kokoro.get_voices())
        if self._speaker not in available:
            raise SpeechError(
                f"voz {self._speaker!r} nao existe no pacote Kokoro "
                f"(ha {len(available)}, por exemplo: {', '.join(sorted(available)[:5])})"
            )

    def close(self) -> None:
        self._kokoro = None

    # -- sintese -----------------------------------------------------------
    def synthesize(self, text: str) -> Iterator[SpeechChunk]:
        if not text.strip():
            return

        self.warm_up()
        assert self._kokoro is not None  # garantido por warm_up

        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=self._speaker,
                speed=1.0 / self._settings.length_scale,
                lang=self._language,
            )
        except Exception as exc:
            raise SpeechError(f"falha na sintese Kokoro: {exc}") from exc

        yield SpeechChunk(
            audio=_to_pcm16(samples),
            format=AudioFormat(sample_rate=sample_rate, channels=1, sample_width=2),
        )


def _to_pcm16(samples: np.ndarray) -> bytes:
    """Converte as amostras de ponto flutuante (-1..1) para PCM de 16 bits."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _ensure_files(model_path: Path, voices_path: Path) -> None:
    if not model_path.is_file():
        raise SpeechError(
            f"modelo Kokoro nao encontrado em {model_path}. Baixe com: roboteye voice download dora"
        )
    if not voices_path.is_file():
        raise SpeechError(
            f"pacote de vozes nao encontrado em {voices_path}. "
            "Baixe com: roboteye voice download dora --force"
        )
