"""Acabamento do audio, entre a sintese e o alto-falante.

Um motor de TTS entrega a fala e mais nada: comeca no primeiro sample e termina
no ultimo, sem margem. Mandar isso direto para a placa produz tres incomodos que
somados sao boa parte do que se ouve como "audio estranho":

**Estalo nas pontas.** Se a forma de onda comeca num valor longe de zero, o
alto-falante recebe um degrau — e um degrau e um clique. Umas poucas
milissegundos de rampa em cada ponta resolvem, e sao curtas demais para se
ouvirem como fade.

**Frases coladas.** O texto e sintetizado frase a frase e cada uma vai para a
placa assim que fica pronta, entao a seguinte comeca no exato sample em que a
anterior acabou. Ninguem fala assim: falta o respiro que separa uma frase da
outra. Um rabicho de silencio no fim de cada uma devolve esse respiro.

**Estouro.** Ganho e uma coisa que se quer poder ajustar, mas multiplicar
amostras de 16 bits sem cuidado faz o sinal dar a volta e virar ruido. O
limitador aqui e o piso de seguranca disso.

Tudo neste modulo sao funcoes sobre bytes, sem estado e sem thread: da para
testar o acabamento sem placa de som nenhuma.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from roboteye.speech.base import AudioFormat, SpeechChunk

#: Rampa de entrada. Curta: o suficiente para matar o degrau, nao o bastante
#: para comer o ataque da primeira silaba.
DEFAULT_FADE_IN = 0.006

#: Rampa de saida, um pouco mais longa — cortar o fim e mais audivel que o comeco.
DEFAULT_FADE_OUT = 0.012

#: Silencio no fim de cada frase. E o respiro entre uma e outra.
DEFAULT_TAIL = 0.14


@dataclass(frozen=True, slots=True)
class AudioPolish:
    """Parametros do acabamento aplicado a cada fala."""

    fade_in: float = DEFAULT_FADE_IN
    fade_out: float = DEFAULT_FADE_OUT
    tail: float = DEFAULT_TAIL
    gain: float = 1.0

    def process(self, chunks: Iterable[SpeechChunk]) -> Iterator[SpeechChunk]:
        """Aplica o acabamento a uma fala inteira, sem esperar por ela.

        As rampas precisam saber onde a fala comeca e onde termina, mas segurar
        todo o audio para descobrir isso jogaria fora a maior vantagem do
        projeto — comecar a tocar antes de a sintese acabar. A saida e olhar um
        bloco a frente: basta para saber se o bloco atual e o ultimo, e atrasa a
        reproducao em um bloco, nao na fala toda.
        """
        pending: SpeechChunk | None = None
        first = True

        for chunk in chunks:
            if pending is not None:
                yield self._polish(pending, fade_in=first, fade_out=False)
                first = False
            pending = chunk

        if pending is None:
            return

        yield self._polish(pending, fade_in=first, fade_out=True)
        if self.tail > 0.0:
            yield SpeechChunk(audio=silence(pending.format, self.tail), format=pending.format)

    def _polish(self, chunk: SpeechChunk, *, fade_in: bool, fade_out: bool) -> SpeechChunk:
        samples = to_float(chunk.audio)
        if samples.size == 0:
            return chunk

        if self.gain != 1.0:
            samples = samples * self.gain

        rate = chunk.format.sample_rate * chunk.format.channels
        if fade_in:
            _ramp_in(samples, int(self.fade_in * rate))
        if fade_out:
            _ramp_out(samples, int(self.fade_out * rate))

        return SpeechChunk(audio=to_pcm16(samples), format=chunk.format)


# ---------------------------------------------------------------------------
# Conversao
# ---------------------------------------------------------------------------
def to_float(pcm: bytes) -> np.ndarray:
    """PCM de 16 bits para ponto flutuante em -1..1."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def to_pcm16(samples: np.ndarray) -> bytes:
    """Ponto flutuante de volta para PCM de 16 bits, com limite.

    O corte em -1..1 e o que impede um ganho alto de fazer o sinal dar a volta:
    sem ele, um pico estourado vira o valor mais negativo possivel, e isso se
    ouve como um estalo seco no meio da palavra.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def silence(audio_format: AudioFormat, seconds: float) -> bytes:
    """Um trecho mudo, no formato dado."""
    count = int(max(0.0, seconds) * audio_format.sample_rate) * audio_format.channels
    return np.zeros(count, dtype="<i2").tobytes()


# ---------------------------------------------------------------------------
# Rampas
# ---------------------------------------------------------------------------
def _ramp_in(samples: np.ndarray, length: int) -> None:
    length = min(length, samples.size)
    if length > 1:
        samples[:length] *= np.linspace(0.0, 1.0, length, dtype=np.float32)


def _ramp_out(samples: np.ndarray, length: int) -> None:
    length = min(length, samples.size)
    if length > 1:
        samples[-length:] *= np.linspace(1.0, 0.0, length, dtype=np.float32)
