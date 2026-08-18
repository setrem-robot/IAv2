"""Locutor assincrono.

O `Speaker` roda em sua propria thread e consome uma fila de frases. Assim o LLM
continua gerando texto enquanto a frase anterior ainda esta tocando, e a face
continua animando a 60 FPS sem engasgar.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from roboteye.core.events import ErrorOccurred, EventBus, SpeechFinished, SpeechStarted
from roboteye.core.text import clean_for_speech, truncate
from roboteye.logging_setup import get_logger
from roboteye.speech.base import SpeechChunk, SpeechError, TTSEngine
from roboteye.speech.envelope import SpeechEnvelope
from roboteye.speech.player import AudioSink
from roboteye.speech.polish import AudioPolish

logger = get_logger(__name__)

#: Teto de caracteres que o locutor junta numa sintese so.
#:
#: Existe para o caso de uma resposta longa: juntar tudo faria a fala demorar a
#: comecar, que e exatamente o problema que cortar em frases resolve. O valor da
#: folga para as duas ou tres frases de uma resposta falada normal.
MAX_BATCH_CHARS = 320


def synthesize_polished(
    engine: TTSEngine,
    text: str,
    *,
    language: str = "",
    polish: AudioPolish | None = None,
) -> Iterator[SpeechChunk]:
    """Texto cru em audio pronto para tocar.

    Reune os tres passos que sempre andam juntos: limpar e normalizar o texto,
    sintetizar, e dar o acabamento no audio. Existe como funcao — e nao so como
    parte do locutor — para que quem sintetiza fora dele (o comando `say`) ouca
    exatamente o mesmo resultado que o robo produz. Quando os dois caminhos eram
    separados, `say` pulava a normalizacao e o acabamento, e servia mal como
    ferramenta de conferencia justamente por isso.
    """
    cleaned = clean_for_speech(text, language=language)
    if not cleaned:
        return
    yield from (polish or AudioPolish()).process(engine.synthesize(cleaned))


@dataclass(frozen=True, slots=True)
class _Utterance:
    """Uma frase a ser falada, marcada com a geracao em que foi enfileirada."""

    text: str
    generation: int
    end_of_turn: bool = False


@dataclass(slots=True)
class _State:
    """Estado compartilhado entre a thread do locutor e quem o comanda."""

    generation: int = 0
    speaking: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class Speaker:
    """Fala textos em segundo plano, publicando eventos de inicio e fim."""

    def __init__(
        self,
        engine: TTSEngine,
        sink: AudioSink,
        bus: EventBus,
        envelope: SpeechEnvelope | None = None,
        polish: AudioPolish | None = None,
        language: str = "",
    ) -> None:
        self._engine = engine
        self._sink = sink
        self._bus = bus
        #: Medidor lido pela face para animar a fala. Opcional: sem ele o
        #: locutor funciona igual, e a face cai no movimento sintetico.
        self._envelope = envelope
        #: Rampas nas pontas e respiro entre as frases.
        self._polish = polish or AudioPolish()
        #: Idioma da voz, que decide como numeros e abreviacoes sao lidos.
        self._language = language

        self._queue: queue.Queue[_Utterance | None] = queue.Queue()
        #: Item retirado da fila que nao coube no lote atual. So a thread do
        #: locutor mexe aqui, entao nao precisa de cadeado.
        self._held: list[_Utterance | None] = []
        self._state = _State()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None

    # -- ciclo de vida -----------------------------------------------------
    def start(self) -> None:
        """Inicia a thread do locutor. Idempotente."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    def warm_up(self) -> None:
        """Pre-carrega o modelo de voz para que a primeira fala nao gaste esse tempo."""
        self._engine.warm_up()

    @property
    def engine_name(self) -> str:
        return self._engine.name

    def close(self, *, timeout: float = 5.0) -> None:
        """Encerra a thread e libera o dispositivo de audio."""
        if self._thread is None:
            return
        self.interrupt()
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None
        self._sink.close()
        self._engine.close()

    def __enter__(self) -> Speaker:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- comandos ----------------------------------------------------------
    def say(self, text: str) -> None:
        """Enfileira uma frase. Retorna imediatamente.

        Aqui o texto so perde marcacao e emojis. A normalizacao de numeros fica
        para a hora da sintese, de proposito: este texto tambem vira legenda na
        tela, e "15:30" se le melhor do que "quinze e trinta".
        """
        cleaned = clean_for_speech(text)
        if not cleaned:
            return
        with self._state.lock:
            generation = self._state.generation
        self._idle.clear()
        self._queue.put(_Utterance(cleaned, generation))

    def end_turn(self) -> None:
        """Marca o fim de uma resposta: dispara `SpeechFinished` apos a ultima frase."""
        with self._state.lock:
            generation = self._state.generation
        self._queue.put(_Utterance("", generation, end_of_turn=True))

    def interrupt(self) -> None:
        """Cala a boca agora: descarta a fila e corta o audio em reproducao."""
        with self._state.lock:
            self._state.generation += 1

        drained = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:  # preserva o pedido de encerramento
                self._queue.put(None)
                break
            drained += 1

        try:
            self._sink.stop()
        except SpeechError:
            logger.debug("falha ao interromper a saida de audio", exc_info=True)

        if self._envelope is not None:
            # O audio em buffer foi descartado; o envelope que o descrevia
            # tambem precisa ir, senao a face continua falando sozinha.
            self._envelope.end()

        # Todo o trabalho pendente foi descartado e a fala em curso sera abortada
        # no proximo bloco de audio: quem esperava o silencio pode seguir.
        self._idle.set()

        if drained:
            logger.debug("fala interrompida (%d frases descartadas)", drained)

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Bloqueia ate a fila esvaziar. Retorna False se estourar o tempo."""
        return self._idle.wait(timeout)

    @property
    def is_speaking(self) -> bool:
        with self._state.lock:
            return self._state.speaking

    # -- thread ------------------------------------------------------------
    def _run(self) -> None:
        logger.debug("locutor iniciado (motor=%s, saida=%s)", self._engine.name, self._sink.name)
        while True:
            item = self._held.pop() if self._held else self._queue.get()
            if item is None:
                break

            try:
                if not self._is_current(item.generation):
                    continue  # sobra de uma fala interrompida
                if item.end_of_turn:
                    if self._envelope is not None:
                        self._envelope.end()
                    self._bus.publish(SpeechFinished())
                    continue
                self._speak(self._batch(item))
            except SpeechError as exc:
                logger.error("falha ao falar: %s", exc)
                self._bus.publish(ErrorOccurred(message=str(exc), source="speech"))
            except Exception as exc:
                logger.exception("erro inesperado no locutor")
                self._bus.publish(ErrorOccurred(message=str(exc), source="speech"))
            finally:
                if self._queue.empty():
                    self._set_speaking(False)
                    self._idle.set()

        self._set_speaking(False)
        self._idle.set()
        logger.debug("locutor encerrado")

    def _batch(self, first: _Utterance) -> _Utterance:
        """Junta numa unica sintese as frases que ja estao esperando na fila.

        Cortar a resposta em frases serve para comecar a falar antes de o modelo
        terminar de escrever. Mas quando as frases *ja chegaram*, sintetizar uma
        de cada vez so cobra caro: cada frase paga o custo fixo do motor — que
        num motor de rede e uma ida e volta inteira — e esse custo vira silencio
        entre uma e outra. Medindo com a voz online, o buraco passava de um
        segundo.

        Juntar tambem melhora o que se ouve, e nao so o tempo. Um motor de voz
        que recebe as duas frases de uma vez entoa a passagem de uma para a
        outra como quem fala, com a pausa e a respiracao no lugar; recebendo uma
        de cada vez, ele produz duas leituras separadas, e da para ouvir a
        emenda.

        O lote leva so o que ja esta na fila: se o modelo ainda nao escreveu a
        proxima frase, esta sai sozinha e a fala comeca na hora, como antes.
        """
        parts = [first.text]
        length = len(first.text)

        while length < MAX_BATCH_CHARS:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            # Fim de turno, encerramento ou sobra de uma fala ja interrompida:
            # nada disso entra no lote. Guarda para o laco principal tratar.
            if item is None or item.end_of_turn or item.generation != first.generation:
                self._held.append(item)
                break

            parts.append(item.text)
            length += len(item.text) + 1

        if len(parts) == 1:
            return first

        logger.debug("%d frases sintetizadas juntas", len(parts))
        return replace(first, text=" ".join(parts))

    def _speak(self, item: _Utterance) -> None:
        logger.info("falando: %s", truncate(item.text))
        self._set_speaking(True)
        self._bus.publish(SpeechStarted(text=item.text))
        if self._envelope is not None:
            self._envelope.begin()

        stream = synthesize_polished(
            self._engine,
            item.text,
            language=self._language,
            polish=self._polish,
        )
        for chunk in stream:
            if not self._is_current(item.generation):
                logger.debug("fala abortada no meio do audio")
                return
            self._sink.start(chunk.format)
            # O medidor e alimentado antes da escrita: `write` bloqueia ate o
            # bloco caber no buffer, e medir depois disso jogaria o envelope
            # para tras justamente nos trechos longos.
            if self._envelope is not None:
                self._envelope.feed(chunk.audio, chunk.format)
            self._sink.write(chunk.audio)

    def _is_current(self, generation: int) -> bool:
        with self._state.lock:
            return generation == self._state.generation

    def _set_speaking(self, value: bool) -> None:
        with self._state.lock:
            self._state.speaking = value
