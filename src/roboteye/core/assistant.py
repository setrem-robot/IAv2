"""Orquestrador: liga mensagem do usuario, LLM e voz.

Roda numa thread propria para que a interface (face ou terminal) nunca bloqueie
esperando o modelo. Cada mensagem recebida vira um "turno":

    usuario -> ThinkingStarted -> LLM (streaming) -> frases -> voz -> SpeechFinished

Como o texto e cortado em frases assim que elas fecham, a primeira frase ja esta
sendo falada enquanto o modelo ainda escreve o resto.
"""

from __future__ import annotations

import queue
import threading

from roboteye.core.events import (
    AssistantReply,
    ErrorOccurred,
    EventBus,
    ThinkingStarted,
    UserMessage,
)
from roboteye.core.text import stream_sentences, truncate
from roboteye.llm.base import LLMClient, LLMError
from roboteye.llm.memory import ConversationMemory
from roboteye.llm.persona import PersonaStore
from roboteye.logging_setup import get_logger
from roboteye.speech.speaker import Speaker

logger = get_logger(__name__)


class Assistant:
    """Consome mensagens do usuario e produz resposta falada."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        memory: ConversationMemory,
        speaker: Speaker,
        bus: EventBus,
        persona: PersonaStore | None = None,
        language: str = "en",
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._speaker = speaker
        self._bus = bus
        self._persona = persona
        self._language = language

        self._inbox: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._busy = threading.Event()
        self._closing = threading.Event()

    # -- ciclo de vida -----------------------------------------------------
    def start(self) -> None:
        """Inicia a thread do assistente. Idempotente."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="assistant", daemon=True)
        self._thread.start()

    def close(self, *, timeout: float = 5.0) -> None:
        if self._thread is None:
            return

        # Sinaliza antes do join: se o modelo ainda estiver respondendo, fechar o
        # cliente HTTP aborta o stream de proposito, e o erro daí decorrente nao
        # deve virar uma mensagem de falha na tela do usuario.
        self._closing.set()
        self._inbox.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None
        self._llm.close()

    def __enter__(self) -> Assistant:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- entrada -----------------------------------------------------------
    def submit(self, text: str) -> None:
        """Envia uma mensagem do usuario. Interrompe a fala em andamento."""
        message = text.strip()
        if not message:
            return

        # Barge-in: quem fala por ultimo e o usuario.
        self._speaker.interrupt()
        self._bus.publish(UserMessage(text=message))
        self._inbox.put(message)

    def say_directly(self, text: str) -> None:
        """Faz o robo falar um texto fixo, sem passar pelo modelo.

        Usado para saudacoes e falas ociosas.
        """
        self._bus.publish(AssistantReply(text=text))
        self._speaker.say(text)
        self._speaker.end_turn()

    def interrupt(self) -> None:
        """Cala a fala em andamento sem descartar o historico."""
        self._speaker.interrupt()

    @property
    def memory(self) -> ConversationMemory:
        """Historico da conversa (usado pelo comando `/limpar`)."""
        return self._memory

    # -- ensinar -----------------------------------------------------------
    def teach(self, fact: str) -> bool:
        """Ensina um fato, que passa a valer da proxima resposta em diante."""
        if self._persona is None:
            return False
        if not self._persona.remember(fact):
            return False
        self.reload_persona()
        return True

    def forget(self, needle: str) -> int:
        """Esquece os fatos que contenham `needle`. Devolve quantos sairam."""
        if self._persona is None:
            return 0
        removidos = self._persona.forget(needle)
        if removidos:
            self.reload_persona()
        return removidos

    def facts(self) -> tuple[str, ...]:
        """Tudo que ela aprendeu ate agora."""
        return self._persona.load_facts() if self._persona else ()

    def reload_persona(self) -> None:
        """Rele a persona do disco e troca o prompt de sistema em uso.

        Permite editar o arquivo com o robo rodando e ver o efeito na proxima
        resposta, sem reiniciar nada.
        """
        if self._persona is None:
            return
        persona = self._persona.load(self._language)
        self._memory.replace_system_prompt(persona.system_prompt())
        logger.info("persona recarregada (%d fatos)", len(persona.facts))

    @property
    def is_busy(self) -> bool:
        return self._busy.is_set()

    # -- thread ------------------------------------------------------------
    def _run(self) -> None:
        logger.debug("assistente iniciado (llm=%s)", self._llm.name)
        while True:
            message = self._inbox.get()
            if message is None:
                break

            # Se o usuario mandou varias mensagens de uma vez, responde so a ultima.
            message = self._latest(message)

            self._busy.set()
            try:
                self._handle_turn(message)
            except LLMError as exc:
                self._report(exc, source="llm")
            except Exception as exc:
                self._report(exc, source="assistant")
            finally:
                self._busy.clear()

        logger.debug("assistente encerrado")

    def _report(self, exc: Exception, *, source: str) -> None:
        """Registra uma falha do turno, exceto quando ela vem do encerramento."""
        if self._closing.is_set():
            # Fechar o cliente HTTP durante um stream levanta erro de socket:
            # e o mecanismo de cancelamento, nao um problema a relatar.
            logger.debug("falha durante o encerramento, ignorada: %s", exc)
            return

        logger.error("falha em %s: %s", source, exc)
        self._bus.publish(ErrorOccurred(message=str(exc), source=source))

    def _latest(self, message: str) -> str:
        """Descarta mensagens enfileiradas mais antigas, mantendo a ultima."""
        while True:
            try:
                queued = self._inbox.get_nowait()
            except queue.Empty:
                return message
            if queued is None:
                self._inbox.put(None)
                return message
            message = queued

    def _handle_turn(self, message: str) -> None:
        logger.info("usuario: %s", truncate(message))
        self._memory.add_user(message)
        self._bus.publish(ThinkingStarted())

        spoken: list[str] = []
        for sentence in stream_sentences(self._llm.stream_reply(self._memory.build_prompt())):
            spoken.append(sentence)
            self._speaker.say(sentence)

        reply = " ".join(spoken).strip()
        if not reply:
            logger.warning("o modelo devolveu uma resposta vazia")
            self._speaker.end_turn()
            return

        self._memory.add_assistant(reply)
        self._bus.publish(AssistantReply(text=reply))
        self._speaker.end_turn()
        logger.info("assistente: %s", truncate(reply))
