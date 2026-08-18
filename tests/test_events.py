"""Testes do barramento de eventos."""

from __future__ import annotations

import queue

from roboteye.core.events import (
    AssistantReply,
    Event,
    EventBus,
    SpeechFinished,
    UserMessage,
    queue_subscriber,
)


class TestEventBus:
    def test_entrega_a_todos_os_assinantes(self, bus: EventBus) -> None:
        recebidos: list[Event] = []
        bus.subscribe(recebidos.append)
        bus.subscribe(recebidos.append)

        bus.publish(UserMessage(text="olá"))

        assert len(recebidos) == 2

    def test_filtra_por_tipo(self, bus: EventBus) -> None:
        somente_falas: list[Event] = []
        bus.subscribe(somente_falas.append, event_type=SpeechFinished)

        bus.publish(UserMessage(text="olá"))
        bus.publish(SpeechFinished())

        assert len(somente_falas) == 1

    def test_handler_com_erro_nao_afeta_os_demais(self, bus: EventBus) -> None:
        def explode(_: Event) -> None:
            raise RuntimeError("falha proposital")

        recebidos: list[Event] = []
        bus.subscribe(explode)
        bus.subscribe(recebidos.append)

        bus.publish(AssistantReply(text="oi"))

        assert len(recebidos) == 1

    def test_unsubscribe_para_de_entregar(self, bus: EventBus) -> None:
        recebidos: list[Event] = []
        bus.subscribe(recebidos.append)
        bus.unsubscribe(recebidos.append)

        bus.publish(UserMessage(text="olá"))

        assert recebidos == []

    def test_queue_subscriber_enfileira(self, bus: EventBus) -> None:
        fila: queue.Queue[Event] = queue.Queue()
        bus.subscribe(queue_subscriber(fila))

        bus.publish(UserMessage(text="olá"))

        assert isinstance(fila.get_nowait(), UserMessage)


class TestEventos:
    def test_carregam_timestamp(self) -> None:
        assert UserMessage(text="oi").timestamp > 0

    def test_sao_imutaveis(self) -> None:
        evento = UserMessage(text="oi")
        try:
            evento.text = "outro"  # type: ignore[misc]
        except (AttributeError, TypeError):
            return
        raise AssertionError("o evento deveria ser imutável")
