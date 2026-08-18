"""Teste de integração da janela da face.

Roda o loop de verdade com o driver `dummy` do SDL, para pegar erros que só
aparecem com a janela montada (fonte, redimensionamento, reação a eventos).
"""

from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("pygame")

from roboteye.config import FaceSettings
from roboteye.core.events import (
    AssistantReply,
    ErrorOccurred,
    EventBus,
    Shutdown,
    SpeechFinished,
    SpeechStarted,
    ThinkingStarted,
    UserMessage,
)
from roboteye.face.app import HINT_TEXT, FaceApp
from roboteye.face.expressions import Expression


@pytest.fixture
def settings() -> FaceSettings:
    return FaceSettings(enabled=True, fullscreen=False, width=640, height=480, fps=60)


def rodar_por(face: FaceApp, segundos: float = 0.4) -> None:
    """Executa o loop da face por um instante e o encerra."""
    parar = threading.Timer(segundos, face.request_stop)
    parar.start()
    try:
        face.run()
    finally:
        parar.cancel()


class TestFaceApp:
    def test_loop_roda_e_encerra_limpo(self, settings: FaceSettings, bus: EventBus) -> None:
        rodar_por(FaceApp(settings, bus))

    def test_reage_aos_eventos_do_sistema(self, settings: FaceSettings, bus: EventBus) -> None:
        face = FaceApp(settings, bus)

        def publicar() -> None:
            time.sleep(0.05)
            bus.publish(UserMessage(text="olá"))
            bus.publish(ThinkingStarted())
            time.sleep(0.05)
            bus.publish(SpeechStarted(text="uma resposta qualquer"))
            time.sleep(0.05)
            bus.publish(SpeechFinished())
            bus.publish(AssistantReply(text="uma resposta qualquer"))

        threading.Thread(target=publicar, daemon=True).start()
        rodar_por(face, 0.4)

    def test_erro_deixa_a_face_brava(self, settings: FaceSettings, bus: EventBus) -> None:
        face = FaceApp(settings, bus)

        threading.Timer(0.05, lambda: bus.publish(ErrorOccurred(message="deu ruim"))).start()
        rodar_por(face, 0.3)

        assert face._animator.current_expression is Expression.ANGRY

    def test_evento_de_encerramento_para_o_loop(
        self, settings: FaceSettings, bus: EventBus
    ) -> None:
        face = FaceApp(settings, bus)

        inicio = time.perf_counter()
        threading.Timer(0.1, lambda: bus.publish(Shutdown())).start()
        face.run()  # deve retornar sozinho, sem o timer de segurança

        assert time.perf_counter() - inicio < 5.0

    def test_ajuda_mostra_a_taxa_de_quadros(self, settings: FaceSettings, bus: EventBus) -> None:
        face = FaceApp(settings, bus)
        rodar_por(face, 0.5)  # tempo de sobra para o Clock formar a media

        assert face._hint().startswith(HINT_TEXT)
        assert face._hint().endswith("FPS")
        assert face._fps > 0.0

    def test_ajuda_escondida_nao_mostra_nada(self, settings: FaceSettings, bus: EventBus) -> None:
        face = FaceApp(settings, bus, show_hint=False)
        rodar_por(face, 0.3)

        assert face._hint() == ""

    def test_face_em_tela_cheia_monta(self, bus: EventBus) -> None:
        settings = FaceSettings(fullscreen=True, width=640, height=480, fps=60)
        rodar_por(FaceApp(settings, bus), 0.2)
