"""Testes do orquestrador e da memória de conversa."""

from __future__ import annotations

from roboteye.core.events import (
    AssistantReply,
    ErrorOccurred,
    SpeechFinished,
    ThinkingStarted,
    UserMessage,
)
from roboteye.llm.base import LLMError
from roboteye.llm.memory import ConversationMemory


class BrokenLLM:
    """Cliente que sempre falha, para exercitar o tratamento de erro."""

    name = "broken"

    def stream_reply(self, messages):
        raise LLMError("ollama fora do ar")
        yield  # pragma: no cover - torna a função um gerador

    def is_available(self) -> bool:
        return False

    def close(self) -> None: ...


class TestAssistant:
    def test_turno_completo_na_ordem_certa(self, make_assistant, make_llm, recorder) -> None:
        llm = make_llm("Que pergunta previsível. Tente de novo, com mais esforço.")
        assistant, _ = make_assistant(llm)

        assistant.submit("olá")

        assert recorder.wait_for(SpeechFinished, timeout=10)
        tipos = recorder.type_names()
        assert tipos.index("UserMessage") < tipos.index("ThinkingStarted")
        assert tipos.index("ThinkingStarted") < tipos.index("SpeechStarted")
        assert "AssistantReply" in tipos

    def test_resposta_e_entregue_a_voz_frase_a_frase(
        self, make_assistant, make_llm, recorder
    ) -> None:
        """O assistente entrega cada frase assim que ela fecha.

        É o que permite começar a falar antes de o modelo terminar de escrever.
        A conferência é feita na fronteira com o locutor, e não no motor de voz:
        o que o motor recebe é decisão do locutor, que junta numa síntese só as
        frases que já chegaram — e é assim que deve ser.
        """
        llm = make_llm("Primeira frase suficientemente longa. Segunda frase igualmente longa.")
        assistant, _ = make_assistant(llm)

        entregues: list[str] = []
        speaker = assistant._speaker
        original = speaker.say
        speaker.say = lambda texto: (entregues.append(texto), original(texto))[1]  # type: ignore[method-assign]

        assistant.submit("olá")

        assert recorder.wait_for(SpeechFinished, timeout=10)
        assert len(entregues) == 2, f"esperava 2 frases, recebi {entregues}"

    def test_pergunta_do_usuario_chega_ao_modelo(self, make_assistant, make_llm, recorder) -> None:
        llm = make_llm("Uma resposta suficientemente longa para o teste.")
        assistant, _ = make_assistant(llm)

        assistant.submit("qual é a resposta?")

        assert recorder.wait_for(SpeechFinished, timeout=10)
        ultima = llm.prompts[-1][-1]
        assert ultima.role == "user"
        assert ultima.content == "qual é a resposta?"

    def test_prompt_de_sistema_vai_junto(self, make_assistant, make_llm, recorder) -> None:
        llm = make_llm("Uma resposta suficientemente longa para o teste.")
        assistant, _ = make_assistant(llm, system_prompt="seja sarcástica")

        assistant.submit("olá")

        assert recorder.wait_for(SpeechFinished, timeout=10)
        primeira = llm.prompts[-1][0]
        assert primeira.role == "system"
        assert primeira.content == "seja sarcástica"

    def test_mensagem_vazia_e_ignorada(self, make_assistant, make_llm, recorder) -> None:
        assistant, _ = make_assistant(make_llm())

        assistant.submit("   ")

        assert recorder.of_type(UserMessage) == []

    def test_historico_guarda_pergunta_e_resposta(self, make_assistant, make_llm, recorder) -> None:
        llm = make_llm("Uma resposta bastante longa para o teste funcionar.")
        assistant, memory = make_assistant(llm)

        assistant.submit("qual é a resposta?")

        assert recorder.wait_for(SpeechFinished, timeout=10)
        assert len(memory) == 2  # pergunta + resposta

    def test_erro_do_llm_vira_evento(self, make_assistant, recorder) -> None:
        assistant, _ = make_assistant(BrokenLLM())

        assistant.submit("olá")

        assert recorder.wait_for(ErrorOccurred, timeout=10)
        assert "ollama fora do ar" in recorder.of_type(ErrorOccurred)[0].message

    def test_erro_nao_derruba_o_assistente(self, make_assistant, make_llm, recorder) -> None:
        # Depois de falhar, o assistente ainda deve atender a próxima mensagem.
        assistant, _ = make_assistant(BrokenLLM())
        assistant.submit("primeira")
        assert recorder.wait_for(ErrorOccurred, timeout=10)

        llm = make_llm("Agora sim, uma resposta suficientemente longa.")
        assistant2, _ = make_assistant(llm)
        assistant2.submit("segunda")

        assert recorder.wait_for(SpeechFinished, timeout=10)

    def test_say_directly_nao_usa_o_llm(self, make_assistant, make_llm, recorder) -> None:
        llm = make_llm()
        assistant, _ = make_assistant(llm)

        assistant.say_directly("Bem-vindo de volta ao centro de testes.")

        assert recorder.wait_for(SpeechFinished, timeout=10)
        assert llm.prompts == []
        assert recorder.of_type(ThinkingStarted) == []
        assert recorder.of_type(AssistantReply)

    def test_interrupt_silencia_a_fala(self, make_assistant, make_llm, sink) -> None:
        assistant, _ = make_assistant(make_llm())

        assistant.interrupt()

        assert sink.stops >= 1


class TestConversationMemory:
    def test_prompt_comeca_com_a_mensagem_de_sistema(self) -> None:
        memory = ConversationMemory("seja breve")
        memory.add_user("olá")

        prompt = memory.build_prompt()

        assert prompt[0].role == "system"
        assert prompt[0].content == "seja breve"
        assert prompt[1].content == "olá"

    def test_janela_descarta_as_mensagens_antigas(self) -> None:
        memory = ConversationMemory("sistema", max_messages=2)
        memory.add_user("primeira")
        memory.add_user("segunda")
        memory.add_user("terceira")

        conteudos = [m.content for m in memory.build_prompt()[1:]]

        assert conteudos == ["segunda", "terceira"]

    def test_mensagens_vazias_sao_ignoradas(self) -> None:
        memory = ConversationMemory("sistema")
        memory.add_user("   ")

        assert len(memory) == 0

    def test_clear_apaga_o_historico_mas_mantem_o_sistema(self) -> None:
        memory = ConversationMemory("sistema")
        memory.add_user("olá")
        memory.clear()

        assert len(memory) == 0
        assert memory.build_prompt()[0].role == "system"

    def test_troca_do_prompt_de_sistema(self) -> None:
        memory = ConversationMemory("antigo")
        memory.replace_system_prompt("novo")

        assert memory.build_prompt()[0].content == "novo"
