"""Testes da persona em arquivo e da memória ensinável."""

from __future__ import annotations

from pathlib import Path

import pytest

from roboteye.llm.persona import Persona, PersonaStore, create_default_persona


@pytest.fixture
def store(tmp_path: Path) -> PersonaStore:
    return PersonaStore(tmp_path, "atlas")


class TestCarregamento:
    def test_usa_a_persona_embutida_sem_arquivo(self, store: PersonaStore) -> None:
        assert "Atlas" in store.load().identity

    def test_le_o_arquivo_quando_existe(self, store: PersonaStore) -> None:
        store.identity_path.write_text("Você é um farol solitário.", encoding="utf-8")
        assert "farol solitário" in store.load().identity

    def test_arquivo_vazio_cai_no_padrao(self, store: PersonaStore) -> None:
        store.identity_path.write_text("   \n  ", encoding="utf-8")
        assert "Atlas" in store.load().identity

    def test_personas_diferentes_usam_arquivos_diferentes(self, tmp_path: Path) -> None:
        atlas = PersonaStore(tmp_path, "atlas")
        jarvis = PersonaStore(tmp_path, "jarvis")

        atlas.identity_path.write_text("Sou a Atlas.", encoding="utf-8")
        jarvis.identity_path.write_text("Sou o Jarvis.", encoding="utf-8")

        assert "Atlas" in atlas.load().identity
        assert "Jarvis" in jarvis.load().identity


class TestPromptDeSistema:
    def test_junta_identidade_fatos_e_regras(self, store: PersonaStore) -> None:
        store.identity_path.write_text("Você é um farol.", encoding="utf-8")
        store.remember("o mar está calmo hoje")

        prompt = store.load("pt").system_prompt()

        assert "farol" in prompt
        assert "o mar está calmo hoje" in prompt
        assert "Brazilian Portuguese" in prompt

    def test_sem_fatos_nao_cria_secao_vazia(self, store: PersonaStore) -> None:
        assert "ensinado por quem te construiu" not in store.load().system_prompt()

    def test_regras_de_saida_estao_sempre_presentes(self, store: PersonaStore) -> None:
        """São requisito do TTS, não do personagem: não podem sumir."""
        store.identity_path.write_text("Apenas isso.", encoding="utf-8")
        prompt = store.load().system_prompt().lower()

        assert "markdown" in prompt
        assert "duas frases" in prompt

    def test_persona_montada_a_mao_tambem_funciona(self) -> None:
        persona = Persona(name="teste", identity="Você existe.", facts=("o céu é azul",))
        prompt = persona.system_prompt()

        assert "Você existe." in prompt
        assert "o céu é azul" in prompt


class TestAprender:
    def test_guarda_um_fato(self, store: PersonaStore) -> None:
        assert store.remember("meu nome é Kerlon")
        assert "meu nome é Kerlon" in store.load_facts()

    def test_nao_guarda_duas_vezes(self, store: PersonaStore) -> None:
        assert store.remember("o robô se chama Bifrost")
        assert not store.remember("o robô se chama Bifrost")
        assert len(store.load_facts()) == 1

    def test_ignora_fato_vazio(self, store: PersonaStore) -> None:
        assert not store.remember("   ")
        assert store.load_facts() == ()

    def test_guarda_varios_na_ordem(self, store: PersonaStore) -> None:
        for fato in ("primeiro", "segundo", "terceiro"):
            store.remember(fato)
        assert store.load_facts() == ("primeiro", "segundo", "terceiro")

    def test_sobrevive_entre_instancias(self, tmp_path: Path) -> None:
        """O ponto do recurso: o que se ensina hoje continua valendo amanhã."""
        PersonaStore(tmp_path, "atlas").remember("o laboratorio fecha as dez")
        assert "o laboratorio fecha as dez" in PersonaStore(tmp_path, "atlas").load_facts()

    def test_arquivo_e_editavel_a_mao(self, store: PersonaStore) -> None:
        store.memory_path.write_text(
            "# um comentário\n\n- fato um\n- fato dois\n", encoding="utf-8"
        )
        assert store.load_facts() == ("fato um", "fato dois")

    def test_aceita_linhas_sem_tracinho(self, store: PersonaStore) -> None:
        store.memory_path.write_text("fato solto\n", encoding="utf-8")
        assert store.load_facts() == ("fato solto",)


class TestEsquecer:
    def test_remove_por_trecho(self, store: PersonaStore) -> None:
        store.remember("meu nome é Kerlon")
        store.remember("o robô é azul")

        assert store.forget("Kerlon") == 1
        assert store.load_facts() == ("o robô é azul",)

    def test_busca_ignora_caixa(self, store: PersonaStore) -> None:
        store.remember("Meu Nome É Kerlon")
        assert store.forget("kerlon") == 1

    def test_remove_todos_os_que_batem(self, store: PersonaStore) -> None:
        store.remember("gosto de café")
        store.remember("café é melhor de manhã")
        store.remember("chá também serve")

        assert store.forget("café") == 2
        assert store.load_facts() == ("chá também serve",)

    def test_sem_memoria_nao_quebra(self, store: PersonaStore) -> None:
        assert store.forget("qualquer coisa") == 0

    def test_trecho_vazio_nao_apaga_nada(self, store: PersonaStore) -> None:
        store.remember("um fato")
        assert store.forget("  ") == 0
        assert len(store.load_facts()) == 1

    def test_preserva_os_comentarios_do_arquivo(self, store: PersonaStore) -> None:
        store.memory_path.write_text("# cabeçalho\n- some\n- fica\n", encoding="utf-8")
        store.forget("some")
        assert "# cabeçalho" in store.memory_path.read_text(encoding="utf-8")


class TestPersonaInicial:
    def test_cria_o_arquivo(self, tmp_path: Path) -> None:
        caminho = create_default_persona(tmp_path, "atlas")
        assert caminho.is_file()
        assert "Atlas" in caminho.read_text(encoding="utf-8")

    def test_nao_sobrescreve_o_que_existe(self, tmp_path: Path) -> None:
        caminho = tmp_path / "atlas.md"
        caminho.write_text("meu texto", encoding="utf-8")

        create_default_persona(tmp_path, "atlas")

        assert caminho.read_text(encoding="utf-8") == "meu texto"


class TestAssistantEnsina:
    """O caminho que o usuário percorre: /lembrar muda a resposta seguinte."""

    def test_ensinar_entra_no_prompt(self, make_assistant, make_llm, tmp_path) -> None:
        llm = make_llm("Uma resposta suficientemente longa para o teste.")
        assistant, memory = make_assistant(llm, persona_dir=tmp_path)

        assert assistant.teach("o robô se chama Bifrost")

        prompt_sistema = memory.build_prompt()[0].content
        assert "o robô se chama Bifrost" in prompt_sistema

    def test_esquecer_sai_do_prompt(self, make_assistant, make_llm, tmp_path) -> None:
        assistant, memory = make_assistant(make_llm(), persona_dir=tmp_path)
        assistant.teach("o robô se chama Bifrost")

        assert assistant.forget("Bifrost") == 1
        assert "Bifrost" not in memory.build_prompt()[0].content

    def test_lista_o_que_aprendeu(self, make_assistant, make_llm, tmp_path) -> None:
        assistant, _ = make_assistant(make_llm(), persona_dir=tmp_path)
        assistant.teach("primeiro fato")
        assistant.teach("segundo fato")

        assert assistant.facts() == ("primeiro fato", "segundo fato")

    def test_sem_persona_ensinar_nao_quebra(self, make_assistant, make_llm) -> None:
        assistant, _ = make_assistant(make_llm())
        assert not assistant.teach("qualquer coisa")
        assert assistant.forget("qualquer coisa") == 0
        assert assistant.facts() == ()

    def test_recarregar_pega_a_edicao_do_arquivo(self, make_assistant, make_llm, tmp_path) -> None:
        assistant, memory = make_assistant(make_llm(), persona_dir=tmp_path)

        (tmp_path / "atlas.md").write_text("Você agora é um farol.", encoding="utf-8")
        assistant.reload_persona()

        assert "farol" in memory.build_prompt()[0].content
