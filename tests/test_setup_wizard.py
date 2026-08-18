"""Testes do assistente de primeira configuração."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from roboteye import setup_wizard
from roboteye.config import Settings
from roboteye.llm.probe import ProbeResult
from roboteye.setup_wizard import Answers, Prompt, run_setup
from roboteye.web import envfile

EXEMPLO = """\
# Comentário que explica a chave abaixo.
ROBOTEYE_LLM_BACKEND=ollama
ROBOTEYE_OLLAMA_HOST=http://localhost:11434
ROBOTEYE_LLM_MODEL=llama3.2:1b

# Voz
ROBOTEYE_VOICE=francisca
ROBOTEYE_PERSONA=atlas
"""


@pytest.fixture
def projeto(tmp_path: Path) -> Path:
    """Um repositório recém-clonado: tem o exemplo, não tem o `.env`."""
    (tmp_path / ".env.example").write_text(EXEMPLO, encoding="utf-8")
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "atlas.md").write_text(
        "<!-- comentário do arquivo -->\n# Quem você é\nVocê é a Atlas, da Setrem.\n",
        encoding="utf-8",
    )
    (persona / "iris.md").write_text(
        "# Quem você é\nVocê é a Íris, uma IA de bordo.\n", encoding="utf-8"
    )
    (persona / "atlas.memoria.md").write_text("fato\n", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def sem_rede_nem_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """O assistente nunca toca na rede durante os testes."""
    monkeypatch.setattr(
        setup_wizard,
        "probe_ollama",
        lambda host, **_: ProbeResult(
            ok=True,
            host=setup_wizard.normalize_host(host),
            latency_ms=7,
            models=("llama3.2:1b", "qwen3:8b"),
        ),
    )
    monkeypatch.setattr(
        "roboteye.voices.download_voice",
        lambda chave, **_: Path(f"/models/{chave}.onnx"),
    )


def falas(respostas: list[str]) -> Iterator[str]:
    return iter(respostas)


def prompt_com(respostas: list[str], saida: list[str] | None = None) -> Prompt:
    """Um terminal dublê: lê de uma fila e guarda o que foi impresso."""
    fila = falas(respostas)
    registro = saida if saida is not None else []
    return Prompt(read=lambda _: next(fila, ""), write=registro.append, interactive=True)


class TestPrompt:
    def test_enter_aceita_o_padrao(self) -> None:
        assert prompt_com([""]).text("modelo", "qwen3:8b") == "qwen3:8b"

    def test_escolha_por_numero(self) -> None:
        prompt = prompt_com(["2"])
        opcoes = [("a", ""), ("b", ""), ("c", "")]
        assert prompt.choice("qual", opcoes, default="a") == "b"

    def test_escolha_pelo_proprio_nome(self) -> None:
        prompt = prompt_com(["c"])
        assert prompt.choice("qual", [("a", ""), ("c", "")], default="a") == "c"

    def test_resposta_invalida_pergunta_de_novo(self) -> None:
        saida: list[str] = []
        prompt = prompt_com(["9", "1"], saida)
        assert prompt.choice("qual", [("a", ""), ("b", "")], default="a") == "a"
        assert any("nao entendi" in linha for linha in saida)

    def test_sem_terminal_devolve_o_padrao_sem_perguntar(self) -> None:
        def explode(_: str) -> str:
            raise AssertionError("nao deveria ter perguntado")

        prompt = Prompt(read=explode, write=lambda _: None, interactive=False)
        assert prompt.text("modelo", "llama3.2:1b") == "llama3.2:1b"
        assert prompt.choice("qual", [("a", "")], default="a") == "a"
        assert prompt.yes_no("baixar?", default=False) is False

    @pytest.mark.parametrize(
        ("resposta", "esperado"), [("s", True), ("sim", True), ("n", False), ("", True)]
    )
    def test_sim_ou_nao(self, resposta: str, esperado: bool) -> None:
        assert prompt_com([resposta]).yes_no("baixar?", default=True) is esperado


class TestGravacao:
    def test_cria_o_env_a_partir_do_exemplo(self, projeto: Path) -> None:
        run_setup(
            Settings.from_env(env_file=projeto / ".nao-existe"),
            Answers(ollama="1.2.3.4", model="qwen3:8b", voice="dii", persona="iris"),
            Prompt(write=lambda _: None, interactive=False),
            env_path=projeto / ".env",
            project_root=projeto,
        )

        conteudo = (projeto / ".env").read_text(encoding="utf-8")
        # O comentário do exemplo é a documentação que fica na máquina: editar o
        # arquivo não pode apagá-lo.
        assert "# Comentário que explica a chave abaixo." in conteudo

        valores = envfile.read(projeto / ".env")
        assert valores["ROBOTEYE_OLLAMA_HOST"] == "http://1.2.3.4:11434"
        assert valores["ROBOTEYE_LLM_MODEL"] == "qwen3:8b"
        assert valores["ROBOTEYE_VOICE"] == "dii"
        assert valores["ROBOTEYE_PERSONA"] == "iris"

    def test_sem_ia_configura_o_modo_echo(self, projeto: Path) -> None:
        plano = run_setup(
            Settings.from_env(env_file=projeto / ".nao-existe"),
            Answers(no_llm=True, voice="dii", persona="atlas"),
            Prompt(write=lambda _: None, interactive=False),
            env_path=projeto / ".env",
            project_root=projeto,
        )

        assert plano.values["ROBOTEYE_LLM_BACKEND"] == "echo"
        # Sem IA não se grava endereço nenhum: o que estava lá continua valendo
        # para quando a máquina da IA voltar.
        assert "ROBOTEYE_OLLAMA_HOST" not in plano.values

    def test_baixa_a_voz_escolhida_e_a_reserva(
        self, projeto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pedidas: list[str] = []
        monkeypatch.setattr(
            "roboteye.voices.download_voice",
            lambda chave, **_: pedidas.append(chave) or Path("x"),
        )
        monkeypatch.setattr(setup_wizard.voice_catalog, "needs_download", lambda _: True)

        plano = run_setup(
            Settings.from_env(env_file=projeto / ".nao-existe"),
            Answers(no_llm=True, voice="francisca", persona="atlas"),
            Prompt(write=lambda _: None, interactive=False),
            env_path=projeto / ".env",
            project_root=projeto,
        )

        # `francisca` fala pela nuvem; a reserva offline é o arquivo que precisa
        # estar no disco antes de a internet cair.
        assert pedidas[0] == "francisca"
        assert len(pedidas) == 2
        assert plano.downloads == tuple(pedidas)

    def test_skip_download_nao_baixa_nada(
        self, projeto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "roboteye.voices.download_voice",
            lambda *_, **__: pytest.fail("não deveria baixar"),
        )
        plano = run_setup(
            Settings.from_env(env_file=projeto / ".nao-existe"),
            Answers(no_llm=True, voice="dii", skip_download=True),
            Prompt(write=lambda _: None, interactive=False),
            env_path=projeto / ".env",
            project_root=projeto,
        )
        assert plano.downloads == ()


class TestDialogo:
    def test_fluxo_completo_escolhendo_pelos_numeros(self, projeto: Path) -> None:
        saida: list[str] = []
        # onde=1 (local), modelo=2 (qwen3:8b), voz=1, persona=2
        prompt = prompt_com(["1", "2", "1", "2"], saida)

        plano = run_setup(
            Settings.from_env(env_file=projeto / ".nao-existe"),
            Answers(skip_download=True),
            prompt,
            env_path=projeto / ".env",
            project_root=projeto,
        )

        assert plano.values["ROBOTEYE_OLLAMA_HOST"] == "http://localhost:11434"
        assert plano.values["ROBOTEYE_LLM_MODEL"] == "qwen3:8b"
        assert plano.values["ROBOTEYE_PERSONA"] == "iris"
        assert any("respondeu em 7 ms" in linha for linha in saida)

    def test_escolher_nenhuma_ia_nao_testa_endereco(
        self, projeto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            setup_wizard,
            "probe_ollama",
            lambda *_, **__: pytest.fail("não deveria sondar sem IA"),
        )
        # onde=3 (nenhuma), voz=1, persona=1
        plano = run_setup(
            Settings.from_env(env_file=projeto / ".nao-existe"),
            Answers(skip_download=True),
            prompt_com(["3", "1", "1"]),
            env_path=projeto / ".env",
            project_root=projeto,
        )
        assert plano.values["ROBOTEYE_LLM_BACKEND"] == "echo"

    def test_endereco_que_falha_pode_ser_corrigido_na_hora(
        self, projeto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tentativas: list[str] = []

        def sonda(host: str, **_: object) -> ProbeResult:
            tentativas.append(host)
            if "9.9.9.9" in host:
                return ProbeResult(ok=False, host=host, error="conexao recusada")
            return ProbeResult(ok=True, host=setup_wizard.normalize_host(host), models=("m:1b",))

        monkeypatch.setattr(setup_wizard, "probe_ollama", sonda)
        saida: list[str] = []
        # onde=2 (rede) -> endereço errado -> corrige -> modelo -> voz -> persona
        prompt = prompt_com(["2", "9.9.9.9", "192.168.0.7", "1", "1", "1"], saida)

        plano = run_setup(
            Settings.from_env(env_file=projeto / ".nao-existe"),
            Answers(skip_download=True),
            prompt,
            env_path=projeto / ".env",
            project_root=projeto,
        )

        assert tentativas == ["9.9.9.9", "192.168.0.7"]
        assert plano.values["ROBOTEYE_OLLAMA_HOST"] == "http://192.168.0.7:11434"
        assert any("falhou: conexao recusada" in linha for linha in saida)

    def test_maquina_sem_modelo_sugere_e_ensina_o_pull(
        self, projeto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            setup_wizard,
            "probe_ollama",
            lambda host, **_: ProbeResult(ok=True, host=setup_wizard.normalize_host(host)),
        )
        # Sem `ollama` nesta máquina, o assistente ensina o comando em vez de tentar.
        monkeypatch.setattr(setup_wizard.shutil, "which", lambda _: None)
        saida: list[str] = []
        prompt = prompt_com(["1", "1", "1", "1"], saida)

        plano = run_setup(
            Settings.from_env(env_file=projeto / ".nao-existe"),
            Answers(skip_download=True),
            prompt,
            env_path=projeto / ".env",
            project_root=projeto,
        )

        assert plano.values["ROBOTEYE_LLM_MODEL"] == setup_wizard.MODEL_SUGGESTIONS[0][0]
        assert any("ollama pull" in linha for linha in saida)


class TestPersonas:
    def test_lista_ignora_os_arquivos_de_memoria(self, projeto: Path) -> None:
        encontradas = setup_wizard.available_personas(projeto / "persona")
        assert [nome for nome, _ in encontradas] == ["atlas", "iris"]
        # O título é o mesmo nas duas ("Quem você é"); o que distingue uma da
        # outra é a primeira frase, e é ela que a lista mostra.
        assert encontradas[0][1] == "Você é a Atlas, da Setrem."
        assert encontradas[1][1] == "Você é a Íris, uma IA de bordo."

    def test_diretorio_vazio_nao_quebra(self, tmp_path: Path) -> None:
        assert setup_wizard.available_personas(tmp_path) == []
