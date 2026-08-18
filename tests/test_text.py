"""Testes da segmentação de texto entre o LLM e a voz."""

from __future__ import annotations

import pytest

from roboteye.core.text import clean_for_speech, split_sentences, stream_sentences, truncate


class TestSplitSentences:
    def test_divide_em_frases_preservando_pontuacao(self) -> None:
        assert split_sentences("Olá. Tudo bem? Claro!") == ["Olá.", "Tudo bem?", "Claro!"]

    def test_texto_sem_pontuacao_final_vira_uma_frase(self) -> None:
        assert split_sentences("sem ponto final") == ["sem ponto final"]

    def test_texto_vazio_nao_gera_frases(self) -> None:
        assert split_sentences("   ") == []

    def test_reticencias_no_meio_nao_quebram_a_frase(self) -> None:
        # Regressão: "Your response is... predictable." era falado como duas frases.
        assert split_sentences("Sua resposta é... previsível.") == ["Sua resposta é... previsível."]

    def test_reticencias_seguidas_de_maiuscula_quebram(self) -> None:
        assert split_sentences("Pense nisso... Depois volte.") == [
            "Pense nisso...",
            "Depois volte.",
        ]


class TestStreamSentences:
    def test_reagrupa_tokens_em_frases(self) -> None:
        tokens = ["A ciência ", "não ", "espera ninguém. ", "Continue ", "o teste agora."]
        assert list(stream_sentences(tokens)) == [
            "A ciência não espera ninguém.",
            "Continue o teste agora.",
        ]

    def test_frases_curtas_sao_agrupadas_com_a_seguinte(self) -> None:
        # "Ok." sozinho soaria picotado no TTS.
        resultado = list(stream_sentences(["Ok. ", "Agora preste muita atenção nisto aqui."]))
        assert resultado == ["Ok. Agora preste muita atenção nisto aqui."]

    def test_resto_sem_pontuacao_e_emitido_no_final(self) -> None:
        assert list(stream_sentences(["Uma frase completa aqui. ", "E um resto"])) == [
            "Uma frase completa aqui.",
            "E um resto",
        ]

    def test_minuscula_apos_o_ponto_nao_quebra(self) -> None:
        # Sinal de continuação (reticências, abreviação): melhor falar junto.
        assert list(stream_sentences(["Uma frase completa aqui. ", "e um resto"])) == [
            "Uma frase completa aqui. e um resto",
        ]

    def test_fluxo_vazio_nao_emite_nada(self) -> None:
        assert list(stream_sentences([])) == []


class TestCleanForSpeech:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("**muito** importante", "muito importante"),
            ("use `código` aqui", "use código aqui"),
            ("## Título", "Título"),
            ("olá 🤖 robô", "olá robô"),
            ("espaços     demais", "espaços demais"),
        ],
    )
    def test_remove_ruido(self, entrada: str, esperado: str) -> None:
        assert clean_for_speech(entrada) == esperado


class TestTruncate:
    def test_encurta_textos_longos(self) -> None:
        assert truncate("a" * 100, limit=10) == "a" * 9 + "…"

    def test_mantem_textos_curtos(self) -> None:
        assert truncate("curto", limit=10) == "curto"
