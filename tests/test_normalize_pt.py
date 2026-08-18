"""Testes da normalização de texto antes da síntese."""

from __future__ import annotations

import pytest

from roboteye.core.normalize_pt import normalize


class TestDinheiro:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("R$ 25,90", "vinte e cinco reais e noventa centavos"),
            ("R$ 1,00", "um real"),
            ("R$ 1,01", "um real e um centavo"),
            ("R$ 100", "cem reais"),
            ("R$1.500,00", "mil e quinhentos reais"),
        ],
    )
    def test_moeda(self, entrada: str, esperado: str) -> None:
        assert normalize(entrada) == esperado

    def test_dinheiro_vem_antes_do_decimal(self) -> None:
        """Se a regra dos decimais rodasse antes, sairia "vinte e cinco vírgula noventa reais"."""
        assert "virgula" not in normalize("R$ 25,90")


class TestHoras:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("15:30", "quinze e trinta"),
            ("15:00", "quinze horas"),
            ("1:00", "uma hora"),
            ("09:05", "nove e cinco"),
            ("23:59", "vinte e tres e cinquenta e nove"),
        ],
    )
    def test_hora(self, entrada: str, esperado: str) -> None:
        assert normalize(entrada) == esperado

    def test_hora_invalida_fica_como_esta(self) -> None:
        """25:00 não é hora; melhor não inventar do que remontar errado."""
        assert normalize("25:00") == "25:00"


class TestOutrasFormas:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("35%", "trinta e cinco por cento"),
            ("100%", "cem por cento"),
            ("23°C", "vinte e tres graus"),
            ("5 km", "cinco quilometros"),
            ("1 km", "um quilometro"),
            ("80 km/h", "oitenta quilometros por hora"),
            ("1º", "primeiro"),
            ("2ª", "segunda"),
        ],
    )
    def test_formas(self, entrada: str, esperado: str) -> None:
        assert normalize(entrada) == esperado

    def test_abreviacoes(self) -> None:
        assert normalize("Dr. Silva") == "doutor Silva"
        assert normalize("Sra. Costa") == "senhora Costa"


class TestNumerosSoltos:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("137", "cento e trinta e sete"),
            ("3,5", "tres virgula cinco"),
            ("1.234", "mil duzentos e trinta e quatro"),
            ("-7", "menos sete"),
        ],
    )
    def test_numeros(self, entrada: str, esperado: str) -> None:
        assert normalize(entrada) == esperado


class TestFrasesInteiras:
    def test_frase_com_varias_formas(self) -> None:
        texto = "Sao 15:30 e a temperatura esta em 23°C, com 40% de umidade."
        assert normalize(texto) == (
            "Sao quinze e trinta e a temperatura esta em vinte e tres graus, "
            "com quarenta por cento de umidade."
        )

    def test_texto_sem_numero_fica_intacto(self) -> None:
        texto = "A prova era amanha, mas voce ja sabia disso."
        assert normalize(texto) == texto

    def test_nao_mexe_em_palavra_que_contem_abreviacao(self) -> None:
        """ "sra" dentro de outra palavra não é "senhora"."""
        assert normalize("compras.") == "compras."
