"""Testes das curvas de aceleração e do tween."""

from __future__ import annotations

from itertools import pairwise

import pytest

from roboteye.face import easing
from roboteye.face.easing import Tween

CURVAS = [
    easing.linear,
    easing.ease_in_quad,
    easing.ease_out_quad,
    easing.ease_out_cubic,
    easing.ease_in_out_cubic,
    easing.ease_out_back,
]


class TestCurvas:
    @pytest.mark.parametrize("curva", CURVAS)
    def test_comecam_em_zero_e_terminam_em_um(self, curva) -> None:
        assert curva(0.0) == pytest.approx(0.0, abs=1e-6)
        assert curva(1.0) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "curva",
        [easing.linear, easing.ease_in_quad, easing.ease_out_quad, easing.ease_in_out_cubic],
    )
    def test_sao_monotonicas(self, curva) -> None:
        valores = [curva(i / 50) for i in range(51)]
        assert all(b >= a - 1e-9 for a, b in pairwise(valores))

    def test_ease_in_e_lento_no_comeco(self) -> None:
        assert easing.ease_in_quad(0.25) < 0.25

    def test_ease_out_e_rapido_no_comeco(self) -> None:
        assert easing.ease_out_quad(0.25) > 0.25

    def test_ease_out_back_passa_do_alvo(self) -> None:
        assert max(easing.ease_out_back(i / 100) for i in range(101)) > 1.0


class TestUtilitarios:
    def test_lerp(self) -> None:
        assert easing.lerp(10.0, 20.0, 0.5) == pytest.approx(15.0)

    @pytest.mark.parametrize(("valor", "esperado"), [(-1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
    def test_clamp(self, valor: float, esperado: float) -> None:
        assert easing.clamp(valor) == esperado

    def test_approach_converge_para_o_alvo(self) -> None:
        valor = 0.0
        for _ in range(200):
            valor = easing.approach(valor, 100.0, 1 / 60, 10.0)
        assert valor == pytest.approx(100.0, abs=0.1)

    def test_approach_independe_da_taxa_de_quadros(self) -> None:
        """O mesmo tempo real deve levar ao mesmo lugar, a 30 ou a 120 FPS."""
        lento = 0.0
        for _ in range(30):
            lento = easing.approach(lento, 100.0, 1 / 30, 8.0)

        rapido = 0.0
        for _ in range(120):
            rapido = easing.approach(rapido, 100.0, 1 / 120, 8.0)

        assert lento == pytest.approx(rapido, abs=0.5)


class TestTween:
    def test_chega_ao_alvo_no_tempo_previsto(self) -> None:
        tween = Tween(0.0)
        tween.to(10.0, 0.5)

        for _ in range(31):  # 31/60 s passa de 0,5 s
            tween.update(1 / 60)

        assert tween.value == pytest.approx(10.0)
        assert tween.done

    def test_ainda_nao_terminou_antes_da_hora(self) -> None:
        tween = Tween(0.0)
        tween.to(10.0, 0.5)
        tween.update(0.4)
        assert not tween.done

    def test_esta_no_meio_do_caminho_na_metade_do_tempo(self) -> None:
        tween = Tween(0.0, easing.linear)
        tween.to(10.0, 1.0)
        tween.update(0.5)

        assert tween.value == pytest.approx(5.0, abs=0.01)

    def test_novo_alvo_parte_de_onde_esta(self) -> None:
        tween = Tween(0.0, easing.linear)
        tween.to(10.0, 1.0)
        tween.update(0.5)
        meio = tween.value

        tween.to(0.0, 1.0)
        assert tween.value == pytest.approx(meio)

    def test_snap_cancela_o_movimento(self) -> None:
        tween = Tween(0.0)
        tween.to(10.0, 1.0)
        tween.snap(3.0)

        assert tween.value == 3.0
        assert tween.done
        assert tween.update(1.0) == 3.0

    def test_duracao_zero_chega_na_hora(self) -> None:
        tween = Tween(0.0)
        tween.to(7.0, 0.0)
        assert tween.value == 7.0

    def test_nao_passa_do_alvo_com_dt_grande(self) -> None:
        tween = Tween(0.0, easing.linear)
        tween.to(10.0, 0.2)
        tween.update(5.0)
        assert tween.value == pytest.approx(10.0)
