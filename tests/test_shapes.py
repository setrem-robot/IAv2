"""Testes do modelo paramétrico do olho."""

from __future__ import annotations

from itertools import pairwise

import pytest

from roboteye.face import shapes
from roboteye.face.expressions import Expression
from roboteye.face.shapes import EyeShape, preset_for


class TestEyeShape:
    def test_padrao_e_um_olho_aberto_e_neutro(self) -> None:
        shape = EyeShape()
        assert shape.width == 1.0
        assert shape.height == 1.0
        assert shape.top_lid == 0.0
        assert shape.bottom_lid == 0.0
        assert not shape.is_closed

    def test_olho_sem_altura_conta_como_fechado(self) -> None:
        assert EyeShape(height=0.0).is_closed
        assert EyeShape(width=0.0).is_closed

    def test_scaled_multiplica(self) -> None:
        shape = EyeShape(width=1.0, height=2.0).scaled(width=0.5, height=0.5)
        assert shape.width == 0.5
        assert shape.height == 1.0

    def test_moved_soma(self) -> None:
        shape = EyeShape(offset_x=10.0).moved(dx=5.0, dy=-3.0)
        assert shape.offset_x == 15.0
        assert shape.offset_y == -3.0

    def test_e_imutavel(self) -> None:
        shape = EyeShape()
        with pytest.raises((AttributeError, TypeError)):
            shape.height = 0.5  # type: ignore[misc]


class TestInterpolacao:
    def test_extremos_devolvem_as_pontas(self) -> None:
        a, b = EyeShape(height=1.0), EyeShape(height=0.0)
        assert a.lerp(b, 0.0) is a
        assert a.lerp(b, 1.0) is b

    def test_meio_do_caminho(self) -> None:
        a = EyeShape(height=1.0, top_lid=0.0)
        b = EyeShape(height=0.0, top_lid=0.4)
        meio = a.lerp(b, 0.5)

        assert meio.height == pytest.approx(0.5)
        assert meio.top_lid == pytest.approx(0.2)

    def test_interpola_todos_os_campos(self) -> None:
        """Se um campo novo ficar de fora do lerp, a transição dele fica seca."""
        a = EyeShape(1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0)
        b = EyeShape(2.0, 2.0, 0.5, 10.0, 20.0, 0.5, 1.0, 0.6)
        meio = a.lerp(b, 0.5)

        assert meio.width == pytest.approx(1.5)
        assert meio.radius == pytest.approx(0.4)
        assert meio.offset_x == pytest.approx(5.0)
        assert meio.offset_y == pytest.approx(10.0)
        assert meio.top_lid_slant == pytest.approx(0.5)
        assert meio.bottom_lid == pytest.approx(0.3)

    def test_caminho_entre_expressoes_e_continuo(self) -> None:
        """Bravo vira feliz sem nenhum salto pelo caminho."""
        origem = preset_for(Expression.ANGRY)
        destino = preset_for(Expression.HAPPY)

        passos = [origem.lerp(destino, i / 40) for i in range(41)]
        for anterior, atual in pairwise(passos):
            assert abs(atual.top_lid - anterior.top_lid) < 0.05
            assert abs(atual.bottom_lid - anterior.bottom_lid) < 0.05


class TestPresets:
    @pytest.mark.parametrize("expressao", list(Expression))
    def test_toda_expressao_tem_forma(self, expressao: Expression) -> None:
        assert isinstance(preset_for(expressao), EyeShape)

    def test_feliz_usa_palpebra_inferior(self) -> None:
        assert preset_for(Expression.HAPPY).bottom_lid > 0.2

    def test_bravo_baixa_o_canto_interno(self) -> None:
        bravo = preset_for(Expression.ANGRY)
        assert bravo.top_lid > 0.2
        assert bravo.top_lid_slant > 0

    def test_cansado_e_o_oposto_de_bravo(self) -> None:
        assert preset_for(Expression.TIRED).top_lid_slant < 0

    def test_dormindo_e_quase_uma_linha(self) -> None:
        assert preset_for(Expression.SLEEP).height < 0.1

    def test_rindo_sorri_mais_que_feliz(self) -> None:
        assert preset_for(Expression.LAUGH).bottom_lid > preset_for(Expression.HAPPY).bottom_lid

    def test_raio_padrao_e_quadrado_de_cantos_macios(self) -> None:
        # 0.5 seria um círculo; o projeto pede canto arredondado, não redondo.
        assert 0.15 < shapes.DEFAULT_RADIUS < 0.45
