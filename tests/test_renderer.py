"""Testes do desenho da face.

Rodam com o driver de vídeo `dummy` do SDL: exercitam o código de desenho de
verdade, sem precisar de tela.
"""

# ruff: noqa: E402 - o driver dummy do SDL precisa ser definido antes de importar pygame

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

from roboteye.face.animator import EyeFrame
from roboteye.face.expressions import Expression
from roboteye.face.layout import EyeLayout
from roboteye.face.renderer import HIGH, LOW, MEDIUM, EyeRenderer, quality_for
from roboteye.face.shapes import EyeShape, preset_for
from roboteye.face.theme import Theme


@pytest.fixture(scope="module", autouse=True)
def pygame_iniciado():
    pygame.init()
    pygame.display.set_mode((320, 240))
    yield
    pygame.quit()


def quadro(shape: EyeShape, expressao: Expression = Expression.NEUTRAL) -> EyeFrame:
    return EyeFrame(expression=expressao, left=shape, right=shape)


@pytest.fixture
def surface() -> pygame.Surface:
    return pygame.Surface((1280, 720))


@pytest.fixture
def renderer(surface: pygame.Surface) -> EyeRenderer:
    return EyeRenderer(surface, EyeLayout.for_screen(1280, 720), Theme())


class TestDesenho:
    @pytest.mark.parametrize("expressao", list(Expression))
    def test_desenha_todas_as_expressoes(
        self, renderer: EyeRenderer, expressao: Expression
    ) -> None:
        renderer.draw(quadro(preset_for(expressao), expressao))

    @pytest.mark.parametrize("altura", [0.03, 0.1, 0.5, 1.0, 1.2])
    def test_desenha_em_qualquer_altura(self, renderer: EyeRenderer, altura: float) -> None:
        renderer.draw(quadro(EyeShape(height=altura)))

    @pytest.mark.parametrize("raio", [0.0, 0.15, 0.3, 0.5])
    def test_desenha_com_qualquer_raio(self, renderer: EyeRenderer, raio: float) -> None:
        renderer.draw(quadro(EyeShape(radius=raio)))

    @pytest.mark.parametrize("inclinacao", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_desenha_palpebra_em_qualquer_inclinacao(
        self, renderer: EyeRenderer, inclinacao: float
    ) -> None:
        renderer.draw(quadro(EyeShape(top_lid=0.3, top_lid_slant=inclinacao)))

    def test_desenha_com_olhar_nos_extremos(self, renderer: EyeRenderer) -> None:
        renderer.draw(quadro(EyeShape(offset_x=-380, offset_y=-120)))
        renderer.draw(quadro(EyeShape(offset_x=380, offset_y=120)))

    def test_olho_fechado_nao_e_desenhado(self, renderer: EyeRenderer) -> None:
        renderer.draw(quadro(EyeShape(height=0.0)))

    def test_desenha_com_legenda_longa(self, renderer: EyeRenderer) -> None:
        renderer.draw(
            quadro(preset_for(Expression.SPEAKING), Expression.SPEAKING),
            caption="Uma legenda bem longa " * 20,
            hint="ESC sair",
        )

    @pytest.mark.parametrize("qualidade", [LOW, MEDIUM, HIGH])
    def test_desenha_em_todos_os_niveis_de_qualidade(
        self, surface: pygame.Surface, qualidade
    ) -> None:
        renderer = EyeRenderer(surface, EyeLayout.for_screen(1280, 720), Theme(), quality=qualidade)
        renderer.draw(quadro(preset_for(Expression.HAPPY), Expression.HAPPY))

    def test_qualidade_auto_resolve_para_um_nivel_conhecido(self) -> None:
        assert quality_for("auto") in (LOW, MEDIUM, HIGH)
        assert quality_for("high") is HIGH
        # Um nome desconhecido nao deve derrubar a face.
        assert quality_for("nao-existe") is MEDIUM

    def test_resize_reconstroi_o_layout(self, renderer: EyeRenderer) -> None:
        nova = pygame.Surface((800, 480))
        renderer.resize(nova, EyeLayout.for_screen(800, 480))
        renderer.draw(quadro(preset_for(Expression.HAPPY), Expression.HAPPY), caption="ok")


class TestPixels:
    """Confere o que efetivamente foi parar na tela.

    Estes testes falam de *forma*, entao desenham no nivel baixo: sem halo e sem
    degrade, um pixel fora do olho e exatamente preto e um pixel dentro e
    exatamente a cor do tema. O halo tem os seus proprios testes mais abaixo.
    """

    def _render(self, shape: EyeShape, **kwargs) -> tuple[pygame.Surface, EyeLayout]:
        surface = pygame.Surface((1280, 720))
        layout = EyeLayout.for_screen(1280, 720)
        kwargs.setdefault("quality", LOW)
        EyeRenderer(surface, layout, Theme(), **kwargs).draw(quadro(shape))
        return surface, layout

    def test_pinta_o_fundo(self) -> None:
        surface, _ = self._render(EyeShape())
        assert surface.get_at((0, 0))[:3] == (0, 0, 0)

    def test_desenha_o_olho_na_cor_do_tema(self) -> None:
        surface, layout = self._render(EyeShape())
        assert surface.get_at((layout.left_eye_x, layout.eye_center_y))[:3] == (4, 201, 253)

    def test_degrade_preserva_a_cor_pedida_no_centro(self) -> None:
        """O degrade clareia o topo tanto quanto escurece a base.

        Sem isso, quem configura uma cor nunca a ve na tela: o olho inteiro sai
        um pouco mais escuro do que foi pedido.
        """
        surface, layout = self._render(EyeShape(), quality=HIGH)
        centro = surface.get_at((layout.left_eye_x, layout.eye_center_y))[:3]
        assert all(abs(a - b) <= 2 for a, b in zip(centro, (4, 201, 253), strict=True))

    def test_palpebra_superior_apaga_o_alto_do_olho(self) -> None:
        layout = EyeLayout.for_screen(1280, 720)
        alto = layout.eye_center_y - layout.eye_height // 2 + 4

        sem_lid, _ = self._render(EyeShape())
        com_lid, _ = self._render(EyeShape(top_lid=0.45))

        assert sem_lid.get_at((layout.left_eye_x, alto))[:3] != (0, 0, 0)
        assert com_lid.get_at((layout.left_eye_x, alto))[:3] == (0, 0, 0)

    def test_palpebra_inferior_apaga_a_base_do_olho(self) -> None:
        layout = EyeLayout.for_screen(1280, 720)
        base = layout.eye_center_y + layout.eye_height // 2 - 4

        sem_lid, _ = self._render(EyeShape())
        com_lid, _ = self._render(EyeShape(bottom_lid=0.5))

        assert sem_lid.get_at((layout.left_eye_x, base))[:3] != (0, 0, 0)
        assert com_lid.get_at((layout.left_eye_x, base))[:3] == (0, 0, 0)

    def test_inclinacao_da_palpebra_e_assimetrica_entre_os_olhos(self) -> None:
        """Bravo baixa o canto interno dos dois olhos, que são lados opostos."""
        surface = pygame.Surface((1280, 720))
        layout = EyeLayout.for_screen(1280, 720)
        shape = EyeShape(top_lid=0.35, top_lid_slant=1.0)
        EyeRenderer(surface, layout, Theme()).draw(quadro(shape))

        topo = layout.eye_center_y - layout.eye_height // 2
        margem = layout.eye_width // 2 - 6

        def coberto(x: int) -> bool:
            for dy in range(0, layout.eye_height // 2):
                if surface.get_at((x, topo + dy))[:3] != (0, 0, 0):
                    return dy > layout.eye_height * 0.12
            return True

        # No olho esquerdo o canto interno é o direito; no direito, o esquerdo.
        assert coberto(layout.left_eye_x + margem)
        assert coberto(layout.right_eye_x - margem)

    def _altura_coberta(self, surface: pygame.Surface, x: int, layout: EyeLayout) -> int:
        """Quantos pixels da palpebra cobrem o olho na coluna `x`."""
        topo = layout.eye_center_y - layout.eye_height // 2
        for dy in range(layout.eye_height):
            if surface.get_at((x, topo + dy))[:3] != (0, 0, 0):
                return dy
        return layout.eye_height

    def test_cansado_baixa_o_canto_externo_dos_dois_olhos(self) -> None:
        """Cansado e o espelho exato de bravo, e o espelho e por olho.

        Bravo ja tinha teste; sem o par, uma troca de sinal passaria despercebida
        num dos dois casos — e e justamente o tipo de erro que nao da pra ver
        olhando, porque os dois olhos continuam simetricos entre si.
        """
        surface = pygame.Surface((1280, 720))
        layout = EyeLayout.for_screen(1280, 720)
        shape = EyeShape(top_lid=0.35, top_lid_slant=-1.0)
        EyeRenderer(surface, layout, Theme(), quality=LOW).draw(quadro(shape))

        margem = layout.eye_width // 2 - 6
        # No olho esquerdo o canto externo e o esquerdo; no direito, o direito.
        esquerdo_fora = self._altura_coberta(surface, layout.left_eye_x - margem, layout)
        esquerdo_dentro = self._altura_coberta(surface, layout.left_eye_x + margem, layout)
        direito_fora = self._altura_coberta(surface, layout.right_eye_x + margem, layout)
        direito_dentro = self._altura_coberta(surface, layout.right_eye_x - margem, layout)

        assert esquerdo_fora > esquerdo_dentro
        assert direito_fora > direito_dentro

    def test_raio_maior_arredonda_mais_os_cantos(self) -> None:
        layout = EyeLayout.for_screen(1280, 720)
        canto_x = layout.left_eye_x - layout.eye_width // 2 + 3
        canto_y = layout.eye_center_y - layout.eye_height // 2 + 3

        quadrado, _ = self._render(EyeShape(radius=0.0))
        redondo, _ = self._render(EyeShape(radius=0.5))

        assert quadrado.get_at((canto_x, canto_y))[:3] != (0, 0, 0)
        assert redondo.get_at((canto_x, canto_y))[:3] == (0, 0, 0)

    def test_configuracao_de_raio_reescala_as_formas(self) -> None:
        layout = EyeLayout.for_screen(1280, 720)
        canto_x = layout.left_eye_x - layout.eye_width // 2 + 3
        canto_y = layout.eye_center_y - layout.eye_height // 2 + 3

        surface, _ = self._render(EyeShape(), corner_radius=0.02)
        assert surface.get_at((canto_x, canto_y))[:3] != (0, 0, 0)


class TestHalo:
    """O brilho ao redor do olho."""

    def _render(self, shape: EyeShape, quality) -> tuple[pygame.Surface, EyeLayout]:
        surface = pygame.Surface((1280, 720))
        layout = EyeLayout.for_screen(1280, 720)
        EyeRenderer(surface, layout, Theme(), quality=quality).draw(quadro(shape))
        return surface, layout

    def test_halo_acende_ao_redor_do_olho(self) -> None:
        fora_x = 40  # bem fora do olho, mas dentro do alcance do halo
        layout = EyeLayout.for_screen(1280, 720)
        ponto = (layout.left_eye_x - layout.eye_width // 2 - fora_x, layout.eye_center_y)

        sem_halo, _ = self._render(EyeShape(), LOW)
        com_halo, _ = self._render(EyeShape(), HIGH)

        assert sem_halo.get_at(ponto)[:3] == (0, 0, 0)
        assert com_halo.get_at(ponto)[:3] != (0, 0, 0)

    def test_halo_nao_vaza_onde_a_palpebra_cortou(self) -> None:
        """Regressao: o halo saia de uma borda que a palpebra havia removido.

        Tirar o halo do campo de distancia parecia economico — o campo ja estava
        calculado — mas uma interseccao feita com `max` so e exata *dentro* da
        forma. Do lado de fora ela subestima a distancia, e o brilho continuava
        saindo da base do olho mesmo depois de o sorriso te-la cortado, o que
        pintava um bloco retangular logo abaixo dos olhos.
        """
        layout = EyeLayout.for_screen(1280, 720)
        surface, _ = self._render(EyeShape(bottom_lid=0.5), HIGH)

        # Uma faixa larga bem abaixo do olho, onde nao ha superficie nenhuma.
        base = layout.eye_center_y + layout.eye_height // 2 + 30
        for x in range(layout.left_eye_x - 200, layout.left_eye_x + 200, 7):
            assert surface.get_at((x, base))[:3] == (0, 0, 0), f"halo vazou em x={x}"


class TestSubPixel:
    """Movimento menor que um pixel precisa aparecer na tela."""

    def _render(self, offset_x: float) -> pygame.Surface:
        surface = pygame.Surface((1280, 720))
        layout = EyeLayout.for_screen(1280, 720)
        EyeRenderer(surface, layout, Theme(), quality=LOW).draw(quadro(EyeShape(offset_x=offset_x)))
        return surface

    def test_deslocamento_fracionario_muda_os_pixels(self) -> None:
        """Antes, `int()` engolia qualquer movimento menor que um pixel.

        Era o que fazia a respiracao e as microssacadas andarem aos pulos: o
        olho ficava parado varios quadros e depois saltava um pixel inteiro.
        """
        layout = EyeLayout.for_screen(1280, 720)
        # Meia unidade base e bem menos que um pixel na tela.
        meio_pixel = 0.5 / layout.scale

        parado = pygame.image.tostring(self._render(0.0), "RGB")
        movido = pygame.image.tostring(self._render(meio_pixel), "RGB")
        assert parado != movido
