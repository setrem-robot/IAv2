"""Gera uma folha de contato com as expressoes da face.

Serve para ajustar a estetica sem precisar rodar o robo inteiro e esperar a hora
de cada expressao aparecer: mexeu num parametro, roda `roboteye preview` e olha.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from roboteye.config import FaceSettings
from roboteye.face.animator import EyeFrame, close_lids
from roboteye.face.expressions import Expression
from roboteye.face.layout import EyeLayout
from roboteye.face.shapes import EyeShape, preset_for
from roboteye.face.theme import Theme

COLUMNS = 3
LABEL_MARGIN = 14


def _panels() -> list[tuple[str, EyeShape, EyeShape]]:
    """Os quadros da folha: expressoes de repouso e instantes de movimento."""
    neutral = preset_for(Expression.NEUTRAL)
    thinking = preset_for(Expression.THINKING)

    panels: list[tuple[str, EyeShape, EyeShape]] = []
    for expression in (
        Expression.NEUTRAL,
        Expression.HAPPY,
        Expression.ANGRY,
        Expression.TIRED,
        Expression.LAUGH,
        Expression.SLEEP,
    ):
        shape = preset_for(expression)
        panels.append((expression.value.upper(), shape, shape))

    panels += [
        (
            "PENSANDO",
            replace(thinking, top_lid=0.09, offset_x=-115, offset_y=-78),
            replace(thinking, top_lid=0.21, offset_x=-115, offset_y=-78),
        ),
        (
            "FALANDO (pico)",
            replace(neutral, height=1.055, width=0.972, offset_y=-6),
            replace(neutral, height=1.055, width=0.972, offset_y=-6),
        ),
        (
            "PISCANDO (meio)",
            close_lids(neutral, 0.55),
            close_lids(neutral, 0.55),
        ),
        (
            "CURIOSO",
            replace(neutral, height=0.94, offset_x=380),
            replace(neutral, height=1.16, offset_x=380),
        ),
        (
            "BRAVO -> FELIZ (meio)",
            preset_for(Expression.ANGRY).lerp(preset_for(Expression.HAPPY), 0.5),
            preset_for(Expression.ANGRY).lerp(preset_for(Expression.HAPPY), 0.5),
        ),
        (
            "PISCANDO (fundo)",
            close_lids(neutral, 1.0),
            close_lids(neutral, 1.0),
        ),
    ]
    return panels


def render_sheet(
    settings: FaceSettings,
    destination: Path,
    *,
    panel_width: int = 640,
    panel_height: int = 360,
) -> Path:
    """Desenha todas as expressoes num unico PNG e devolve o caminho salvo."""
    # O driver dummy permite gerar a folha sem abrir janela nenhuma.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from roboteye.face.renderer import EyeRenderer, quality_for

    pygame.init()
    pygame.display.set_mode((64, 64))

    # A folha e estatica: nao ha orcamento de quadro a respeitar, entao ela sai
    # sempre no nivel mais alto, independente do que o robo usaria em execucao.
    quality = quality_for("high")
    panels = _panels()
    rows = (len(panels) + COLUMNS - 1) // COLUMNS
    theme = Theme.from_settings(settings)

    sheet = pygame.Surface((panel_width * COLUMNS, panel_height * rows))
    sheet.fill(theme.background)
    font = pygame.font.Font(None, max(16, panel_height // 16))

    layout = EyeLayout.for_screen(panel_width, panel_height)
    for index, (label, left, right) in enumerate(panels):
        panel = pygame.Surface((panel_width, panel_height))
        renderer = EyeRenderer(
            panel,
            layout,
            theme,
            quality=quality,
            corner_radius=settings.corner_radius,
        )
        renderer.draw(EyeFrame(expression=Expression.NEUTRAL, left=left, right=right))

        panel.blit(font.render(label, True, theme.caption), (LABEL_MARGIN, LABEL_MARGIN))
        pygame.draw.rect(panel, theme.caption, (0, 0, panel_width, panel_height), 1)
        sheet.blit(panel, ((index % COLUMNS) * panel_width, (index // COLUMNS) * panel_height))

    destination.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(sheet, str(destination))
    pygame.quit()
    return destination
