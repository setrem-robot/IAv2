"""Geometria da face.

As medidas sao escritas num referencial fixo de 2560x1440 e convertidas para a
resolucao real por um unico fator de escala. Assim a face fica igual num monitor
4K e numa telinha de 800x480 do Raspberry Pi.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Resolucao de referencia em que as medidas abaixo foram desenhadas.
BASE_WIDTH = 2560
BASE_HEIGHT = 1440

# Medidas em unidades base.
EYE_WIDTH = 640
EYE_HEIGHT = 640
EYE_Y_OFFSET = -150
LOOK_RANGE = 380
CAPTION_MARGIN = 40
CAPTION_SIZE = 46


@dataclass(frozen=True, slots=True)
class EyeLayout:
    """Posicoes e tamanhos ja convertidos para pixels."""

    screen_width: int
    screen_height: int
    scale: float

    eye_width: int
    eye_height: int

    left_eye_x: int
    right_eye_x: int
    eye_center_y: int

    @classmethod
    def for_screen(cls, width: int, height: int) -> EyeLayout:
        """Calcula o layout para uma tela de `width` x `height` pixels."""
        scale = min(width / BASE_WIDTH, height / BASE_HEIGHT)

        def px(value: float) -> int:
            return max(1, int(value * scale))

        return cls(
            screen_width=width,
            screen_height=height,
            scale=scale,
            eye_width=px(EYE_WIDTH),
            eye_height=px(EYE_HEIGHT),
            left_eye_x=width // 3,
            right_eye_x=2 * width // 3,
            eye_center_y=height // 2 + int(EYE_Y_OFFSET * scale),
        )

    def px(self, base_value: float) -> int:
        """Converte uma medida do referencial base para pixels."""
        return int(base_value * self.scale)

    @property
    def caption_font_size(self) -> int:
        return max(12, self.px(CAPTION_SIZE))

    @property
    def caption_margin(self) -> int:
        return max(8, self.px(CAPTION_MARGIN))
