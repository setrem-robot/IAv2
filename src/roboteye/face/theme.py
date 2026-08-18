"""Cores da face."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roboteye.config import FaceSettings

Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Theme:
    """Paleta usada pelo renderizador."""

    eye: Color = (4, 201, 253)
    background: Color = (0, 0, 0)
    caption: Color = (130, 130, 130)
    caption_highlight: Color = (200, 200, 200)

    @classmethod
    def from_settings(cls, settings: FaceSettings) -> Theme:
        return cls(eye=settings.eye_color, background=settings.background_color)
