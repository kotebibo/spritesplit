"""Background detection: alpha first, then a flood fill over palette indices.

No model is needed. Pixel art on a flat background has an exact background
colour, and flooding over *indices* (not RGB distances) cannot bleed into
a sprite colour that is merely similar.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .palette import TRANSPARENT
from .regions import label_components


@dataclass(frozen=True)
class Background:
    mask: np.ndarray  # (H, W) bool: pixels that are background (incl. transparent)
    mode: str  # "transparent" | "color" | "none"
    color_index: int | None  # palette index of a flat background colour


def find_background(index: np.ndarray) -> Background:
    transparent = index == TRANSPARENT
    if transparent.any():
        return Background(transparent, "transparent", None)

    h, w = index.shape
    corners = [
        int(index[0, 0]),
        int(index[0, w - 1]),
        int(index[h - 1, 0]),
        int(index[h - 1, w - 1]),
    ]
    # A flat background shows in at least 2 corners; a sprite rarely covers 3.
    # Most frequent corner colour wins; ties go to the lower palette index.
    color, hits = min(Counter(corners).items(), key=lambda kv: (-kv[1], kv[0]))
    if hits < 2:
        return Background(np.zeros_like(transparent), "none", None)

    labels, _ = label_components(index, index == color)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    seeds = np.unique(border[border >= 0])
    return Background(np.isin(labels, seeds), "color", color)
