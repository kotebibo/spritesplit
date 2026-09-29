"""Palette extraction: every opaque pixel becomes an index into a small color table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import pack_rgba, srgb_to_oklab, unpack_rgba

TRANSPARENT = -1


class NotPixelArtError(ValueError):
    """Raised when an image has too many colors to be treated as quantized pixel art."""


@dataclass(frozen=True)
class Palette:
    colors: np.ndarray  # (k, 4) uint8 RGBA, sorted by packed RGBA value
    counts: np.ndarray  # (k,) pixel count per color
    lab: np.ndarray  # (k, 3) OKLab, rounded (see color.LAB_DECIMALS)

    def __len__(self) -> int:
        return len(self.colors)

    @property
    def lightness(self) -> np.ndarray:
        return self.lab[:, 0]


def extract_palette(rgba: np.ndarray, max_colors: int = 256) -> tuple[Palette, np.ndarray]:
    """Return the palette and an (H, W) int32 index map (TRANSPARENT where alpha == 0).

    np.unique sorts, so palette order depends only on the colours themselves,
    never on where they appear, which keeps every downstream tie-break stable.
    """
    keys = pack_rgba(rgba)
    opaque = rgba[..., 3] > 0
    index = np.full(keys.shape, TRANSPARENT, dtype=np.int32)
    if not opaque.any():
        empty = Palette(np.zeros((0, 4), np.uint8), np.zeros(0, np.int64), np.zeros((0, 3)))
        return empty, index

    unique, inverse, counts = np.unique(keys[opaque], return_inverse=True, return_counts=True)
    if len(unique) > max_colors:
        raise NotPixelArtError(
            f"image has {len(unique)} colors (limit {max_colors}); "
            "quantize it first, or raise --max-colors"
        )
    index[opaque] = inverse.reshape(-1)
    colors = unpack_rgba(unique)
    return Palette(colors, counts.astype(np.int64), srgb_to_oklab(colors[:, :3])), index
