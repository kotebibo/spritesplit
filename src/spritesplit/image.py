"""Image loading and integer-upscale detection."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from PIL import Image

from .color import pack_rgba


def load_rgba(path: str | Path) -> np.ndarray:
    """Load any Pillow-readable image as (H, W, 4) uint8 RGBA.

    Fully transparent pixels are normalised to (0, 0, 0, 0): their RGB is
    invisible noise that would otherwise pollute the palette.
    """
    with Image.open(path) as im:
        # 16-bit and float modes: Pillow's down-conversion has changed between
        # versions, which would break byte-identical output. Pixel art is 8-bit.
        if im.mode.startswith(("I", "F")):
            raise ValueError(f"unsupported image mode {im.mode!r}: export as an 8-bit PNG")
        rgba = np.array(im.convert("RGBA"), dtype=np.uint8)
    rgba[rgba[..., 3] == 0] = 0
    return rgba


def detect_scale(rgba: np.ndarray) -> int:
    """Return k if the image is an exact k-times nearest-neighbour upscale, else 1.

    If a sprite was upscaled by k from the origin, every run of identical pixels
    starts at a multiple of k, in every row and every column. So k is the GCD of
    the image size and every position where the colour changes. When all
    boundaries sit on multiples of k both horizontally and vertically, each k*k
    block has no internal change, i.e. it is uniform, so no extra check is needed.
    """
    keys = pack_rgba(rgba)
    h, w = keys.shape
    xs = np.nonzero((keys[:, 1:] != keys[:, :-1]).any(axis=0))[0] + 1
    ys = np.nonzero((keys[1:, :] != keys[:-1, :]).any(axis=1))[0] + 1
    if xs.size == 0 and ys.size == 0:
        return 1  # a single flat colour: no evidence of any scale
    k = gcd(w, h)
    for positions in (xs, ys):
        if positions.size:
            k = gcd(k, int(np.gcd.reduce(positions)))
    return max(k, 1)


def downscale(rgba: np.ndarray, k: int) -> np.ndarray:
    return rgba if k == 1 else np.ascontiguousarray(rgba[::k, ::k])


def upscale(rgba: np.ndarray, k: int) -> np.ndarray:
    return rgba if k == 1 else np.repeat(np.repeat(rgba, k, axis=0), k, axis=1)
