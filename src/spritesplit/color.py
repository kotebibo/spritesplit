"""sRGB -> OKLab conversion and color helpers.

Why OKLab: it is perceptually uniform, so Euclidean distances mean roughly the
same visual difference everywhere, and L (lightness) is decoupled from the a/b
chroma plane. That lets ramp detection reason about "lighter/darker" (L) and
"same material" (a/b displacement) separately.

Determinism: cube roots and powers can differ in the last ulp across libm
implementations. Every OKLab value is rounded to ``LAB_DECIMALS`` so later
threshold comparisons see identical numbers on every platform.
"""

from __future__ import annotations

import numpy as np

LAB_DECIMALS = 4

# Björn Ottosson's OKLab matrices (https://bottosson.github.io/posts/oklab/).
_LINEAR_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LMS_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert (..., 3) uint8 sRGB to (..., 3) float64 OKLab, rounded for determinism."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lms = linear @ _LINEAR_TO_LMS.T
    lab = np.cbrt(lms) @ _LMS_TO_LAB.T
    # +0.0 turns -0.0 into 0.0 so rounded values print and compare identically.
    return np.round(lab, LAB_DECIMALS) + 0.0


def pack_rgba(rgba: np.ndarray) -> np.ndarray:
    """Pack (..., 4) uint8 RGBA into (...) uint32 keys, preserving RGBA sort order."""
    c = rgba.astype(np.uint32)
    return (c[..., 0] << 24) | (c[..., 1] << 16) | (c[..., 2] << 8) | c[..., 3]


def unpack_rgba(keys: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_rgba`."""
    k = np.asarray(keys, dtype=np.uint32)
    return np.stack(
        [(k >> 24) & 0xFF, (k >> 16) & 0xFF, (k >> 8) & 0xFF, k & 0xFF], axis=-1
    ).astype(np.uint8)


def to_hex(rgba) -> str:
    """'#rrggbb', or '#rrggbbaa' when the color is not fully opaque."""
    r, g, b, a = (int(v) for v in rgba)
    return f"#{r:02x}{g:02x}{b:02x}" + ("" if a == 255 else f"{a:02x}")


def from_hex(text: str) -> tuple[int, int, int, int]:
    s = text.lstrip("#")
    if len(s) not in (6, 8):
        raise ValueError(f"expected #rrggbb or #rrggbbaa, got {text!r}")
    a = int(s[6:8], 16) if len(s) == 8 else 255
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a
