"""Generate synthetic fixture sprites with known ground truth.

Each fixture targets one failure mode from the design:
  ball_hueshift   hue-shifted ramp (purple shadows -> yellow highlights), black outline
  mushroom_shared two materials sharing their darkest colour, flat background
  ball_selout     selective outline: no outline colour, contour uses dark ramp shades
  black_hat       black fill that must NOT become outline, next to a black outline
  slime_noline    no outline at all, flat background
  ball_4x         ball_hueshift upscaled 4x (scale detection)
Real sprites (kenney_*.png, CC0) are committed as-is; only their expectations
are written here.

Ground truth goes to tests/fixtures/expected.json: each ramp lists every
colour of one material, dark -> light, including colours it shares with other
materials. Run:  uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"

TRANSPARENT = (0, 0, 0, 0)


def rgba(hex_color: str) -> tuple[int, int, int, int]:
    s = hex_color.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255


def canvas(w: int, h: int, bg=TRANSPARENT) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:] = bg
    return img


def ellipse_mask(w, h, cx, cy, rx, ry) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1.0


def lit_levels(mask, cx, cy, rx, ry, cuts, light=(-0.55, -0.65, 0.52)) -> np.ndarray:
    """Lambert-lit ellipsoid banded into len(cuts)+1 levels (0 = darkest)."""
    h, w = mask.shape
    y, x = np.mgrid[0:h, 0:w]
    nx = (x + 0.5 - cx) / rx
    ny = (y + 0.5 - cy) / ry
    nz = np.sqrt(np.clip(1 - nx**2 - ny**2, 0, 1))
    lx, ly, lz = np.array(light) / np.linalg.norm(light)
    intensity = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    return np.digitize(intensity, cuts)


def ring(mask: np.ndarray) -> np.ndarray:
    """Pixels outside ``mask`` that 4-touch it: a 1-px outline."""
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown & ~mask


def paint_levels(img, mask, levels, ramp):
    for i, color in enumerate(ramp):
        img[mask & (levels == i)] = rgba(color)


# --- palettes --------------------------------------------------------------

BALL_RAMP = ["#4a1d4f", "#8c2a4a", "#d0463a", "#f08a3c", "#ffd27a"]
OUTLINE = "#1b1025"
BALL_CUTS = [0.12, 0.38, 0.78, 0.93]  # base (level 2) covers the most area


def ball() -> np.ndarray:
    w = h = 18
    img = canvas(w, h)
    m = ellipse_mask(w, h, 9, 9, 7.5, 7.5)
    paint_levels(img, m, lit_levels(m, 9, 9, 7.5, 7.5, BALL_CUTS), BALL_RAMP)
    img[ring(m)] = rgba(OUTLINE)
    return img


SHARED_DARK = "#3d1f2e"
CAP_RAMP = [SHARED_DARK, "#7a2336", "#c23b3b", "#ee7b6a"]
STEM_RAMP = [SHARED_DARK, "#6e4f5a", "#b89c8c", "#eadcc8"]
MUSHROOM_BG = "#5fb3b3"


def mushroom() -> np.ndarray:
    w, h = 22, 22
    img = canvas(w, h, rgba(MUSHROOM_BG))
    cap = ellipse_mask(w, h, 11, 10, 9.5, 7.5)
    cap[10:] = False  # flat underside
    stem = np.zeros((h, w), bool)
    stem[11:19, 7:15] = True
    # Stem: shadow column on the right, highlight column on the left.
    stem_levels = np.full((h, w), 2)
    stem_levels[:, 13:15] = 1
    stem_levels[:, 14] = 0
    stem_levels[:, 8] = 3
    paint_levels(img, stem, stem_levels, STEM_RAMP)
    paint_levels(img, cap, lit_levels(cap, 11, 10, 9.5, 7.5, [0.15, 0.45, 0.85]), CAP_RAMP)
    img[ring(cap | stem)] = rgba(OUTLINE)
    img[10, 3:19] = rgba(OUTLINE)  # line under the cap separates the two materials
    return img


GREEN_RAMP = ["#1f3a3d", "#2f6b4f", "#4fa04a", "#9bd35a", "#e4f59a"]


def ball_selout() -> np.ndarray:
    w = h = 18
    img = canvas(w, h)
    m = ellipse_mask(w, h, 9, 9, 7.5, 7.5)
    levels = lit_levels(m, 9, 9, 7.5, 7.5, [0.06, 0.3, 0.78, 0.93])
    # Sel-out: the edge ring of the shape is drawn one ramp step darker.
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    levels = np.where(edge, np.maximum(levels - 1, 0), levels)
    paint_levels(img, m, levels, GREEN_RAMP)
    return img


SKIN_RAMP = ["#7a4a3a", "#c7805e", "#f0b890", "#ffe0c0"]
HAT_RAMP = [OUTLINE, "#3c3a52"]  # the hat fill is the outline colour itself


def black_hat() -> np.ndarray:
    w, h = 20, 22
    img = canvas(w, h)
    head = ellipse_mask(w, h, 10, 14, 6.5, 6.5)
    paint_levels(img, head, lit_levels(head, 10, 14, 6.5, 6.5, [0.15, 0.7, 0.9]), SKIN_RAMP)
    hat = np.zeros((h, w), bool)
    hat[2:9, 6:14] = True  # crown
    hat[8:10, 3:17] = True  # brim
    img[hat] = rgba(HAT_RAMP[0])
    img[3:8, 7] = rgba(HAT_RAMP[1])  # hat highlight stripe
    img[ring(head | hat)] = rgba(OUTLINE)
    return img


SLIME_RAMP = ["#23305e", "#3b5dc9", "#41a6f6", "#9ad7ff"]
SLIME_BG = "#f4f4e8"


def slime_noline() -> np.ndarray:
    w, h = 20, 14
    img = canvas(w, h, rgba(SLIME_BG))
    m = ellipse_mask(w, h, 10, 10, 8.5, 7.5)
    m[11:] = False
    m[11, 3:17] = True
    paint_levels(img, m, lit_levels(m, 10, 10, 8.5, 7.5, [0.15, 0.5, 0.9]), SLIME_RAMP)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ball_img = ball()
    fixtures = {
        "ball_hueshift": (ball_img, 1),
        "mushroom_shared": (mushroom(), 1),
        "ball_selout": (ball_selout(), 1),
        "black_hat": (black_hat(), 1),
        "slime_noline": (slime_noline(), 1),
        "ball_4x": (np.repeat(np.repeat(ball_img, 4, 0), 4, 1), 4),
    }
    for name, (img, _) in fixtures.items():
        Image.fromarray(img, "RGBA").save(OUT / f"{name}.png")

    ball_truth = {
        "background": "transparent",
        "outline": {"mode": "color", "colors": [OUTLINE], "thickness": 1},
        "ramps": [BALL_RAMP],
        "bases": [BALL_RAMP[2]],
    }
    expected = {
        "ball_hueshift": {"scale": 1, **ball_truth},
        "ball_4x": {"scale": 4, **ball_truth},
        "mushroom_shared": {
            "scale": 1,
            "background": "color",
            "outline": {"mode": "color", "colors": [OUTLINE], "thickness": 1},
            # Both materials contain the shared dark; its pixels resolve per region.
            "ramps": [CAP_RAMP, STEM_RAMP],
            "bases": [CAP_RAMP[2], STEM_RAMP[2]],
        },
        "ball_selout": {
            "scale": 1,
            "background": "transparent",
            "outline": {"mode": "selout", "colors": []},
            "ramps": [GREEN_RAMP],
            "bases": [GREEN_RAMP[2]],
        },
        "black_hat": {
            "scale": 1,
            "background": "transparent",
            "outline": {"mode": "color", "colors": [OUTLINE], "thickness": 1},
            "ramps": [SKIN_RAMP, HAT_RAMP],
            "bases": [SKIN_RAMP[1], HAT_RAMP[0]],
        },
        "slime_noline": {
            "scale": 1,
            "background": "color",
            "outline": {"mode": "none", "colors": []},
            "ramps": [SLIME_RAMP],
            "bases": [SLIME_RAMP[2]],
        },
    }
    # Real CC0 sprites (Kenney "Tiny Dungeon", see tests/fixtures/SOURCES.md).
    # No exact ground truth for every ramp, so only what is certain is pinned:
    # the 2 px outline and the bases of clearly separate materials.
    kenney_outline = {"mode": "color", "colors": ["#3f2631"], "thickness": 2}
    kenney_bases = {
        "kenney_wizard": ["#9b4ca3", "#c0cbdc", "#f7c282"],  # robe, beard, skin
        "kenney_knight": ["#8b9bb4", "#f7c282"],  # armour, skin
        "kenney_rogue": ["#8b9bb4", "#f7c282"],  # hair, skin
        "kenney_ghost": ["#c0cbdc"],  # sheet
    }
    for name, bases in kenney_bases.items():
        expected[name] = {
            "scale": 1,
            "background": "transparent",
            "outline": kenney_outline,
            "bases_include": bases,
        }

    (OUT / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {len(fixtures)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
