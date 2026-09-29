"""Render the README demo GIF (layers building up one by one) and a still PNG sheet.

uv run python scripts/make_demo.py [sprite.png ...] [-o docs/demo.gif]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from spritesplit import LAYER_ORDER, decompose
from spritesplit.image import load_rgba

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPRITES = [
    ROOT / "tests/fixtures/kenney_knight.png",  # CC0, Kenney "Tiny Dungeon"
    ROOT / "tests/fixtures/mushroom_shared.png",
]
ZOOM = 8
PAD = 12
LABEL_H = 18
BG = (40, 38, 52, 255)
CHECKER = ((70, 68, 84, 255), (58, 56, 70, 255))
TEXT = (235, 230, 220, 255)


def checker(w: int, h: int, cell: int = 4) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w]
    tiles = ((x // cell + y // cell) % 2).astype(bool)
    arr = np.where(tiles[..., None], CHECKER[0], CHECKER[1]).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def tile(rgba: np.ndarray, label: str) -> Image.Image:
    h, w = rgba.shape[:2]
    img = checker(w * ZOOM, h * ZOOM)
    img.alpha_composite(Image.fromarray(rgba, "RGBA").resize((w * ZOOM, h * ZOOM), Image.NEAREST))
    out = Image.new("RGBA", (img.width, img.height + LABEL_H), BG)
    out.paste(img, (0, LABEL_H))
    ImageDraw.Draw(out).text((2, 3), label, fill=TEXT)
    return out


def frames_for(path: Path) -> list[Image.Image]:
    d = decompose(load_rgba(path))
    bottom_up = list(reversed(LAYER_ORDER))
    empty = np.zeros_like(d.image)
    frames = []
    # Frame k: the stack so far (left), the layers revealed so far (right).
    for k in range(len(bottom_up) + 1):
        stack = empty.copy()
        for name in bottom_up[:k]:
            m = d.layers[name][..., 3] > 0
            stack[m] = d.layers[name][m]
        tiles = [tile(d.image, "input"), tile(stack, "stack")]
        tiles += [tile(d.layers[n] if i < k else empty, n) for i, n in enumerate(bottom_up)]
        frames.append(tiles)
    return [compose(row) for row in frames]


def compose(tiles: list[Image.Image]) -> Image.Image:
    w = sum(t.width for t in tiles) + PAD * (len(tiles) + 1)
    h = max(t.height for t in tiles) + 2 * PAD
    sheet = Image.new("RGBA", (w, h), BG)
    x = PAD
    for t in tiles:
        sheet.paste(t, (x, PAD))
        x += t.width + PAD
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sprites", nargs="*", type=Path, default=DEFAULT_SPRITES)
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "docs" / "demo.gif")
    args = ap.parse_args()

    per_sprite = [frames_for(p) for p in args.sprites]
    width = max(f.width for fs in per_sprite for f in fs)
    frames, durations = [], []
    for fs in per_sprite:
        for i, f in enumerate(fs):
            canvas = Image.new("RGBA", (width, f.height), BG)
            canvas.paste(f, (0, 0))
            frames.append(canvas.convert("RGB"))
            durations.append(1600 if i in (0, len(fs) - 1) else 700)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    # Still image: every sprite's full breakdown, stacked.
    finals = [fs[-1] for fs in per_sprite]
    sheet = Image.new("RGBA", (width, sum(f.height for f in finals)), BG)
    y = 0
    for f in finals:
        sheet.paste(f, (0, y))
        y += f.height
    sheet.save(args.output.with_suffix(".png"))
    print(f"wrote {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
