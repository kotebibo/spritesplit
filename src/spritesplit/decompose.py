"""The decomposition pipeline: sprite in, named layers out.

    load -> detect upscale -> palette -> background -> outline -> ramps
         -> per-pixel ramp assignment -> base colour per ramp -> layers

Layers (top to bottom): outline, highlights, shading, flats, background.
Invariant: stacking them back (each opaque pixel replacing what is below)
reproduces the input exactly. That makes the split lossless and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .background import Background, find_background
from .color import to_hex
from .image import detect_scale, downscale
from .outline import Outline, OutlineParams, find_outline
from .palette import Palette, extract_palette
from .ramps import NO_RAMP, RampParams, Ramps, detect_ramps
from .regions import NEIGHBOURS_4, label_components, shifted

LAYER_ORDER = ("outline", "highlights", "shading", "flats", "background")  # top -> bottom
REPORT_VERSION = 1


@dataclass(frozen=True)
class Params:
    max_colors: int = 256
    outline: OutlineParams = field(default_factory=OutlineParams)
    ramps: RampParams = field(default_factory=RampParams)


@dataclass(frozen=True)
class Decomposition:
    image: np.ndarray  # (H, W, 4) native-resolution input
    scale: int  # detected integer upscale of the original file
    palette: Palette
    index: np.ndarray  # (H, W) palette index, -1 transparent
    background: Background
    outline: Outline
    ramps: Ramps
    pixel_ramp: np.ndarray  # (H, W) ramp id of each fill pixel, NO_RAMP elsewhere
    bases: tuple[int, ...]  # base (flat) palette index per ramp
    layers: dict[str, np.ndarray]  # name -> (H, W, 4) RGBA, in LAYER_ORDER

    def report(self) -> dict:
        """JSON-serialisable summary; also the raw material for generation metrics."""
        colors = self.palette.colors
        h, w = self.index.shape
        ramps = []
        for r, chain in enumerate(self.ramps.chains):
            ramps.append(
                {
                    "members": [to_hex(colors[c]) for c in chain],
                    "base": to_hex(colors[self.bases[r]]),
                    "pixels": int((self.pixel_ramp == r).sum()),
                }
            )
        return {
            "version": REPORT_VERSION,
            "size": {"width": w, "height": h, "scale": self.scale},
            "palette_size": len(self.palette),
            "background": {
                "mode": self.background.mode,
                "color": None
                if self.background.color_index is None
                else to_hex(colors[self.background.color_index]),
            },
            "outline": {
                "mode": self.outline.mode,
                "colors": [to_hex(colors[c]) for c in self.outline.colors],
                "pixels": int(self.outline.mask.sum()),
            },
            "ramps": ramps,
            "shared_colors": {
                to_hex(colors[c]): [int(r) for r in rs] for c, rs in self.ramps.shared.items()
            },
            "layers": {name: int((layer[..., 3] > 0).sum()) for name, layer in self.layers.items()},
        }


def decompose(rgba: np.ndarray, params: Params = Params()) -> Decomposition:
    scale = detect_scale(rgba)
    image = downscale(rgba, scale)
    palette, index = extract_palette(image, params.max_colors)
    if len(palette) == 0:
        raise ValueError("image is fully transparent: nothing to decompose")

    background = find_background(index)
    sprite = (index >= 0) & ~background.mask
    outline = find_outline(index, sprite, palette, params.outline)
    fill = sprite & ~outline.mask

    ramps = detect_ramps(palette, index, fill, params.ramps)
    pixel_ramp = assign_pixels(index, fill, ramps, params.ramps.shared_override_ratio)
    bases = choose_bases(index, pixel_ramp, ramps, len(palette))
    layers = build_layers(image, index, palette, background, outline, fill, pixel_ramp, bases)
    return Decomposition(
        image, scale, palette, index, background, outline, ramps, pixel_ramp, bases, layers
    )


def assign_pixels(
    index: np.ndarray, fill: np.ndarray, ramps: Ramps, override_ratio: float
) -> np.ndarray:
    """Ramp id per fill pixel. Shared colours are resolved by their neighbours.

    Each connected blob of a shared colour votes: every 4-neighbour pair with an
    unambiguous fill pixel is one vote for that pixel's ramp. The primary ramp
    came from a mutual-best link, which is stronger evidence than the one-sided
    link behind a shared ramp, so a shared ramp wins only with a clear majority:
    more than ``shared_override_ratio`` times the primary's votes.
    """
    safe = np.maximum(index, 0)
    pixel_ramp = np.where(fill, ramps.ramp_of[safe], NO_RAMP)
    if not ramps.shared:
        return pixel_ramp

    ambiguous = fill & np.isin(index, list(ramps.shared))
    blobs, n = label_components(index, ambiguous)
    votes = np.zeros((n, len(ramps.chains)), dtype=np.int64)
    unambiguous = fill & ~ambiguous
    for dy, dx in NEIGHBOURS_4:
        sel = ambiguous & shifted(unambiguous, dy, dx, False)
        np.add.at(votes, (blobs[sel], shifted(pixel_ramp, dy, dx, NO_RAMP)[sel]), 1)

    blob_color = np.zeros(n, dtype=np.int64)
    blob_color[blobs[ambiguous]] = index[ambiguous]
    choice = np.empty(n, dtype=np.int64)
    for b in range(n):
        primary, *others = ramps.candidates(int(blob_color[b]))
        choice[b] = primary
        if others:
            # max() keeps the first of equal scores: lowest ramp id, deterministic.
            best = max(others, key=lambda r: votes[b, r])
            if votes[b, best] > override_ratio * votes[b, primary]:
                choice[b] = best
    pixel_ramp[ambiguous] = choice[blobs[ambiguous]]
    return pixel_ramp


def choose_bases(
    index: np.ndarray, pixel_ramp: np.ndarray, ramps: Ramps, k: int
) -> tuple[int, ...]:
    """Base colour of each ramp = its chain member covering the most pixels.

    Artists paint the base first and shade on top, so the base is almost always
    the largest area. Ties go to the darker member (earlier in the chain).
    """
    assigned = pixel_ramp != NO_RAMP
    # 2-D histogram in one bincount: pack (ramp, colour) into ramp * k + colour,
    # count, then reshape so area[r, c] = pixels of colour c assigned to ramp r.
    area = np.bincount(
        pixel_ramp[assigned] * k + index[assigned], minlength=len(ramps.chains) * k
    ).reshape(len(ramps.chains), k)
    return tuple(
        int(chain[int(np.argmax(area[r, list(chain)]))]) for r, chain in enumerate(ramps.chains)
    )


def build_layers(image, index, palette, background, outline, fill, pixel_ramp, bases):
    empty = np.zeros_like(image)

    def take(mask: np.ndarray) -> np.ndarray:
        layer = empty.copy()
        layer[mask] = image[mask]
        return layer

    safe_index = np.maximum(index, 0)
    # Only meaningful on fill pixels (elsewhere 0); there may be no ramps at all.
    base_of_pixel = np.zeros(index.shape, dtype=np.int64)
    base_of_pixel[fill] = np.asarray(bases, dtype=np.int64)[pixel_ramp[fill]]
    L = palette.lightness
    darker = fill & (L[safe_index] < L[base_of_pixel])
    # ">=" with "not the base": a shared colour exactly as light as another
    # ramp's base must still land somewhere, or the split would lose pixels.
    lighter = fill & (L[safe_index] >= L[base_of_pixel]) & (index != base_of_pixel)

    flats = empty.copy()
    flats[fill] = palette.colors[base_of_pixel[fill]]

    return {
        "outline": take(outline.mask),
        "highlights": take(lighter),
        "shading": take(darker),
        "flats": flats,
        "background": take(background.mask & (index >= 0)),
    }


def composite(layers: dict[str, np.ndarray]) -> np.ndarray:
    """Stack layers bottom to top, each opaque pixel replacing what is below."""
    out = np.zeros_like(next(iter(layers.values())))
    for name in reversed(LAYER_ORDER):
        layer = layers[name]
        mask = layer[..., 3] > 0
        out[mask] = layer[mask]
    return out
