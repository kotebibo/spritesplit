"""Outline detection: topological, not luminance based.

An outline is a band of one colour that hugs the silhouette and then *stops*:
what is inside it is something else. "Darkest colour = outline" fails on
black hats, dark boots and shadowed sides, so darkness is only used for
selective outlining (sel-out), where there is no single outline colour.

Algorithm:
1. Depth. Peel the sprite like an onion: shell 1 = sprite pixels 4-adjacent to
   the outside (background, transparency, or the image edge), shell 2 = those
   adjacent to shell 1, and so on. The *edge* is the part of shell 1 that
   touches real background; an image border cutting the sprite is not edge.
2. A colour c is an outline of thickness t (tried t = 1, then 2, ...) when
     - it covers >= min_contour_share of the edge  (it draws on the silhouette)
     - it fills shells 2..t too                    (the band really is t thick)
     - it mostly stops at shell t+1                (a fill colour keeps going)
   The "stops" test is relative (c in shell t+1 vs. c in shell t), so a black
   hat drawn with the outline colour does not disqualify the outline: only the
   hat continues inward, the rest of the black line stops.
   Enclosure: an outline goes *around* the sprite. The candidate colours
   together must cover >= min_side_coverage of the edge on each of the four
   sides (top/right/bottom/left sectors around the centroid). A dark
   terminator crescent is also a thin band that stops, but only on one side.
3. Outline pixels = outline-colour pixels within the band (depth <= t), plus
   outline-colour pixels anywhere that are thin (not inside a one-colour 2x2
   block): internal lines. Fat blobs of that colour deeper inside (the hat)
   stay in the fills and become a material.
4. No outline colour? Try sel-out: edge pixels noticeably darker than the
   pixels just inside them. Accept only if that holds for most of the edge, so
   a single dark shading band doesn't qualify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .palette import Palette
from .regions import NEIGHBOURS_4, in_2x2_block, shifted


@dataclass(frozen=True)
class OutlineParams:
    max_thickness: int = 2  # try outline bands of 1..max_thickness pixels
    min_contour_share: float = 0.25  # colour must cover this fraction of the edge
    min_band_fill: float = 0.5  # c in shell s >= this * (c in shell 1), for s in 2..t
    max_continuation: float = 0.6  # c in shell t+1 <= this * (c in shell t)
    min_side_coverage: float = 0.4  # outline colours must cover this much of every side
    selout_min_darker: float = 0.03  # OKLab L drop vs. inner neighbours to count as sel-out
    selout_min_fraction: float = 0.5  # fraction of the edge that must be darker


@dataclass(frozen=True)
class Outline:
    mask: np.ndarray  # (H, W) bool
    mode: str  # "color" | "selout" | "none"
    colors: tuple[int, ...]  # palette indices of outline colours (mode == "color")
    thickness: int = 0  # band thickness in pixels (mode == "color")


def depth_from_outside(sprite: np.ndarray, max_depth: int) -> np.ndarray:
    """4-connected distance from outside the sprite: 0 outside, 1 = edge shell, ...

    Pixels deeper than ``max_depth`` get ``max_depth + 1``. The image border
    counts as outside, so tiles whose sprite touches the edge are handled.
    """
    depth = np.zeros(sprite.shape, dtype=np.int32)
    remaining = sprite.copy()
    for level in range(1, max_depth + 1):
        exposed = np.zeros_like(remaining)
        for dy, dx in NEIGHBOURS_4:
            exposed |= ~shifted(remaining, dy, dx, False)
        shell = remaining & exposed
        depth[shell] = level
        remaining &= ~shell
    depth[remaining] = max_depth + 1
    return depth


def find_outline(
    index: np.ndarray, sprite: np.ndarray, palette: Palette, params: OutlineParams = OutlineParams()
) -> Outline:
    none = Outline(np.zeros_like(sprite), "none", ())
    depth = depth_from_outside(sprite, params.max_thickness + 1)
    k = len(palette)
    # per_shell[s][c] = number of colour-c pixels at depth s (s = 0, outside, unused)
    per_shell = [np.zeros(k, dtype=np.int64)] + [
        np.bincount(index[depth == s], minlength=k) for s in range(1, params.max_thickness + 2)
    ]
    # The *true* edge touches background inside the image. A sprite cropped by
    # the image border has no outline along the cut, so the cut is not
    # evidence for or against an outline colour.
    touches_bg = np.zeros_like(sprite)
    for dy, dx in NEIGHBOURS_4:
        touches_bg |= ~shifted(sprite, dy, dx, True)
    edge = sprite & touches_bg
    n_edge = int(edge.sum())
    if n_edge == 0:
        return none

    on_edge = np.bincount(index[edge], minlength=k) >= params.min_contour_share * n_edge
    band = on_edge.copy()
    for t in range(1, params.max_thickness + 1):
        if t > 1:
            band &= per_shell[t] >= params.min_band_fill * per_shell[1]
        has_inside = per_shell[t + 1].sum() > 0  # a sprite that is all band has no outline
        stops = per_shell[t + 1] <= params.max_continuation * per_shell[t]
        colors = tuple(int(c) for c in np.nonzero(band & stops)[0]) if has_inside else ()
        if colors and _encloses(np.isin(index, colors) & edge, edge, params):
            thin = ~in_2x2_block(index, sprite)
            mask = sprite & np.isin(index, colors) & ((depth <= t) | thin)
            return Outline(mask, "color", colors, t)

    return _selout(index, sprite, edge, palette, params) or none


def _encloses(covered: np.ndarray, edge: np.ndarray, params: OutlineParams) -> bool:
    """True if ``covered`` is >= min_side_coverage of ``edge`` on every side that has edge."""
    ys, xs = np.nonzero(edge)
    dy = ys - ys.mean()
    dx = xs - xs.mean()
    sides = (dy <= -np.abs(dx), dx >= np.abs(dy), dy >= np.abs(dx), dx <= -np.abs(dy))
    hit = covered[ys, xs]
    return all(hit[side].sum() >= params.min_side_coverage * side.sum() for side in sides)


def _selout(index, sprite, edge, palette, params) -> Outline | None:
    L = np.where(sprite, palette.lightness[np.maximum(index, 0)], 0.0)
    inner = sprite & ~edge
    total = np.zeros(L.shape)
    count = np.zeros(L.shape, dtype=np.int64)
    for dy, dx in NEIGHBOURS_4:
        nb = shifted(inner, dy, dx, False)
        total += np.where(nb, shifted(L, dy, dx, 0.0), 0.0)
        count += nb
    with np.errstate(invalid="ignore", divide="ignore"):
        # No inner neighbour (a 1-2 px wide part): no evidence, never sel-out.
        mean_inner = np.where(count > 0, total / np.maximum(count, 1), -np.inf)
    darker = edge & (L <= mean_inner - params.selout_min_darker)
    if darker.sum() < params.selout_min_fraction * edge.sum():
        return None
    return Outline(darker, "selout", ())
