"""4-connected component labelling and neighbour helpers.

Pure Python BFS: sprites are small (a 256x256 sprite is 65k pixels), and this
avoids pulling in scipy for one function. Labels are assigned in raster scan
order, so the output is deterministic.
"""

from __future__ import annotations

import numpy as np

# (dy, dx) for the 4-neighbourhood.
NEIGHBOURS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def label_components(key: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected regions of equal ``key`` among ``valid`` pixels.

    Returns (labels, n) where labels is int32 with -1 for invalid pixels.
    """
    h, w = key.shape
    keys = key.ravel().tolist()
    ok = valid.ravel().tolist()
    labels = [-1] * (h * w)
    n = 0
    for start in range(h * w):
        if not ok[start] or labels[start] != -1:
            continue
        k = keys[start]
        labels[start] = n
        stack = [start]
        while stack:
            p = stack.pop()
            y, x = divmod(p, w)
            for q, inside in (
                (p - w, y > 0),
                (p + w, y < h - 1),
                (p - 1, x > 0),
                (p + 1, x < w - 1),
            ):
                if inside and ok[q] and labels[q] == -1 and keys[q] == k:
                    labels[q] = n
                    stack.append(q)
        n += 1
    return np.array(labels, dtype=np.int32).reshape(h, w), n


def shifted(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """Return b where b[y, x] = a[y + dy, x + dx], or ``fill`` outside the image."""
    h, w = a.shape[:2]
    out = np.full_like(a, fill)
    ys, yd = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
    xs, xd = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
    out[yd, xd] = a[ys, xs]
    return out


def in_2x2_block(index: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """True for valid pixels that belong to some 2x2 square of one single colour.

    A 1-pixel line (straight, diagonal or staircase) never contains such a
    square; a filled area almost always does. This is the thinness test.
    """
    tl = index[:-1, :-1]
    block = (
        (tl == index[1:, :-1])
        & (tl == index[:-1, 1:])
        & (tl == index[1:, 1:])
        & valid[:-1, :-1]
        & valid[1:, :-1]
        & valid[:-1, 1:]
        & valid[1:, 1:]
    )
    out = np.zeros(index.shape, dtype=bool)
    out[:-1, :-1] |= block
    out[1:, :-1] |= block
    out[:-1, 1:] |= block
    out[1:, 1:] |= block
    return out
