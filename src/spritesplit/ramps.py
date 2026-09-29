"""Palette ramp detection: the core algorithm.

A ramp is one material at several light levels (deep shadow, shadow, base,
highlight...). Skilled artists *hue-shift* ramps (shadows drift toward
blue/purple, highlights toward yellow), so a ramp is NOT "similar hue, varying
lightness". The hue at one end can be far from the other end. What stays small
is each individual *step*. So ramps are found by chaining, not clustering:

1. Adjacency. Count how often each pair of palette colours are 4-neighbours
   among fill pixels (not background, not outline). Shade pixels touch their
   own material's other shades; different materials are usually kept apart by
   outline. Colours that never touch are not linked, however similar they are.

2. Step compatibility, for a darker colour i and a lighter colour j (OKLab):
     min_step_l <= L_j - L_i <= max_step_l   (a real lightness step, not a jump)
     |(a_j, b_j) - (a_i, b_i)| <= max_step_ab (small drift in the chroma plane)
   Using a/b-plane distance instead of hue angle means greys (where hue is
   noise) and hue wrap-around need no special cases.

3. Best step. Each colour picks its best lighter partner (best_up) and its
   best darker partner (best_down): the compatible, adjacent candidate with
   the smallest normalised step cost. Ties go to the lower palette index.

4. Mutual best. Keep the link i -> j only if best_up[i] == j AND
   best_down[j] == i. Every colour then has at most one link up and one down,
   so the links form disjoint chains ordered by lightness: those are the ramps.
   No global threshold on the ramp as a whole is needed.

5. Shared colours. Limited palettes reuse colours across ramps (the darkest
   skin shadow is also the dark wood; one white specular on every material).
   A one-sided best link marks the colour as *shared*: if j wants i as its
   darker step but i chose someone else, i is shared into j's ramp; if i
   wants j as its lighter step but j chose someone else, j is shared into i's.
   Which ramp a shared colour belongs to is decided per pixel later, from its
   spatial neighbours (see decompose.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .palette import Palette

NO_RAMP = -1


@dataclass(frozen=True)
class RampParams:
    min_step_l: float = 0.02  # smaller L differences are not a light-level step
    max_step_l: float = 0.40  # larger jumps are a different material
    max_step_ab: float = 0.10  # max a/b-plane drift per step (allows ~20-40 deg hue shift)
    min_adjacency: int = 1  # neighbour pairs required to link two colours
    # A shared colour leaves its primary (mutual-best) ramp for a one-sided one
    # only when that ramp gets more than this many times the primary's votes.
    shared_override_ratio: float = 2.0


@dataclass(frozen=True)
class Ramps:
    chains: tuple[tuple[int, ...], ...]  # palette indices, dark -> light
    ramp_of: np.ndarray  # (k,) primary ramp id per colour, NO_RAMP if unused in fills
    shared: dict[int, tuple[int, ...]] = field(default_factory=dict)  # colour -> extra ramps
    best_up: np.ndarray | None = None  # (k,) diagnostic: best lighter partner or -1
    best_down: np.ndarray | None = None  # (k,) diagnostic: best darker partner or -1

    def candidates(self, color: int) -> tuple[int, ...]:
        """All ramps a colour may belong to, primary first."""
        return (int(self.ramp_of[color]), *self.shared.get(color, ()))


def adjacency_counts(index: np.ndarray, fill: np.ndarray, k: int) -> np.ndarray:
    """(k, k) symmetric matrix: number of 4-neighbour pairs between colour i and j."""
    adj = np.zeros((k, k), dtype=np.int64)
    pairs = (
        (index[:, :-1], index[:, 1:], fill[:, :-1] & fill[:, 1:]),
        (index[:-1, :], index[1:, :], fill[:-1, :] & fill[1:, :]),
    )
    for a, b, both in pairs:
        ok = both & (a != b)
        np.add.at(adj, (a[ok], b[ok]), 1)
    return adj + adj.T


def step_costs(palette: Palette, adjacency: np.ndarray, params: RampParams) -> np.ndarray:
    """(k, k) matrix: squared cost[i, j] of stepping from darker i up to lighter j; inf = no link.

    Squared distances only: +, -, * are exactly rounded in IEEE 754, whereas
    sqrt/hypot go through libm and could differ by an ulp between platforms,
    flipping a tie or a threshold and breaking byte-identical output.
    """
    L = palette.lab[:, 0]
    ab = palette.lab[:, 1:]
    d_l = L[None, :] - L[:, None]
    d_a = ab[None, :, 0] - ab[:, None, 0]
    d_b = ab[None, :, 1] - ab[:, None, 1]
    d_ab2 = d_a * d_a + d_b * d_b
    ok = (
        (d_l >= params.min_step_l)
        & (d_l <= params.max_step_l)
        & (d_ab2 <= params.max_step_ab * params.max_step_ab)
        & (adjacency >= params.min_adjacency)
    )
    cost = (d_l / params.max_step_l) ** 2 + d_ab2 / (params.max_step_ab * params.max_step_ab)
    return np.where(ok, cost, np.inf)


def _argmin_or_none(cost: np.ndarray, axis: int) -> np.ndarray:
    """argmin along axis (first index wins ties), or -1 where every entry is inf."""
    best = np.argmin(cost, axis=axis)
    return np.where(np.isfinite(cost.min(axis=axis)), best, -1)


def detect_ramps(
    palette: Palette,
    index: np.ndarray,
    fill: np.ndarray,
    params: RampParams = RampParams(),
) -> Ramps:
    k = len(palette)
    present = np.bincount(index[fill], minlength=k) > 0
    adjacency = adjacency_counts(index, fill, k)
    cost = step_costs(palette, adjacency, params)
    cost[~present, :] = np.inf
    cost[:, ~present] = np.inf

    best_up = _argmin_or_none(cost, axis=1)  # for each darker i: best lighter j
    best_down = _argmin_or_none(cost, axis=0)  # for each lighter j: best darker i

    nxt = np.full(k, -1)
    prv = np.full(k, -1)
    for i in range(k):
        j = best_up[i]
        if j >= 0 and best_down[j] == i:
            nxt[i], prv[j] = j, i

    chains: list[tuple[int, ...]] = []
    ramp_of = np.full(k, NO_RAMP, dtype=np.int64)
    for start in range(k):
        if not present[start] or prv[start] != -1:
            continue
        chain = [start]
        while nxt[chain[-1]] != -1:
            chain.append(int(nxt[chain[-1]]))
        ramp_of[chain] = len(chains)
        chains.append(tuple(chain))

    shared: dict[int, set[int]] = {}
    for i in range(k):
        # i is j's preferred darker step, but i chose another lighter partner.
        for j in np.nonzero(best_down == i)[0]:
            if nxt[i] != j and ramp_of[j] != ramp_of[i]:
                shared.setdefault(i, set()).add(int(ramp_of[j]))
        # j is i's preferred lighter step, but j chose another darker partner:
        # a shared *highlight* (one white specular used by several materials).
        j = best_up[i]
        if j >= 0 and prv[j] != i and ramp_of[j] != ramp_of[i]:
            shared.setdefault(int(j), set()).add(int(ramp_of[i]))

    return Ramps(
        chains=tuple(chains),
        ramp_of=ramp_of,
        shared={c: tuple(sorted(r)) for c, r in sorted(shared.items())},
        best_up=best_up,
        best_down=best_down,
    )
