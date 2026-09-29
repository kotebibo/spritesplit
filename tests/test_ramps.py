"""Unit tests for ramp detection on hand-drawn micro sprites."""

import numpy as np

from spritesplit.palette import extract_palette
from spritesplit.ramps import RampParams, adjacency_counts, detect_ramps

# A hue-shifted red ramp: purple -> red -> orange -> yellow.
RED = ["#4a1d4f", "#8c2a4a", "#d0463a", "#f08a3c", "#ffd27a"]
# A blue ramp, a different material.
BLUE = ["#23305e", "#3b5dc9", "#41a6f6"]


def ramps_of(img, params=RampParams()):
    pal, idx = extract_palette(img)
    fill = idx >= 0
    ramps = detect_ramps(pal, idx, fill, params)
    from spritesplit.color import to_hex

    return {tuple(to_hex(pal.colors[c]) for c in chain) for chain in ramps.chains}, ramps, pal


def test_adjacency_counts_are_symmetric_and_ignore_same_colour():
    idx = np.array([[0, 0, 1], [2, 1, 1]])
    adj = adjacency_counts(idx, np.ones_like(idx, dtype=bool), 3)
    np.testing.assert_array_equal(adj, adj.T)
    assert adj[0, 1] == 2 and adj[0, 2] == 1 and adj[1, 2] == 1
    assert np.all(np.diag(adj) == 0)


def test_hue_shifted_ramp_is_one_chain(rows_to_sprite):
    legend = dict(zip("12345", RED, strict=True))
    img = rows_to_sprite(["12345", "12345", "12345"], legend)
    chains, _, _ = ramps_of(img)
    assert chains == {tuple(RED)}


def test_two_materials_are_separate_ramps(rows_to_sprite):
    legend = {**dict(zip("123", RED[1:4], strict=True)), **dict(zip("abc", BLUE, strict=True))}
    img = rows_to_sprite(["123abc", "123abc"], legend)
    chains, _, _ = ramps_of(img)
    assert chains == {tuple(RED[1:4]), tuple(BLUE)}


def test_colours_that_never_touch_are_not_linked(rows_to_sprite):
    legend = {"1": RED[1], "2": RED[2], "x": "#808080"}
    img = rows_to_sprite(["1x2"], legend)  # compatible colours, separated by grey
    chains, _, _ = ramps_of(img)
    assert (RED[1], RED[2]) not in chains


def test_mutual_best_skips_to_nearest_step(rows_to_sprite):
    # 1 touches both 2 and 3; the nearer step (2) wins, giving a single chain.
    legend = dict(zip("123", RED[1:4], strict=True))
    img = rows_to_sprite(["123", "131"], legend)
    chains, _, _ = ramps_of(img)
    assert chains == {tuple(RED[1:4])}


def test_shared_colour_is_marked_shared(rows_to_sprite):
    # One dark maroon used as the shadow of two materials.
    dark, cap, stem = "#3d1f2e", ["#7a2336", "#c23b3b"], ["#6e4f5a", "#b89c8c"]
    legend = {"d": dark, "c": cap[0], "C": cap[1], "s": stem[0], "S": stem[1], "x": "#ffffff"}
    img = rows_to_sprite(
        [
            "dcCxxxx",
            "dcCxdsS",
            "dcCxdsS",
        ],
        legend,
    )
    _, ramps, pal = ramps_of(img)
    from spritesplit.color import to_hex

    shared = {to_hex(pal.colors[c]) for c in ramps.shared}
    assert dark in shared


def test_shared_highlight_resolves_per_region(rows_to_sprite):
    # One near-white specular on two soft materials: it can chain into only one
    # ramp, so it is shared into the other and resolved per blob. (Saturated
    # tones are > max_step_ab from white and would leave it a separate flat.)
    from spritesplit.color import to_hex
    from spritesplit.decompose import assign_pixels

    legend = {
        "r": "#c06878",
        "R": "#e8a8b0",
        "b": "#6878c0",
        "B": "#a8b8e8",
        "w": "#f4f0f4",
    }
    img = rows_to_sprite(
        [
            "rrRRR.bbBBB",
            "rrRwR.bbBwB",
            "rrRRR.bbBBB",
        ],
        legend,
    )
    pal, idx = extract_palette(img)
    fill = idx >= 0
    ramps = detect_ramps(pal, idx, fill)
    assert {to_hex(pal.colors[c]) for c in ramps.shared} == {"#f4f0f4"}
    pixel_ramp = assign_pixels(idx, fill, ramps, RampParams().shared_override_ratio)
    assert pixel_ramp[1, 3] == pixel_ramp[1, 2]  # white on red joins red
    assert pixel_ramp[1, 9] == pixel_ramp[1, 8]  # white on blue joins blue
    assert pixel_ramp[1, 2] != pixel_ramp[1, 8]
