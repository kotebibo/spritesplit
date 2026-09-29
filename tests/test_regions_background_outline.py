import numpy as np

from spritesplit.background import find_background
from spritesplit.outline import find_outline
from spritesplit.palette import extract_palette
from spritesplit.regions import in_2x2_block, label_components

BG = "#ffffff"
LEGEND = {"o": "#101010", "a": "#c04040", "b": "#e08080", "w": BG, "k": "#101010"}


def test_label_components_4_connectivity():
    key = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    labels, n = label_components(key, np.ones_like(key, dtype=bool))
    # Diagonal 1s are not 4-connected; the bottom row joins the centre.
    assert n == 7 - 1
    assert labels[1, 1] == labels[2, 0] == labels[2, 2]
    assert labels[0, 0] != labels[0, 2]


def test_in_2x2_block_separates_lines_from_areas():
    rows = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 2, 2],
            [0, 0, 1, 2, 2],
        ]
    )
    block = in_2x2_block(rows, np.ones_like(rows, dtype=bool))
    assert not block[0, 0] and not block[1, 1] and not block[2, 2]  # diagonal line
    assert block[1:, 3:].all()  # 2x2 area


def test_background_flat_colour_even_if_sprite_touches_a_corner(rows_to_sprite):
    img = rows_to_sprite(
        [
            "oooww",
            "oaaow",
            "oaaow",
            "oooww",
            "wwwww",
        ],
        LEGEND,
    )
    pal, idx = extract_palette(img)
    bg = find_background(idx)
    assert bg.mode == "color"
    np.testing.assert_array_equal(bg.mask, (img == [255, 255, 255, 255]).all(axis=-1))
    assert not bg.mask[0, 0]


def test_background_prefers_alpha(rows_to_sprite):
    img = rows_to_sprite(["..", ".a"], LEGEND)
    _, idx = extract_palette(img)
    bg = find_background(idx)
    assert bg.mode == "transparent" and bg.mask.sum() == 3


def test_background_none_when_corners_disagree(rows_to_sprite):
    img = rows_to_sprite(["ao", "bw"], LEGEND)
    _, idx = extract_palette(img)
    assert find_background(idx).mode == "none"


def test_dark_fill_is_not_outline(rows_to_sprite):
    # A black blob ('k', same colour as the outline) inside the sprite.
    img = rows_to_sprite(
        [
            "........",
            ".oooooo.",
            ".oaakko.",
            ".oaakko.",
            ".obbbbo.",
            ".oooooo.",
            "........",
        ],
        LEGEND,
    )
    pal, idx = extract_palette(img)
    sprite = idx >= 0
    out = find_outline(idx, sprite, pal)
    assert out.mode == "color"
    # The 2x2 black blob and the outline pixels that square off with it are fill.
    assert not out.mask[2:4, 4:6].any()
    assert out.mask[5, 1:7].all() and out.mask[1:5, 1].all()
