import numpy as np
import pytest

from spritesplit.image import detect_scale, downscale, upscale
from spritesplit.palette import TRANSPARENT, NotPixelArtError, extract_palette


def random_sprite(h=9, w=7, colors=5, seed=0):
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(colors, 4), dtype=np.uint8)
    palette[:, 3] = 255
    return palette[rng.integers(0, colors, size=(h, w))]


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_detect_scale_recovers_integer_upscale(k):
    sprite = random_sprite()
    big = upscale(sprite, k)
    assert detect_scale(big) == k
    np.testing.assert_array_equal(downscale(big, k), sprite)


def test_detect_scale_rejects_offset_blocks():
    big = upscale(random_sprite(), 4)
    # Crop one column: runs are no longer aligned to multiples of 4.
    assert detect_scale(big[:, 1:]) == 1


def test_detect_scale_flat_image_is_one():
    assert detect_scale(np.full((8, 8, 4), 255, dtype=np.uint8)) == 1


def test_palette_order_depends_only_on_colours():
    sprite = random_sprite(seed=3)
    pal_a, idx_a = extract_palette(sprite)
    pal_b, idx_b = extract_palette(sprite[::-1, ::-1])
    np.testing.assert_array_equal(pal_a.colors, pal_b.colors)
    np.testing.assert_array_equal(pal_a.colors[idx_a], sprite)
    np.testing.assert_array_equal(pal_b.colors[idx_b], sprite[::-1, ::-1])


def test_palette_transparent_pixels():
    sprite = random_sprite()
    sprite[0, 0] = 0
    pal, idx = extract_palette(sprite)
    assert idx[0, 0] == TRANSPARENT
    assert (pal.counts.sum()) == sprite.shape[0] * sprite.shape[1] - 1


def test_palette_limit():
    with pytest.raises(NotPixelArtError):
        extract_palette(random_sprite(colors=20, h=20, w=20), max_colors=8)
