import numpy as np

from spritesplit.color import from_hex, pack_rgba, srgb_to_oklab, to_hex, unpack_rgba


def test_oklab_reference_points():
    lab = srgb_to_oklab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8))
    np.testing.assert_allclose(lab[0], [1.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-3)
    # Published OKLab value for sRGB red.
    np.testing.assert_allclose(lab[2], [0.6279, 0.2249, 0.1258], atol=1e-3)


def test_oklab_is_rounded_for_determinism():
    lab = srgb_to_oklab(np.array([[12, 200, 77]], dtype=np.uint8))
    np.testing.assert_array_equal(lab, np.round(lab, 4))


def test_grey_has_no_chroma():
    lab = srgb_to_oklab(np.array([[128, 128, 128]], dtype=np.uint8))
    np.testing.assert_allclose(lab[0, 1:], [0.0, 0.0], atol=2e-4)


def test_pack_roundtrip_and_order():
    colors = np.array([[1, 2, 3, 4], [255, 0, 0, 255], [0, 0, 0, 0]], dtype=np.uint8)
    keys = pack_rgba(colors)
    np.testing.assert_array_equal(unpack_rgba(keys), colors)
    assert keys[2] < keys[0] < keys[1]


def test_hex_roundtrip():
    assert to_hex((255, 128, 0, 255)) == "#ff8000"
    assert to_hex((255, 128, 0, 16)) == "#ff800010"
    assert from_hex("#ff8000") == (255, 128, 0, 255)
    assert from_hex("ff800010") == (255, 128, 0, 16)
