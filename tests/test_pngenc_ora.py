import io
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

from spritesplit.ora import ora_bytes
from spritesplit.pngenc import encode_png


def rand_rgba(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 4), dtype=np.uint8)


def decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert("RGBA"))


def test_png_roundtrip_small():
    img = rand_rgba(5, 3)
    np.testing.assert_array_equal(decode(encode_png(img)), img)


def test_png_roundtrip_spans_multiple_stored_blocks():
    img = rand_rgba(200, 100)  # 80 KB of raw scanlines > one 64 KB stored block
    np.testing.assert_array_equal(decode(encode_png(img)), img)


def test_png_is_deterministic():
    img = rand_rgba(16, 16, seed=7)
    assert encode_png(img) == encode_png(img.copy())


def test_ora_structure():
    layers = [("top", rand_rgba(4, 6, 1)), ("bottom", rand_rgba(4, 6, 2))]
    merged = rand_rgba(4, 6, 3)
    data = ora_bytes(layers, merged)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        assert infos[0].filename == "mimetype"
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert zf.read("mimetype") == b"image/openraster"
        # mimetype must start at byte 38 so file(1) can sniff it.
        assert data[30:38] == b"mimetype"
        root = ET.fromstring(zf.read("stack.xml"))
        assert (root.get("w"), root.get("h")) == ("6", "4")
        srcs = [el.get("src") for el in root.iter("layer")]
        assert srcs == ["data/top.png", "data/bottom.png"]
        for (_, rgba), src in zip(layers, srcs, strict=True):
            np.testing.assert_array_equal(decode(zf.read(src)), rgba)
        np.testing.assert_array_equal(decode(zf.read("mergedimage.png")), merged)
        assert "Thumbnails/thumbnail.png" in zf.namelist()


def test_ora_is_deterministic():
    layers = [("a", rand_rgba(8, 8, 1))]
    assert ora_bytes(layers, layers[0][1]) == ora_bytes(layers, layers[0][1].copy())


def test_thumbnail_capped_at_256():
    big = rand_rgba(300, 520)
    with zipfile.ZipFile(io.BytesIO(ora_bytes([("a", big)], big))) as zf:
        thumb = decode(zf.read("Thumbnails/thumbnail.png"))
    assert max(thumb.shape[:2]) <= 256
