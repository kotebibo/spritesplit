import io
import json
import zipfile

import numpy as np
from conftest import FIXTURES
from PIL import Image

from spritesplit.cli import main

BALL = FIXTURES / "ball_hueshift.png"


def ora_size(path):
    with zipfile.ZipFile(path) as zf, Image.open(io.BytesIO(zf.read("mergedimage.png"))) as im:
        return im.size


def test_default_output_path_and_native_size(tmp_path):
    src = tmp_path / "ball.png"
    src.write_bytes((FIXTURES / "ball_4x.png").read_bytes())
    assert main([str(src)]) == 0
    assert ora_size(tmp_path / "ball.ora") == (18, 18)  # 4x input reduced to native


def test_scale_upscales_output(tmp_path):
    out = tmp_path / "ball.ora"
    assert main([str(BALL), "-o", str(out), "--scale", "3"]) == 0
    assert ora_size(out) == (54, 54)


def test_report_to_stdout(tmp_path, capsys):
    assert main([str(BALL), "-o", str(tmp_path / "b.ora"), "--report", "-"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["outline"]["mode"] == "color"
    assert report["size"] == {"width": 18, "height": 18, "scale": 1}


def test_bad_scale(tmp_path, capsys):
    assert main([str(BALL), "--scale", "0"]) == 2
    assert "--scale" in capsys.readouterr().err


def test_missing_input(tmp_path, capsys):
    assert main([str(tmp_path / "nope.png")]) == 1
    assert capsys.readouterr().err.startswith("spritesplit:")


def test_too_many_colours(tmp_path, capsys):
    noisy = np.random.default_rng(0).integers(0, 256, size=(32, 32, 4), dtype=np.uint8)
    noisy[..., 3] = 255
    src = tmp_path / "noisy.png"
    Image.fromarray(noisy, "RGBA").save(src)
    assert main([str(src), "--max-colors", "64"]) == 1
    assert "colors" in capsys.readouterr().err


def test_unwritable_output_is_a_clean_error(tmp_path, capsys):
    assert main([str(BALL), "-o", str(tmp_path / "no" / "such" / "dir.ora")]) == 1
    assert capsys.readouterr().err.startswith("spritesplit:")


def test_16_bit_input_is_rejected(tmp_path, capsys):
    src = tmp_path / "deep.png"
    Image.fromarray(np.full((4, 4), 40000, dtype=np.uint16)).save(src)
    assert main([str(src)]) == 1
    assert "8-bit" in capsys.readouterr().err
