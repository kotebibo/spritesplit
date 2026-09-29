"""Semantic checks against ground truth in tests/fixtures/expected.json.

These say *what* the decomposition must mean (ramps, bases, outline mode).
test_golden.py separately pins the exact output bytes.
"""

import numpy as np
import pytest
from conftest import EXPECTED, FIXTURE_NAMES, FIXTURES

from spritesplit import composite, decompose
from spritesplit.color import to_hex
from spritesplit.image import load_rgba


@pytest.fixture(scope="module", params=FIXTURE_NAMES)
def case(request):
    name = request.param
    return name, EXPECTED[name], decompose(load_rgba(FIXTURES / f"{name}.png"))


def materials(d):
    """Each ramp as the colours its pixels actually use, dark -> light, plus its base."""
    out = {}
    for r in range(len(d.ramps.chains)):
        used = np.unique(d.index[d.pixel_ramp == r])
        used = sorted(used, key=lambda c: d.palette.lightness[c])
        out[tuple(to_hex(d.palette.colors[c]) for c in used)] = to_hex(d.palette.colors[d.bases[r]])
    return out


def test_scale(case):
    _, exp, d = case
    assert d.scale == exp["scale"]


def test_background(case):
    _, exp, d = case
    assert d.background.mode == exp["background"]


def test_outline(case):
    _, exp, d = case
    assert d.outline.mode == exp["outline"]["mode"]
    assert [to_hex(d.palette.colors[c]) for c in d.outline.colors] == exp["outline"]["colors"]
    if "thickness" in exp["outline"]:
        assert d.outline.thickness == exp["outline"]["thickness"]


def test_ramps_and_bases(case):
    _, exp, d = case
    if "ramps" not in exp:
        pytest.skip("no exact ramp ground truth for this fixture")
    expected = {tuple(r): b for r, b in zip(exp["ramps"], exp["bases"], strict=True)}
    assert materials(d) == expected


def test_lossless(case):
    _, _, d = case
    np.testing.assert_array_equal(composite(d.layers), d.image)


def test_layers_do_not_overlap(case):
    _, _, d = case
    detail = ("outline", "highlights", "shading", "background")
    coverage = sum((d.layers[n][..., 3] > 0).astype(int) for n in detail)
    assert coverage.max() <= 1
    # Flats cover exactly the fill: everything that is neither outline nor background.
    fill = (d.index >= 0) & ~d.background.mask & ~d.outline.mask
    np.testing.assert_array_equal(d.layers["flats"][..., 3] > 0, fill)


def test_bases_include(case):
    _, exp, d = case
    if "bases_include" not in exp:
        pytest.skip("fixture has exact ramps instead")
    bases = {to_hex(d.palette.colors[b]) for b in d.bases}
    assert set(exp["bases_include"]) <= bases
