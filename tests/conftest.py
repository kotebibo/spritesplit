from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).parent
FIXTURES = TESTS / "fixtures"
GOLDEN = TESTS / "golden"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
FIXTURE_NAMES = sorted(EXPECTED)


def sprite_from_rows(rows: list[str], legend: dict[str, str]) -> np.ndarray:
    """Build an RGBA sprite from ASCII rows; '.' is transparent, others map via legend."""
    from spritesplit.color import from_hex

    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 4), dtype=np.uint8)
    for y, row in enumerate(rows):
        assert len(row) == w, "ragged rows"
        for x, ch in enumerate(row):
            if ch != ".":
                img[y, x] = from_hex(legend[ch])
    return img


@pytest.fixture
def rows_to_sprite():
    return sprite_from_rows
