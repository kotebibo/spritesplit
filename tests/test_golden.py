"""Golden files: the CLI output for every fixture must match tests/golden byte for byte.

This is the determinism guarantee: CI runs it on Linux, Windows and macOS against
the same committed bytes. After an intentional change, regenerate with
    uv run python scripts/update_golden.py
and review the layer PNG diffs (GitHub renders them) before committing.
"""

from pathlib import Path

import pytest
from conftest import FIXTURE_NAMES, FIXTURES, GOLDEN

from spritesplit.cli import main


def render(name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    code = main(
        [
            str(FIXTURES / f"{name}.png"),
            "-o",
            str(out / "layers.ora"),
            "--layers-dir",
            str(out),
            "--report",
            str(out / "report.json"),
        ]
    )
    assert code == 0


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_matches_golden(name, tmp_path):
    render(name, tmp_path)
    golden = GOLDEN / name
    expected = sorted(p.name for p in golden.iterdir())
    assert sorted(p.name for p in tmp_path.iterdir()) == expected
    for filename in expected:
        assert (tmp_path / filename).read_bytes() == (golden / filename).read_bytes(), filename


def test_repeat_runs_are_byte_identical(tmp_path):
    render("mushroom_shared", tmp_path / "a")
    render("mushroom_shared", tmp_path / "b")
    for f in (tmp_path / "a").iterdir():
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes()
