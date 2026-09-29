"""Regenerate tests/golden from tests/fixtures. Review the diff before committing!

uv run python scripts/update_golden.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from conftest import FIXTURE_NAMES, GOLDEN  # noqa: E402
from test_golden import render  # noqa: E402


def main() -> None:
    for name in FIXTURE_NAMES:
        out = GOLDEN / name
        shutil.rmtree(out, ignore_errors=True)
        render(name, out)
    print(f"regenerated {len(FIXTURE_NAMES)} golden sets in {GOLDEN}")


if __name__ == "__main__":
    main()
