# spritesplit

Pixel-art sprite → editable layers (outline, highlights, shading, flats, background) as `.ora`.
Algorithm walkthrough: `docs/ALGORITHM.md`. Read it before changing `outline.py` or `ramps.py`.

## Commands
- `uv run pytest` · `uv run ruff check . && uv run ruff format --check .`
- `uv run python scripts/update_golden.py` after an *intentional* output change; view the layer PNGs before accepting.
- `uv run python scripts/make_fixtures.py` regenerates synthetic fixtures + `tests/fixtures/expected.json`.

## Invariants (CI enforces them)
- **Byte-identical output across OSes.** No zlib compression (`pngenc.py` uses stored blocks), fixed ZIP metadata,
  OKLab rounded, only `+ - * /` in threshold math (no sqrt/hypot/pow on compared values), ties broken by palette index.
- **Lossless:** `composite(layers) == input`.
- Core path: numpy + Pillow only. No GPU, network, ML, or paid dependency (Aseprite must stay optional).

## Scope guards
- Pixel art only; per-body-part splitting, animation consistency: out of scope.
- The decomposer is the product; generation (later phases) is optional and must not become the identity.
- Fixtures: generated or CC0 only, each listed in `tests/fixtures/SOURCES.md`.

## Git
- Never add Claude/AI attribution to commits or PRs: no `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" line.
- Release in phases: don't push work for a later roadmap phase until the current phase is released.
