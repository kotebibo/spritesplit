"""Command line: spritesplit sprite.png -o sprite.ora"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .decompose import LAYER_ORDER, Params, decompose
from .image import load_rgba, upscale
from .ora import write_ora
from .palette import NotPixelArtError
from .pngenc import encode_png


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spritesplit",
        description="Decompose a pixel-art sprite into editable layers "
        "(outline, highlights, shading, flats, background).",
    )
    p.add_argument("input", type=Path, help="pixel-art image (PNG recommended)")
    p.add_argument("-o", "--output", type=Path, help="output .ora (default: INPUT.ora)")
    p.add_argument("--layers-dir", type=Path, help="also write each layer as a PNG here")
    p.add_argument("--report", metavar="FILE", help="write a JSON report ('-' for stdout)")
    p.add_argument(
        "--scale",
        type=int,
        default=1,
        help="upscale output layers by this integer factor (default 1: native pixel size; "
        "upscaled inputs are detected and reduced to native size automatically)",
    )
    p.add_argument("--max-colors", type=int, default=256, help="refuse images with more colors")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scale < 1:
        print("spritesplit: --scale must be >= 1", file=sys.stderr)
        return 2
    try:
        result = decompose(load_rgba(args.input), Params(max_colors=args.max_colors))
    except (OSError, NotPixelArtError, ValueError) as exc:
        print(f"spritesplit: {exc}", file=sys.stderr)
        return 1

    layers = [(name, upscale(result.layers[name], args.scale)) for name in LAYER_ORDER]
    output = args.output or args.input.with_suffix(".ora")
    info = result.report()
    try:
        write_ora(output, layers, upscale(result.image, args.scale))
        if args.layers_dir:
            args.layers_dir.mkdir(parents=True, exist_ok=True)
            for name, rgba in layers:
                (args.layers_dir / f"{name}.png").write_bytes(encode_png(rgba))
        if args.report:
            text = json.dumps(info, indent=2, sort_keys=True) + "\n"
            if args.report == "-":
                sys.stdout.write(text)
            else:
                Path(args.report).write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"spritesplit: {exc}", file=sys.stderr)
        return 1

    note = f" (input was {result.scale}x upscaled)" if result.scale > 1 else ""
    print(
        f"{output}: {info['palette_size']} colors, {len(info['ramps'])} ramps, "
        f"outline={info['outline']['mode']}, background={info['background']['mode']}{note}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
