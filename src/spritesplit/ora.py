"""OpenRaster (.ora) writer: a ZIP of PNGs plus stack.xml. Stdlib only.

Opens in Krita, GIMP (2.10+ via plugin, 3.0 natively) and MyPaint.
Spec: https://www.openraster.org/

Determinism: entries are written in a fixed order, uncompressed, with a fixed
timestamp and fixed "made by" system, so the archive bytes depend only on
the layer pixels.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .pngenc import encode_png

_EPOCH = (1980, 1, 1, 0, 0, 0)
_THUMB_MAX = 256


def _entry(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_EPOCH)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3  # unix; otherwise zipfile uses the host OS
    info.external_attr = 0o100644 << 16
    zf.writestr(info, data)


def _thumbnail(rgba: np.ndarray) -> np.ndarray:
    """Nearest-neighbour shrink so the longest side is <= 256 (the spec maximum)."""
    step = -(-max(rgba.shape[:2]) // _THUMB_MAX)
    return np.ascontiguousarray(rgba[::step, ::step]) if step > 1 else rgba


def stack_xml(width: int, height: int, names: list[str]) -> str:
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f'<image version="0.0.5" w="{width}" h="{height}">',
        "  <stack>",
    ]
    for name in names:
        lines.append(
            f"    <layer name={quoteattr(name)} src={quoteattr(f'data/{name}.png')} "
            'x="0" y="0" opacity="1.000" visibility="visible" composite-op="svg:src-over"/>'
        )
    lines += ["  </stack>", "</image>", ""]
    return "\n".join(lines)


def ora_bytes(layers: list[tuple[str, np.ndarray]], merged: np.ndarray) -> bytes:
    """Build an .ora archive. ``layers`` is ordered top to bottom, as in stack.xml."""
    h, w = merged.shape[:2]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        _entry(zf, "mimetype", b"image/openraster")  # must be first and uncompressed
        _entry(zf, "stack.xml", stack_xml(w, h, [n for n, _ in layers]).encode("utf-8"))
        for name, rgba in layers:
            _entry(zf, f"data/{name}.png", encode_png(rgba))
        _entry(zf, "mergedimage.png", encode_png(merged))
        _entry(zf, "Thumbnails/thumbnail.png", encode_png(_thumbnail(merged)))
    return buf.getvalue()


def write_ora(path: str | Path, layers: list[tuple[str, np.ndarray]], merged: np.ndarray) -> None:
    Path(path).write_bytes(ora_bytes(layers, merged))
