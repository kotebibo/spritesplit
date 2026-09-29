"""A deterministic PNG encoder.

Pillow and zlib's compressed output depends on the zlib version (Linux links
the system zlib; Windows/macOS wheels bundle their own), so the same pixels
can produce different bytes on different machines. This encoder emits deflate
*stored* blocks, whose layout is fully specified by RFC 1951, plus CRC32 and
Adler-32 checksums, which are also exactly specified. Same pixels -> same
bytes, everywhere.

The cost is size (no compression), which is irrelevant at sprite sizes:
a 128x128 RGBA layer is ~64 KiB.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_STORED = 0xFFFF


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _zlib_stored(raw: bytes) -> bytes:
    out = bytearray(b"\x78\x01")  # zlib header: deflate, 32K window, no dict; (0x7801 % 31 == 0)
    pos = 0
    while True:
        block = raw[pos : pos + _MAX_STORED]
        pos += len(block)
        final = pos >= len(raw)
        out += bytes([1 if final else 0]) + struct.pack("<HH", len(block), len(block) ^ 0xFFFF)
        out += block
        if final:
            break
    out += struct.pack(">I", zlib.adler32(raw))
    return bytes(out)


def encode_png(rgba: np.ndarray) -> bytes:
    """Encode (H, W, 4) uint8 RGBA as PNG bytes (8-bit RGBA, no filtering, stored deflate)."""
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("expected an (H, W, 4) uint8 array")
    h, w = rgba.shape[:2]
    rows = np.zeros((h, 1 + 4 * w), dtype=np.uint8)  # leading 0 byte = filter type "None"
    rows[:, 1:] = rgba.reshape(h, 4 * w)
    # 8-bit depth, colour type 6 (RGBA), compression 0 (deflate), filter method 0
    # (the only one; every row above uses filter type None), no interlace
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", _zlib_stored(rows.tobytes()))
        + _chunk(b"IEND", b"")
    )
