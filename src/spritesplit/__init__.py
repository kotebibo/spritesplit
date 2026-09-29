"""Spritesplit: decompose a pixel-art sprite into the layers artists work in."""

__version__ = "0.1.0"

from .decompose import LAYER_ORDER, Decomposition, Params, composite, decompose  # noqa: E402

__all__ = ["LAYER_ORDER", "Decomposition", "Params", "composite", "decompose", "__version__"]
