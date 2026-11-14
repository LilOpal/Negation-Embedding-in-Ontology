"""Fuzzy logical connectives with switchable t-norms.

All operators take and return truth values in [0, 1]. `or_` and `implies`
are defined as the De Morgan dual and S-implication of `and_`, so De Morgan's
laws hold by construction under every supported t-norm.
"""

from __future__ import annotations

import torch
from torch import Tensor

TNORMS = ("product", "godel", "lukasiewicz")


def neg(x: Tensor) -> Tensor:
    """Standard fuzzy negation."""
    return 1.0 - x


def and_(x: Tensor, y: Tensor, tnorm: str = "product") -> Tensor:
    if tnorm == "product":
        return x * y
    if tnorm == "godel":
        return torch.minimum(x, y)
    if tnorm == "lukasiewicz":
        return torch.clamp(x + y - 1.0, min=0.0)
    raise ValueError(f"unknown t-norm: {tnorm!r}, expected one of {TNORMS}")


def or_(x: Tensor, y: Tensor, tnorm: str = "product") -> Tensor:
    """De Morgan dual of `and_`."""
    return neg(and_(neg(x), neg(y), tnorm))


def implies(x: Tensor, y: Tensor, tnorm: str = "product") -> Tensor:
    """S-implication: x -> y := ~x or y."""
    return or_(neg(x), y, tnorm)
