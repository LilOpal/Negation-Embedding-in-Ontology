"""Fuzzy quantifier aggregators based on generalized (power) means.

`dim=None` aggregates the whole tensor to a scalar; pass a dimension to
aggregate per-slice (e.g. exists-over-objects for each subject).

For `exists`, `p=float("inf")` selects the p -> infinity limit (hard max).
This matters over large domains: a finite-p mean dilutes a single true
witness (e.g. one child among 50 candidates) towards zero, making
existential definitions unsatisfiable.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_EPS = 1e-9


def forall(xs: Tensor, p: float = 2.0, dim: int | None = None) -> Tensor:
    """Universal quantification via p-mean error (higher p = stricter)."""
    errs = torch.pow(1.0 - xs, p)
    return 1.0 - torch.pow(errs.mean(dim=dim).clamp(min=_EPS), 1.0 / p)


def exists(xs: Tensor, p: float = 2.0, dim: int | None = None) -> Tensor:
    """Existential quantification via p-mean (p=inf gives the max)."""
    if math.isinf(p):
        if dim is None:
            return xs.max()
        return xs.max(dim=dim).values
    return torch.pow(torch.pow(xs.clamp(min=_EPS), p).mean(dim=dim), 1.0 / p)

