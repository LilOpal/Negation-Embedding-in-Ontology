"""Learnable logical operators and quantifiers (neural t-norm setting).

Two-layer MLPs replace the fixed fuzzy connectives and quantifier
aggregators, trained end-to-end with the embeddings through the KB loss.
`init_fuzzy` pretrains every operator on a grid to match its fuzzy
counterpart (product t-norm, standard negation, p-mean quantifiers), so
training starts at the known-good fuzzy semantics and deviates only where
the data prefers it.

Quantifiers aggregate truth sets via summary features — `[mean, min,
log1p(count)]` for `forall`, `[mean, max, log1p(count)]` for `exists` —
which makes them arity-independent and keeps the vectorized loss fast.
Empty sets are masked to 0 by the caller (`exists` of nothing is false).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from negemb.ltn.operators import and_, implies, neg, or_
from negemb.ltn.quantifiers import forall


class OpMLP(nn.Module):
    """Two-layer MLP truth function: (..., in_features) -> (...,)."""

    def __init__(self, in_features: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def _log1p_count(n: int, like: Tensor) -> Tensor:
    return torch.full_like(like, math.log1p(n))


class NeuralOps(nn.Module):
    """Learnable connectives and quantifier aggregators."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.neg_mlp = OpMLP(1, hidden)
        self.and_mlp = OpMLP(2, hidden)
        self.or_mlp = OpMLP(2, hidden)
        self.implies_mlp = OpMLP(2, hidden)
        self.forall_mlp = OpMLP(3, hidden)
        self.exists_mlp = OpMLP(3, hidden)

    def neg(self, a: Tensor) -> Tensor:
        return self.neg_mlp(a.unsqueeze(-1))

    def and_(self, a: Tensor, b: Tensor) -> Tensor:
        return self.and_mlp(torch.stack([a, b], dim=-1))

    def or_(self, a: Tensor, b: Tensor) -> Tensor:
        return self.or_mlp(torch.stack([a, b], dim=-1))

    def implies(self, a: Tensor, b: Tensor) -> Tensor:
        return self.implies_mlp(torch.stack([a, b], dim=-1))

    def forall_agg(self, truths: Tensor, dim: int | None = None) -> Tensor:
        """Aggregate a truth set into a satisfiability in [0, 1]."""
        if dim is None:
            truths = truths.flatten()
            dim = 0
            feats = torch.stack(
                [truths.mean(), truths.amin(), _log1p_count(truths.numel(), truths.mean())]
            )
            return self.forall_mlp(feats)
        mean = truths.mean(dim=dim)
        feats = torch.stack(
            [mean, truths.amin(dim=dim), _log1p_count(truths.shape[dim], mean)], dim=-1
        )
        return self.forall_mlp(feats)

    def exists_agg(self, truths: Tensor, dim: int | None = None) -> Tensor:
        """Aggregate a truth set into an existence degree in [0, 1]."""
        if dim is None:
            truths = truths.flatten()
            dim = 0
            feats = torch.stack(
                [truths.mean(), truths.amax(), _log1p_count(truths.numel(), truths.mean())]
            )
            return self.exists_mlp(feats)
        mean = truths.mean(dim=dim)
        feats = torch.stack(
            [mean, truths.amax(dim=dim), _log1p_count(truths.shape[dim], mean)], dim=-1
        )
        return self.exists_mlp(feats)

    def init_fuzzy(self, tnorm: str = "product", p: float = 2.0, steps: int = 4000) -> None:
        """Fit every operator to its fuzzy counterpart (deterministic)."""
        device = self.neg_mlp.net[0].weight.device
        gen = torch.Generator(device="cpu").manual_seed(0)

        def fit(module: OpMLP, xs: Tensor, ys: Tensor) -> None:
            xs, ys = xs.to(device), ys.to(device).flatten()
            opt = torch.optim.Adam(module.parameters(), lr=0.02)
            for _ in range(steps):
                opt.zero_grad()
                loss = ((module(xs) - ys) ** 2).mean()
                loss.backward()
                opt.step()

        x1 = torch.rand(512, 1, generator=gen)
        fit(self.neg_mlp, x1, neg(x1))

        x2 = torch.rand(1024, 2, generator=gen)
        fit(self.and_mlp, x2, and_(x2[:, 0], x2[:, 1], tnorm))
        fit(self.or_mlp, x2, or_(x2[:, 0], x2[:, 1], tnorm))
        fit(self.implies_mlp, x2, implies(x2[:, 0], x2[:, 1], tnorm))

        # random truth sets of log-uniform size, mixed value distributions
        forall_feats, forall_tgt, exists_feats, exists_tgt = [], [], [], []
        for _ in range(1024):
            n = int(math.exp(float(torch.rand(1, generator=gen)) * math.log(20000)))
            n = max(n, 1)
            kind = int(torch.randint(4, (1,), generator=gen))
            if kind == 0:
                vals = torch.rand(n, generator=gen)
            elif kind == 1:
                vals = 1.0 - torch.rand(n, generator=gen) ** 2  # skewed high
            elif kind == 2:
                vals = torch.rand(n, generator=gen) ** 2  # skewed low
            else:
                vals = (torch.rand(n, generator=gen) > 0.5).float()  # bimodal
            log_n = math.log1p(n)
            forall_feats.append([float(vals.mean()), float(vals.min()), log_n])
            forall_tgt.append(float(forall(vals, p)))
            exists_feats.append([float(vals.mean()), float(vals.max()), log_n])
            exists_tgt.append(float(vals.max()))
        fit(self.forall_mlp, torch.tensor(forall_feats), torch.tensor(forall_tgt))
        fit(self.exists_mlp, torch.tensor(exists_feats), torch.tensor(exists_tgt))
