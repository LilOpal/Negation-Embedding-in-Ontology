"""Traditional ontology-embedding metrics: corrupted-entity ranking.

For a held-out membership query `C(e)`, the true entity `e` is ranked against
all other entities by the truth value of the membership, and standard ranking
metrics (MRR, Hits@k) are computed. Uses the average rank on ties.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor

from negemb.data.schema import Index, Ontology
from negemb.ltn.grounding import LTNModel


def average_rank(scores: Tensor, target: int) -> float:
    t = scores[target]
    greater = int((scores > t).sum())
    ties = int((scores == t).sum()) - 1
    return 1.0 + greater + ties / 2.0


def ranking_metrics(ranks: list[float], ks: tuple[int, ...] = (1, 10)) -> dict[str, float]:
    if not ranks:
        return {"mrr": 0.0, "n": 0, **{f"hits@{k}": 0.0 for k in ks}}
    out = {"mrr": sum(1.0 / r for r in ranks) / len(ranks), "n": len(ranks)}
    for k in ks:
        out[f"hits@{k}"] = sum(r <= k for r in ranks) / len(ranks)
    return out


def membership_ranking(
    model: LTNModel,
    idx: Index,
    n_entities: int,
    queries: Iterable[tuple[str, str]],
    score_fn: Callable[[int, Tensor], Tensor],
    ks: tuple[int, ...] = (1, 10),
    known_true: dict[str, set[str]] | None = None,
) -> dict[str, float]:
    """Rank the true entity against all others, per query.

    If `known_true` maps class name -> known-true entities (filtered
    protocol), those entities are excluded from the ranking so that other
    correct answers do not count as errors.
    """
    device = next(model.parameters()).device
    ent_all = torch.arange(n_entities, device=device)
    ranks = []
    with torch.no_grad():
        for c_name, e_name in sorted(queries):
            scores = score_fn(idx.cls[c_name], ent_all).clone()
            if known_true is not None:
                others = known_true.get(c_name, set()) - {e_name}
                if others:
                    scores[torch.tensor([idx.entity[o] for o in others], device=device)] = (
                        float("-inf")
                    )
            ranks.append(average_rank(scores, idx.entity[e_name]))
    return ranking_metrics(ranks, ks)


def known_members(*assertion_sets: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Class name -> entities with a known assertion, for filtered ranking."""
    known: dict[str, set[str]] = {}
    for assertions in assertion_sets:
        for c, e in assertions:
            known.setdefault(c, set()).add(e)
    return known


def evaluate_standard(model: LTNModel, ont: Ontology, idx: Index) -> dict[str, float]:
    """Filtered ranking metrics on held-out positive membership queries."""
    return membership_ranking(
        model, idx, len(ont.entities), ont.test_positive, model.class_truth,
        known_true=known_members(ont.train_positive, ont.test_positive),
    )
