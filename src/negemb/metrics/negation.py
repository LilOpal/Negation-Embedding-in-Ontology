"""Negation-oriented evaluation metrics.

- complement_consistency: mean |C(x) + ~C(x) - 1| over classes/entities.
  Trivially 0 in "derived" mode (by construction); diagnostic in "explicit".
- non_contradiction_rate: fraction of (class, entity) with both C(x) and
  ~C(x) above threshold. 0 in "derived" mode by construction.
- disjointness_violation: fraction of entities violating declared
  Disjoint/ComplementOf pairs (both classes above threshold).
- de_morgan_gap: mean |truth(~(A and B)) - truth(~A or ~B)| over class pairs.
  0 in "derived" mode by construction; probes the consistency of explicit
  negation embeddings with the connectives.
- negative_query_ranking: MRR/Hits@k on held-out *negative* assertions,
  scored with the negation grounding. NOTE: this metric floods under a
  closed-world prior — thousands of unasserted entities sit at ~C ≈ 1,
  so a true negative cannot stand out. Prefer negation_discrimination.
- negation_discrimination: Mann-Whitney AUC of ~C on *known* negatives
  vs *known* positives per class: does the negation grounding rank
  non-members above members? Immune to flooding because it never
  considers unasserted entities.
"""

from __future__ import annotations

import itertools

import torch

from negemb.data.schema import ComplementOf, Disjoint, Index, Ontology
from negemb.ltn.grounding import LTNModel
from negemb.ltn.operators import and_, neg, or_
from negemb.metrics.standard import known_members, membership_ranking


def _entity_ids(model: LTNModel, n_entities: int) -> torch.Tensor:
    device = next(model.parameters()).device
    return torch.arange(n_entities, device=device)


def complement_consistency(model: LTNModel, idx: Index, n_entities: int) -> float:
    ent_all = _entity_ids(model, n_entities)
    with torch.no_grad():
        gaps = [
            (model.class_truth(cid, ent_all) + model.neg_class_truth(cid, ent_all) - 1.0)
            .abs()
            .mean()
            for cid in idx.cls.values()
        ]
    return float(torch.stack(gaps).mean())


def non_contradiction_rate(
    model: LTNModel, idx: Index, n_entities: int, threshold: float = 0.5
) -> float:
    ent_all = _entity_ids(model, n_entities)
    both = total = 0
    with torch.no_grad():
        for cid in idx.cls.values():
            c = model.class_truth(cid, ent_all)
            nc = model.neg_class_truth(cid, ent_all)
            both += int(((c > threshold) & (nc > threshold)).sum())
            total += c.numel()
    return both / total


def disjointness_violation(
    model: LTNModel, ont: Ontology, idx: Index, n_entities: int, threshold: float = 0.5
) -> float:
    pairs = [(ax.a, ax.b) for ax in ont.axioms if isinstance(ax, Disjoint)]
    pairs += [(ax.cls, ax.complement) for ax in ont.axioms if isinstance(ax, ComplementOf)]
    if not pairs:
        return 0.0
    ent_all = _entity_ids(model, n_entities)
    violations = 0
    with torch.no_grad():
        for a, b in pairs:
            ta = model.class_truth(idx.cls[a], ent_all)
            tb = model.class_truth(idx.cls[b], ent_all)
            violations += int(((ta > threshold) & (tb > threshold)).sum())
    return violations / (len(pairs) * n_entities)


def de_morgan_gap(
    model: LTNModel, idx: Index, n_entities: int, tnorm: str = "product", max_pairs: int = 20
) -> float:
    pairs = list(itertools.combinations(sorted(idx.cls.values()), 2))[:max_pairs]
    ent_all = _entity_ids(model, n_entities)
    with torch.no_grad():
        gaps = []
        for a, b in pairs:
            lhs = neg(and_(model.class_truth(a, ent_all), model.class_truth(b, ent_all), tnorm))
            rhs = or_(model.neg_class_truth(a, ent_all), model.neg_class_truth(b, ent_all), tnorm)
            gaps.append((lhs - rhs).abs().mean())
    return float(torch.stack(gaps).mean())


def negative_query_ranking(model: LTNModel, ont: Ontology, idx: Index) -> dict[str, float]:
    return membership_ranking(
        model, idx, len(ont.entities), ont.test_negative, model.neg_class_truth,
        known_true=known_members(ont.train_negative, ont.test_negative),
    )


def _auc(neg_scores: torch.Tensor, pos_scores: torch.Tensor) -> float:
    """Mann-Whitney AUC P(neg > pos) with exact average-tie correction."""
    device = neg_scores.device
    scores = torch.cat([pos_scores, neg_scores])
    order = scores.argsort()
    ranks = torch.empty(len(scores), dtype=torch.float, device=device)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float, device=device)
    _, inv, counts = scores.unique(return_inverse=True, return_counts=True)
    sums = torch.zeros(len(counts), dtype=torch.float, device=device).index_add_(
        0, inv, ranks
    )
    avg_ranks = (sums / counts)[inv]
    n_neg = len(neg_scores)
    rank_sum = float(avg_ranks[len(pos_scores):].sum())
    return (rank_sum - n_neg * (n_neg + 1) / 2) / (len(pos_scores) * n_neg)


def negation_discrimination(model: LTNModel, ont: Ontology, idx: Index) -> dict[str, float]:
    """AUC of ~C on known negatives vs known positives, mean over classes."""
    device = next(model.parameters()).device
    pos = known_members(ont.train_positive, ont.test_positive)
    neg_ = known_members(ont.train_negative, ont.test_negative)
    aucs = []
    with torch.no_grad():
        for c_name in sorted(set(pos) & set(neg_)):
            cid = idx.cls[c_name]
            p_ids = torch.tensor([idx.entity[e] for e in pos[c_name]], device=device)
            n_ids = torch.tensor([idx.entity[e] for e in neg_[c_name]], device=device)
            aucs.append(
                _auc(model.neg_class_truth(cid, n_ids), model.neg_class_truth(cid, p_ids))
            )
    return {
        "negation_auc": sum(aucs) / len(aucs) if aucs else 0.0,
        "n_classes": len(aucs),
    }


def evaluate_negation(
    model: LTNModel, ont: Ontology, idx: Index, tnorm: str = "product"
) -> dict[str, object]:
    n = len(ont.entities)
    return {
        "complement_consistency": complement_consistency(model, idx, n),
        "non_contradiction_rate": non_contradiction_rate(model, idx, n),
        "disjointness_violation": disjointness_violation(model, ont, idx, n),
        "de_morgan_gap": de_morgan_gap(model, idx, n, tnorm),
        "negative_query_ranking": negative_query_ranking(model, ont, idx),
        "negation_discrimination": negation_discrimination(model, ont, idx),
    }
