"""UOBM sanity diagnostic: train with mini-batched quantifiers and probe
whether rule-defined classes separate entities that satisfy the rule body
from those that do not.

Usage:
    python experiments/diagnose_uobm.py [--epochs 200] [--batch-size 1024]
"""

from __future__ import annotations

import argparse

import torch

from negemb.axioms import kb_loss
from negemb.data import uobm
from negemb.data.schema import EquivalentTo, build_index
from negemb.metrics.standard import evaluate_standard, known_members, membership_ranking
from negemb.train import TrainConfig, train


def mean_truth(model, idx, cls: str, entities: list[str]) -> float:
    if not entities:
        return float("nan")
    device = next(model.parameters()).device
    ids = torch.tensor([idx.entity[e] for e in entities], device=device)
    return float(model.class_truth(idx.cls[cls], ids).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="quantifier mini-batch size; omit for full batch")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negation", choices=["derived", "explicit", "neural"],
                        default="derived")
    parser.add_argument("--ops", choices=["fuzzy", "neural"], default="fuzzy")
    parser.add_argument("--prior-weight", type=float, default=0.0)
    args = parser.parse_args()

    ont = uobm.load(seed=args.seed)
    cfg = TrainConfig(
        dim=args.dim, lr=args.lr, epochs=args.epochs, negation=args.negation,
        seed=args.seed, batch_size=args.batch_size, log_every=20,
        prior_weight=args.prior_weight, ops_mode=args.ops,
    )
    model, history = train(ont, cfg)
    idx = build_index(ont)

    _, details = kb_loss(model, ont, idx, ops=getattr(model, "ops", None))
    by_type: dict[str, list[float]] = {}
    for name, sat in details.items():
        by_type.setdefault(name.split(":", 1)[0], []).append(sat)
    print("\n== per-type satisfiability (full-batch) ==")
    for t, vals in sorted(by_type.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {t:28s} {sum(vals) / len(vals):.3f}  (n={len(vals)})")

    # probes -----------------------------------------------------------------
    baseball_cls_members = {e for c, e in ont.train_positive if c == "BaseballClass"}
    likers = sorted(
        {s for r, s, o in ont.relational if r == "like" and o in baseball_cls_members}
    )
    persons = sorted({e for c, e in ont.train_positive if c == "Person"})
    non_likers = sorted(set(persons) - set(likers))
    full_profs = sorted({e for c, e in ont.train_positive if c == "FullProfessor"})
    anchors = [e for e in ("Baseball",) if e in idx.entity]

    print("\n== probes ==")
    print(f"  BaseballLover(likers)      {mean_truth(model, idx, 'BaseballLover', likers):.3f}"
          f"  (n={len(likers)})")
    print(f"  BaseballLover(non-likers)  {mean_truth(model, idx, 'BaseballLover', non_likers):.3f}"
          f"  (n={len(non_likers)})")
    print(f"  Professor(FullProfessor)   {mean_truth(model, idx, 'Professor', full_profs):.3f}"
          f"  (n={len(full_profs)})")
    print(f"  BaseballClass(Baseball)    {mean_truth(model, idx, 'BaseballClass', anchors):.3f}")

    # metrics ----------------------------------------------------------------
    n = len(ont.entities)
    defined = {ax.cls for ax in ont.axioms if isinstance(ax, EquivalentTo)}
    standard = evaluate_standard(model, ont, idx)
    standard_def = membership_ranking(
        model, idx, n, {(c, e) for c, e in ont.test_positive if c in defined},
        model.class_truth,
        known_true=known_members(ont.train_positive, ont.test_positive),
    )
    neg_def = membership_ranking(
        model, idx, n, {(c, e) for c, e in ont.test_negative if c in defined},
        model.neg_class_truth,
        known_true=known_members(ont.train_negative, ont.test_negative),
    )
    print("\n== metrics (filtered) ==")
    print(f"  final_loss                       {history[-1]:.4f}")
    print(f"  standard            MRR {standard['mrr']:.4f}  H@1 {standard['hits@1']:.4f}"
          f"  H@10 {standard['hits@10']:.4f}  (n={standard['n']})")
    print(f"  standard_defined    MRR {standard_def['mrr']:.4f}  H@1 {standard_def['hits@1']:.4f}"
          f"  H@10 {standard_def['hits@10']:.4f}  (n={standard_def['n']})")
    print(f"  neg_query_defined   MRR {neg_def['mrr']:.4f}  H@1 {neg_def['hits@1']:.4f}"
          f"  H@10 {neg_def['hits@10']:.4f}  (n={neg_def['n']})")


if __name__ == "__main__":
    main()
