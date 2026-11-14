"""Controlled-dataset sanity check: isolate why rule-defined class members
stay low on UOBM despite assertions + closed-world prior.

World (fully known, no noise):
- classes Person, Sport, Lover; relation `like`
- Lover ≡ Person ⊓ ∃like.Sport   (single EquivalentTo rule)
- `n_likers` persons each like one sport; other persons like nothing;
  a few non-persons also like a sport (the Person conjunct must exclude
  them from Lover)

The pipeline (schema, EquivalentTo grounding, closed-world prior,
materialization, filtered ranking) is the same as for UOBM, but a run
takes seconds, so failure modes can be attributed precisely.

Usage:
    python experiments/controlled_sanity.py [--prior-weight 0.1] [--epochs 800]
"""

from __future__ import annotations

import argparse
import random

import torch

from negemb.axioms import kb_loss
from negemb.data.schema import EquivalentTo, Ontology, SomeValuesFrom, build_index
from negemb.metrics.standard import known_members, membership_ranking
from negemb.train import TrainConfig, train


def generate_controlled(
    seed: int = 0,
    n_entities: int = 500,
    n_persons: int = 300,
    n_sports: int = 5,
    n_likers: int = 100,
    n_nonperson_likers: int = 10,
    test_frac: float = 0.2,
) -> Ontology:
    rng = random.Random(seed)
    entities = [f"e{i}" for i in range(n_entities)]
    sports = entities[:n_sports]
    persons = entities[n_sports:n_sports + n_persons]
    likers = persons[:n_likers]
    nonperson_likers = entities[n_sports + n_persons:n_sports + n_persons + n_nonperson_likers]

    relational = {
        ("like", e, rng.choice(sports)) for e in likers + nonperson_likers
    }

    def split(items: set[tuple]) -> tuple[set[tuple], set[tuple]]:
        ordered = sorted(items)
        rng.shuffle(ordered)
        k = round(len(ordered) * test_frac) if len(ordered) > 1 else 0
        return set(ordered[k:]), set(ordered[:k])

    direct = {("Person", e) for e in persons} | {("Sport", e) for e in sports}
    materialized = {("Lover", e) for e in likers}
    train_pos, test_pos = split(direct)
    m_train, m_test = split(materialized)
    train_pos |= m_train
    test_pos |= m_test

    members = {"Person": set(persons), "Sport": set(sports), "Lover": set(likers)}
    train_neg: set[tuple[str, str]] = set()
    test_neg: set[tuple[str, str]] = set()
    for c, mem in members.items():
        pool = sorted(set(entities) - mem)
        negs = rng.sample(pool, min(len(mem), len(pool)))
        n_train, n_test = split({(c, e) for e in negs})
        train_neg |= n_train
        test_neg |= n_test

    rel_neg: set[tuple[str, str, str]] = set()
    while len(rel_neg) < len(relational):
        s, o = rng.choice(entities), rng.choice(entities)
        if ("like", s, o) not in relational:
            rel_neg.add(("like", s, o))

    return Ontology(
        entities=entities,
        classes=["Person", "Sport", "Lover"],
        relations=["like"],
        axioms=[EquivalentTo("Lover", ("Person", SomeValuesFrom("like", "Sport")))],
        train_positive=train_pos,
        train_negative=train_neg,
        test_positive=test_pos,
        test_negative=test_neg,
        relational=relational,
        relational_negative=rel_neg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--prior-p", type=float, default=8.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--negation", choices=["derived", "explicit", "neural"],
                        default="derived")
    parser.add_argument("--ops", choices=["fuzzy", "neural"], default="fuzzy")
    args = parser.parse_args()

    ont = generate_controlled(seed=args.seed)
    cfg = TrainConfig(
        dim=args.dim, lr=args.lr, epochs=args.epochs, negation=args.negation,
        seed=args.seed, prior_weight=args.prior_weight, prior_p=args.prior_p,
        weight_decay=args.weight_decay, log_every=100, ops_mode=args.ops,
    )
    model, history = train(ont, cfg)
    idx = build_index(ont)
    device = next(model.parameters()).device

    def mean_truth(cls: str, names: list[str]) -> float:
        ids = torch.tensor([idx.entity[e] for e in names], device=device)
        return float(model.class_truth(idx.cls[cls], ids).mean())

    persons = [e for c, e in ont.train_positive | ont.test_positive if c == "Person"]
    likers = [e for c, e in ont.train_positive if c == "Lover"]
    held_out_likers = [e for c, e in ont.test_positive if c == "Lover"]
    non_likers = sorted(
        {e for e in persons} - {e for c, e in ont.train_positive | ont.test_positive
                                if c == "Lover"}
    )

    with torch.no_grad():
        _, details = kb_loss(model, ont, idx)
        print("\n== probes ==")
        print(f"  Lover(train likers)    {mean_truth('Lover', likers):.3f}  (n={len(likers)})")
        print(f"  Lover(heldout likers)  {mean_truth('Lover', held_out_likers):.3f}"
              f"  (n={len(held_out_likers)})")
        print(f"  Lover(non-likers)      {mean_truth('Lover', non_likers):.3f}"
              f"  (n={len(non_likers)})")
        print(f"  equivdef sat           {details['equivdef:Lover']:.3f}")
        print(f"  assert:positive sat    {details['assert:positive']:.3f}")
        print(f"  assert:negative sat    {details['assert:negative']:.3f}")

        n = len(ont.entities)
        ranking = membership_ranking(
            model, idx, n, {(c, e) for c, e in ont.test_positive if c == "Lover"},
            model.class_truth,
            known_true=known_members(ont.train_positive, ont.test_positive),
        )
        print("\n== held-out Lover ranking (filtered) ==")
        print(f"  MRR {ranking['mrr']:.4f}  H@1 {ranking['hits@1']:.4f}"
              f"  H@10 {ranking['hits@10']:.4f}  (n={ranking['n']})")
    print(f"\nfinal_loss {history[-1]:.4f}")


if __name__ == "__main__":
    main()
