"""Train a baseline on the synthetic toy ontology and print evaluation.

Usage:
    python experiments/run_baseline.py --negation derived
    python experiments/run_baseline.py --negation explicit --tnorm lukasiewicz
"""

from __future__ import annotations

import argparse
import json

from negemb.data.schema import EquivalentTo, build_index
from negemb.data.synthetic import generate
from negemb.ltn.operators import TNORMS
from negemb.metrics.negation import evaluate_negation
from negemb.metrics.standard import evaluate_standard, known_members, membership_ranking
from negemb.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "uobm"], default="synthetic")
    parser.add_argument("--max-entities", type=int, default=None,
                        help="subsample entities (uobm only)")
    parser.add_argument("--negation", choices=["derived", "explicit", "neural"],
                        default="derived")
    parser.add_argument("--ops", choices=["fuzzy", "neural"], default="fuzzy",
                        help="neural = learnable connectives + quantifiers")
    parser.add_argument("--tnorm", choices=list(TNORMS), default="product")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--complement-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="mini-batch size for quantifier domains (default: full batch)")
    parser.add_argument("--prior-weight", type=float, default=0.0,
                        help="closed-world prior weight on unasserted memberships")
    parser.add_argument("--save", default=None,
                        help="optional path to save the trained model state_dict")
    args = parser.parse_args()

    if args.dataset == "uobm":
        from negemb.data import uobm

        ont = uobm.load(seed=args.seed, max_entities=args.max_entities)
    else:
        ont = generate(seed=args.seed)
    cfg = TrainConfig(
        dim=args.dim,
        lr=args.lr,
        epochs=args.epochs,
        tnorm=args.tnorm,
        negation=args.negation,
        seed=args.seed,
        device=args.device,
        complement_weight=args.complement_weight,
        prior_weight=args.prior_weight,
        batch_size=args.batch_size,
        ops_mode=args.ops,
    )
    model, history = train(ont, cfg)
    if args.save:
        import torch

        torch.save({"state_dict": model.state_dict(), "config": vars(args)}, args.save)

    idx = build_index(ont)
    results = {
        "config": vars(args),
        "final_loss": history[-1],
        "standard": evaluate_standard(model, ont, idx),
        "negation": evaluate_negation(model, ont, idx, tnorm=args.tnorm),
    }
    # breakdown for classes defined via relations (EquivalentTo axioms):
    # the part of the benchmark where generalization is possible
    defined = {ax.cls for ax in ont.axioms if isinstance(ax, EquivalentTo)}
    if defined:
        n = len(ont.entities)
        results["standard_defined"] = membership_ranking(
            model, idx, n, {(c, e) for c, e in ont.test_positive if c in defined},
            model.class_truth,
            known_true=known_members(ont.train_positive, ont.test_positive),
        )
        results["negative_query_ranking_defined"] = membership_ranking(
            model, idx, n, {(c, e) for c, e in ont.test_negative if c in defined},
            model.neg_class_truth,
            known_true=known_members(ont.train_negative, ont.test_negative),
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
