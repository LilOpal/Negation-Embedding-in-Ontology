"""Inspect learned neural operators against the fuzzy semantics.

Loads a checkpoint saved by run_baseline.py --save and prints the learned
operator shapes: negation vs the complement, binary connectives vs the
product t-norm, and quantifier behavior on probe sets.

Usage:
    python experiments/inspect_ops.py experiments/runs/uobm_neural_seed0.pt
"""

from __future__ import annotations

import argparse

import torch

from negemb.ltn.grounding import LTNModel, _squash
from negemb.ltn.neural_ops import NeuralOps
from negemb.ltn.quantifiers import forall


def _table(title: str, rows: list[str]) -> None:
    print(f"\n== {title} ==")
    for row in rows:
        print(f"  {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["state_dict"]
    model = LTNModel(
        num_entities=sd["entity_emb.weight"].shape[0],
        num_classes=sd["class_emb.weight"].shape[0],
        num_relations=sd["rel_emb.weight"].shape[0],
        dim=sd["entity_emb.weight"].shape[1],
        negation=cfg["negation"],
    )
    has_ops = any(k.startswith("ops.") for k in sd)
    if has_ops:
        model.ops = NeuralOps()
    model.load_state_dict(sd)
    model.eval()

    print(f"checkpoint: {args.checkpoint}")
    print(f"negation={cfg['negation']}  ops={cfg.get('ops')}  dim={cfg['dim']}")

    with torch.no_grad():
        if cfg["negation"] == "neural":
            x = torch.linspace(0, 1, 11)
            learned = _squash(model.neg_mlp(x.unsqueeze(-1)))
            rows = [f"neg({v:.1f}) = {float(lv):.3f}   (fuzzy {1 - v:.1f})"
                    for v, lv in zip(x, learned, strict=True)]
            mad = float((learned - (1 - x)).abs().mean())
            rows.append(f"mean |learned - fuzzy| = {mad:.4f}")
            _table("learned negation vs fuzzy complement", rows)

        if has_ops:
            ops = model.ops
            grid = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
            a, b = torch.meshgrid(grid, grid, indexing="ij")
            a, b = a.flatten(), b.flatten()
            devs = {
                "and": (ops.and_(a, b) - a * b).abs().mean(),
                "or": (ops.or_(a, b) - (a + b - a * b)).abs().mean(),
                "implies": (ops.implies(a, b) - (1 - a + a * b)).abs().mean(),
                "neg": (ops.neg(a) - (1 - a)).abs().mean(),
            }
            _table(
                "connectives vs product t-norm (mean abs deviation over grid)",
                [f"{name:8s} {float(d):.4f}" for name, d in devs.items()],
            )
            sets = {
                "all high (0.95)": torch.full((100,), 0.95),
                "one violator": torch.cat(
                    [torch.full((99,), 0.95), torch.tensor([0.05])]
                ),
                "mixed (0.5)": torch.full((100,), 0.5),
                "all low (0.05)": torch.full((100,), 0.05),
            }
            _table(
                "forall on probe sets (learned vs fuzzy p=2)",
                [
                    f"{name:18s} learned {float(ops.forall_agg(v)):.3f}"
                    f"   fuzzy {float(forall(v, 2.0)):.3f}"
                    for name, v in sets.items()
                ],
            )
            _table(
                "exists on probe sets (learned vs max)",
                [
                    f"{name:18s} learned {float(ops.exists_agg(v)):.3f}"
                    f"   max {float(v.max()):.3f}"
                    for name, v in sets.items()
                ],
            )


if __name__ == "__main__":
    main()
