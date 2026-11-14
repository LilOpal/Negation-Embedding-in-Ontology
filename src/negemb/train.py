"""Training loop: maximize knowledge-base satisfiability."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from negemb.axioms import compile_kb, kb_loss
from negemb.data.schema import Ontology, build_index
from negemb.ltn.grounding import LTNModel


@dataclass
class TrainConfig:
    dim: int = 16
    lr: float = 0.01
    epochs: int = 2000
    tnorm: str = "product"
    p: float = 2.0
    negation: str = "derived"
    seed: int = 0
    device: str = "auto"
    complement_weight: float = 1.0
    prior_weight: float = 0.0  # closed-world prior on unasserted memberships
    prior_p: float = 8.0  # high p = worst-case focused (ranking punishes drifters)
    weight_decay: float = 0.0  # AdamW decay; >0 un-traps sigmoid-saturated pairs
    batch_size: int | None = None  # None = full-batch; set for mini-batch quantifiers
    ops_mode: str = "fuzzy"  # "neural" = learnable connectives + quantifiers
    log_every: int = 200


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(ont: Ontology, cfg: TrainConfig) -> tuple[LTNModel, list[float]]:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    idx = build_index(ont)
    model = LTNModel(
        num_entities=len(ont.entities),
        num_classes=len(ont.classes),
        num_relations=len(ont.relations),
        dim=cfg.dim,
        negation=cfg.negation,
    ).to(device)
    ops = None
    if cfg.ops_mode == "neural":
        from negemb.ltn.neural_ops import NeuralOps

        ops = NeuralOps().to(device)
        ops.init_fuzzy(tnorm=cfg.tnorm, p=cfg.p)
        model.ops = ops  # optimized and checkpointed together with the model
    if cfg.negation == "neural":
        model.init_neural_neg()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    compiled = compile_kb(ont, idx, torch.device(device))
    n = len(ont.entities)

    history: list[float] = []
    for epoch in range(cfg.epochs):
        log = bool(
            cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1)
        )
        if cfg.batch_size is None:
            opt.zero_grad()
            loss, details = kb_loss(
                model, ont, idx, tnorm=cfg.tnorm, p=cfg.p,
                complement_weight=cfg.complement_weight, compiled=compiled,
                with_details=log, prior_weight=cfg.prior_weight,
                prior_p=cfg.prior_p, ops=ops,
            )
            loss.backward()
            opt.step()
            epoch_loss = float(loss.detach())
        else:
            perm = torch.randperm(n, device=device)
            losses = []
            for start in range(0, n, cfg.batch_size):
                opt.zero_grad()
                loss, details = kb_loss(
                    model, ont, idx, tnorm=cfg.tnorm, p=cfg.p,
                    complement_weight=cfg.complement_weight,
                    ent_ids=perm[start:start + cfg.batch_size], compiled=compiled,
                    with_details=log, prior_weight=cfg.prior_weight,
                    prior_p=cfg.prior_p, ops=ops,
                )
                loss.backward()
                opt.step()
                losses.append(loss.detach())
            epoch_loss = float(torch.stack(losses).mean())
        history.append(epoch_loss)
        if log:
            worst = min(details.items(), key=lambda kv: kv[1])
            print(
                f"epoch {epoch:5d}/{cfg.epochs}  loss={history[-1]:.4f}  "
                f"worst={worst[0]}:{worst[1]:.3f}"
            )
    return model, history
