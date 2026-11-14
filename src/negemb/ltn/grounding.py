"""Grounding of ontology symbols into tensors.

Entities, classes, and relations are embedding vectors. The truth value of a
membership `C(x)` is a small MLP over the elementwise product of the class
and entity embeddings, squashed to [0, 1].

Negation modes:
- "derived"  (baseline A): `~C(x)` is defined as `1 - C(x)`; negation has no
  parameters of its own.
- "explicit" (baseline B): every class gets a separate negation embedding
  used to ground `~C(x)`; consistency with `C(x)` is enforced softly through
  axioms in `negemb.axioms`.
- "neural" (setting C): `~C(x)` is a learnable two-layer MLP over `C(x)`,
  initialized to the fuzzy complement (`init_neural_neg`) and then trained
  by the KB loss alone (no consistency axioms).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

NEGATION_MODES = ("derived", "explicit", "neural")

# Bound pre-activations to [-6, 6]: an unbounded MLP can push unconstrained
# pairs into full sigmoid saturation (observed: z=+28, score pinned at 1.0,
# gradient ~1e-12 — drifters the prior could never pull back down, ruining
# filtered ranking). At |z|=6 the sigmoid gradient is still ~0.005, so every
# pair keeps trainable signal while scores can reach 0.9975 / 0.0025.
_LOGIT_BOUND = 6.0


def _squash(z: Tensor) -> Tensor:
    return torch.sigmoid(_LOGIT_BOUND * torch.tanh(z / _LOGIT_BOUND))


class LTNModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_classes: int,
        num_relations: int,
        dim: int = 16,
        negation: str = "derived",
    ) -> None:
        super().__init__()
        if negation not in NEGATION_MODES:
            raise ValueError(f"unknown negation mode: {negation!r}")
        self.negation = negation
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.class_emb = nn.Embedding(num_classes, dim)
        self.rel_emb = nn.Embedding(max(num_relations, 1), dim)
        if negation == "explicit":
            self.neg_class_emb = nn.Embedding(num_classes, dim)
            # separate scorer for the negation path: sharing class_mlp lets
            # the 2K negation-consistency groups (e⊙~c) interfere with
            # membership structure (e⊙c) at scale — observed on UOBM (107
            # classes): standard_defined MRR collapsed to random while the
            # negation metrics were near-perfect. Harmless at toy scale
            # (controlled dataset, 3 classes), destructive beyond it.
            self.neg_class_mlp = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)
            )
        elif negation == "neural":
            # learnable negation: ~C(x) = MLP(C(x)), initialized to the fuzzy
            # complement (init_neural_neg); no separate embeddings and no
            # hard-wired consistency axioms, so the data alone shapes it
            self.neg_mlp = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)
            )
        self.class_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.rel_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def class_truth(self, class_idx: int, ent_ids: Tensor) -> Tensor:
        """Truth of C(x) for one class over a batch of entities."""
        e = self.entity_emb(ent_ids)
        c = self.class_emb.weight[class_idx].expand_as(e)
        return _squash(self.class_mlp(e * c)).squeeze(-1)

    def neg_class_truth(self, class_idx: int, ent_ids: Tensor) -> Tensor:
        """Truth of ~C(x) for one class over a batch of entities."""
        if self.negation == "explicit":
            e = self.entity_emb(ent_ids)
            c = self.neg_class_emb.weight[class_idx].expand_as(e)
            return _squash(self.neg_class_mlp(e * c)).squeeze(-1)
        if self.negation == "neural":
            base = self.class_truth(class_idx, ent_ids)
            return _squash(self.neg_mlp(base.unsqueeze(-1))).squeeze(-1)
        return 1.0 - self.class_truth(class_idx, ent_ids)

    def rel_truth(self, rel_idx: int, subj_ids: Tensor, obj_ids: Tensor) -> Tensor:
        """Truth of R(s, o) for one relation over a batch of pairs."""
        s = self.entity_emb(subj_ids)
        o = self.entity_emb(obj_ids)
        r = self.rel_emb.weight[rel_idx].expand_as(s)
        return _squash(self.rel_mlp(s * o * r)).squeeze(-1)

    def class_truth_matrix(self, ent_ids: Tensor) -> Tensor:
        """Truth of C(x) for every class (rows) over a batch of entities (cols)."""
        e = self.entity_emb(ent_ids)  # (B, dim)
        prod = self.class_emb.weight.unsqueeze(1) * e.unsqueeze(0)  # (K, B, dim)
        return _squash(self.class_mlp(prod)).squeeze(-1)

    def neg_class_truth_matrix(self, ent_ids: Tensor) -> Tensor:
        """Truth of ~C(x) for every class (rows) over a batch of entities (cols)."""
        if self.negation == "explicit":
            e = self.entity_emb(ent_ids)
            prod = self.neg_class_emb.weight.unsqueeze(1) * e.unsqueeze(0)
            return _squash(self.neg_class_mlp(prod)).squeeze(-1)
        if self.negation == "neural":
            base = self.class_truth_matrix(ent_ids)
            return _squash(self.neg_mlp(base.unsqueeze(-1))).squeeze(-1)
        return 1.0 - self.class_truth_matrix(ent_ids)

    def class_truth_pairs(self, class_ids: Tensor, ent_ids: Tensor) -> Tensor:
        """Truth of C_i(x_i) for paired class/entity index tensors."""
        prod = self.class_emb(class_ids) * self.entity_emb(ent_ids)
        return _squash(self.class_mlp(prod)).squeeze(-1)

    def neg_class_truth_pairs(self, class_ids: Tensor, ent_ids: Tensor) -> Tensor:
        """Truth of ~C_i(x_i) for paired class/entity index tensors."""
        if self.negation == "explicit":
            prod = self.neg_class_emb(class_ids) * self.entity_emb(ent_ids)
            return _squash(self.neg_class_mlp(prod)).squeeze(-1)
        if self.negation == "neural":
            base = self.class_truth_pairs(class_ids, ent_ids)
            return _squash(self.neg_mlp(base.unsqueeze(-1))).squeeze(-1)
        return 1.0 - self.class_truth_pairs(class_ids, ent_ids)

    def rel_truth_pairs(
        self, rel_ids: Tensor, subj_ids: Tensor, obj_ids: Tensor
    ) -> Tensor:
        """Truth of R_i(s_i, o_i) for paired relation/subject/object tensors."""
        prod = self.rel_emb(rel_ids) * self.entity_emb(subj_ids) * self.entity_emb(obj_ids)
        return _squash(self.rel_mlp(prod)).squeeze(-1)

    def init_neural_neg(self, steps: int = 1000) -> None:
        """Fit the learnable negation to the fuzzy complement 1 - x."""
        if self.negation != "neural":
            return
        device = self.neg_mlp[0].weight.device
        x = torch.linspace(0.0, 1.0, 256, device=device).unsqueeze(-1)
        y = 1.0 - x.squeeze(-1)
        opt = torch.optim.Adam(self.neg_mlp.parameters(), lr=0.02)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((_squash(self.neg_mlp(x)).squeeze(-1) - y) ** 2).mean()
            loss.backward()
            opt.step()
