"""Compile ontology axioms and assertions into a differentiable KB loss.

Each axiom/assertion group is aggregated into a satisfiability value in
[0, 1]; groups are then combined per axiom *type* (subclass, disjoint,
assertions, ...) and the loss is the weighted mean of `1 - sat` over types,
so frequent axiom types cannot crowd out rare ones.

- `SubClassOf(C, D)`   -> forall x: C(x) -> D(x)
- `Disjoint(C, D)`     -> forall x: ~(C(x) and D(x))
- `ComplementOf(C, D)` -> forall x: ~(C(x) and D(x))  (non-contradiction)
                          forall x: C(x) or D(x)      (exhaustiveness)
- `ExistentialDef(C, R)` -> forall x: C(x) <-> exists y: R(x, y)
- `EquivalentTo(C, parts)` -> forall x: (parts at x) -> C(x), where a part
  is a named class or `SomeValuesFrom(R, F)` (grounded over asserted
  R-edges; see below).

  NOTE: compiled as a *sufficient condition* (body -> head) only, matching
  rule-injection practice in KGE. The reverse direction (C(x) -> parts) is
  unsound under open-world semantics: the ABox does not assert every
  implied edge (e.g. UOBM has 264 isStudentOf edges for 12.5k students), so
  requiring `C -> exists R` over asserted edges contradicts the positive
  assertions and collapses training.

In "explicit" negation mode, every class additionally gets soft consistency
axioms tying its negation embedding to the class embedding:
non-contradiction `~(C and ~C)` and excluded middle `C or ~C`, weighted by
`complement_weight`.

Batching: `kb_loss(..., ent_ids=batch)` restricts every `forall` and the
unary assertions to the batch entities; relational assertions keep triples
whose *subject* is in the batch (object embeddings still receive gradients),
and existential restrictions are evaluated for batch subjects only.
Filtering by subject rather than both endpoints avoids the (batch/n)^2
edge-survival collapse.

Performance: everything is vectorized over axiom groups and assertion
groups (one class-truth matrix per call, pairwise truth ops, scatter-based
segment aggregation). A naive per-axiom/per-class Python loop launches
~2000 tiny kernels per step (~600 ms); the vectorized form is ~100 ops.
`compile_kb` precomputes the static index tensors once, and
`with_details=False` skips the per-group GPU->CPU syncs on non-logged steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from negemb.data.schema import (
    ComplementOf,
    Disjoint,
    EquivalentTo,
    ExistentialDef,
    Index,
    Ontology,
    SomeValuesFrom,
    SubClassOf,
)
from negemb.ltn.grounding import LTNModel
from negemb.ltn.neural_ops import NeuralOps
from negemb.ltn.operators import and_, implies, neg, or_
from negemb.ltn.quantifiers import exists, forall

_EPS = 1e-9


@dataclass
class EquivEntry:
    cls: int
    named: Tensor  # named-class conjunct ids, (m,)
    svf_occs: list[int]  # indices into the SVF occurrence axis
    name: str


@dataclass
class FlatGroup:
    """Concatenated assertion groups with a group id per item."""

    gid: Tensor
    n_groups: int
    a: Tensor  # class id / relation id
    b: Tensor  # entity id / subject id
    c: Tensor | None = None  # object id (relational only)


@dataclass
class CompiledKB:
    """Static index tensors for axioms and assertions, built once per run."""

    sub_pairs: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    sub_names: list[str] = field(default_factory=list)
    dis_pairs: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    dis_names: list[str] = field(default_factory=list)
    compl_pairs: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    compl_names: list[str] = field(default_factory=list)
    existsdef: list[ExistentialDef] = field(default_factory=list)
    equiv: list[EquivEntry] = field(default_factory=list)
    svf_src: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    svf_dst: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    svf_seg: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    svf_rid: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    svf_fid: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    n_svf: int = 0
    unary_pos: FlatGroup | None = None
    unary_neg: FlatGroup | None = None
    rel_pos: FlatGroup | None = None
    rel_neg: FlatGroup | None = None
    cls_names: list[str] = field(default_factory=list)  # class name per class id
    pos_mask: Tensor | None = None  # (K, n) bool: known members (train + test)


def _flat_unary(
    idx: Index, assertions: set[tuple[str, str]], device: torch.device
) -> FlatGroup | None:
    cids: list[int] = []
    ents: list[int] = []
    gids: list[int] = []
    gid_of: dict[int, int] = {}
    for c_name, e_name in assertions:
        cid = idx.cls[c_name]
        gid = gid_of.setdefault(cid, len(gid_of))
        cids.append(cid)
        ents.append(idx.entity[e_name])
        gids.append(gid)
    if not cids:
        return None
    return FlatGroup(
        gid=torch.tensor(gids, device=device),
        n_groups=len(gid_of),
        a=torch.tensor(cids, device=device),
        b=torch.tensor(ents, device=device),
    )


def _flat_relational(
    idx: Index, triples: set[tuple[str, str, str]], device: torch.device
) -> FlatGroup | None:
    rids: list[int] = []
    srcs: list[int] = []
    dsts: list[int] = []
    gids: list[int] = []
    gid_of: dict[int, int] = {}
    for r_name, s_name, o_name in triples:
        rid = idx.rel[r_name]
        gid = gid_of.setdefault(rid, len(gid_of))
        rids.append(rid)
        srcs.append(idx.entity[s_name])
        dsts.append(idx.entity[o_name])
        gids.append(gid)
    if not rids:
        return None
    return FlatGroup(
        gid=torch.tensor(gids, device=device),
        n_groups=len(gid_of),
        a=torch.tensor(rids, device=device),
        b=torch.tensor(srcs, device=device),
        c=torch.tensor(dsts, device=device),
    )


def compile_kb(ont: Ontology, idx: Index, device: torch.device) -> CompiledKB:
    kb = CompiledKB(cls_names=[""] * len(idx.cls))
    for name, cid in idx.cls.items():
        kb.cls_names[cid] = name

    def t(pairs: list[tuple[int, int]]) -> Tensor:
        return torch.tensor(pairs, device=device).t() if pairs else (
            torch.empty((2, 0), dtype=torch.long, device=device)
        )

    sub, dis, compl = [], [], []
    svf_src, svf_dst, svf_seg, svf_rid, svf_fid = [], [], [], [], []
    for ax in ont.axioms:
        if isinstance(ax, SubClassOf):
            sub.append((idx.cls[ax.sub], idx.cls[ax.sup]))
            kb.sub_names.append(f"subclass:{ax.sub}->{ax.sup}")
        elif isinstance(ax, Disjoint):
            dis.append((idx.cls[ax.a], idx.cls[ax.b]))
            kb.dis_names.append(f"disjoint:{ax.a}|{ax.b}")
        elif isinstance(ax, ComplementOf):
            compl.append((idx.cls[ax.cls], idx.cls[ax.complement]))
            kb.compl_names.append(f"{ax.cls}|{ax.complement}")
        elif isinstance(ax, ExistentialDef):
            kb.existsdef.append(ax)
        elif isinstance(ax, EquivalentTo):
            named: list[int] = []
            occs: list[int] = []
            for part in ax.conjuncts:
                if isinstance(part, SomeValuesFrom):
                    occ = len(svf_rid)
                    edges = [
                        (idx.entity[s], idx.entity[o])
                        for r, s, o in ont.relational
                        if r == part.relation
                    ]
                    svf_src.extend(a for a, _ in edges)
                    svf_dst.extend(b for _, b in edges)
                    svf_seg.extend([occ] * len(edges))
                    svf_rid.append(idx.rel[part.relation])
                    svf_fid.append(idx.cls[part.filler])
                    occs.append(occ)
                else:
                    named.append(idx.cls[part])
            kb.equiv.append(
                EquivEntry(
                    cls=idx.cls[ax.cls],
                    named=torch.tensor(named, device=device),
                    svf_occs=occs,
                    name=f"equivdef:{ax.cls}",
                )
            )
    kb.sub_pairs = t(sub)
    kb.dis_pairs = t(dis)
    kb.compl_pairs = t(compl)
    kb.n_svf = len(svf_rid)
    kb.svf_src = torch.tensor(svf_src, device=device)
    kb.svf_dst = torch.tensor(svf_dst, device=device)
    kb.svf_seg = torch.tensor(svf_seg, device=device)
    kb.svf_rid = torch.tensor(svf_rid, device=device)
    kb.svf_fid = torch.tensor(svf_fid, device=device)
    kb.unary_pos = _flat_unary(idx, ont.train_positive, device)
    kb.unary_neg = _flat_unary(idx, ont.train_negative, device)
    kb.rel_pos = _flat_relational(idx, ont.relational, device)
    kb.rel_neg = _flat_relational(idx, ont.relational_negative, device)
    # known members (train + held-out test) are excluded from the
    # closed-world prior: the prior must suppress drift of *unasserted*
    # pairs, not fight the assertions (observed: an unmasked prior pulled
    # true members down to ~0.1 alongside the drifters)
    mask = torch.zeros(len(idx.cls), len(idx.entity), dtype=torch.bool, device=device)
    for c_name, e_name in ont.train_positive | ont.test_positive:
        mask[idx.cls[c_name], idx.entity[e_name]] = True
    kb.pos_mask = mask
    return kb


def _seg_forall(
    truths: Tensor, gid: Tensor, n_groups: int, p: float, ops: NeuralOps | None = None
) -> tuple[Tensor, Tensor]:
    """Per-group aggregation; returns (per_group, present_mask).

    Fuzzy mode: p-mean error. Neural mode: per-group [mean, min, log1p(n)]
    features through the learnable forall MLP.
    """
    cnts = torch.zeros(n_groups, device=truths.device).scatter_add_(
        0, gid, torch.ones_like(truths)
    )
    if ops is None:
        errs = torch.pow(1.0 - truths, p)
        sums = torch.zeros(n_groups, device=truths.device).scatter_add_(0, gid, errs)
        mean = sums / cnts.clamp(min=1.0)
        per_group = 1.0 - torch.pow(mean.clamp(min=_EPS), 1.0 / p)
    else:
        sums = torch.zeros(n_groups, device=truths.device).scatter_add_(0, gid, truths)
        mins = torch.ones(n_groups, device=truths.device).scatter_reduce_(
            0, gid, truths, reduce="amin", include_self=True
        )
        feats = torch.stack(
            [sums / cnts.clamp(min=1.0), mins, torch.log1p(cnts)], dim=-1
        )
        per_group = ops.forall_mlp(feats)
    return per_group, cnts > 0


def kb_loss(
    model: LTNModel,
    ont: Ontology,
    idx: Index,
    tnorm: str = "product",
    p: float = 2.0,
    complement_weight: float = 1.0,
    ent_ids: Tensor | None = None,
    compiled: CompiledKB | None = None,
    with_details: bool = True,
    prior_weight: float = 0.0,
    prior_p: float = 8.0,
    ops: NeuralOps | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Return (loss, per-group satisfiability for logging).

    `ent_ids=None` evaluates over the whole domain (full batch). `compiled`
    caches the static index tensors; pass the result of `compile_kb` when
    calling in a training loop. `with_details=False` skips the per-group
    GPU->CPU syncs (use it for non-logged training steps). `ops` swaps the
    fuzzy connectives and instance-level quantifiers for learnable MLPs
    (neural t-norm setting); the engineered aggregation mechanisms —
    per-type weighting, firing-normalized rules, the closed-world prior —
    stay fixed either way, so the comparison isolates the operators.
    """
    device = next(model.parameters()).device
    n = len(ont.entities)
    if compiled is None:
        compiled = compile_kb(ont, idx, device)
    if ent_ids is None:
        ent_all = torch.arange(n, device=device)
        pos = None
    else:
        ent_all = ent_ids.to(device)
        pos = torch.full((n,), -1, dtype=torch.long, device=device)
        pos[ent_all] = torch.arange(len(ent_all), device=device)
    batch = len(ent_all)

    if ops is None:
        def conn_and(a: Tensor, b: Tensor) -> Tensor:
            return and_(a, b, tnorm)

        def conn_or(a: Tensor, b: Tensor) -> Tensor:
            return or_(a, b, tnorm)

        def conn_imp(a: Tensor, b: Tensor) -> Tensor:
            return implies(a, b, tnorm)

        conn_neg = neg

        def agg_forall(t: Tensor, dim: int | None = None) -> Tensor:
            return forall(t, p, dim)

        def agg_exists(t: Tensor, dim: int | None = None) -> Tensor:
            return exists(t, float("inf"), dim)
    else:
        conn_and, conn_or = ops.and_, ops.or_
        conn_imp, conn_neg = ops.implies, ops.neg
        agg_forall, agg_exists = ops.forall_agg, ops.exists_agg

    c_mat = model.class_truth_matrix(ent_all)  # (K, B)
    nc_mat = model.neg_class_truth_matrix(ent_all)

    type_sats: list[tuple[Tensor, float]] = []
    detail_vecs: list[tuple[Tensor, list[str]]] = []
    detail_scalars: list[tuple[Tensor, str]] = []

    def add_axiom_type(vec: Tensor, names: list[str], weight: float = 1.0) -> None:
        type_sats.append((forall(vec, p), weight))
        if with_details:
            detail_vecs.append((vec, names))

    if compiled.sub_pairs.numel():
        # firing-normalized like EquivalentTo below: the rule C(x) -> D(x)
        # only constrains entities where C(x) fires; averaging the violation
        # over the whole domain dilutes rare subclasses (130 FullProfessors
        # among 25k entities) to nothing against the closed-world prior
        s, d = compiled.sub_pairs
        body, head = c_mat[s], c_mat[d]  # (n_ax, B)
        err = torch.pow(1.0 - conn_imp(body, head), p)
        norm = torch.pow(body, p).sum(dim=1).clamp(min=_EPS)
        sats = 1.0 - torch.pow((err.sum(dim=1) / norm).clamp(min=_EPS), 1.0 / p)
        add_axiom_type(sats, compiled.sub_names)
    if compiled.dis_pairs.numel():
        a, b = compiled.dis_pairs
        add_axiom_type(agg_forall(conn_neg(conn_and(c_mat[a], c_mat[b])), dim=1),
                       compiled.dis_names)
    if compiled.compl_pairs.numel():
        a, b = compiled.compl_pairs
        cons = agg_forall(conn_neg(conn_and(c_mat[a], c_mat[b])), dim=1)
        exh = agg_forall(conn_or(c_mat[a], c_mat[b]), dim=1)
        add_axiom_type(cons, [f"compl-consistent:{n}" for n in compiled.compl_names])
        add_axiom_type(exh, [f"compl-exhaustive:{n}" for n in compiled.compl_names])

    if compiled.existsdef:
        cols = torch.arange(n, device=device)
        sats = []
        for ax in compiled.existsdef:
            r = model.rel_truth(
                idx.rel[ax.relation],
                ent_all.repeat_interleave(n),
                cols.repeat(batch),
            ).view(batch, n)
            ex = agg_exists(r, dim=1)
            c = c_mat[idx.cls[ax.cls]]
            equiv = conn_and(conn_imp(ex, c), conn_imp(c, ex))
            sats.append(agg_forall(equiv))
        add_axiom_type(
            torch.stack(sats),
            [f"existsdef:{ax.cls}={ax.relation}" for ax in compiled.existsdef],
        )

    if compiled.equiv:
        # Ground every SomeValuesFrom occurrence over asserted edges at once:
        # max over edges with a batch subject, per occurrence.
        if compiled.n_svf:
            if pos is not None:
                keep = pos[compiled.svf_src] >= 0
                src = compiled.svf_src[keep]
                dst = compiled.svf_dst[keep]
                seg = compiled.svf_seg[keep]
                rows = pos[src]
            else:
                src, dst, seg = compiled.svf_src, compiled.svf_dst, compiled.svf_seg
                rows = src
            edge_truth = conn_and(
                model.rel_truth_pairs(compiled.svf_rid[seg], src, dst),
                model.class_truth_pairs(compiled.svf_fid[seg], dst),
            )
            flat_idx = seg * batch + rows
            if ops is None:
                svf_vals = torch.zeros(
                    compiled.n_svf * batch, device=device
                ).scatter_reduce(
                    0, flat_idx, edge_truth, reduce="amax", include_self=False
                ).view(compiled.n_svf, batch)
            else:
                # learnable existential over asserted edges: per-subject
                # [mean, max, log1p(degree)] features; no edges -> 0
                cnt = torch.zeros(compiled.n_svf * batch, device=device).scatter_add_(
                    0, flat_idx, torch.ones_like(edge_truth)
                )
                sums = torch.zeros(compiled.n_svf * batch, device=device).scatter_add_(
                    0, flat_idx, edge_truth
                )
                mx = torch.zeros(compiled.n_svf * batch, device=device).scatter_reduce_(
                    0, flat_idx, edge_truth, reduce="amax", include_self=False
                )
                feats = torch.stack(
                    [sums / cnt.clamp(min=1.0), mx, torch.log1p(cnt)], dim=-1
                )
                svf_vals = (ops.exists_mlp(feats) * (cnt > 0)).view(
                    compiled.n_svf, batch
                )
        else:
            svf_vals = c_mat.new_zeros((0, batch))
        sats = []
        for eq in compiled.equiv:
            conj = None
            for cid in eq.named:
                t = c_mat[cid]
                conj = t if conj is None else conn_and(conj, t)
            for occ in eq.svf_occs:
                t = svf_vals[occ]
                conj = t if conj is None else conn_and(conj, t)
            # Sufficient condition only (see module docstring). The
            # violation mean is normalized by firing strength (sum of
            # body^p), NOT by domain size: a plain forall dilutes the few
            # entities the rule fires on by |domain| (observed: 20 held-out
            # likers' violations got weight 20/500 and the rule could not
            # lift them against the closed-world prior).
            rule = conn_imp(conj, c_mat[eq.cls])
            err = torch.pow(1.0 - rule, p)
            norm = torch.pow(conj, p).sum().clamp(min=_EPS)
            sats.append(1.0 - torch.pow((err.sum() / norm).clamp(min=_EPS), 1.0 / p))
        add_axiom_type(torch.stack(sats), [eq.name for eq in compiled.equiv])

    if model.negation == "explicit":
        cons = agg_forall(conn_neg(conn_and(c_mat, nc_mat)), dim=1)
        exh = agg_forall(conn_or(c_mat, nc_mat), dim=1)
        add_axiom_type(cons, [f"neg-consistent:{n}" for n in compiled.cls_names],
                       complement_weight)
        add_axiom_type(exh, [f"neg-exhaustive:{n}" for n in compiled.cls_names],
                       complement_weight)

    if prior_weight > 0:
        # closed-world prior: unasserted memberships default to false with a
        # small weight. Only ~members-many negatives per class are asserted,
        # so without a prior the vast majority of non-members is
        # unconstrained and drifts arbitrarily high (observed: non-likers
        # scored 0.71 on BaseballLover, above true likers at 0.59, and
        # filtered MRR stayed random). Assertions (weight 1) and axioms
        # easily override the prior where they apply. Known members
        # (train + test positives) are masked out of the prior. The prior
        # uses a high p (worst-case focused): a mean over ~25k pairs
        # tolerates a handful of high drifters (observed: non-person
        # likers at ~0.9 in the controlled dataset, destroying filtered
        # ranking while prior sat read 0.86) — ranking metrics punish
        # exactly those few violators.
        mask = compiled.pos_mask[:, ent_all]
        errs = torch.pow(c_mat, prior_p).masked_fill(mask, 0.0)
        counts = (~mask).sum(dim=1).clamp(min=1)
        per_class = 1.0 - torch.pow(
            (errs.sum(dim=1) / counts).clamp(min=_EPS), 1.0 / prior_p
        )
        sat = forall(per_class, p)
        type_sats.append((sat, prior_weight))
        if with_details:
            detail_scalars.append((sat, "prior:closed-world"))

    def add_assertion_type(
        name: str, group: FlatGroup | None, pair_fn
    ) -> None:
        """Per-class (or per-relation) aggregation first: a single forall over
        all assertions would dilute rare classes (1 anchor among 35k
        assertions gets ~zero gradient, and p-mean error then ignores them).
        """
        if group is None:
            return
        a, b, gid = group.a, group.b, group.gid
        c = group.c
        if pos is not None:
            keep = pos[b] >= 0
            a, b, gid = a[keep], b[keep], gid[keep]
            c = c[keep] if c is not None else None
        if b.numel() == 0:
            return
        truths = pair_fn(a, b) if c is None else pair_fn(a, b, c)
        per_group, present = _seg_forall(truths, gid, group.n_groups, p, ops)
        sat = forall(per_group[present], p)
        type_sats.append((sat, 1.0))
        if with_details:
            detail_scalars.append((sat, name))

    add_assertion_type("assert:positive", compiled.unary_pos, model.class_truth_pairs)
    add_assertion_type("assert:negative", compiled.unary_neg, model.neg_class_truth_pairs)
    # relational groups keep triples whose subject (FlatGroup.b) is in the
    # batch; filtering both endpoints would survive only ~(batch/n)^2 of edges
    add_assertion_type("assert:relational", compiled.rel_pos, model.rel_truth_pairs)
    add_assertion_type(
        "assert:relational-negative", compiled.rel_neg,
        lambda r, s, o: conn_neg(model.rel_truth_pairs(r, s, o)),
    )

    # Aggregate per axiom type before averaging: with a flat mean over
    # groups, frequent types (e.g. 83 subclass axioms) dominate and rare but
    # critical groups (positive assertions) can be sacrificed entirely --
    # the model collapses all class memberships to 0 and satisfies every
    # implication vacuously (observed on UOBM: assert:positive sat ~ 0).
    total_weight = sum(w for _, w in type_sats)
    loss = sum(w * (1.0 - sat) for sat, w in type_sats) / total_weight
    details: dict[str, float] = {}
    if with_details:
        for vec, names in detail_vecs:
            for name, sat in zip(names, vec.detach().tolist(), strict=True):
                details[name] = sat
        for sat, name in detail_scalars:
            details[name] = float(sat.detach())
    return loss, details
