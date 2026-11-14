# Negation Embedding in Ontology

This project attacks the problem of negation in ontology embeddings.
We categorize existing methods into 3 types: Geometric, Sequence and Neural.
Geometric methods represent axioms and entities as geometric shapes in the space
(e.g., boxes, balls, or cones).
Closed shapes like boxes and balls are inherently hard to represent negation.

Cones don't have empirical experiments so far.

Sequence methods do not deal with negation so far.

Neural methods like Logic Tensor Networks (LTN) seem promising in this topic.

## What this repository is

A first **runnable** LTN-style ontology embedding that treats negation as an
explicit modelling choice, so the three attitudes towards negation can be
compared under one pipeline and one set of metrics:

- **`derived` (baseline A)** — `~C(x)` is simply `1 - C(x)`. No parameters, no
  consistency axioms; a strong reference point.
- **`explicit` (baseline B)** — `~C(x)` comes from its own embedding. No
  complement relation is built in; consistency is enforced only softly through
  axioms.
- **`neural` (setting C)** — `~C(x)` is a small learnable function of `C(x)`,
  initialized to the fuzzy complement and then shaped by the knowledge-base
  loss alone.

Additionally, the logical connectives and the instance-level quantifiers can
either be fixed fuzzy operators (product/Gödel/Łukasiewicz t-norm, p-mean
quantifiers) or **learnable operators** initialized to those same fuzzy
semantics — the `--ops fuzzy|neural` switch. Everything else (axiom
compilation, type weighting, closed-world prior, metrics) is held fixed, so
the comparison isolates one factor at a time.

The knowledge base (subsumption, disjointness, complement, existential
definitions, conjunctive definitions, positive and negative assertions) is
compiled into a single differentiable satisfiability loss and trained with
standard entity/class/relation embeddings.

Two datasets are supported: a fully controlled synthetic family ontology, and
UOBM-1 (University Ontology Benchmark).

## How to run

```
pip install -e ".[dev]"

# synthetic toy ontology
python experiments/run_baseline.py --negation derived
python experiments/run_baseline.py --negation explicit --tnorm lukasiewicz
python experiments/run_baseline.py --negation neural --ops neural

# UOBM (files under data/raw/uobm/)
python experiments/run_baseline.py --dataset uobm --negation derived --prior-weight 0.1
```

Experiments print a JSON report with standard filtered-ranking metrics
(MRR / Hits@1 / Hits@10 on held-out positive memberships) and a block of
negation-oriented metrics: complement consistency, non-contradiction rate,
disjointness violations, De Morgan gap, ranking of held-out negative
assertions, and a Mann-Whitney AUC measuring whether `~C` separates known
non-members from known members.

`pytest` runs the test suite (operators, quantifiers, grounding/negation
modes, axiom compilation, batching, metrics, datasets, training smoke test).

Extra scripts: `experiments/controlled_sanity.py` (a small fully-known world
that reproduces the UOBM failure modes in seconds),
`experiments/diagnose_uobm.py` (per-axiom-type satisfiability and probes on
UOBM), and `experiments/inspect_ops.py` (compare a learned operator against
its fuzzy counterpart).

## Conclusions so far

The pipeline runs end-to-end on both datasets, and working through it produced
the following observations. They are the main result of this stage: they say
what a negation-aware LTN embedding *needs*, not which variant "wins" yet.

1. **Explicit negation needs a separate scoring head.** Giving `~C` its own
   embedding while reusing the membership scorer works at toy scale but
   destroys standard ranking at UOBM scale: the negation-consistency groups
   and the membership structure compete for the same weights. A dedicated
   scorer for the negation path removes the interference.

2. **Bounded truth functions matter more than the negation scheme.** With
   unbounded MLP logits, pairs the axioms do not constrain drift into sigmoid
   saturation, where the gradient vanishes and the pair is effectively lost;
   ranking metrics punish exactly those drifters. Bounding the pre-activation
   keeps every pair trainable.

3. **Open-world embeddings do not rank without a closed-world prior.** Only a
   few negatives per class are asserted, so most non-members are unconstrained
   and drift high — above true members — which floods held-out negative
   queries. A closed-world prior on *unasserted* memberships (masking known
   members, focused on the worst pairs) is what makes filtered ranking work.
   For the same reason, AUC on known negatives vs known positives is a more
   trustworthy negation metric here than ranking of sampled negatives, which
   floods.

4. **Axioms must be aggregated carefully or they are vacuous.** With a flat
   mean over axioms, frequent types dominate and rare but critical groups
   (positive assertions, rare subclasses) get sacrificed: the model satisfies
   every implication by driving all memberships to zero. Per-type weighting,
   plus normalizing subsumption and rule axioms by how strongly their body
   fires, fixes this.

5. **Definitions may only be injected as sufficient conditions.** Under
   open-world semantics the ABox does not contain every implied edge,
   so requiring `C(x) -> body` contradicts the assertions and collapses
   training. Injecting `body -> C(x)` (standard rule injection) and
   materializing body matches as positives is what lets rule-defined classes
   generalize instead of memorizing.

6. **Learnable operators stay close to fuzzy semantics.** Initializing the
   neural connectives and quantifiers to the fuzzy ones gives a well-defined
   starting point; the remainder of the comparisons (`fuzzy` vs `neural`
   operators, `derived` vs `explicit` vs `neural` negation) is what the
   experiments are set up to measure.

## Open questions

- Do learnable operators buy anything over fuzzy ones, or do they mostly
  rediscover them?
- Is a learnable negation better than a derived one once ranking is stable,
  or does it only trade consistency for expressivity?
- How do these findings transfer beyond UOBM-1, and to ontologies whose
  negation is not expressible as a complement of named classes?
- Where geometry would fail, can a neural/LTN formulation actually deliver —
  i.e., the original question behind this project?