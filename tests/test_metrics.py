import torch

from negemb.data.schema import Index, build_index
from negemb.data.synthetic import generate
from negemb.ltn.grounding import LTNModel
from negemb.metrics.negation import (
    _auc,
    complement_consistency,
    de_morgan_gap,
    negation_discrimination,
    non_contradiction_rate,
)
from negemb.metrics.standard import (
    average_rank,
    known_members,
    membership_ranking,
    ranking_metrics,
)
from negemb.train import TrainConfig, train


class _StubModel:
    """Fixed scores: entity i scores scores[i] for every class."""

    def __init__(self, scores):
        self.scores = scores

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def class_truth(self, class_idx, ent_ids):
        return self.scores[ent_ids]


def test_filtered_ranking_ignores_other_true_answers():
    model = _StubModel(torch.tensor([1.0, 1.0, 0.0]))
    idx = Index(entity={"e0": 0, "e1": 1, "e2": 2}, cls={"C": 0}, rel={})
    queries = {("C", "e0")}
    unfiltered = membership_ranking(model, idx, 3, queries, model.class_truth)
    assert unfiltered["mrr"] == 1 / 1.5  # tied with e1
    filtered = membership_ranking(
        model, idx, 3, queries, model.class_truth,
        known_true={"C": {"e0", "e1"}},
    )
    assert filtered["mrr"] == 1.0
    assert known_members({("C", "e0")}, {("C", "e1")}) == {"C": {"e0", "e1"}}


def test_average_rank_no_ties():
    scores = torch.tensor([0.9, 0.5, 0.7])
    assert average_rank(scores, 1) == 3.0
    assert average_rank(scores, 0) == 1.0


def test_average_rank_with_ties():
    scores = torch.tensor([0.5, 0.5, 0.5])
    assert average_rank(scores, 0) == 2.0


def test_ranking_metrics():
    out = ranking_metrics([1.0, 2.0, 4.0], ks=(1, 10))
    assert out["hits@1"] == 1 / 3
    assert out["hits@10"] == 1.0
    assert abs(out["mrr"] - (1.0 + 0.5 + 0.25) / 3) < 1e-9


def _toy_model(negation):
    torch.manual_seed(0)
    return LTNModel(num_entities=8, num_classes=3, num_relations=1, dim=4, negation=negation)


def _toy_index():
    return Index(
        entity={f"e{i}": i for i in range(8)},
        cls={f"C{i}": i for i in range(3)},
        rel={"R": 0},
    )


def test_derived_negation_metrics_are_trivially_zero():
    model = _toy_model("derived")
    idx = _toy_index()
    assert complement_consistency(model, idx, 8) == 0.0
    assert non_contradiction_rate(model, idx, 8) == 0.0
    for tnorm in ("product", "godel", "lukasiewicz"):
        assert de_morgan_gap(model, idx, 8, tnorm) < 1e-6


def test_explicit_negation_metrics_are_nontrivial():
    model = _toy_model("explicit")
    idx = _toy_index()
    # random independent negation embeddings should not be perfectly consistent
    assert complement_consistency(model, idx, 8) > 0.0


def test_auc_tie_and_order_handling():
    assert _auc(torch.tensor([0.9, 0.8]), torch.tensor([0.1, 0.2])) == 1.0
    assert _auc(torch.tensor([0.1]), torch.tensor([0.9])) == 0.0
    assert _auc(torch.tensor([0.5]), torch.tensor([0.5])) == 0.5


def test_negation_discrimination_bounds_and_direction():
    ont = generate(seed=0, n_people=16)
    idx = build_index(ont)
    model = LTNModel(
        num_entities=len(ont.entities),
        num_classes=len(ont.classes),
        num_relations=len(ont.relations),
        dim=8,
    )
    out = negation_discrimination(model, ont, idx)
    assert 0.0 <= out["negation_auc"] <= 1.0
    assert out["n_classes"] > 0
    # a model trained for a few steps should beat the random 0.5 level
    model, _ = train(ont, TrainConfig(dim=8, epochs=200, device="cpu", log_every=0))
    trained = negation_discrimination(model, ont, idx)
    assert trained["negation_auc"] > 0.5
