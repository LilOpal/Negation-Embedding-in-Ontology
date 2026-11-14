import math

import pytest

from negemb.data.schema import build_index
from negemb.data.synthetic import generate
from negemb.metrics.negation import evaluate_negation
from negemb.metrics.standard import evaluate_standard
from negemb.train import TrainConfig, train


@pytest.mark.parametrize("negation", ["derived", "explicit"])
def test_smoke_training_reduces_loss(negation):
    ont = generate(seed=0, n_people=24)
    cfg = TrainConfig(
        dim=8, epochs=300, negation=negation, device="cpu", log_every=0
    )
    model, history = train(ont, cfg)
    assert history[-1] < history[0]
    assert all(math.isfinite(v) for v in history)


@pytest.mark.parametrize("negation", ["derived", "explicit"])
def test_evaluation_runs_and_is_finite(negation):
    ont = generate(seed=0, n_people=24)
    cfg = TrainConfig(dim=8, epochs=100, negation=negation, device="cpu", log_every=0)
    model, _ = train(ont, cfg)
    idx = build_index(ont)

    standard = evaluate_standard(model, ont, idx)
    negation_metrics = evaluate_negation(model, ont, idx, tnorm="product")

    assert 0.0 <= standard["mrr"] <= 1.0
    neg_rank = negation_metrics["negative_query_ranking"]
    assert 0.0 <= neg_rank["mrr"] <= 1.0
    for key in ("complement_consistency", "non_contradiction_rate",
                "disjointness_violation", "de_morgan_gap"):
        assert math.isfinite(negation_metrics[key])
