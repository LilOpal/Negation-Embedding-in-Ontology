import math

import torch

from negemb.data.synthetic import generate
from negemb.ltn.neural_ops import NeuralOps
from negemb.train import TrainConfig, train


def test_init_fuzzy_matches_fuzzy_operators():
    ops = NeuralOps()
    ops.init_fuzzy()
    a = torch.tensor([0.7, 0.2, 0.5])
    b = torch.tensor([0.6, 0.9, 0.4])
    assert torch.allclose(ops.neg(a), 1 - a, atol=0.05)
    assert torch.allclose(ops.and_(a, b), a * b, atol=0.05)
    assert torch.allclose(ops.or_(a, b), a + b - a * b, atol=0.05)
    assert torch.allclose(ops.implies(a, b), 1 - a + a * b, atol=0.06)


def test_quantifier_aggregates_behave_after_init():
    ops = NeuralOps()
    ops.init_fuzzy()
    high = torch.full((50,), 0.95)
    mixed = torch.cat([torch.full((49,), 0.95), torch.tensor([0.05])])
    low = torch.full((50,), 0.05)
    assert ops.forall_agg(high) > 0.8
    assert ops.forall_agg(mixed) < ops.forall_agg(high)
    assert ops.exists_agg(low) < 0.3
    assert ops.exists_agg(mixed) > 0.8


def test_neural_training_smoke():
    ont = generate(seed=0, n_people=24)
    cfg = TrainConfig(
        dim=8, epochs=200, negation="neural", ops_mode="neural",
        device="cpu", log_every=0,
    )
    model, history = train(ont, cfg)
    assert history[-1] < history[0]
    assert all(math.isfinite(v) for v in history)
    vals = model.neg_class_truth(0, torch.arange(len(ont.entities)))
    assert bool(((vals >= 0) & (vals <= 1)).all())
