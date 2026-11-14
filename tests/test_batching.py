import math

import torch

from negemb.axioms import kb_loss
from negemb.data.schema import build_index
from negemb.data.synthetic import generate
from negemb.ltn.grounding import LTNModel
from negemb.train import TrainConfig, train


def test_batched_training_reduces_loss():
    ont = generate(seed=0, n_people=24)
    cfg = TrainConfig(
        dim=8, epochs=300, batch_size=8, negation="derived",
        device="cpu", log_every=0,
    )
    model, history = train(ont, cfg)
    assert history[-1] < history[0]
    assert all(math.isfinite(v) for v in history)


def test_batched_loss_matches_full_batch_on_full_domain():
    ont = generate(seed=0, n_people=24)
    idx = build_index(ont)
    torch.manual_seed(0)
    model = LTNModel(
        num_entities=len(ont.entities),
        num_classes=len(ont.classes),
        num_relations=len(ont.relations),
        dim=8,
    )
    full_loss, _ = kb_loss(model, ont, idx)
    ent_ids = torch.arange(len(ont.entities))
    batched_loss, _ = kb_loss(model, ont, idx, ent_ids=ent_ids)
    assert torch.isclose(full_loss, batched_loss, atol=1e-6)


def test_batched_loss_is_finite_on_small_batch():
    ont = generate(seed=0, n_people=24)
    idx = build_index(ont)
    torch.manual_seed(0)
    model = LTNModel(
        num_entities=len(ont.entities),
        num_classes=len(ont.classes),
        num_relations=len(ont.relations),
        dim=8,
    )
    loss, details = kb_loss(model, ont, idx, ent_ids=torch.arange(4))
    assert math.isfinite(float(loss))
    assert all(math.isfinite(v) for v in details.values())
