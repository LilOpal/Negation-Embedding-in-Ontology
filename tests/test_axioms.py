import math

import torch

from negemb.axioms import kb_loss
from negemb.data.schema import (
    EquivalentTo,
    Index,
    Ontology,
    SomeValuesFrom,
)
from negemb.ltn.grounding import LTNModel


def _toy_ontology() -> Ontology:
    return Ontology(
        entities=["a", "b", "c"],
        classes=["C", "D", "E"],
        relations=["R"],
        axioms=[EquivalentTo("C", ("D", SomeValuesFrom("R", "E")))],
        train_positive={("D", "a"), ("E", "b")},
        train_negative={("C", "c")},
        relational={("R", "a", "b")},
        relational_negative={("R", "c", "b")},
    )


def test_equivalent_to_compiles_and_loss_is_finite():
    torch.manual_seed(0)
    ont = _toy_ontology()
    idx = Index(
        entity={e: i for i, e in enumerate(ont.entities)},
        cls={c: i for i, c in enumerate(ont.classes)},
        rel={"R": 0},
    )
    model = LTNModel(3, 3, 1, dim=8)
    loss, details = kb_loss(model, ont, idx)
    assert math.isfinite(float(loss))
    assert "equivdef:C" in details


def test_exists_restriction_uses_asserted_edges():
    """Entities with a matching edge should score higher than those without."""
    torch.manual_seed(0)
    ont = _toy_ontology()
    idx = Index(
        entity={e: i for i, e in enumerate(ont.entities)},
        cls={c: i for i, c in enumerate(ont.classes)},
        rel={"R": 0},
    )
    model = LTNModel(3, 3, 1, dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss, _ = kb_loss(model, ont, idx)
        loss.backward()
        opt.step()
    with torch.no_grad():
        c_truth = model.class_truth(idx.cls["C"], torch.arange(3))
    # "a" is D and has an R-edge to E-instance "b"; "c" is neither
    assert c_truth[idx.entity["a"]] > c_truth[idx.entity["c"]]
