import torch

from negemb.ltn.operators import TNORMS, and_, implies, neg, or_
from negemb.ltn.quantifiers import exists, forall


def test_neg_involution():
    x = torch.rand(100)
    assert torch.allclose(neg(neg(x)), x)


def test_de_morgan_holds_for_all_tnorms():
    a, b = torch.rand(64), torch.rand(64)
    for t in TNORMS:
        assert torch.allclose(or_(a, b, t), neg(and_(neg(a), neg(b), t)), atol=1e-6)


def test_and_boundaries():
    x = torch.rand(64)
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    for t in TNORMS:
        assert torch.allclose(and_(one, x, t), x)
        assert torch.allclose(and_(zero, x, t), zero)


def test_implies_classical_corners():
    one, zero = torch.tensor(1.0), torch.tensor(0.0)
    for t in TNORMS:
        assert torch.isclose(implies(one, one, t), one)
        assert torch.isclose(implies(one, zero, t), zero)
        assert torch.isclose(implies(zero, one, t), one)
        assert torch.isclose(implies(zero, zero, t), one)


def test_unknown_tnorm_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown t-norm"):
        and_(torch.tensor(0.5), torch.tensor(0.5), "bogus")


def test_quantifier_bounds_and_extremes():
    xs = torch.rand(128)
    for agg in (forall, exists):
        out = agg(xs)
        assert 0.0 <= out <= 1.0
    assert forall(torch.ones(10)) > 0.999
    assert exists(torch.zeros(10)) < 1e-3


def test_quantifiers_aggregate_per_dimension():
    xs = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    ex = exists(xs, dim=1)
    fa = forall(xs, dim=1)
    assert ex.shape == fa.shape == (2,)
    assert ex[1] > ex[0]  # row with two true values has a stronger exists
    assert fa[1] > fa[0]  # and a stronger forall


def test_exists_with_infinite_p_is_max():
    xs = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    ex = exists(xs, p=float("inf"), dim=1)
    assert torch.allclose(ex, torch.tensor([1.0, 0.0]))
    # no dilution of a single witness over a large domain
    one_witness = torch.zeros(50)
    one_witness[7] = 1.0
    assert exists(one_witness, p=float("inf")) == 1.0
