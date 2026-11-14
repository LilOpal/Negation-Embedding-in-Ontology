import pytest

from negemb.data.schema import ComplementOf, Disjoint, EquivalentTo, SomeValuesFrom, SubClassOf

uobm = pytest.importorskip("negemb.data.uobm")
pytestmark = pytest.mark.skipif(
    not (uobm.RAW_DIR / uobm.TBOX_FILE).exists(), reason="UOBM raw data not downloaded"
)


def _axiom_classes(ax) -> set[str]:
    if isinstance(ax, EquivalentTo):
        return {ax.cls} | {p if isinstance(p, str) else p.filler for p in ax.conjuncts}
    return set(ax.__dict__.values())


def test_load_deterministic_and_sane():
    ont = uobm.load(seed=0, max_entities=1000)
    assert ont == uobm.load(seed=0, max_entities=1000)
    assert 0 < len(ont.entities) <= 1000
    assert ont.axioms and ont.train_positive and ont.train_negative
    assert ont.test_positive and ont.test_negative
    assert ont.train_positive.isdisjoint(ont.test_positive)
    assert ont.relational.isdisjoint(ont.relational_negative)
    referenced = {n for ax in ont.axioms for n in _axiom_classes(ax)}
    assert referenced <= set(ont.classes)


def test_negation_axioms_present():
    ont = uobm.load(seed=0, max_entities=1000)
    assert any(isinstance(ax, (Disjoint, ComplementOf)) for ax in ont.axioms)
    assert any(isinstance(ax, SubClassOf) for ax in ont.axioms)


def test_equivalent_class_axioms_extracted():
    ont = uobm.load(seed=0, max_entities=1000)
    equiv = [ax for ax in ont.axioms if isinstance(ax, EquivalentTo)]
    assert equiv, "expected equivalentClass intersection axioms"
    assert any(
        isinstance(p, SomeValuesFrom) for ax in equiv for p in ax.conjuncts
    )


def test_rule_members_materialized_as_positives():
    """Entities satisfying a rule body via asserted edges must appear as
    positive assertions of the defined class (rule injection), and never as
    negatives of that class or its superclasses."""
    ont = uobm.load(seed=0, max_entities=1000)
    pos = ont.train_positive | ont.test_positive
    neg = ont.train_negative | ont.test_negative
    fillers = {e for c, e in pos if c == "BaseballClass"}
    likers = {s for r, s, o in ont.relational if r == "like" and o in fillers}
    persons = {e for c, e in pos if c == "Person"}
    expected = likers & persons
    assert expected, "subsample should keep some rule-satisfying likers"
    assert all(("BaseballLover", e) in pos for e in expected)
    assert not pos & neg
