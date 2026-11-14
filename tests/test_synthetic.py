from negemb.data.synthetic import CLASSES, generate


def test_deterministic_per_seed():
    assert generate(seed=42) == generate(seed=42)
    assert generate(seed=0) != generate(seed=1)


def test_ontology_sanity():
    ont = generate(seed=0)

    def members(cls, universe):
        return {e for c, e in universe if c == cls}

    pos = ont.train_positive | ont.test_positive
    neg = ont.train_negative | ont.test_negative
    male, female = members("Male", pos), members("Female", pos)
    parent, childless = members("Parent", pos), members("Childless", pos)
    grandparent = members("Grandparent", pos)

    assert male.isdisjoint(female)
    assert parent.isdisjoint(childless)
    assert parent | childless == set(ont.entities)
    assert grandparent <= parent
    for cls in CLASSES:
        assert members(cls, pos).isdisjoint(members(cls, neg))


def test_train_test_split_disjoint():
    ont = generate(seed=0)
    assert ont.train_positive.isdisjoint(ont.test_positive)
    assert ont.train_negative.isdisjoint(ont.test_negative)
    assert ont.test_positive and ont.test_negative
    # assertions reference declared symbols only
    for cls, ent in ont.train_positive | ont.test_positive | ont.train_negative | ont.test_negative:
        assert cls in ont.classes and ent in ont.entities
    for rel, s, o in ont.relational:
        assert rel in ont.relations and s in ont.entities and o in ont.entities
    assert ont.relational.isdisjoint(ont.relational_negative)
