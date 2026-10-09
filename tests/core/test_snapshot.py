"""The feature snapshot: sorted by construction, and refusing what cannot be read back."""

from __future__ import annotations

import math

import pytest

from neurotrade.core.snapshot import NO_FEATURES, FeatureSnapshot

# ── construction and ordering ───────────────────────────────────────────


def test_of_sorts_by_name_whatever_order_the_mapping_iterated_in() -> None:
    """The ordering is a property of the value, not of the caller's dict. A run
    digest folds this object, so inheriting insertion order would let two
    identical runs disagree."""
    one = FeatureSnapshot.of({"z": 1.0, "a": 2.0, "m": 3.0})
    other = FeatureSnapshot.of({"m": 3.0, "z": 1.0, "a": 2.0})
    assert one.values == (("a", 2.0), ("m", 3.0), ("z", 1.0))
    assert one == other


def test_the_empty_snapshot_is_the_shared_constant() -> None:
    assert FeatureSnapshot() == NO_FEATURES
    assert FeatureSnapshot.of({}) == NO_FEATURES
    assert NO_FEATURES.is_empty
    assert len(NO_FEATURES) == 0


def test_a_snapshot_is_hashable() -> None:
    """`Intent` is a frozen dataclass that carries one; a `dict` field would
    make every intent unhashable."""
    assert len({FeatureSnapshot.of({"a": 1.0}), FeatureSnapshot.of({"a": 1.0})}) == 1


def test_names_and_as_dict_report_the_sorted_content() -> None:
    snapshot = FeatureSnapshot.of({"b": None, "a": 1.5})
    assert snapshot.names == ("a", "b")
    assert snapshot.as_dict() == {"a": 1.5, "b": None}
    assert list(snapshot.as_dict()) == ["a", "b"]


def test_as_dict_hands_back_a_copy() -> None:
    snapshot = FeatureSnapshot.of({"a": 1.0})
    snapshot.as_dict()["a"] = 99.0
    assert snapshot.get("a") == 1.0


# ── reading a value ─────────────────────────────────────────────────────


def test_a_recorded_none_means_warming_up_and_is_not_an_error() -> None:
    assert FeatureSnapshot.of({"rvol": None}).get("rvol") is None


def test_an_unrecorded_name_raises_rather_than_reading_as_none() -> None:
    """ "Never declared" and "declared but cold" are different facts about the
    decision. Collapsing them into one `None` is how a model ends up trained on
    a feature the strategy could not see."""
    with pytest.raises(KeyError, match="was not recorded"):
        FeatureSnapshot.of({"rvol": 1.0}).get("atr")


# ── rejection ───────────────────────────────────────────────────────────


def test_unsorted_pairs_are_refused_by_the_constructor() -> None:
    """`of()` is the way in. A hand-built snapshot in the wrong order could
    never be caught later, because nothing downstream re-sorts."""
    with pytest.raises(ValueError, match="strictly ascending"):
        FeatureSnapshot(values=(("z", 1.0), ("a", 2.0)))


def test_a_duplicate_name_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly ascending"):
        FeatureSnapshot(values=(("a", 1.0), ("a", 2.0)))


@pytest.mark.parametrize("name", ["", " ", "\t"])
def test_a_blank_feature_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        FeatureSnapshot(values=((name, 1.0),))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_value_is_refused(value: float) -> None:
    """A NaN would travel into the run digest, into a model's input matrix, and
    into the journal as the literal token `NaN`, which is not valid JSON."""
    with pytest.raises(ValueError, match="is not finite"):
        FeatureSnapshot.of({"atr": value})


@pytest.mark.parametrize("values", [[("a", 1.0)], (("a", 1.0),), "a=1", 5])
def test_of_refuses_anything_that_is_not_a_mapping(values: object) -> None:
    """Checked explicitly so a malformed decoded row raises TypeError, which the
    journal and the codec translate into a named error, rather than
    AttributeError, which neither catches."""
    with pytest.raises(TypeError, match="built from a mapping"):
        FeatureSnapshot.of(values)  # type: ignore[arg-type]
