"""What a strategy had read at the moment it decided.

A `FeatureSnapshot` is the point-in-time record the Invariants require on every
trade record: the resolved value of each feature the strategy declared, as of
the bar it acted on. Without it a journal row says what happened and nothing
about why, so a meta-labeller (§7.3) has outcomes to learn from and no inputs.

**Why it is a sorted tuple of pairs rather than a mapping.** Two reasons, both
structural rather than stylistic:

- `Intent` is a frozen dataclass and an `Event`. A `dict` field would make it
  unhashable and would make its equality depend on a mutable object.
- Determinism is testable (Invariants), and a run digest folds every event. Had
  the order come from a `dict`, the bytes behind a digest would inherit that
  `dict`'s insertion order; sorting on construction makes the ordering a
  property of the value, so no caller can move a digest by building its
  mapping in a different order.

**`None` means warming up, never zero.** The same convention `StrategyContext`
uses: a feature that has not filled its lookback has no value, and treating
that as `0.0` is the silent version of acting on data that does not exist. A
name absent altogether means the strategy never declared it.

**Non-finite values are refused.** A `NaN` would travel into a run digest, into
a model's input matrix, and into the journal as the literal token `NaN`, which
is not valid JSON and would make the file unreadable by the analysis path. A
feature producing one is a bug in the feature; failing here is the direction
that surfaces it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Final

__all__ = ["NO_FEATURES", "FeatureSnapshot"]


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Resolved feature values at one decision moment, sorted by name.

    Construct with `of()` from whatever mapping the context holds; the
    constructor itself takes the already-sorted pairs and refuses anything else,
    so an unsorted snapshot cannot exist to be hashed or encoded.

    Example:
        >>> snapshot = FeatureSnapshot.of({"rvol": 1.8, "atr": None})
        >>> snapshot.names
        ('atr', 'rvol')
        >>> snapshot.get("rvol")
        1.8
        >>> snapshot.get("atr") is None        # declared, still warming up
        True
    """

    values: tuple[tuple[str, float | None], ...] = ()
    """One `(name, value)` pair per declared feature, ascending by name. The
    value is a `float` because derived features are floats (Invariants) — never
    a `Decimal`, which is reserved for money and prices — or `None` while the
    feature is still warming up."""

    def __post_init__(self) -> None:
        """Validate the ordering, the names and the values.

        Raises:
            ValueError: If the pairs are not strictly ascending by name (which
                includes any duplicate), if a name is blank, or if a value is
                not finite. See the module docstring for why each of the three
                is fatal rather than repaired.
        """
        names = [name for name, _ in self.values]
        if any(not name.strip() for name in names):
            raise ValueError(f"a feature name must not be blank: {names}")
        if names != sorted(set(names)):
            raise ValueError(
                f"snapshot pairs must be strictly ascending by name; got {names} — "
                f"build it with FeatureSnapshot.of()"
            )
        for name, value in self.values:
            if value is not None and not isfinite(value):
                raise ValueError(f"feature {name!r} is not finite: {value!r}")

    @classmethod
    def of(cls, values: Mapping[str, float | None]) -> FeatureSnapshot:
        """Build a snapshot from a context's resolved values.

        Args:
            values: Feature name to value, in any order. `None` for a feature
                that has not warmed up.

        Returns:
            The same content, sorted by name.

        Raises:
            TypeError: If `values` is not a mapping. Checked rather than left to
                `.items()` because this is also the decode boundary for the
                event log and the trade journal, and `AttributeError` escaping
                from a malformed row would bypass the callers that translate a
                bad record into a named, line-numbered error.

        Example:
            >>> FeatureSnapshot.of({"b": 2.0, "a": 1.0}).values
            (('a', 1.0), ('b', 2.0))
            >>> FeatureSnapshot.of({}) == FeatureSnapshot()
            True
        """
        if not isinstance(values, Mapping):
            raise TypeError(f"a snapshot is built from a mapping, not {type(values).__name__}")
        return cls(values=tuple(sorted(values.items())))

    @property
    def names(self) -> tuple[str, ...]:
        """Every feature recorded, ascending."""
        return tuple(name for name, _ in self.values)

    @property
    def is_empty(self) -> bool:
        """Whether nothing was recorded.

        True both for a strategy that declared no features and for a decision
        taken where no snapshot was captured — the two are indistinguishable
        here, and a caller needing to tell them apart must look at the
        strategy's declaration rather than at the row.
        """
        return not self.values

    def get(self, name: str) -> float | None:
        """Read one recorded feature.

        Args:
            name: The feature's declared name.

        Returns:
            Its value, or `None` if it had not warmed up at the decision.

        Raises:
            KeyError: If the name was not recorded at all. Distinguished from a
                recorded `None` on purpose: "never declared" and "declared but
                cold" are different facts about the decision, and collapsing
                them into one `None` is how a model ends up trained on a
                feature the strategy could not see.

        Example:
            >>> FeatureSnapshot.of({"rvol": 1.8}).get("rvol")
            1.8
        """
        for recorded, value in self.values:
            if recorded == name:
                return value
        raise KeyError(f"{name!r} was not recorded; snapshot holds {list(self.names)}")

    def as_dict(self) -> dict[str, float | None]:
        """A fresh mutable copy, for serialisation and for model input assembly.

        Insertion order is the sorted order, so a JSON object written from it is
        byte-stable without relying on the encoder to sort keys.

        Example:
            >>> FeatureSnapshot.of({"b": None, "a": 1.0}).as_dict()
            {'a': 1.0, 'b': None}
        """
        return dict(self.values)

    def __len__(self) -> int:
        """How many features were recorded."""
        return len(self.values)


NO_FEATURES: Final = FeatureSnapshot()
"""The empty snapshot. A module-level constant because it is the default of a
dataclass field (`Intent.features`), and a frozen value can be shared by every
instance that has nothing to report — "no snapshot was recorded here", never
"every feature read zero"."""
