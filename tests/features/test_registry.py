"""Tests for feature specs and the feature registry.

The lookahead tests are the ones that matter. Everything else here is
bookkeeping; that check is the difference between a backtest that means
something and one that does not.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.registry import DuplicateRegistration, RegistryFrozen, Version
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.features.registry import (
    FeatureRegistry,
    FeatureSpec,
    LookaheadError,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MINUTE = 60_000_000_000
OPEN_NS = 1_773_495_000_000_000_000


def bar(ts: int, close: str = "100") -> Bar:
    return Bar(
        symbol=AAPL,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=Price(close),
        high=Price(close),
        low=Price(close),
        close=Price(close),
        volume=Quantity(1_000),
    )


def series(count: int, start: int = OPEN_NS) -> list[Bar]:
    """`count` consecutive one-minute bars, closing at 100, 101, 102, …"""
    return [bar(start + i * MINUTE, close=str(100 + i)) for i in range(count)]


def mean_close(bars: Sequence[Bar]) -> float:
    return sum(float(b.close.value) for b in bars) / len(bars)


def spec(lookback: int = 3) -> FeatureSpec:
    return FeatureSpec(
        name="mean_close",
        version=Version.parse("1.0.0"),
        lookback=lookback,
        interval=BarInterval.MIN_1,
        compute=mean_close,
        description="mean of recent closes",
    )


# ── Point-in-time correctness ────────────────────────────────


def test_a_bar_from_the_future_is_rejected() -> None:
    """The check that makes lookahead impossible rather than unlikely."""
    history = series(3)
    too_early = history[-1].ts_event - 1  # the last bar has not happened yet
    with pytest.raises(LookaheadError, match="after the moment being modelled"):
        spec().evaluate(history, as_of=too_early)


def test_a_window_ending_exactly_at_now_is_allowed() -> None:
    """A bar's close time is the instant it becomes observable, so it counts."""
    history = series(3)
    assert spec().evaluate(history, as_of=history[-1].ts_event) == 101.0


def test_a_window_ending_before_now_is_allowed() -> None:
    history = series(3)
    assert spec().evaluate(history, as_of=history[-1].ts_event + MINUTE) == 101.0


def test_out_of_order_bars_are_rejected() -> None:
    """Catches a bad feed where it is used, not as a wrong number weeks later."""
    history = series(3)
    scrambled = [history[0], history[2], history[1]]
    with pytest.raises(LookaheadError, match="out of order"):
        spec().evaluate(scrambled, as_of=OPEN_NS + 10 * MINUTE)


def test_only_the_evaluated_window_is_checked() -> None:
    """Older history outside the lookback cannot affect the result, so it is
    not the spec's business whether it is tidy."""
    history = series(10)
    assert spec(lookback=2).evaluate(history, as_of=history[-1].ts_event) == 108.5


# ── Warm-up ──────────────────────────────────────────────────


def test_returns_none_until_the_window_is_full() -> None:
    """A 3-bar mean computed from 2 bars is a different number that looks fine."""
    for count in (0, 1, 2):
        assert spec(lookback=3).evaluate(series(count), as_of=OPEN_NS + 100 * MINUTE) is None


def test_produces_a_value_once_warm() -> None:
    history = series(3)
    assert spec(lookback=3).evaluate(history, as_of=history[-1].ts_event) == 101.0


def test_only_the_last_lookback_bars_reach_the_computation() -> None:
    """Otherwise a 20-bar average silently becomes a 400-bar average."""
    seen: list[int] = []

    def record(bars: Sequence[Bar]) -> float:
        seen.append(len(bars))
        return 0.0

    FeatureSpec(
        name="counter",
        version=Version.parse("1.0.0"),
        lookback=5,
        interval=BarInterval.MIN_1,
        compute=record,
        description="counts what it received",
    ).evaluate(series(50), as_of=OPEN_NS + 100 * MINUTE)
    assert seen == [5]


# ── Declaration validation ───────────────────────────────────


@pytest.mark.parametrize("bad", [0, -1])
def test_lookback_must_be_at_least_one(bad: int) -> None:
    with pytest.raises(ValueError, match="lookback must be at least 1"):
        spec(lookback=bad)


def test_name_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        FeatureSpec(
            name="  ",
            version=Version.parse("1.0.0"),
            lookback=1,
            interval=BarInterval.MIN_1,
            compute=mean_close,
            description="x",
        )


def test_description_is_required() -> None:
    """It reaches the dashboard (§14); an unexplained number is not useful."""
    with pytest.raises(ValueError, match="describe what it measures"):
        FeatureSpec(
            name="x",
            version=Version.parse("1.0.0"),
            lookback=1,
            interval=BarInterval.MIN_1,
            compute=mean_close,
            description="",
        )


def test_qualified_name() -> None:
    assert spec().qualified_name == "mean_close@1.0.0"


# ── Registry ─────────────────────────────────────────────────


def test_decorator_registers_and_returns_the_function() -> None:
    registry = FeatureRegistry()

    @registry.feature("last_close", "1.0.0", lookback=1, description="most recent close")
    def last_close(bars: Sequence[Bar]) -> float:
        return float(bars[-1].close.value)

    assert registry.get("last_close").lookback == 1
    assert last_close(series(1)) == 100.0  # still directly callable


def test_registered_spec_carries_the_declaration() -> None:
    registry = FeatureRegistry()

    @registry.feature("atr", "1.2.0", lookback=14, description="average true range")
    def atr(bars: Sequence[Bar]) -> float:
        return 0.0

    found = registry.get("atr")
    assert (found.name, str(found.version), found.lookback) == ("atr", "1.2.0", 14)
    assert found.description == "average true range"


def test_two_versions_coexist() -> None:
    registry = FeatureRegistry()

    @registry.feature("rvol", "1.0.0", lookback=20, description="relative volume")
    def rvol_v1(bars: Sequence[Bar]) -> float:
        return 1.0

    @registry.feature("rvol", "2.0.0", lookback=50, description="relative volume, longer")
    def rvol_v2(bars: Sequence[Bar]) -> float:
        return 2.0

    assert registry.get("rvol", "1.0.0").lookback == 20
    assert registry.get("rvol").lookback == 50  # latest


def test_duplicate_registration_is_rejected() -> None:
    registry = FeatureRegistry()

    @registry.feature("x", "1.0.0", lookback=1, description="x")
    def first(bars: Sequence[Bar]) -> float:
        return 0.0

    with pytest.raises(DuplicateRegistration):

        @registry.feature("x", "1.0.0", lookback=1, description="x again")
        def second(bars: Sequence[Bar]) -> float:
            return 1.0


def test_freeze_closes_registration() -> None:
    registry = FeatureRegistry()
    registry.freeze()
    with pytest.raises(RegistryFrozen):

        @registry.feature("late", "1.0.0", lookback=1, description="late")
        def late(bars: Sequence[Bar]) -> float:
            return 0.0


def test_unknown_feature_raises() -> None:
    with pytest.raises(KeyError, match="no feature registered"):
        FeatureRegistry().get("nope")


# ── max_lookback: what the engine loads before the open ──────


def test_max_lookback_is_the_largest_declared() -> None:
    registry = FeatureRegistry()
    for name, lookback in (("a", 20), ("b", 200), ("c", 14)):

        @registry.feature(name, "1.0.0", lookback=lookback, description=name)
        def _(bars: Sequence[Bar]) -> float:
            return 0.0

    assert registry.max_lookback() == 200


def test_max_lookback_is_per_interval() -> None:
    """A 200-bar daily feature does not mean loading 200 minutes of intraday."""
    registry = FeatureRegistry()

    @registry.feature("intraday", "1.0.0", lookback=20, description="intraday")
    def intraday(bars: Sequence[Bar]) -> float:
        return 0.0

    @registry.feature(
        "daily", "1.0.0", lookback=200, description="daily", interval=BarInterval.DAY_1
    )
    def daily(bars: Sequence[Bar]) -> float:
        return 0.0

    assert registry.max_lookback(BarInterval.MIN_1) == 20
    assert registry.max_lookback(BarInterval.DAY_1) == 200


def test_max_lookback_of_an_empty_registry_is_zero() -> None:
    assert FeatureRegistry().max_lookback() == 0


# ── Snapshot ─────────────────────────────────────────────────


def test_snapshot_is_deterministic() -> None:
    registry = FeatureRegistry()
    for name, version in (("vwap", "2.0.0"), ("atr", "1.0.0")):

        @registry.feature(name, version, lookback=1, description=name)
        def _(bars: Sequence[Bar]) -> float:
            return 0.0

    assert registry.snapshot() == ("atr@1.0.0", "vwap@2.0.0")


def test_membership_and_length() -> None:
    registry = FeatureRegistry()

    @registry.feature("atr", "1.0.0", lookback=1, description="atr")
    def atr(bars: Sequence[Bar]) -> float:
        return 0.0

    assert "atr" in registry
    assert "nope" not in registry
    assert len(registry) == 1
    assert registry.names() == ("atr",)
