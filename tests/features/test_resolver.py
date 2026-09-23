"""The rolling window features are computed from.

The properties that matter are the ones a wrong number would not announce: that
one symbol's bars never enter another's window, that trimming keeps the tail a
feature actually reads, and that a duplicate or backwards bar is refused rather
than folded in.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.features.registry import FeatureRegistry, LookaheadError
from neurotrade.features.resolver import FeatureResolver

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)

MINUTE = 60_000_000_000


def bar(
    symbol: Symbol, minute: int, close: str = "100", interval: BarInterval = BarInterval.MIN_1
) -> Bar:
    price = Price(close)
    return Bar(
        symbol=symbol,
        ts_event=minute * MINUTE,
        ts_init=minute * MINUTE,
        interval=interval,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Quantity(1_000),
    )


def library() -> FeatureRegistry:
    """Two features with different lookbacks, and two versions of one of them."""
    registry = FeatureRegistry()

    @registry.feature("last", "1.0.0", lookback=1, description="the last close")
    def last(bars: Sequence[Bar]) -> float:
        return float(bars[-1].close.value)

    @registry.feature("last", "2.0.0", lookback=1, description="the last close, doubled")
    def last_doubled(bars: Sequence[Bar]) -> float:
        return 2 * float(bars[-1].close.value)

    @registry.feature("mean3", "1.0.0", lookback=3, description="mean of three closes")
    def mean3(bars: Sequence[Bar]) -> float:
        return sum(float(b.close.value) for b in bars) / len(bars)

    @registry.feature("daily", "1.0.0", lookback=2, description="daily", interval=BarInterval.DAY_1)
    def daily(bars: Sequence[Bar]) -> float:
        return 0.0

    return registry


def resolver_over(*labels: str) -> FeatureResolver:
    registry = library()
    specs = {}
    for label in labels:
        name, _, version = label.partition("@")
        specs[label] = registry.get(name, version or None)
    return FeatureResolver(specs, interval=BarInterval.MIN_1)


# ── Declaration ──────────────────────────────────────────────


def test_a_feature_for_another_bar_size_is_refused() -> None:
    """A daily feature fed minute bars returns a plausible number, not an error."""
    registry = library()
    with pytest.raises(ValueError, match=r"declared for 1m bars: daily \(1d\)"):
        FeatureResolver({"daily": registry.get("daily")}, interval=BarInterval.MIN_1)


def test_lookback_is_the_widest_declared() -> None:
    assert resolver_over("last", "mean3").lookback == 3


def test_an_empty_resolver_needs_no_history() -> None:
    resolver = FeatureResolver({}, interval=BarInterval.MIN_1)
    resolver.observe(bar(AAPL, 1))
    assert (resolver.lookback, resolver.resolve(bar(AAPL, 1))) == (0, {})


def test_resolution_order_follows_the_label_not_the_declaration() -> None:
    """Iteration order must be a property of the declaration, not of registration."""
    resolver = resolver_over("mean3", "last")
    resolver.observe(bar(AAPL, 1))
    assert list(resolver.resolve(bar(AAPL, 1))) == ["last", "mean3"]


# ── Warm-up ──────────────────────────────────────────────────


def test_a_feature_is_none_until_its_window_is_full() -> None:
    resolver = resolver_over("mean3")
    for minute in (1, 2):
        resolver.observe(bar(AAPL, minute))
    assert resolver.resolve(bar(AAPL, 2))["mean3"] is None


def test_a_warm_feature_produces_a_value() -> None:
    resolver = resolver_over("mean3")
    for minute, close in ((1, "100"), (2, "101"), (3, "102")):
        resolver.observe(bar(AAPL, minute, close))
    assert resolver.resolve(bar(AAPL, 3))["mean3"] == pytest.approx(101.0)


# ── History ──────────────────────────────────────────────────


def test_one_symbols_bars_never_enter_anothers_window() -> None:
    resolver = resolver_over("mean3")
    for minute in (1, 2, 3):
        resolver.observe(bar(AAPL, minute))
    resolver.observe(bar(MSFT, 4))
    assert resolver.resolve(bar(MSFT, 4))["mean3"] is None


def test_trimming_keeps_the_tail_a_feature_reads() -> None:
    """The buffer is trimmed in batches; the last `lookback` bars must survive."""
    resolver = resolver_over("mean3")
    for minute in range(1, 41):
        resolver.observe(bar(AAPL, minute, str(100 + minute)))
    # Last three closes are 138, 139, 140.
    assert resolver.resolve(bar(AAPL, 40))["mean3"] == pytest.approx(139.0)


def test_history_is_continuous_across_a_gap() -> None:
    """Warm at the open is the point: a session boundary does not reset it."""
    resolver = resolver_over("mean3")
    for minute in (1, 2, 1_000_000):
        resolver.observe(bar(AAPL, minute))
    assert resolver.resolve(bar(AAPL, 1_000_000))["mean3"] is not None


# ── Rejection ────────────────────────────────────────────────


def test_a_bar_of_the_wrong_interval_is_refused() -> None:
    resolver = resolver_over("last")
    with pytest.raises(ValueError, match=r"fed 1m bars, got 5m"):
        resolver.observe(bar(AAPL, 1, interval=BarInterval.MIN_5))


def test_bars_out_of_order_raise_rather_than_compute() -> None:
    resolver = resolver_over("mean3")
    for minute in (3, 2, 1):
        resolver.observe(bar(AAPL, minute))
    with pytest.raises(LookaheadError, match=r"out of order"):
        resolver.resolve(bar(AAPL, 3))


def test_a_value_may_not_be_read_from_the_future() -> None:
    resolver = resolver_over("last")
    resolver.observe(bar(AAPL, 10))
    with pytest.raises(LookaheadError, match=r"after the moment being modelled"):
        resolver.resolve(bar(AAPL, 9))


# ── Versions ─────────────────────────────────────────────────


def test_two_versions_of_one_feature_resolve_separately() -> None:
    """§10.2 runs a pinned champion beside a challenger tracking latest."""
    resolver = resolver_over("last@1.0.0", "last@2.0.0")
    resolver.observe(bar(AAPL, 1, "50"))
    values = resolver.resolve(bar(AAPL, 1))
    assert (values["last@1.0.0"], values["last@2.0.0"]) == (50.0, 100.0)
