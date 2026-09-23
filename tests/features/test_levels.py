"""Tests for the session-anchored reference levels.

The rejection cases carry the weight. A partial opening range and a real one
have the same type and the same fields, so the only thing stopping a strategy
trading a level nobody else is watching is the refusal to build one.
"""

from __future__ import annotations

from datetime import date

import pytest

from neurotrade.core.calendar import TradingSession
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.features.levels import (
    OpeningRange,
    SessionLevelTracker,
    opening_range,
    session_vwap,
    vwap_distance,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MINUTE = 60_000_000_000


def bar(
    index: int,
    close: str,
    *,
    high: str | None = None,
    low: str | None = None,
    volume: str = "100",
    vwap: str | None = None,
) -> Bar:
    price = Price(close)
    ts = index * MINUTE
    return Bar(
        symbol=AAPL,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=price,
        high=Price(high) if high else price,
        low=Price(low) if low else price,
        close=price,
        volume=Quantity(volume),
        vwap=Price(vwap) if vwap else None,
    )


# ── OpeningRange ─────────────────────────────────────────────────────────────


def test_an_inverted_range_is_refused() -> None:
    """Every breakout test would invert silently."""
    with pytest.raises(ValueError, match="is below low"):
        OpeningRange(high=Price("99"), low=Price("101"), bar_count=5)


def test_a_trade_at_the_high_is_not_a_breakout() -> None:
    """Otherwise the signal fires on the bar that set the level."""
    rng = OpeningRange(high=Price("101"), low=Price("99"), bar_count=5)
    assert not rng.breaks_up(Price("101"))
    assert rng.breaks_up(Price("101.01"))


def test_a_trade_at_the_low_is_not_a_breakdown() -> None:
    rng = OpeningRange(high=Price("101"), low=Price("99"), bar_count=5)
    assert not rng.breaks_down(Price("99"))
    assert rng.breaks_down(Price("98.99"))


def test_width_is_the_span_of_the_range() -> None:
    rng = OpeningRange(high=Price("101.5"), low=Price("99"), bar_count=5)
    assert str(rng.width) == "2.5"


# ── opening_range ────────────────────────────────────────────────────────────


def test_the_range_spans_the_whole_opening_window() -> None:
    bars = [bar(0, "100", high="101", low="99"), bar(1, "100", high="102", low="98")]
    found = opening_range(bars, minutes=2)
    assert found is not None
    assert (str(found.high), str(found.low)) == ("102", "98")


def test_bars_after_the_window_do_not_widen_the_range() -> None:
    """The level is the first N minutes, not the session so far."""
    bars = [bar(0, "100", high="101", low="99"), bar(1, "100", high="500", low="1")]
    found = opening_range(bars, minutes=1)
    assert found is not None
    assert (str(found.high), str(found.low)) == ("101", "99")


def test_a_short_session_yields_no_range_rather_than_a_partial_one() -> None:
    """A 15-minute range from 4 bars is a different statistic with the same name."""
    assert opening_range([bar(0, "100")], minutes=15) is None


def test_an_empty_session_yields_no_range() -> None:
    assert opening_range([], minutes=5) is None


@pytest.mark.parametrize("minutes", [0, -1])
def test_a_non_positive_window_is_refused(minutes: int) -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        opening_range([bar(0, "100")], minutes=minutes)


# ── session_vwap ─────────────────────────────────────────────────────────────


def test_vwap_weights_by_volume_not_by_bar() -> None:
    """The whole point: a big print moves it more than a small one."""
    assert str(session_vwap([bar(0, "10", volume="100"), bar(1, "20", volume="300")])) == "17.5"


def test_vwap_prefers_the_feeds_own_figure_when_present() -> None:
    """It is computed from every print inside the bar, so it is strictly better."""
    with_vwap = session_vwap([bar(0, "10", high="12", low="8", volume="100", vwap="11")])
    assert str(with_vwap) == "11"


def test_vwap_falls_back_to_the_typical_price() -> None:
    assert str(session_vwap([bar(0, "9", high="12", low="6", volume="100")])) == "9"


def test_vwap_ignores_bars_after_the_point_in_time_bound() -> None:
    """The one lookahead guard these functions have; the caller must pass it."""
    bars = [bar(0, "10", volume="100"), bar(1, "1000", volume="100")]
    assert str(session_vwap(bars, up_to=0)) == "10"


def test_zero_volume_bars_do_not_contribute() -> None:
    bars = [bar(0, "10", volume="100"), bar(1, "1000", volume="0")]
    assert str(session_vwap(bars)) == "10"


def test_a_session_with_no_volume_has_no_vwap() -> None:
    """None rather than zero: zero would read as a price."""
    assert session_vwap([bar(0, "10", volume="0")]) is None


def test_an_empty_session_has_no_vwap() -> None:
    assert session_vwap([]) is None


# ── vwap_distance ────────────────────────────────────────────────────────────


def test_distance_is_a_fraction_so_it_compares_across_instruments() -> None:
    cheap = vwap_distance(Price("8.16"), Price("8"))
    dear = vwap_distance(Price("408"), Price("400"))
    assert cheap == pytest.approx(dear)


def test_distance_is_negative_below_vwap() -> None:
    assert vwap_distance(Price("98"), Price("100")) == pytest.approx(-0.02)


def test_a_zero_vwap_cannot_be_constructed_at_all() -> None:
    """Why `vwap_distance` carries no zero-division guard: it is unreachable."""
    with pytest.raises(ValueError, match="must be positive"):
        Price("0")


def test_vwap_is_quantized_to_a_storable_price() -> None:
    """The raw division runs to context precision; a Price is an order price.

    A real session produced `330.2556630065255193527176013`, which
    `decimal128(18, 8)` refuses outright rather than rounding.
    """
    bars = [bar(0, "10", volume="3"), bar(1, "20", volume="7")]
    found = session_vwap(bars)
    assert found is not None
    assert -found.value.as_tuple().exponent <= 8  # type: ignore[operator]


# ── SessionLevelTracker ──────────────────────────────────────────────────────

MSFT = Symbol("MSFT", Venue.NASDAQ)


def session(day: int, *, first_minute: int, minutes: int = 390) -> TradingSession:
    """A session opening at `first_minute - 1` so bar `first_minute` is its first."""
    open_ns = (first_minute - 1) * MINUTE
    return TradingSession(
        venue=Venue.NASDAQ,
        session_date=date(2024, 7, day),
        open_ns=open_ns,
        close_ns=open_ns + minutes * MINUTE,
        is_early_close=False,
    )


def fold(tracker: SessionLevelTracker, bars: list[Bar], day: TradingSession) -> None:
    for one in bars:
        tracker.update(one, day)


def test_nothing_is_reported_before_the_first_bar() -> None:
    assert SessionLevelTracker().levels(AAPL) is None


def test_a_bar_outside_any_session_is_not_folded_in() -> None:
    """An RTH corpus has none; one would put after-hours prints in a session VWAP."""
    tracker = SessionLevelTracker()
    tracker.update(bar(1, "100"), None)
    assert tracker.levels(AAPL) is None


def test_the_open_is_the_first_bars_open_not_its_close() -> None:
    tracker = SessionLevelTracker()
    fold(tracker, [bar(1, "101", high="102", low="99"), bar(2, "103")], session(8, first_minute=1))
    levels = tracker.levels(AAPL)
    assert levels is not None
    assert str(levels.session_open) == "101"


def test_high_and_low_run_across_the_session() -> None:
    tracker = SessionLevelTracker()
    fold(
        tracker,
        [bar(1, "100", high="101", low="99"), bar(2, "100", high="105", low="95")],
        session(8, first_minute=1),
    )
    levels = tracker.levels(AAPL)
    assert levels is not None
    assert (str(levels.high), str(levels.low)) == ("105", "95")


def test_the_running_vwap_agrees_with_the_batch_function() -> None:
    """One implementation: the streaming fold and `session_vwap` cannot diverge."""
    bars = [bar(i, str(100 + i), volume=str(100 * i)) for i in range(1, 8)]
    tracker = SessionLevelTracker()
    fold(tracker, bars, session(8, first_minute=1))
    levels = tracker.levels(AAPL)
    assert levels is not None
    assert levels.vwap == session_vwap(bars)


def test_a_zero_volume_bar_does_not_move_the_vwap() -> None:
    tracker = SessionLevelTracker()
    fold(tracker, [bar(1, "100"), bar(2, "200", volume="0")], session(8, first_minute=1))
    levels = tracker.levels(AAPL)
    assert levels is not None
    assert str(levels.vwap) == "100"


def test_an_opening_range_appears_on_the_bar_that_completes_it() -> None:
    tracker = SessionLevelTracker()
    day = session(8, first_minute=1)
    bars = [bar(i, "100", high=str(100 + i), low=str(100 - i)) for i in range(1, 6)]
    for index, one in enumerate(bars, start=1):
        tracker.update(one, day)
        levels = tracker.levels(AAPL)
        assert levels is not None
        assert (levels.opening_range(5) is None) == (index < 5)
    final = tracker.levels(AAPL)
    assert final is not None
    assert final.opening_range(5) == OpeningRange(high=Price("105"), low=Price("95"), bar_count=5)


def test_an_untracked_opening_window_is_refused() -> None:
    """Each window is a separate trial; a silent None would hide a typo."""
    tracker = SessionLevelTracker()
    fold(tracker, [bar(1, "100")], session(8, first_minute=1))
    levels = tracker.levels(AAPL)
    assert levels is not None
    with pytest.raises(ValueError, match=r"opening range 7 is not tracked"):
        levels.opening_range(7)


def test_a_new_session_resets_everything_but_the_prior_close() -> None:
    tracker = SessionLevelTracker()
    fold(tracker, [bar(1, "100", high="110", low="90")], session(8, first_minute=1))
    fold(tracker, [bar(400, "50", high="51", low="49")], session(9, first_minute=400))
    levels = tracker.levels(AAPL)
    assert levels is not None
    assert (str(levels.high), str(levels.low), str(levels.prior_close)) == ("51", "49", "100")


def test_the_first_session_seen_has_no_prior_close() -> None:
    tracker = SessionLevelTracker()
    fold(tracker, [bar(1, "100")], session(8, first_minute=1))
    levels = tracker.levels(AAPL)
    assert levels is not None
    assert (levels.prior_close, levels.gap) == (None, None)


def test_the_gap_is_signed_and_relative() -> None:
    tracker = SessionLevelTracker()
    fold(tracker, [bar(1, "100")], session(8, first_minute=1))
    fold(tracker, [bar(400, "99", high="99", low="99")], session(9, first_minute=400))
    levels = tracker.levels(AAPL)
    assert levels is not None
    assert levels.gap == pytest.approx(-0.01)


def test_symbols_do_not_share_levels() -> None:
    tracker = SessionLevelTracker()
    day = session(8, first_minute=1)
    tracker.update(bar(1, "100"), day)
    other = Bar(
        symbol=MSFT,
        ts_event=MINUTE,
        ts_init=MINUTE,
        interval=BarInterval.MIN_1,
        open=Price("200"),
        high=Price("200"),
        low=Price("200"),
        close=Price("200"),
        volume=Quantity(100),
    )
    tracker.update(other, day)
    aapl, msft = tracker.levels(AAPL), tracker.levels(MSFT)
    assert aapl is not None and msft is not None
    assert (str(aapl.close), str(msft.close)) == ("100", "200")
