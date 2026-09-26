"""The first strategy: trade a gap too wide to expect a fill.

Most of these are refusals. A strategy that fires where it should not is far
more expensive than one that misses a trade, and every guard below corresponds
to a way the proposal would be wrong rather than merely unprofitable.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.features.levels import SessionLevels
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, StrategyContext
from neurotrade.strategies.gap_continuation import GapContinuation

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)

MINUTE = 60_000_000_000
CLOSE_NS = 390 * MINUTE
JULY_8 = date(2024, 7, 8)


def bar(close: str = "104", ts: int = MINUTE, symbol: Symbol = AAPL) -> Bar:
    price = Price(close)
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Quantity(1_000),
    )


def levels(
    *,
    session_open: str = "103",
    prior_close: str | None = "100",
    range_mean: str | None = "2",
    bar_count: int = 1,
    day: date = JULY_8,
) -> SessionLevels:
    """Levels whose default gap is +1.5 session ranges — inside the band."""
    return SessionLevels(
        session_date=day,
        open_ns=0,
        close_ns=CLOSE_NS,
        session_open=Price(session_open),
        high=Price("105"),
        low=Price("99"),
        close=Price("104"),
        vwap=Price("103"),
        bar_count=bar_count,
        prior_close=Price(prior_close) if prior_close is not None else None,
        prior_range_mean=Decimal(range_mean) if range_mean is not None else None,
    )


def context(
    found: SessionLevels | None = None,
    *,
    session: MarketSession = MarketSession.REGULAR,
    symbol: Symbol = AAPL,
    as_of: int = MINUTE,
) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        as_of=as_of,
        session=session,
        regime=Regime.TREND_UP,
        levels=found if found is not None else levels(),
    )


# ── It fires ─────────────────────────────────────────────────


def test_a_wide_gap_up_proposes_a_long() -> None:
    (intent,) = GapContinuation().on_bar(bar(), context())
    assert intent.side is Side.BUY


def test_a_wide_gap_down_proposes_a_short() -> None:
    down = levels(session_open="97")
    (intent,) = GapContinuation().on_bar(bar("96"), context(down))
    assert intent.side is Side.SELL


def test_the_session_open_is_what_proves_it_wrong() -> None:
    """Price back through the open means the fill is under way."""
    (intent,) = GapContinuation().on_bar(bar(), context())
    assert intent.invalidation == Price("103")


def test_the_position_is_flat_by_the_bell() -> None:
    (intent,) = GapContinuation().on_bar(bar(), context())
    assert intent.ts_event + intent.horizon_ns == CLOSE_NS


def test_the_rationale_carries_the_gap_that_caused_it() -> None:
    (intent,) = GapContinuation().on_bar(bar(), context())
    assert "+1.50" in intent.rationale


def test_the_target_is_two_r() -> None:
    (intent,) = GapContinuation().on_bar(bar(), context())
    assert intent.target_r == Decimal(2)


def test_it_is_registered_in_the_arsenal() -> None:
    assert arsenal.get("gap_continuation") is GapContinuation


# ── It does not fire ─────────────────────────────────────────


def test_nothing_without_levels() -> None:
    """Before the session's first bar there is no open to gap from."""
    bare = StrategyContext(
        symbol=AAPL, as_of=MINUTE, session=MarketSession.REGULAR, regime=Regime.TREND_UP
    )
    assert GapContinuation().on_bar(bar(), bare) == ()


def test_nothing_outside_the_regular_session() -> None:
    assert GapContinuation().on_bar(bar(), context(session=MarketSession.CLOSED)) == ()


@pytest.mark.parametrize("open_at", ["100.5", "101", "102.3"])
def test_a_narrow_gap_is_left_alone(open_at: str) -> None:
    """Under the band is the fade trade, which the sweep rejected as a primary."""
    assert GapContinuation().on_bar(bar(), context(levels(session_open=open_at))) == ()


def test_nothing_without_a_prior_close() -> None:
    """The first session the tracker sees has nothing to gap from."""
    assert GapContinuation().on_bar(bar(), context(levels(prior_close=None))) == ()


def test_nothing_until_the_range_history_is_full() -> None:
    """A gap quoted against a partial history is not the statistic thresholded."""
    assert GapContinuation().on_bar(bar(), context(levels(range_mean=None))) == ()


def test_nothing_once_the_open_has_been_lost() -> None:
    assert GapContinuation().on_bar(bar("102.5"), context()) == ()


def test_a_touch_of_the_open_is_not_holding_it() -> None:
    """Equal is not above: the stop would sit on the entry, and `Intent` refuses it."""
    assert GapContinuation().on_bar(bar("103"), context()) == ()


def test_nothing_after_the_entry_window_closes() -> None:
    late = levels(bar_count=GapContinuation.entry_window_bars + 1)
    assert GapContinuation().on_bar(bar(), context(late)) == ()


def test_nothing_at_or_past_the_closing_bell() -> None:
    """A zero or negative horizon is not a proposal `Intent` will accept."""
    assert GapContinuation().on_bar(bar(ts=CLOSE_NS), context(as_of=CLOSE_NS)) == ()


# ── One trade per instrument per session ─────────────────────


def test_the_same_session_is_traded_once() -> None:
    strategy = GapContinuation()
    first = strategy.on_bar(bar(), context())
    second = strategy.on_bar(bar(ts=2 * MINUTE), context(levels(bar_count=2)))
    assert (len(first), second) == (1, ())


def test_the_next_session_may_trade_again() -> None:
    strategy = GapContinuation()
    strategy.on_bar(bar(), context())
    tomorrow = levels(day=date(2024, 7, 9))
    assert len(strategy.on_bar(bar(), context(tomorrow))) == 1


def test_one_instrument_being_traded_does_not_silence_another() -> None:
    strategy = GapContinuation()
    strategy.on_bar(bar(), context())
    other = strategy.on_bar(bar(symbol=MSFT), context(symbol=MSFT))
    assert len(other) == 1
