"""ORB fade: trade a break of the opening range that fails back inside it.

The rule is two events in order, so the tests are mostly about the order: a bar
inside the range is not a failure unless price left first, and a strategy that
fired on the second condition alone would be a mean-reversion rule wearing a
breakout-failure name. The regime exclusion against a breakout family is pinned
here too, because that pairing is the reason `Strategy.regimes` exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

import pytest

from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.intent import Intent
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.features.levels import OPENING_WINDOWS, OpeningRange, SessionLevels
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, StrategyContext
from neurotrade.strategies.intraday_momentum import IntradayMomentum
from neurotrade.strategies.orb_fade import (
    MIN_EXCURSION_FRACTION,
    SWEEP_OPENING_MINUTES,
    Excursion,
    OrbFade,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)

MINUTE = 60_000_000_000
NOON = 120 * MINUTE
CLOSE_NS = 390 * MINUTE
JULY_8 = date(2024, 7, 8)
JULY_9 = date(2024, 7, 9)

# A 30-minute range of [99, 101]: width 2, so a real excursion is >= 0.5 beyond
# an edge at MIN_EXCURSION_FRACTION = 0.25.
RANGE = OpeningRange(high=Price("101"), low=Price("99"), bar_count=30)


def bar(
    *,
    high: str,
    low: str,
    close: str,
    ts: int = NOON,
    symbol: Symbol = AAPL,
) -> Bar:
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=Price(close),
        high=Price(high),
        low=Price(low),
        close=Price(close),
        volume=Quantity(1_000),
    )


def context(
    *,
    day: date = JULY_8,
    symbol: Symbol = AAPL,
    regime: Regime = Regime.CHOP,
    session: MarketSession = MarketSession.REGULAR,
    ranges: dict[int, OpeningRange] | None = None,
    ts: int = NOON,
) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        as_of=ts,
        session=session,
        regime=regime,
        levels=SessionLevels(
            session_date=day,
            open_ns=0,
            close_ns=CLOSE_NS,
            session_open=Price("100"),
            high=Price("102"),
            low=Price("98"),
            close=Price("100"),
            vwap=Price("100"),
            bar_count=120,
            prior_close=Price("100"),
            opening_ranges={30: RANGE} if ranges is None else ranges,
        ),
    )


def broke_up_then_back(
    strategy: OrbFade, *, day: date = JULY_8, symbol: Symbol = AAPL
) -> Sequence[Intent]:
    """Drive the two events the rule needs: an excursion, then a close inside."""
    strategy.on_bar(
        bar(high="102", low="100.5", close="101.8", symbol=symbol),
        context(day=day, symbol=symbol),
    )
    return strategy.on_bar(
        bar(high="101.2", low="99.8", close="100", ts=NOON + MINUTE, symbol=symbol),
        context(day=day, symbol=symbol, ts=NOON + MINUTE),
    )


# ── Registration and the declared search ────────────────────────────────


def test_the_strategy_is_in_the_arsenal() -> None:
    assert arsenal.get("orb_fade") is OrbFade


def test_the_sweep_is_one_trial_per_opening_window() -> None:
    """ORB at 15, 30 and 60 minutes is three hypotheses, not one knob."""
    sweep = OrbFade.sweep()
    assert [instance.opening_minutes for _, instance in sweep] == list(SWEEP_OPENING_MINUTES)


def test_the_five_minute_window_is_deliberately_not_swept() -> None:
    """Five one-minute bars at the open is mostly the auction's spread."""
    assert 5 in OPENING_WINDOWS
    assert 5 not in SWEEP_OPENING_MINUTES


def test_an_untracked_window_is_refused_at_construction_not_mid_run() -> None:
    """`levels.opening_range` raises, which would kill the measurement."""
    with pytest.raises(ValueError, match="opening range 7 is not tracked"):
        OrbFade(opening_minutes=7)


def test_the_regimes_are_the_exact_complement_of_a_breakout_family() -> None:
    """An ORB and an ORB fade must never be live together (`strategies/base.py`)."""
    assert set(OrbFade.regimes) == {Regime.CHOP, Regime.REVERSAL}
    assert not set(OrbFade.regimes) & set(IntradayMomentum.regimes)


# ── It fires, on the second event ───────────────────────────────────────


def test_a_failed_upward_break_is_shorted_back_toward_the_far_edge() -> None:
    (intent,) = broke_up_then_back(OrbFade())
    assert intent.side is Side.SELL
    assert intent.invalidation == Price("102")  # the excursion's extreme
    # Target is the range's opposite edge, so target_r is computed, not declared:
    # entry 100, stop 102 (risk 2), target 99 (reward 1).
    assert intent.target_r == Decimal("0.5")
    assert intent.horizon_ns == CLOSE_NS - (NOON + MINUTE)


def test_a_failed_downward_break_is_bought() -> None:
    strategy = OrbFade()
    strategy.on_bar(bar(high="99.5", low="98", close="98.2"), context())
    (intent,) = strategy.on_bar(
        bar(high="100.2", low="98.8", close="100", ts=NOON + MINUTE),
        context(ts=NOON + MINUTE),
    )
    assert intent.side is Side.BUY
    assert intent.invalidation == Price("98")


def test_an_excursion_is_read_off_the_bar_high_not_its_close() -> None:
    """A spike through the range that closed back inside within one minute still counts.

    Reading the close only would miss exactly the fastest failures the strategy
    exists to trade.
    """
    strategy = OrbFade()
    # One bar: high 102 is outside, close 100 is back inside.
    assert strategy.on_bar(bar(high="102", low="99.5", close="100"), context())


# ── The order of the two events is the rule ─────────────────────────────


def test_a_bar_inside_the_range_with_no_prior_excursion_is_not_a_failure() -> None:
    assert OrbFade().on_bar(bar(high="100.5", low="99.5", close="100"), context()) == ()


def test_a_break_that_has_not_come_back_is_not_yet_a_failure() -> None:
    assert OrbFade().on_bar(bar(high="102", low="100.5", close="101.8"), context()) == ()


def test_a_shallow_poke_through_the_edge_is_not_an_excursion() -> None:
    """The range's edge being the range's edge, not a breakout that failed."""
    strategy = OrbFade()
    strategy.on_bar(bar(high="101.1", low="100.5", close="101.05"), context())
    assert (
        strategy.on_bar(
            bar(high="100.8", low="99.8", close="100", ts=NOON + MINUTE),
            context(ts=NOON + MINUTE),
        )
        == ()
    )
    assert MIN_EXCURSION_FRACTION == 0.25


def test_an_outside_bar_names_no_failed_side() -> None:
    """Both edges left inside one minute; declined rather than tie-broken."""
    strategy = OrbFade()
    strategy.on_bar(bar(high="102", low="98", close="100"), context())
    assert (
        strategy.on_bar(
            bar(high="100.5", low="99.5", close="100", ts=NOON + MINUTE),
            context(ts=NOON + MINUTE),
        )
        == ()
    )


# ── Refusals ────────────────────────────────────────────────────────────


def test_nothing_fires_before_the_opening_range_has_completed() -> None:
    """A partial range is a different statistic wearing the same name."""
    assert OrbFade().on_bar(bar(high="102", low="99", close="100"), context(ranges={})) == ()


def test_nothing_fires_outside_a_regular_session() -> None:
    strategy = OrbFade()
    strategy.on_bar(bar(high="102", low="100.5", close="101.8"), context())
    assert (
        strategy.on_bar(
            bar(high="101.2", low="99.8", close="100", ts=NOON + MINUTE),
            context(ts=NOON + MINUTE, session=MarketSession.CLOSED),
        )
        == ()
    )


def test_an_entry_sitting_exactly_on_the_far_edge_has_no_reward_to_state() -> None:
    """`Intent` refuses a zero target, so the degenerate close is declined here."""
    strategy = OrbFade()
    strategy.on_bar(bar(high="102", low="100.5", close="101.8"), context())
    assert (
        strategy.on_bar(
            bar(high="101", low="99", close="99", ts=NOON + MINUTE),
            context(ts=NOON + MINUTE),
        )
        == ()
    )


# ── One per instrument per session ──────────────────────────────────────


def test_only_one_fade_per_instrument_per_session() -> None:
    strategy = OrbFade()
    assert broke_up_then_back(strategy)
    assert (
        strategy.on_bar(
            bar(high="101.3", low="99.8", close="100", ts=NOON + 2 * MINUTE),
            context(ts=NOON + 2 * MINUTE),
        )
        == ()
    )


def test_a_new_session_clears_yesterdays_excursion() -> None:
    strategy = OrbFade()
    assert broke_up_then_back(strategy)
    assert broke_up_then_back(strategy, day=JULY_9)


def test_state_is_keyed_by_symbol() -> None:
    strategy = OrbFade()
    assert broke_up_then_back(strategy)
    assert broke_up_then_back(strategy, symbol=MSFT)


# ── The excursion record ────────────────────────────────────────────────


def test_an_excursion_knows_which_way_a_fade_of_it_trades() -> None:
    upward = Excursion(session_date=JULY_8, side=Side.BUY, extreme=Price("102"))
    assert upward.faded_side is Side.SELL
    assert (
        Excursion(session_date=JULY_8, side=Side.SELL, extreme=Price("98")).faded_side is Side.BUY
    )
