"""Intraday momentum: trade a move that has left the day's noise area.

Most of these are refusals, for the reason the gap-continuation suite gives: a
strategy that fires where it should not is far more expensive than one that
misses a trade. The boundary arithmetic is pinned with exact prices rather than
approximations, because the published rule is an exact formula and a band that
is quietly 1% wide instead of 0.5% is a different strategy with the same name.
"""

from __future__ import annotations

from datetime import date

import pytest

from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.features.levels import SessionLevels
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, StrategyContext
from neurotrade.strategies.intraday_momentum import (
    NO_TARGET_R,
    SWEEP_VOLATILITY_MULTIPLIERS,
    IntradayMomentum,
    noise_bounds,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)

MINUTE = 60_000_000_000
CHECKPOINT = 30 * MINUTE
CLOSE_NS = 390 * MINUTE
JULY_8 = date(2024, 7, 8)
JULY_9 = date(2024, 7, 9)


def bar(close: str = "102", ts: int = CHECKPOINT, symbol: Symbol = AAPL) -> Bar:
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
    session_open: str = "100",
    prior_close: str | None = "100",
    mean_move: float | None = 0.01,
    close: str = "102",
    day: date = JULY_8,
) -> SessionLevels:
    """Levels whose default noise area is [99, 101] at a multiplier of 1."""
    return SessionLevels(
        session_date=day,
        open_ns=0,
        close_ns=CLOSE_NS,
        session_open=Price(session_open),
        high=Price("105"),
        low=Price("95"),
        close=Price(close),
        vwap=Price("100.5"),
        bar_count=30,
        prior_close=None if prior_close is None else Price(prior_close),
        mean_abs_move_from_open=mean_move,
    )


def context(
    *,
    session: MarketSession = MarketSession.REGULAR,
    symbol: Symbol = AAPL,
    **kwargs: object,
) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        as_of=CHECKPOINT,
        session=session,
        regime=Regime.TREND_UP,
        levels=levels(**kwargs),  # type: ignore[arg-type]
    )


# ── The noise area ──────────────────────────────────────────────────────


def test_the_boundaries_sit_a_multiplier_of_mean_moves_outside_the_anchors() -> None:
    upper, lower = noise_bounds(levels(), multiplier=1.0) or pytest.fail("no bounds")
    assert (str(upper), str(lower)) == ("101", "99")


def test_a_wider_multiplier_widens_the_area_proportionally() -> None:
    upper, lower = noise_bounds(levels(), multiplier=1.5) or pytest.fail("no bounds")
    assert (str(upper), str(lower)) == ("101.5", "98.5")


def test_a_gap_widens_the_area_rather_than_being_traded_twice() -> None:
    """After a gap down, the upper boundary is lifted by the size of the gap.

    The anchors are `max(open, prior_close)` and `min(open, prior_close)`, so an
    open of 98 under a prior close of 100 leaves the upper boundary measured from
    100 and the lower from 98 — the published gap adjustment.
    """
    upper, lower = noise_bounds(
        levels(session_open="98", prior_close="100"), multiplier=1.0
    ) or pytest.fail("no bounds")
    assert (str(upper), str(lower)) == ("101", "97.02")


def test_a_session_with_no_prior_close_anchors_on_the_open_alone() -> None:
    upper, lower = noise_bounds(levels(prior_close=None), multiplier=1.0) or pytest.fail(
        "no bounds"
    )
    assert (str(upper), str(lower)) == ("101", "99")


def test_a_cold_move_history_has_no_boundaries() -> None:
    assert noise_bounds(levels(mean_move=None), multiplier=1.0) is None


@pytest.mark.parametrize("half_width", [0.0, 1.0, 2.0])
def test_a_half_width_outside_the_unit_interval_has_no_boundaries(half_width: float) -> None:
    """A move of zero carries no information and one of 100% cannot be priced.

    The lower boundary would land at or below zero, which `Price` refuses — and
    a strategy that raises kills the run it is being measured in.
    """
    assert noise_bounds(levels(mean_move=half_width), multiplier=1.0) is None


# ── Firing ──────────────────────────────────────────────────────────────


def test_it_is_registered_in_the_arsenal() -> None:
    assert arsenal.get("intraday_momentum") is IntradayMomentum


def test_a_close_above_the_area_proposes_a_long_stopped_at_the_far_side() -> None:
    (intent,) = IntradayMomentum().on_bar(bar(close="102"), context())
    assert intent.side is Side.BUY
    assert str(intent.invalidation) == "99"
    assert intent.target_r == NO_TARGET_R
    assert intent.horizon_ns == CLOSE_NS - CHECKPOINT
    assert intent.strategy == "intraday_momentum"


def test_a_close_below_the_area_proposes_a_short_stopped_at_the_far_side() -> None:
    (intent,) = IntradayMomentum().on_bar(bar(close="98"), context())
    assert intent.side is Side.SELL
    assert str(intent.invalidation) == "101"


def test_the_rationale_names_the_area_the_close_left() -> None:
    (intent,) = IntradayMomentum().on_bar(bar(close="102"), context())
    assert "102" in intent.rationale
    assert "[99, 101]" in intent.rationale
    assert "VM 1" in intent.rationale


# ── Refusals ────────────────────────────────────────────────────────────


def test_a_close_inside_the_area_proposes_nothing() -> None:
    assert IntradayMomentum().on_bar(bar(close="100.5"), context()) == ()


@pytest.mark.parametrize("edge", ["101", "99"])
def test_a_close_exactly_on_a_boundary_proposes_nothing(edge: str) -> None:
    """Strictly outside, or the band would fire on the day it merely matched."""
    assert IntradayMomentum().on_bar(bar(close=edge), context()) == ()


@pytest.mark.parametrize("minute", [1, 15, 29, 31, 45, 389])
def test_nothing_fires_away_from_a_checkpoint(minute: int) -> None:
    assert IntradayMomentum().on_bar(bar(ts=minute * MINUTE), context()) == ()


@pytest.mark.parametrize("minute", [30, 60, 300])
def test_every_half_hour_of_the_session_is_a_checkpoint(minute: int) -> None:
    assert len(IntradayMomentum().on_bar(bar(ts=minute * MINUTE), context())) == 1


def test_the_session_open_itself_is_not_a_checkpoint() -> None:
    """Minute zero has no move from the open to compare with anything."""
    assert IntradayMomentum().on_bar(bar(ts=0), context()) == ()


def test_a_cold_move_history_proposes_nothing() -> None:
    assert IntradayMomentum().on_bar(bar(), context(mean_move=None)) == ()


def test_nothing_fires_at_or_after_the_bell() -> None:
    """The horizon would be zero or negative, which `Intent` refuses."""
    assert IntradayMomentum().on_bar(bar(ts=CLOSE_NS), context()) == ()


def test_nothing_fires_outside_regular_hours() -> None:
    assert IntradayMomentum().on_bar(bar(), context(session=MarketSession.CLOSED)) == ()


def test_nothing_fires_before_the_first_bar_of_a_session() -> None:
    bare = StrategyContext(
        symbol=AAPL, as_of=CHECKPOINT, session=MarketSession.REGULAR, regime=Regime.TREND_UP
    )
    assert IntradayMomentum().on_bar(bar(), bare) == ()


# ── One proposal per direction per session ──────────────────────────────


def test_the_same_direction_is_not_proposed_twice_in_one_session() -> None:
    strategy = IntradayMomentum()
    assert len(strategy.on_bar(bar(ts=CHECKPOINT), context())) == 1
    assert strategy.on_bar(bar(ts=2 * CHECKPOINT), context()) == ()


def test_a_crossover_to_the_opposite_boundary_flips_the_position() -> None:
    """New evidence, which the published rule acts on — unlike a second long."""
    strategy = IntradayMomentum()
    (first,) = strategy.on_bar(bar(close="102", ts=CHECKPOINT), context())
    (second,) = strategy.on_bar(bar(close="98", ts=2 * CHECKPOINT), context())
    assert (first.side, second.side) == (Side.BUY, Side.SELL)


def test_the_next_session_starts_clean() -> None:
    strategy = IntradayMomentum()
    assert len(strategy.on_bar(bar(), context())) == 1
    assert len(strategy.on_bar(bar(), context(day=JULY_9))) == 1


def test_state_is_kept_per_symbol() -> None:
    """One instance sees the whole universe, so AAPL must not silence MSFT."""
    strategy = IntradayMomentum()
    assert len(strategy.on_bar(bar(symbol=AAPL), context(symbol=AAPL))) == 1
    assert len(strategy.on_bar(bar(symbol=MSFT), context(symbol=MSFT))) == 1


# ── The declared search ─────────────────────────────────────────────────


def test_the_sweep_declares_one_trial_per_published_multiplier() -> None:
    variants = IntradayMomentum.sweep()
    assert [label for label, _ in variants] == ["VM 1", "VM 1.5"]
    assert [s.volatility_multiplier for _, s in variants] == list(SWEEP_VOLATILITY_MULTIPLIERS)


def test_the_gate_is_trend_and_volatility_never_chop() -> None:
    strategy = IntradayMomentum()
    assert strategy.is_eligible(Regime.TREND_UP)
    assert strategy.is_eligible(Regime.HIGH_VOLATILITY)
    assert not strategy.is_eligible(Regime.CHOP)
    assert not strategy.is_eligible(Regime.LIQUIDITY_LULL)
