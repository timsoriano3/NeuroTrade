"""Prior-close pressure reversal — the overnight family, and its quarantine.

The refusals matter more here than anywhere else in the arsenal, because this is
the one strategy that carries risk past the bell. Three of them are structural
rather than about the rule: the instrument set, the calendar's next open, and the
engine that will not host it beside a day strategy.
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
from neurotrade.strategies.prior_close_reversal import (
    EXIT_MINUTES_AFTER_OPEN,
    OVERNIGHT_INSTRUMENTS,
    STOP_SIGMAS,
    SWEEP_ENTRY_MINUTES,
    TARGET_R,
    PriorCloseReversal,
)
from neurotrade.strategies.vwap_band_reversion import VwapBandReversion

SPY = Symbol("SPY", Venue.ARCA)
AAPL = Symbol("AAPL", Venue.NASDAQ)
XIU = Symbol("XIU", Venue.TSX)

MINUTE = 60_000_000_000
CLOSE_NS = 390 * MINUTE
NEXT_OPEN = CLOSE_NS + 1_050 * MINUTE  # 17.5 hours later
LATE = CLOSE_NS - 5 * MINUTE
JULY_8 = date(2024, 7, 8)
JULY_9 = date(2024, 7, 9)


def bar(close: str = "103", ts: int = LATE, symbol: Symbol = SPY) -> Bar:
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


def context(
    *,
    symbol: Symbol = SPY,
    vwap: str | None = "100",
    sigma: float | None = 0.01,
    next_open: int | None = NEXT_OPEN,
    bar_count: int = 380,
    session: MarketSession = MarketSession.REGULAR,
    regime: Regime = Regime.CHOP,
    day: date = JULY_8,
    ts: int = LATE,
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
            high=Price("104"),
            low=Price("99"),
            close=Price("103"),
            vwap=None if vwap is None else Price(vwap),
            bar_count=bar_count,
            prior_close=Price("100"),
            vwap_sigma=sigma,
            next_open_ns=next_open,
        ),
    )


# ── Registration and the declared search ────────────────────────────────


def test_the_strategy_is_in_the_arsenal() -> None:
    assert arsenal.get("prior_close_reversal") is PriorCloseReversal


def test_it_is_the_only_strategy_that_holds_overnight() -> None:
    """The quarantine is declared, and nothing else in the arsenal declares it."""
    overnight = [name for name in arsenal.names() if arsenal.get(name).holds_overnight]
    assert overnight == ["prior_close_reversal"]


def test_the_sweep_is_one_trial_per_entry_window() -> None:
    sweep = PriorCloseReversal.sweep()
    assert [instance.entry_minutes_before_close for _, instance in sweep] == list(
        SWEEP_ENTRY_MINUTES
    )


def test_the_band_is_not_swept_here_because_the_intraday_fade_already_sweeps_it() -> None:
    """Searching one axis against two families deflates it twice."""
    bands = {instance.band_sigmas for _, instance in PriorCloseReversal.sweep()}
    assert len(bands) == 1
    assert len({instance.band_sigmas for _, instance in VwapBandReversion.sweep()}) == 3


def test_the_regimes_are_reversion_only() -> None:
    """A close pushed to the high of a trend day is continuation, not pressure."""
    assert set(PriorCloseReversal.regimes) == {Regime.CHOP, Regime.REVERSAL}


# ── It fires ────────────────────────────────────────────────────────────


def test_a_close_pushed_above_vwap_is_sold_into_the_next_open() -> None:
    (intent,) = PriorCloseReversal().on_bar(bar(), context())
    assert intent.side is Side.SELL
    assert intent.invalidation > Price("103")
    assert intent.target_r == TARGET_R
    # The exit is the calendar's next open plus the published 30 minutes, not a
    # guessed span — and the offset is what makes the labeller find a bar there.
    assert intent.ts_event + intent.horizon_ns == NEXT_OPEN + EXIT_MINUTES_AFTER_OPEN * MINUTE


def test_a_close_pushed_below_vwap_is_bought() -> None:
    (intent,) = PriorCloseReversal().on_bar(bar(close="97"), context())
    assert intent.side is Side.BUY
    assert intent.invalidation < Price("97")


def test_the_stop_is_far_enough_out_that_the_time_barrier_resolves_the_trade() -> None:
    """The labeller would record a gapped stop as filling at the next bar's low.

    It would not have; it would have filled at the open. Placing the barriers out
    of reach makes the position resolve on the time barrier, which is the quantity
    the literature reports.
    """
    (intent,) = PriorCloseReversal().on_bar(bar(), context())
    # Six sigmas of a 1% dispersion is 6% above a close of 103.
    assert intent.invalidation.value == pytest.approx(Decimal("103") * Decimal("1.06"), abs=0.01)
    assert STOP_SIGMAS == 6.0


def test_a_canadian_index_proxy_is_in_the_family() -> None:
    (intent,) = PriorCloseReversal().on_bar(bar(symbol=XIU), context(symbol=XIU))
    assert intent.symbol == XIU


# ── The structural refusals ─────────────────────────────────────────────


def test_a_single_name_is_never_held_overnight() -> None:
    """A name can gap 20% on an earnings release; an index cannot."""
    assert "AAPL" not in OVERNIGHT_INSTRUMENTS
    assert PriorCloseReversal().on_bar(bar(symbol=AAPL), context(symbol=AAPL)) == ()


def test_an_emerging_markets_basket_is_deliberately_excluded() -> None:
    """EEM's overnight move is other markets' regular session, not a reversal."""
    assert "EEM" not in OVERNIGHT_INSTRUMENTS


def test_without_a_next_open_there_is_no_exit_to_state() -> None:
    """The end of the calendar. Declining beats guessing a span."""
    assert PriorCloseReversal().on_bar(bar(), context(next_open=None)) == ()


def test_a_next_open_that_is_not_in_the_future_is_refused() -> None:
    assert PriorCloseReversal().on_bar(bar(), context(next_open=LATE - MINUTE)) == ()


# ── The rule's own refusals ─────────────────────────────────────────────


def test_nothing_fires_earlier_in_the_session() -> None:
    """A deviation at 15:00 has most of an hour to resolve inside the session."""
    assert (
        PriorCloseReversal().on_bar(
            bar(ts=CLOSE_NS - 60 * MINUTE), context(ts=CLOSE_NS - 60 * MINUTE)
        )
        == ()
    )


def test_nothing_fires_on_the_closing_bar_itself() -> None:
    """The last print; there is nothing left to enter on."""
    assert PriorCloseReversal().on_bar(bar(ts=CLOSE_NS), context(ts=CLOSE_NS)) == ()


def test_a_close_inside_the_band_is_not_a_push() -> None:
    assert PriorCloseReversal().on_bar(bar(close="100.5"), context()) == ()


@pytest.mark.parametrize(("vwap", "sigma"), [(None, 0.01), ("100", None), ("100", 0.0)])
def test_an_unusable_dispersion_declines(vwap: str | None, sigma: float | None) -> None:
    assert PriorCloseReversal().on_bar(bar(), context(vwap=vwap, sigma=sigma)) == ()


def test_a_thin_session_has_no_dispersion_to_quote() -> None:
    assert PriorCloseReversal().on_bar(bar(), context(bar_count=4)) == ()


def test_nothing_fires_outside_a_regular_session() -> None:
    assert PriorCloseReversal().on_bar(bar(), context(session=MarketSession.CLOSED)) == ()


def test_only_one_proposal_per_instrument_per_session() -> None:
    strategy = PriorCloseReversal()
    assert strategy.on_bar(bar(), context())
    assert strategy.on_bar(bar(ts=LATE + MINUTE), context(ts=LATE + MINUTE)) == ()


def test_a_new_session_starts_clean() -> None:
    strategy = PriorCloseReversal()
    assert strategy.on_bar(bar(), context())
    assert strategy.on_bar(bar(ts=LATE + MINUTE), context(ts=LATE + MINUTE, day=JULY_9))


def test_the_horizon_reaches_past_the_next_open_so_a_bar_exists_there() -> None:
    """`next_open_ns` is the open *instant*; the first bar of that session closes later.

    `signals_from_intents` counts bars with `bisect_right(timestamps, deadline)`,
    so a deadline exactly at the open finds no bar of the next session and the
    time barrier collapses back to tonight's bell — measuring the last minutes of
    today instead of the overnight. The offset is load-bearing.
    """
    (intent,) = PriorCloseReversal().on_bar(bar(), context())
    deadline = intent.ts_event + intent.horizon_ns
    assert deadline > NEXT_OPEN
    # A minute would be enough to fix the barrier; thirty is where the published
    # number is, so the two reasons agree rather than one overriding the other.
    assert EXIT_MINUTES_AFTER_OPEN == 30
