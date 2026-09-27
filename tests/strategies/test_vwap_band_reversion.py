"""Fade a stretch away from VWAP that volume does not support.

Refusal-heavy for the reason the other two suites are: the session windows and
the dispersion floor are the strategy, not decoration, and a fade taken outside
them is a different hypothesis with the same name on it.
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
from neurotrade.strategies.vwap_band_reversion import (
    MIN_DISPERSION_BARS,
    SWEEP_BAND_SIGMAS,
    VwapBandReversion,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)

MINUTE = 60_000_000_000
CLOSE_NS = 390 * MINUTE
IN_OPENING_WINDOW = 60 * MINUTE
JULY_8 = date(2024, 7, 8)
JULY_9 = date(2024, 7, 9)


def bar(close: str = "102", ts: int = IN_OPENING_WINDOW, symbol: Symbol = AAPL) -> Bar:
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
    vwap: str | None = "100",
    sigma: float | None = 0.01,
    bar_count: int = MIN_DISPERSION_BARS,
    day: date = JULY_8,
) -> SessionLevels:
    """Levels whose 2-sigma band sits at 102 and 98, with the stop at 103 / 97."""
    return SessionLevels(
        session_date=day,
        open_ns=0,
        close_ns=CLOSE_NS,
        session_open=Price("100"),
        high=Price("103"),
        low=Price("97"),
        close=Price("102"),
        vwap=None if vwap is None else Price(vwap),
        bar_count=bar_count,
        prior_close=Price("100"),
        vwap_sigma=sigma,
    )


def context(
    *,
    session: MarketSession = MarketSession.REGULAR,
    symbol: Symbol = AAPL,
    **kwargs: object,
) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        as_of=IN_OPENING_WINDOW,
        session=session,
        regime=Regime.CHOP,
        levels=levels(**kwargs),  # type: ignore[arg-type]
    )


# ── Firing ──────────────────────────────────────────────────────────────


def test_it_is_registered_in_the_arsenal() -> None:
    assert arsenal.get("vwap_band_reversion") is VwapBandReversion


def test_a_stretch_above_the_band_is_sold_with_the_stop_a_sigma_beyond() -> None:
    (intent,) = VwapBandReversion().on_bar(bar(close="102"), context())
    assert intent.side is Side.SELL
    assert str(intent.invalidation) == "103"
    assert intent.target_r == Decimal("2.0")
    assert intent.horizon_ns == CLOSE_NS - IN_OPENING_WINDOW


def test_a_stretch_below_the_band_is_bought_with_the_stop_a_sigma_beyond() -> None:
    (intent,) = VwapBandReversion().on_bar(bar(close="98"), context())
    assert intent.side is Side.BUY
    assert str(intent.invalidation) == "97"


def test_the_target_is_vwap_itself_when_the_fill_is_at_the_band() -> None:
    """R is one dispersion and VWAP is `band_sigmas` of them away, so the ratio
    is the band width rather than a number chosen to look good."""
    (intent,) = VwapBandReversion().on_bar(bar(close="102"), context())
    # "100.0" rather than "100": the scale comes from `target_r`, which is the
    # band width as a decimal, and `Price` keeps what it is handed.
    assert str(intent.target_price(Price("102"))) == "100.0"


def test_a_wider_band_moves_both_barriers_out() -> None:
    (intent,) = VwapBandReversion(band_sigmas=2.5).on_bar(bar(close="102.5"), context())
    assert str(intent.invalidation) == "103.5"
    assert intent.target_r == Decimal("2.5")


def test_the_rationale_names_the_stretch_in_sigmas() -> None:
    (intent,) = VwapBandReversion().on_bar(bar(close="102"), context())
    assert "+2.00 session sigmas" in intent.rationale


def test_the_closing_window_is_open_for_business() -> None:
    """§5.3 puts the effect in the first 90 minutes and the final 60."""
    assert len(VwapBandReversion().on_bar(bar(ts=330 * MINUTE), context())) == 1


@pytest.mark.parametrize("minute", [30, 90])
def test_the_opening_window_runs_to_its_ninetieth_minute(minute: int) -> None:
    assert len(VwapBandReversion().on_bar(bar(ts=minute * MINUTE), context())) == 1


# ── Refusals ────────────────────────────────────────────────────────────


def test_a_price_inside_the_band_is_left_alone() -> None:
    assert VwapBandReversion().on_bar(bar(close="101.99"), context()) == ()


def test_the_midday_stretch_is_not_traded() -> None:
    """Neither window holds it, and the lull is a no-trade regime besides."""
    assert VwapBandReversion().on_bar(bar(ts=200 * MINUTE), context()) == ()


def test_a_dispersion_measured_on_too_few_bars_is_not_a_dispersion() -> None:
    assert VwapBandReversion().on_bar(bar(), context(bar_count=MIN_DISPERSION_BARS - 1)) == ()


def test_a_session_with_no_vwap_proposes_nothing() -> None:
    assert VwapBandReversion().on_bar(bar(), context(vwap=None, sigma=None)) == ()


@pytest.mark.parametrize("sigma", [None, 0.0])
def test_a_session_with_no_dispersion_proposes_nothing(sigma: float | None) -> None:
    """Every print at one price makes every deviation infinite in band units."""
    assert VwapBandReversion().on_bar(bar(), context(sigma=sigma)) == ()


def test_a_dispersion_wider_than_the_price_cannot_be_priced() -> None:
    """The stop would land at or below zero, which `Price` refuses."""
    assert VwapBandReversion().on_bar(bar(close="200"), context(sigma=0.5)) == ()


def test_nothing_fires_at_or_after_the_bell() -> None:
    assert VwapBandReversion().on_bar(bar(ts=CLOSE_NS), context()) == ()


def test_nothing_fires_outside_regular_hours() -> None:
    assert VwapBandReversion().on_bar(bar(), context(session=MarketSession.CLOSED)) == ()


def test_nothing_fires_before_the_first_bar_of_a_session() -> None:
    bare = StrategyContext(
        symbol=AAPL, as_of=IN_OPENING_WINDOW, session=MarketSession.REGULAR, regime=Regime.CHOP
    )
    assert VwapBandReversion().on_bar(bar(), bare) == ()


# ── One fade per direction per session ──────────────────────────────────


def test_the_same_direction_is_not_faded_twice_in_one_session() -> None:
    strategy = VwapBandReversion()
    assert len(strategy.on_bar(bar(ts=60 * MINUTE), context())) == 1
    assert strategy.on_bar(bar(ts=80 * MINUTE), context()) == ()


def test_a_stretch_the_other_way_is_a_new_fade() -> None:
    strategy = VwapBandReversion()
    (first,) = strategy.on_bar(bar(close="102", ts=60 * MINUTE), context())
    (second,) = strategy.on_bar(bar(close="98", ts=80 * MINUTE), context())
    assert (first.side, second.side) == (Side.SELL, Side.BUY)


def test_the_next_session_starts_clean() -> None:
    strategy = VwapBandReversion()
    assert len(strategy.on_bar(bar(), context())) == 1
    assert len(strategy.on_bar(bar(), context(day=JULY_9))) == 1


def test_state_is_kept_per_symbol() -> None:
    strategy = VwapBandReversion()
    assert len(strategy.on_bar(bar(symbol=AAPL), context(symbol=AAPL))) == 1
    assert len(strategy.on_bar(bar(symbol=MSFT), context(symbol=MSFT))) == 1


# ── The declared search, and the gate ───────────────────────────────────


def test_the_sweep_declares_one_trial_per_band_width() -> None:
    variants = VwapBandReversion.sweep()
    assert [label for label, _ in variants] == ["1.5 sigma", "2 sigma", "2.5 sigma"]
    assert [s.band_sigmas for _, s in variants] == list(SWEEP_BAND_SIGMAS)


def test_it_can_never_be_live_beside_intraday_momentum() -> None:
    """The same observation read two ways: the regimes make them exclusive."""
    from neurotrade.strategies.intraday_momentum import IntradayMomentum

    fade, trend = VwapBandReversion(), IntradayMomentum()
    assert not set(fade.regimes) & set(trend.regimes)
    assert fade.is_eligible(Regime.CHOP) and not trend.is_eligible(Regime.CHOP)
    assert trend.is_eligible(Regime.TREND_UP) and not fade.is_eligible(Regime.TREND_UP)
    assert not fade.is_eligible(Regime.LIQUIDITY_LULL)
