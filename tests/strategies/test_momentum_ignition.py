"""Momentum ignition: trade a bar abnormal in volume and in range at once.

Mostly refusals. The conjunction is the whole rule — either condition alone
names a different and much more common event — so the tests that matter are the
ones proving each half can veto on its own, and that the direction comes off the
bar's body rather than off anything the bar did not say.
"""

from __future__ import annotations

from datetime import date

import pytest

from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.features.levels import SessionLevels
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, StrategyContext, UndeclaredFeature
from neurotrade.strategies.momentum_ignition import (
    DEFAULT_TARGET_R,
    MIN_SESSION_BARS,
    SWEEP_MIN_RVOL,
    MomentumIgnition,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)

MINUTE = 60_000_000_000
NOON = 120 * MINUTE
CLOSE_NS = 390 * MINUTE
JULY_8 = date(2024, 7, 8)
JULY_9 = date(2024, 7, 9)


def bar(
    *,
    open_: str = "100",
    high: str = "103",
    low: str = "99.5",
    close: str = "102.5",
    ts: int = NOON,
    symbol: Symbol = AAPL,
) -> Bar:
    """A bar covering 3.5 in range — 3.5 ATRs at the fixture's ATR of 1.0."""
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=Price(open_),
        high=Price(high),
        low=Price(low),
        close=Price(close),
        volume=Quantity(50_000),
    )


def levels(*, day: date = JULY_8, bar_count: int = 120) -> SessionLevels:
    return SessionLevels(
        session_date=day,
        open_ns=0,
        close_ns=CLOSE_NS,
        session_open=Price("100"),
        high=Price("103"),
        low=Price("99"),
        close=Price("102.5"),
        vwap=Price("101"),
        bar_count=bar_count,
        prior_close=Price("100"),
    )


def context(
    *,
    rvol: float | None = 5.0,
    atr: float | None = 1.0,
    regime: Regime = Regime.TREND_UP,
    session: MarketSession = MarketSession.REGULAR,
    symbol: Symbol = AAPL,
    day: date = JULY_8,
    bar_count: int = 120,
) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        as_of=NOON,
        session=session,
        regime=regime,
        values={"rvol": rvol, "atr": atr},
        levels=levels(day=day, bar_count=bar_count),
    )


# ── Registration and the declared search ────────────────────────────────


def test_the_strategy_is_in_the_arsenal() -> None:
    assert arsenal.get("momentum_ignition") is MomentumIgnition


def test_the_sweep_is_one_trial_per_burst_threshold() -> None:
    """Every entry is a hypothesis the ledger counts (§17)."""
    sweep = MomentumIgnition.sweep()
    assert [instance.min_rvol for _, instance in sweep] == list(SWEEP_MIN_RVOL)
    assert len({label for label, _ in sweep}) == len(SWEEP_MIN_RVOL)


def test_both_features_are_declared() -> None:
    """The engine resolves exactly what is declared, so reading one undeclared raises."""
    assert {ref.name for ref in MomentumIgnition.features} == {"rvol", "atr"}


def test_the_regimes_never_overlap_the_fade_families() -> None:
    """A burst continuing is a trend day; CHOP and REVERSAL are where it is the extreme."""
    assert Regime.CHOP not in MomentumIgnition.regimes
    assert Regime.REVERSAL not in MomentumIgnition.regimes
    assert Regime.LIQUIDITY_LULL not in MomentumIgnition.regimes


# ── It fires ────────────────────────────────────────────────────────────


def test_a_burst_in_both_volume_and_range_is_traded_with_the_body() -> None:
    (intent,) = MomentumIgnition().on_bar(bar(), context())
    assert intent.side is Side.BUY
    assert intent.invalidation == Price("99.5")  # the igniting bar's low
    assert intent.target_r == DEFAULT_TARGET_R
    assert intent.horizon_ns == CLOSE_NS - NOON  # flat at the bell
    assert intent.strategy == "momentum_ignition"


def test_a_down_burst_is_shorted_and_stopped_at_its_high() -> None:
    down = bar(open_="103", high="103", low="99.5", close="100")
    (intent,) = MomentumIgnition().on_bar(down, context())
    assert intent.side is Side.SELL
    assert intent.invalidation == Price("103")


# ── Each half of the conjunction can veto alone ─────────────────────────


def test_range_without_volume_is_not_a_burst() -> None:
    """A thin bar in a name nobody is trading."""
    assert MomentumIgnition().on_bar(bar(), context(rvol=1.2)) == ()


def test_volume_without_range_is_not_a_burst() -> None:
    """An auction print or a block crossed away from the tape moves no price."""
    flat = bar(open_="100", high="100.2", low="99.9", close="100.1")
    assert MomentumIgnition().on_bar(flat, context()) == ()


def test_a_bar_with_no_body_names_no_direction() -> None:
    """Declined rather than guessed — a doji burst says who traded, not which way."""
    doji = bar(open_="101", high="103", low="99.5", close="101")
    assert MomentumIgnition().on_bar(doji, context()) == ()


# ── Refusals ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("rvol", "atr"), [(None, 1.0), (5.0, None)])
def test_a_cold_feature_declines_rather_than_reading_none_as_zero(
    rvol: float | None, atr: float | None
) -> None:
    assert MomentumIgnition().on_bar(bar(), context(rvol=rvol, atr=atr)) == ()


def test_nothing_fires_before_the_session_is_old_enough() -> None:
    """`atr`'s window is fifteen bars and the open is the widest part of the day."""
    assert MomentumIgnition().on_bar(bar(), context(bar_count=MIN_SESSION_BARS - 1)) == ()


def test_nothing_fires_outside_a_regular_session() -> None:
    assert MomentumIgnition().on_bar(bar(), context(session=MarketSession.CLOSED)) == ()


def test_nothing_fires_on_or_after_the_bell() -> None:
    """There is no hold left, so the horizon would be zero or negative."""
    assert MomentumIgnition().on_bar(bar(ts=CLOSE_NS), context()) == ()


def test_a_zero_atr_cannot_divide_the_range_test() -> None:
    assert MomentumIgnition().on_bar(bar(), context(atr=0.0)) == ()


def test_no_levels_means_no_session_to_be_flat_by() -> None:
    view = StrategyContext(
        symbol=AAPL,
        as_of=NOON,
        session=MarketSession.REGULAR,
        regime=Regime.TREND_UP,
        values={"rvol": 5.0, "atr": 1.0},
    )
    assert MomentumIgnition().on_bar(bar(), view) == ()


# ── One per instrument per session ──────────────────────────────────────


def test_a_second_burst_in_the_same_session_is_the_same_evidence_twice() -> None:
    strategy = MomentumIgnition()
    assert strategy.on_bar(bar(), context())
    assert strategy.on_bar(bar(ts=NOON + MINUTE), context()) == ()


def test_a_new_session_starts_clean() -> None:
    strategy = MomentumIgnition()
    assert strategy.on_bar(bar(), context())
    assert strategy.on_bar(bar(ts=NOON + MINUTE), context(day=JULY_9))


def test_state_is_keyed_by_symbol_because_one_instance_sees_the_universe() -> None:
    """`add_strategy` subscribes one instance to every instrument."""
    strategy = MomentumIgnition()
    assert strategy.on_bar(bar(), context())
    assert strategy.on_bar(bar(symbol=MSFT), context(symbol=MSFT))


def test_reading_an_undeclared_feature_still_raises_through_this_strategy() -> None:
    """The declaration guarantee is the engine's, and this confirms it is not bypassed."""
    with pytest.raises(UndeclaredFeature):
        context().feature("realised_vol")
