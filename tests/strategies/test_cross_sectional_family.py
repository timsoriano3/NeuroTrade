"""The four strategies that read the rest of the universe.

Three of them share `_cross_sectional.py`, so the shared behaviour is tested
once and each strategy is tested for the two things that make it itself: what
it ranks on, and which end it buys. The pairings matter most — `relative_strength`
against `residual_reversion`, and `opening_range_breakout` against `orb_fade` —
because a sign error in either pair turns two opposite rules into the same one.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.features.cross_section import CrossSection, InstrumentSnapshot
from neurotrade.features.levels import OpeningRange, SessionLevels
from neurotrade.strategies._cross_sectional import MIN_RANKED, SWEEP_EXTREME_NAMES
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, Strategy, StrategyContext
from neurotrade.strategies.intraday_reversal import IntradayReversal
from neurotrade.strategies.opening_range_breakout import OpeningRangeBreakout
from neurotrade.strategies.orb_fade import OrbFade
from neurotrade.strategies.relative_strength import RelativeStrength
from neurotrade.strategies.residual_reversion import ResidualReversion

MINUTE = 60_000_000_000
NOON = 120 * MINUTE
CLOSE_NS = 390 * MINUTE
JULY_8 = date(2024, 7, 8)

UNIVERSE = [Symbol(f"N{index:02d}", Venue.NASDAQ) for index in range(12)]
SUBJECT = UNIVERSE[0]  # will be given the most extreme reading
LAGGARD = UNIVERSE[11]

CROSS_SECTIONAL = (IntradayReversal, RelativeStrength, ResidualReversion)


def bar(symbol: Symbol = SUBJECT, close: str = "100", ts: int = NOON) -> Bar:
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
    day: date = JULY_8,
    rvol: float | None = None,
    ranges: dict[int, OpeningRange] | None = None,
) -> SessionLevels:
    return SessionLevels(
        session_date=day,
        open_ns=0,
        close_ns=CLOSE_NS,
        session_open=Price("100"),
        high=Price("101"),
        low=Price("99"),
        close=Price("100"),
        vwap=Price("100"),
        bar_count=120,
        prior_close=Price("100"),
        opening_ranges=ranges or {},
        relative_volume_from_open=rvol,
    )


def spread(measure: str, *, n: int = 12) -> CrossSection:
    """A universe whose readings fan out, `SUBJECT` highest and `LAGGARD` lowest."""
    rows = {}
    for index, symbol in enumerate(UNIVERSE[:n]):
        value = 0.02 - 0.003 * index
        fields: dict[str, float | None] = {"trailing_return": None, "residual": None}
        fields[measure] = value
        rows[symbol] = InstrumentSnapshot(
            symbol=symbol,
            session_return=value,
            trailing_return=fields["trailing_return"],
            relative_volume=None,
            beta=1.0,
            residual=fields["residual"],
        )
    return CrossSection(as_of=NOON - MINUTE, rows=rows)


def context(
    *,
    symbol: Symbol = SUBJECT,
    section: CrossSection | None = None,
    regime: Regime = Regime.CHOP,
    session: MarketSession = MarketSession.REGULAR,
    day: date = JULY_8,
    rvol: float | None = None,
    ranges: dict[int, OpeningRange] | None = None,
) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        as_of=NOON,
        session=session,
        regime=regime,
        levels=levels(day=day, rvol=rvol, ranges=ranges),
        cross_section=section,
    )


def measure_of(strategy: Strategy) -> str:
    return "trailing_return" if isinstance(strategy, IntradayReversal) else "residual"


# ── Registration and declarations ───────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("intraday_reversal", IntradayReversal),
        ("opening_range_breakout", OpeningRangeBreakout),
        ("relative_strength", RelativeStrength),
        ("residual_reversion", ResidualReversion),
    ],
)
def test_each_is_registered(name: str, cls: type[Strategy]) -> None:
    assert arsenal.get(name) is cls


@pytest.mark.parametrize(
    "cls", [IntradayReversal, OpeningRangeBreakout, RelativeStrength, ResidualReversion]
)
def test_each_declares_that_it_reads_the_universe(cls: type[Strategy]) -> None:
    """Undeclared means `context.cross_section` is `None` and nothing fires."""
    assert cls.needs_cross_section is True


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_the_sweep_is_one_trial_per_extremity(
    cls: type[IntradayReversal] | type[RelativeStrength] | type[ResidualReversion],
) -> None:
    sweep = cls.sweep()
    assert [instance.extreme_names for _, instance in sweep] == list(SWEEP_EXTREME_NAMES)


def test_the_two_residual_strategies_can_never_be_live_together() -> None:
    """Same number, opposite ends — so the regimes must not intersect (§5.7)."""
    assert not set(RelativeStrength.regimes) & set(ResidualReversion.regimes)


def test_the_two_orb_strategies_can_never_be_live_together() -> None:
    assert not set(OpeningRangeBreakout.regimes) & set(OrbFade.regimes)


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_the_lull_is_never_a_tradable_regime(cls: type[Strategy]) -> None:
    assert Regime.LIQUIDITY_LULL not in cls.regimes


# ── The pairings: same input, opposite sides ────────────────────────────


def test_relative_strength_buys_the_top_residual() -> None:
    (intent,) = RelativeStrength().on_bar(bar(), context(section=spread("residual")))
    assert intent.side is Side.BUY
    assert intent.invalidation < Price("100")  # the stop is below a long's entry


def test_residual_reversion_sells_the_same_top_residual() -> None:
    (intent,) = ResidualReversion().on_bar(bar(), context(section=spread("residual")))
    assert intent.side is Side.SELL
    assert intent.invalidation > Price("100")


def test_the_pair_takes_opposite_sides_at_the_bottom_too() -> None:
    section = spread("residual")
    (strong,) = RelativeStrength().on_bar(bar(LAGGARD), context(symbol=LAGGARD, section=section))
    (fade,) = ResidualReversion().on_bar(bar(LAGGARD), context(symbol=LAGGARD, section=section))
    assert strong.side is Side.SELL
    assert fade.side is Side.BUY


def test_intraday_reversal_fades_the_biggest_trailing_gainer() -> None:
    (intent,) = IntradayReversal().on_bar(bar(), context(section=spread("trailing_return")))
    assert intent.side is Side.SELL


def test_each_ranks_on_its_own_measure_and_no_other() -> None:
    """A residual strategy handed only trailing returns must decline, and vice versa."""
    assert RelativeStrength().on_bar(bar(), context(section=spread("trailing_return"))) == ()
    assert IntradayReversal().on_bar(bar(), context(section=spread("residual"))) == ()


# ── Shared behaviour, tested once per strategy ──────────────────────────


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_a_name_in_the_middle_of_the_ranking_is_not_traded(cls: type[Strategy]) -> None:
    middle = UNIVERSE[6]
    section = spread(measure_of(cls()))
    assert cls().on_bar(bar(middle), context(symbol=middle, section=section)) == ()


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_a_cross_section_of_six_is_not_a_cross_section(cls: type[Strategy]) -> None:
    """Below `MIN_RANKED` the two ends overlap and the rule stops being cross-sectional."""
    section = spread(measure_of(cls()), n=MIN_RANKED - 1)
    assert cls().on_bar(bar(), context(section=section)) == ()


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_nothing_fires_without_a_cross_section(cls: type[Strategy]) -> None:
    assert cls().on_bar(bar(), context(section=None)) == ()


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_nothing_fires_outside_a_regular_session(cls: type[Strategy]) -> None:
    section = spread(measure_of(cls()))
    assert cls().on_bar(bar(), context(section=section, session=MarketSession.CLOSED)) == ()


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_nothing_fires_on_or_after_the_bell(cls: type[Strategy]) -> None:
    section = spread(measure_of(cls()))
    assert cls().on_bar(bar(ts=CLOSE_NS), context(section=section)) == ()


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_one_trade_per_instrument_per_session(cls: type[Strategy]) -> None:
    strategy = cls()
    section = spread(measure_of(strategy))
    assert strategy.on_bar(bar(), context(section=section))
    assert strategy.on_bar(bar(ts=NOON + MINUTE), context(section=section)) == ()


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_a_new_session_starts_clean(cls: type[Strategy]) -> None:
    strategy = cls()
    section = spread(measure_of(strategy))
    assert strategy.on_bar(bar(), context(section=section))
    assert strategy.on_bar(bar(ts=NOON + MINUTE), context(section=section, day=date(2024, 7, 9)))


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_state_is_keyed_by_symbol(cls: type[Strategy]) -> None:
    strategy = cls()
    section = spread(measure_of(strategy))
    assert strategy.on_bar(bar(), context(section=section))
    assert strategy.on_bar(bar(LAGGARD), context(symbol=LAGGARD, section=section))


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_the_risk_scales_with_the_move_being_traded(cls: type[Strategy]) -> None:
    """A signal from a small move must not buy the same stop as one from a large move."""
    strategy = cls()
    measure = measure_of(strategy)
    big = spread(measure)
    small_rows = {
        symbol: InstrumentSnapshot(
            symbol=row.symbol,
            session_return=row.session_return / 10,
            trailing_return=None if row.trailing_return is None else row.trailing_return / 10,
            relative_volume=None,
            beta=1.0,
            residual=None if row.residual is None else row.residual / 10,
        )
        for symbol, row in big.rows.items()
    }
    small = CrossSection(as_of=big.as_of, rows=small_rows)
    (wide,) = strategy.on_bar(bar(), context(section=big))
    (tight,) = cls().on_bar(bar(), context(section=small))
    assert abs(wide.invalidation.value - Decimal(100)) > abs(
        tight.invalidation.value - Decimal(100)
    )


@pytest.mark.parametrize("cls", CROSS_SECTIONAL)
def test_a_move_too_small_to_move_the_stop_is_declined(cls: type[Strategy]) -> None:
    """R would be zero and `signals_from_intents` would drop it silently."""
    strategy = cls()
    measure = measure_of(strategy)
    rows = {
        symbol: InstrumentSnapshot(
            symbol=symbol,
            session_return=0.0,
            trailing_return=0.0 if measure == "trailing_return" else None,
            relative_volume=None,
            beta=1.0,
            residual=0.0 if measure == "residual" else None,
        )
        for symbol in UNIVERSE
    }
    assert strategy.on_bar(bar(), context(section=CrossSection(as_of=0, rows=rows))) == ()


# ── ORB on stocks in play ───────────────────────────────────────────────


RANGE = {30: OpeningRange(high=Price("101"), low=Price("99"), bar_count=30)}


def in_play(rvol: float) -> CrossSection:
    rows = {
        symbol: InstrumentSnapshot(symbol, 0.01, None, rvol if symbol is SUBJECT else 1.0, None)
        for symbol in UNIVERSE
    }
    return CrossSection(as_of=NOON - MINUTE, rows=rows)


def test_a_breakout_on_a_selected_name_is_traded() -> None:
    (intent,) = OpeningRangeBreakout().on_bar(
        bar(close="102"), context(section=in_play(5.0), ranges=RANGE, regime=Regime.TREND_UP)
    )
    assert intent.side is Side.BUY
    assert intent.invalidation == Price("99")  # the other side of the range
    assert intent.target_r == Decimal(2)


def test_a_downward_break_is_shorted_and_stopped_at_the_range_high() -> None:
    (intent,) = OpeningRangeBreakout().on_bar(
        bar(close="98"), context(section=in_play(5.0), ranges=RANGE)
    )
    assert intent.side is Side.SELL
    assert intent.invalidation == Price("101")


def test_the_same_breakout_on_a_name_not_in_play_is_declined() -> None:
    """The selection is the hypothesis; the breakout alone is the half known to fail."""
    assert (
        OpeningRangeBreakout().on_bar(bar(close="102"), context(section=in_play(1.1), ranges=RANGE))
        == ()
    )


def test_a_selected_name_inside_its_range_is_not_a_breakout() -> None:
    assert (
        OpeningRangeBreakout().on_bar(bar(close="100"), context(section=in_play(5.0), ranges=RANGE))
        == ()
    )


def test_nothing_fires_before_the_opening_range_completes() -> None:
    assert OpeningRangeBreakout().on_bar(bar(close="102"), context(section=in_play(5.0))) == ()


def test_an_untracked_window_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="opening range 7 is not tracked"):
        OpeningRangeBreakout(opening_minutes=7)


def test_only_one_breakout_per_instrument_per_session() -> None:
    strategy = OpeningRangeBreakout()
    assert strategy.on_bar(bar(close="102"), context(section=in_play(5.0), ranges=RANGE))
    assert (
        strategy.on_bar(
            bar(close="103", ts=NOON + MINUTE), context(section=in_play(5.0), ranges=RANGE)
        )
        == ()
    )
