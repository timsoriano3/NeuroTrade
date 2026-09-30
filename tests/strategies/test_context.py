"""Assembling the view a strategy is given: phase, regime and features.

Three things are worth testing here rather than trusting: that the lull window
lands where §5.7 puts it on a normal day and on a half day, that a strategy
cannot see a feature another strategy declared, and that a context is refused
rather than built from stale history.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

import pytest

from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import Nanos, to_nanos
from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.features.registry import FeatureRegistry
from neurotrade.strategies.base import FeatureRef, Regime, Strategy
from neurotrade.strategies.context import (
    LULL_ENDS_AFTER_OPEN_NS,
    LULL_STARTS_AFTER_OPEN_NS,
    NEXT_SESSION_SEARCH_DAYS,
    MarketContext,
    market_phase,
    time_of_day_regime,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MINUTE = 60_000_000_000
JULY_8 = date(2024, 7, 8)


def utc(hour: int, minute: int = 0, day: date = JULY_8) -> Nanos:
    return to_nanos(datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC))


#: A full NYSE session: 09:30-16:00 ET, which in July is 13:30-20:00 UTC.
FULL_DAY = TradingSession(
    venue=Venue.NASDAQ,
    session_date=JULY_8,
    open_ns=utc(13, 30),
    close_ns=utc(20, 0),
    is_early_close=False,
)

#: 3 July 2024: the same open, closing at 13:00 ET.
HALF_DAY = TradingSession(
    venue=Venue.NASDAQ,
    session_date=date(2024, 7, 3),
    open_ns=utc(13, 30, date(2024, 7, 3)),
    close_ns=utc(17, 0, date(2024, 7, 3)),
    is_early_close=True,
)


def bar(ts: Nanos, close: str = "100", symbol: Symbol = AAPL) -> Bar:
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


class Calendar:
    """A `CalendarPort` over a fixed set of sessions, counting its lookups."""

    def __init__(self, *sessions: TradingSession) -> None:
        self._sessions = {(s.venue, s.session_date): s for s in sessions}
        self.lookups = 0

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        return tuple(d for (v, d) in self._sessions if v is venue and start <= d <= end)

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        self.lookups += 1
        return self._sessions.get((venue, session_date))


def library() -> FeatureRegistry:
    registry = FeatureRegistry()

    @registry.feature("last", "1.0.0", lookback=1, description="the last close")
    def last(bars: Sequence[Bar]) -> float:
        return float(bars[-1].close.value)

    @registry.feature("mean2", "1.0.0", lookback=2, description="mean of two closes")
    def mean2(bars: Sequence[Bar]) -> float:
        return sum(float(b.close.value) for b in bars) / len(bars)

    return registry


class Reader(Strategy):
    name, version = "reader", "1.0.0"
    regimes: ClassVar[tuple[Regime, ...]] = (Regime.UNKNOWN,)
    features: ClassVar[tuple[FeatureRef, ...]] = (FeatureRef("last"),)


class Other(Strategy):
    name, version = "other", "1.0.0"
    features: ClassVar[tuple[FeatureRef, ...]] = (FeatureRef("mean2"),)


def context_over(*sessions: TradingSession) -> MarketContext:
    return MarketContext(features=library(), calendar=Calendar(*sessions))


# ── Venue phase ──────────────────────────────────────────────


def test_a_bar_a_session_holds_is_regular() -> None:
    assert market_phase(FULL_DAY, utc(14, 0)) is MarketSession.REGULAR


def test_a_bar_no_session_holds_is_closed_not_guessed() -> None:
    """The calendar models regular hours only; PRE and POST would be invented."""
    assert market_phase(None, utc(11, 0)) is MarketSession.CLOSED


def test_a_closed_bar_is_not_tradable() -> None:
    assert not market_phase(None, utc(11, 0)).is_tradable


# ── The liquidity lull ───────────────────────────────────────


def test_the_lull_offsets_are_noon_to_two_eastern() -> None:
    """The constants are offsets; on a 09:30 open they must be 12:00 and 14:00."""
    assert (
        FULL_DAY.open_ns + LULL_STARTS_AFTER_OPEN_NS,
        FULL_DAY.open_ns + LULL_ENDS_AFTER_OPEN_NS,
    ) == (utc(16, 0), utc(18, 0))


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        (utc(14, 0), Regime.UNKNOWN),  # 10:00 ET, well before
        (utc(16, 0), Regime.UNKNOWN),  # the bar closing at 12:00 covers 11:59-12:00
        (utc(16, 1), Regime.LIQUIDITY_LULL),  # 12:01 ET, first bar inside
        (utc(18, 0), Regime.LIQUIDITY_LULL),  # 14:00 ET, covers 13:59-14:00
        (utc(18, 1), Regime.UNKNOWN),  # 14:01 ET, the first bar past it
    ],
)
def test_the_lull_window_is_open_exclusive_and_close_inclusive(ts: Nanos, expected: Regime) -> None:
    assert time_of_day_regime(FULL_DAY, ts) is expected


def test_a_half_day_clips_the_lull_at_the_early_close() -> None:
    """13:00 ET close: the window cannot run to 14:00, and must not."""
    early = HALF_DAY.close_ns
    assert (
        time_of_day_regime(HALF_DAY, early) is Regime.LIQUIDITY_LULL
        and time_of_day_regime(HALF_DAY, early + MINUTE) is Regime.UNKNOWN
    )


def test_outside_a_session_there_is_nothing_to_classify() -> None:
    assert time_of_day_regime(None, utc(11, 0)) is Regime.UNKNOWN


def test_everything_else_is_unknown_until_phase_5() -> None:
    """§5.7's classifier is an HMM; nothing here pretends to be one."""
    assert time_of_day_regime(FULL_DAY, utc(14, 0)) is Regime.UNKNOWN


# ── What a strategy may see ──────────────────────────────────


def test_a_strategy_sees_only_what_it_declared() -> None:
    context = context_over(FULL_DAY)
    for strategy in (Reader(), Other()):
        context.declare(strategy)
    context.observe(bar(utc(14, 0), "101"))
    assert context(bar(utc(14, 0), "101"), Reader()).values == {"last": 101.0}


def test_a_declared_feature_resolves_against_the_bars_observed() -> None:
    context = context_over(FULL_DAY)
    context.declare(Other())
    for minute, close in ((0, "100"), (1, "102")):
        context.observe(bar(utc(14, minute), close))
    assert context(bar(utc(14, 1), "102"), Other()).values == {"mean2": 101.0}


def test_lookback_reports_the_widest_declared_feature() -> None:
    context = context_over(FULL_DAY)
    context.declare(Other())
    assert context.lookback == 2


def test_the_context_is_stamped_at_the_bar_close() -> None:
    context = context_over(FULL_DAY)
    context.declare(Reader())
    context.observe(bar(utc(14, 0)))
    assert context(bar(utc(14, 0)), Reader()).as_of == utc(14, 0)


def test_the_regime_source_is_a_seam() -> None:
    """Phase 5 swaps the HMM in here without touching the engine."""
    context = MarketContext(
        features=library(),
        calendar=Calendar(FULL_DAY),
        regime_source=lambda session, ts: Regime.TREND_UP,
    )
    context.declare(Reader())
    context.observe(bar(utc(14, 0)))
    assert context(bar(utc(14, 0)), Reader()).regime is Regime.TREND_UP


# ── Rejection ────────────────────────────────────────────────


def test_an_undeclared_strategy_is_refused() -> None:
    context = context_over(FULL_DAY)
    context.observe(bar(utc(14, 0)))
    with pytest.raises(ValueError, match=r"reader@1\.0\.0 was never declared"):
        context(bar(utc(14, 0)), Reader())


def test_a_bar_that_was_not_observed_is_refused() -> None:
    """A context built from stale history is an unreproducible backtest."""
    context = context_over(FULL_DAY)
    context.declare(Reader())
    context.observe(bar(utc(14, 0)))
    with pytest.raises(ValueError, match=r"was not observed before use"):
        context(bar(utc(14, 1)), Reader())


def test_declaring_after_the_run_began_is_refused() -> None:
    """A feature added mid-run gets a window that starts in the middle of the data."""
    context = context_over(FULL_DAY)
    context.observe(bar(utc(14, 0)))
    with pytest.raises(ValueError, match=r"declared after the run began"):
        context.declare(Reader())


def test_a_feature_the_library_does_not_hold_is_refused() -> None:
    class Missing(Strategy):
        name, version = "missing", "1.0.0"
        features: ClassVar[tuple[FeatureRef, ...]] = (FeatureRef("nonesuch"),)

    with pytest.raises(KeyError):
        context_over(FULL_DAY).declare(Missing())


# ── Session lookup ───────────────────────────────────────────


def test_a_duplicate_bar_does_not_enter_a_window_twice() -> None:
    context = context_over(FULL_DAY)
    context.declare(Other())
    context.observe(bar(utc(14, 0), "100"))
    context.observe(bar(utc(14, 0), "100"))
    assert context(bar(utc(14, 0), "100"), Other()).values == {"mean2": None}


def test_sessions_are_memoised_per_venue_and_day() -> None:
    """A backtest asks for the same session 390 times a day."""
    calendar = Calendar(FULL_DAY)
    context = MarketContext(features=library(), calendar=calendar)
    context.declare(Reader())
    for minute in range(3):
        context.observe(bar(utc(14, minute)))
        context(bar(utc(14, minute)), Reader())
    assert calendar.lookups == 1


def test_a_bar_after_utc_midnight_finds_its_own_session() -> None:
    """A November close at 21:00 UTC is the same date; the guard is the day before."""
    november = TradingSession(
        venue=Venue.NASDAQ,
        session_date=date(2024, 11, 5),
        open_ns=utc(14, 30, date(2024, 11, 5)),
        close_ns=utc(21, 0, date(2024, 11, 5)),
        is_early_close=False,
    )
    context = MarketContext(features=library(), calendar=Calendar(november))
    context.declare(Reader())
    late = bar(utc(21, 0, date(2024, 11, 5)))
    context.observe(late)
    assert context(late, Reader()).session is MarketSession.REGULAR


# ── Session levels ───────────────────────────────────────────

JULY_9 = TradingSession(
    venue=Venue.NASDAQ,
    session_date=date(2024, 7, 9),
    open_ns=utc(13, 30, date(2024, 7, 9)),
    close_ns=utc(20, 0, date(2024, 7, 9)),
    is_early_close=False,
)


def test_levels_are_supplied_without_being_declared() -> None:
    """§5.3 shared infrastructure: no `FeatureRef` names them, everyone gets them."""
    context = context_over(FULL_DAY)
    context.declare(Reader())
    context.observe(bar(utc(14, 0), "101"))
    levels = context(bar(utc(14, 0), "101"), Reader()).levels
    assert levels is not None
    assert (str(levels.session_open), levels.bar_count) == ("101", 1)


def test_levels_are_absent_outside_a_session() -> None:
    """A CLOSED bar must not put an after-hours print into a session VWAP."""
    context = context_over(FULL_DAY)
    context.declare(Reader())
    context.observe(bar(utc(11, 0)))
    assert context(bar(utc(11, 0)), Reader()).levels is None


def test_levels_reset_at_the_next_session_and_carry_the_prior_close() -> None:
    context = MarketContext(features=library(), calendar=Calendar(FULL_DAY, JULY_9))
    context.declare(Reader())
    for one in (bar(utc(14, 0), "100"), bar(utc(14, 0, date(2024, 7, 9)), "80")):
        context.observe(one)
    last = bar(utc(14, 0, date(2024, 7, 9)), "80")
    levels = context(last, Reader()).levels
    assert levels is not None
    assert (levels.session_date, str(levels.prior_close), levels.bar_count) == (
        date(2024, 7, 9),
        "100",
        1,
    )


# ── The cross-section ────────────────────────────────────────


class Watcher(Strategy):
    """A strategy that reads the universe. Records what it was shown."""

    name, version = "watcher", "1.0.0"
    needs_cross_section: ClassVar[bool] = True


class Loner(Strategy):
    """A strategy that does not. Must never be handed one."""

    name, version = "loner", "1.0.0"


MSFT = Symbol("MSFT", Venue.NASDAQ)


def priced(symbol: Symbol, ts: Nanos, close: str) -> Bar:
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


def session_on(day: date) -> TradingSession:
    """A full regular session on one date, for the next-open lookups."""
    return TradingSession(
        venue=Venue.NASDAQ,
        session_date=day,
        open_ns=to_nanos(datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)),
        close_ns=to_nanos(datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC)),
        is_early_close=False,
    )


def test_no_cross_section_is_built_when_nothing_asks_for_one() -> None:
    """What keeps every measurement taken before this existed byte-identical."""
    context = context_over(FULL_DAY)
    context.declare(Loner())
    bar = priced(AAPL, utc(14, 0), "100")
    context.observe(bar)
    assert context(bar, Loner()).cross_section is None


def test_a_declaring_strategy_gets_one_and_a_non_declaring_one_does_not() -> None:
    """Both registered on the same context, so the gate is per strategy, not per run."""
    context = context_over(FULL_DAY)
    context.declare(Watcher())
    context.declare(Loner())
    first = priced(AAPL, utc(14, 0), "100")
    context.observe(first)
    second = priced(AAPL, utc(14, 1), "101")
    context.observe(second)
    assert context(second, Watcher()).cross_section is not None
    assert context(second, Loner()).cross_section is None


def test_the_cross_section_a_strategy_sees_is_always_an_earlier_tick() -> None:
    """The no-lookahead guarantee, through the real context rather than the tracker.

    Two instruments print at 14:01. Whichever is dispatched first must not be
    shown the other's 14:01 bar, so the section it gets is stamped 14:00.
    """
    context = context_over(FULL_DAY)
    context.declare(Watcher())
    for symbol in (AAPL, MSFT):
        context.observe(priced(symbol, utc(14, 0), "100"))
    later = priced(AAPL, utc(14, 1), "101")
    context.observe(later)
    section = context(later, Watcher()).cross_section
    assert section is not None
    assert section.as_of == utc(14, 0)
    assert set(section.symbols) == {AAPL, MSFT}


def test_a_bar_outside_any_session_folds_into_no_cross_section() -> None:
    """No session means no levels, and the tracker is fed from the levels."""
    context = context_over(FULL_DAY)
    context.declare(Watcher())
    stray = priced(AAPL, utc(11, 0), "100")
    context.observe(stray)
    assert context(stray, Watcher()).cross_section is None


def test_the_repr_says_whether_a_cross_section_is_being_built() -> None:
    plain = context_over(FULL_DAY)
    plain.declare(Loner())
    watching = context_over(FULL_DAY)
    watching.declare(Watcher())
    assert "cross-section" not in repr(plain)
    assert "cross-section" in repr(watching)


# ── The next session's open ──────────────────────────────────────────────


def test_the_next_open_comes_from_the_calendar_not_from_a_fixed_span() -> None:
    """The overnight family's exit. A fixed 17.5 hours lands on a Saturday one
    night in five, where the labeller finds no bars at all."""
    tomorrow = session_on(date(2024, 7, 9))
    context = context_over(FULL_DAY, tomorrow)
    context.declare(Loner())
    bar = priced(AAPL, utc(14, 0), "100")
    context.observe(bar)
    levels = context(bar, Loner()).levels
    assert levels is not None
    assert levels.next_open_ns == tomorrow.open_ns


def test_a_holiday_weekend_is_skipped_rather_than_stepped() -> None:
    """The calendar is asked for trading days, so a four-day gap costs one lookup."""
    after = session_on(date(2024, 7, 15))
    context = context_over(FULL_DAY, after)
    context.declare(Loner())
    bar = priced(AAPL, utc(14, 0), "100")
    context.observe(bar)
    levels = context(bar, Loner()).levels
    assert levels is not None
    assert levels.next_open_ns == after.open_ns


def test_no_later_session_leaves_the_next_open_unset() -> None:
    """The end of the calendar, which an overnight strategy declines on."""
    context = context_over(FULL_DAY)
    context.declare(Loner())
    bar = priced(AAPL, utc(14, 0), "100")
    context.observe(bar)
    levels = context(bar, Loner()).levels
    assert levels is not None
    assert levels.next_open_ns is None


def test_a_session_beyond_the_search_horizon_is_not_found() -> None:
    """Bounded, so a venue whose schedule has ended is not walked indefinitely."""
    far = session_on(date(2024, 7, 8) + timedelta(days=NEXT_SESSION_SEARCH_DAYS + 1))
    context = context_over(FULL_DAY, far)
    context.declare(Loner())
    bar = priced(AAPL, utc(14, 0), "100")
    context.observe(bar)
    levels = context(bar, Loner()).levels
    assert levels is not None
    assert levels.next_open_ns is None
