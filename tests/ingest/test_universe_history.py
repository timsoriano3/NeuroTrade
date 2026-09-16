"""Tests for the point-in-time universe screen.

The leakage test is the one that matters. Every other failure here is visible —
a wrong count, a missing symbol — but a window that reaches into the session it
is deciding produces a *better-looking* backtest, silently, forever.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import to_nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Currency, Money, Price, Quantity, Symbol, Venue
from neurotrade.core.universe import Universe, UniverseHistory
from neurotrade.ingest.universe_history import LiquidityFloor, ScreenRules, screen_universe

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)

# Far enough into the fake calendar that the reach-back for a trailing window
# always lands inside it.
FIRST = date(2024, 1, 1)
DECIDE = date(2024, 6, 3)
LAST = date(2024, 6, 7)

USD_FLOOR = LiquidityFloor(Money("1000000", Currency.USD), Price("5"))
CAD_FLOOR = LiquidityFloor(Money("1000000", Currency.CAD), Price("5"))


def default_rules(
    lookback: int = 3, floors: tuple[LiquidityFloor, ...] = (USD_FLOOR, CAD_FLOOR)
) -> ScreenRules:
    return ScreenRules(lookback_sessions=lookback, floors=floors)


class FakeCalendar:
    """Weekdays are sessions, minus any date the venue is told to skip."""

    def __init__(
        self, venues: Sequence[Venue], *, closed: dict[Venue, set[date]] | None = None
    ) -> None:
        self._venues = tuple(venues)
        self._closed = closed or {}

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        if venue not in self._venues:
            return ()
        closed = self._closed.get(venue, set())
        days = []
        day = start
        while day <= end:
            if day.weekday() < 5 and day not in closed:
                days.append(day)
            day += timedelta(days=1)
        return tuple(days)

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        if session_date not in self.sessions(venue, session_date, session_date):
            return None
        open_ns = to_nanos(
            datetime(session_date.year, session_date.month, session_date.day, 13, 30, tzinfo=UTC)
        )
        return TradingSession(
            venue=venue,
            session_date=session_date,
            open_ns=open_ns,
            close_ns=open_ns + 390 * BarInterval.MIN_1.nanos,
            is_early_close=False,
        )


class FakeStore:
    """An in-memory `StoragePort`. The screen only ever reads."""

    def __init__(self, bars: dict[Symbol, list[Bar]]) -> None:
        self._bars = bars

    def write_bars(
        self, bars: Sequence[Bar], *, source: str, session_date: date
    ) -> None:  # pragma: no cover - never called
        raise NotImplementedError("the screen does not write")

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: int, end: int
    ) -> Iterator[Bar]:
        return iter([bar for bar in self._bars.get(symbol, []) if start <= bar.ts_event < end])


def a_bar(symbol: Symbol, close_ns: int, *, close: str, volume: str) -> Bar:
    price = Price(close)
    return Bar(
        symbol=symbol,
        ts_event=close_ns,
        ts_init=close_ns,
        interval=BarInterval.DAY_1,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Quantity(volume),
    )


def corpus(
    calendar: FakeCalendar,
    symbol: Symbol,
    *,
    close: str = "100",
    volume: str = "1000000",
    through: date = LAST,
    overrides: dict[date, tuple[str, str]] | None = None,
) -> list[Bar]:
    """Daily bars on every session the venue had, stamped at its close."""
    bars = []
    for day in calendar.sessions(symbol.venue, FIRST, through):
        session = calendar.session(symbol.venue, day)
        assert session is not None
        day_close, day_volume = (overrides or {}).get(day, (close, volume))
        bars.append(a_bar(symbol, session.close_ns, close=day_close, volume=day_volume))
    return bars


def build(
    universe: Universe,
    calendar: FakeCalendar,
    store: FakeStore,
    *,
    start: date = DECIDE,
    end: date = DECIDE,
    rules: ScreenRules | None = None,
    survivorship_biased: bool = False,
) -> UniverseHistory:
    return screen_universe(
        universe,
        calendar,
        store,
        start=start,
        end=end,
        rules=rules if rules is not None else default_rules(),
        survivorship_biased=survivorship_biased,
    )


# ── Point-in-time: the whole reason this module exists ───────


def test_the_session_being_decided_does_not_vote_on_itself() -> None:
    """The leak. AAPL is far too thin on every prior session and only clears the
    floor on the decision date itself — which is not knowable before the open."""
    calendar = FakeCalendar([Venue.NASDAQ])
    thin = dict.fromkeys(calendar.sessions(Venue.NASDAQ, FIRST, LAST), ("100", "1"))
    thin[DECIDE] = ("100", "999999999")
    store = FakeStore({AAPL: corpus(calendar, AAPL, overrides=thin)})

    history = build(Universe([AAPL]), calendar, store)

    assert history.as_of(DECIDE) == ()


def test_a_name_liquid_only_before_the_decision_is_admitted() -> None:
    """The mirror image: the window ends at t-1, so t-1 must count."""
    calendar = FakeCalendar([Venue.NASDAQ])
    store = FakeStore({AAPL: corpus(calendar, AAPL, volume="1000000")})

    history = build(Universe([AAPL]), calendar, store)

    assert history.as_of(DECIDE) == (AAPL,)


def test_a_window_shorter_than_the_lookback_admits_nobody() -> None:
    """Warm-up. A short window is a different screen, not a weaker one."""
    calendar = FakeCalendar([Venue.NASDAQ])
    two_sessions = calendar.sessions(Venue.NASDAQ, FIRST, DECIDE)[-3:-1]
    store = FakeStore(
        {
            AAPL: [
                a_bar(AAPL, _close_ns(calendar, day), close="100", volume="1000000")
                for day in two_sessions
            ]
        }
    )

    history = build(Universe([AAPL]), calendar, store, rules=default_rules(lookback=3))

    assert history.as_of(DECIDE) == ()


# ── The floors ───────────────────────────────────────────────


def test_a_price_below_the_floor_is_rejected() -> None:
    calendar = FakeCalendar([Venue.NASDAQ])
    store = FakeStore({AAPL: corpus(calendar, AAPL, close="4.99", volume="99999999")})

    assert build(Universe([AAPL]), calendar, store).as_of(DECIDE) == ()


def test_the_price_floor_reads_the_last_close_not_the_first() -> None:
    """A name that has fallen below the floor is out today, however it traded a
    month ago."""
    calendar = FakeCalendar([Venue.NASDAQ])
    sessions = calendar.sessions(Venue.NASDAQ, FIRST, LAST)
    fallen = {sessions[-1 - i]: ("1", "1000000") for i in range(3, 6)}
    fallen[_previous(calendar, DECIDE)] = ("1", "1000000")
    store = FakeStore({AAPL: corpus(calendar, AAPL, overrides=fallen)})

    assert build(Universe([AAPL]), calendar, store).as_of(DECIDE) == ()


def test_the_volume_test_is_a_median_not_a_mean() -> None:
    """One frantic session must not carry a name that is otherwise untradeable.
    Mean dollar volume here is far above the floor; the median is far below."""
    calendar = FakeCalendar([Venue.NASDAQ])
    spike = dict.fromkeys(calendar.sessions(Venue.NASDAQ, FIRST, LAST), ("100", "1"))
    spike[_previous(calendar, DECIDE)] = ("100", "100000000")
    store = FakeStore({AAPL: corpus(calendar, AAPL, overrides=spike)})

    assert build(Universe([AAPL]), calendar, store).as_of(DECIDE) == ()


def test_dollar_volume_is_price_times_shares_not_shares() -> None:
    """A $100 name trading 20k shares clears $1M; the share count alone does
    not, and screening on shares would rank a penny stock alongside it."""
    calendar = FakeCalendar([Venue.NASDAQ])
    store = FakeStore({AAPL: corpus(calendar, AAPL, close="100", volume="20000")})

    assert build(Universe([AAPL]), calendar, store).as_of(DECIDE) == (AAPL,)


# ── Currency: the invariant, not a convenience ───────────────


def test_each_currency_is_judged_against_its_own_floor() -> None:
    """A CAD name is never compared to a USD threshold. Both clear their own
    floor here, and neither floor was converted."""
    calendar = FakeCalendar([Venue.NASDAQ, Venue.TSX])
    store = FakeStore(
        {
            AAPL: corpus(calendar, AAPL),
            SHOP: corpus(calendar, SHOP),
        }
    )

    history = build(Universe([AAPL, SHOP]), calendar, store)

    assert history.as_of(DECIDE) == (AAPL, SHOP)


def test_a_currency_with_no_floor_is_an_error_not_a_free_pass() -> None:
    calendar = FakeCalendar([Venue.NASDAQ, Venue.TSX])
    store = FakeStore({SHOP: corpus(calendar, SHOP)})

    with pytest.raises(KeyError, match="no liquidity floor configured for CAD"):
        build(Universe([SHOP]), calendar, store, rules=default_rules(floors=(USD_FLOOR,)))


# ── Venues keep their own holidays ───────────────────────────


def test_a_name_is_not_eligible_on_a_day_its_own_venue_was_shut() -> None:
    """Canada Day: the TSX is closed, US venues are not. Carrying the Canadian
    names into that date would let a backtest trade a shut market."""
    calendar = FakeCalendar([Venue.NASDAQ, Venue.TSX], closed={Venue.TSX: {DECIDE}})
    store = FakeStore({AAPL: corpus(calendar, AAPL), SHOP: corpus(calendar, SHOP)})

    history = build(Universe([AAPL, SHOP]), calendar, store)

    assert history.as_of(DECIDE) == (AAPL,)


def test_a_date_is_evaluated_when_any_venue_traded() -> None:
    calendar = FakeCalendar([Venue.NASDAQ, Venue.TSX], closed={Venue.NASDAQ: {DECIDE}})
    store = FakeStore({AAPL: corpus(calendar, AAPL), SHOP: corpus(calendar, SHOP)})

    history = build(Universe([AAPL, SHOP]), calendar, store)

    assert history.dates == (DECIDE,)
    assert history.as_of(DECIDE) == (SHOP,)


# ── Refusals ─────────────────────────────────────────────────


def test_an_inverted_range_is_refused() -> None:
    calendar = FakeCalendar([Venue.NASDAQ])
    with pytest.raises(ValueError, match="precedes start"):
        build(Universe([AAPL]), calendar, FakeStore({}), start=LAST, end=FIRST)


def test_a_bar_matching_no_session_close_is_refused() -> None:
    """The corpus and the calendar disagreeing makes every window boundary
    suspect; recompute derived/, do not screen around it."""
    calendar = FakeCalendar([Venue.NASDAQ])
    stray = _close_ns(calendar, _previous(calendar, DECIDE)) + 1
    store = FakeStore({AAPL: [a_bar(AAPL, stray, close="100", volume="1000000")]})

    with pytest.raises(ValueError, match="matching no NASDAQ session close"):
        build(Universe([AAPL]), calendar, store)


def test_an_empty_corpus_admits_nobody_rather_than_raising() -> None:
    """A corpus that has not been filled yet is an ordinary state during a
    build-out, not a fault."""
    calendar = FakeCalendar([Venue.NASDAQ])

    history = build(Universe([AAPL]), calendar, FakeStore({}))

    assert history.as_of(DECIDE) == ()


# ── The bias flag travels ────────────────────────────────────


def test_the_survivorship_flag_reaches_the_result() -> None:
    calendar = FakeCalendar([Venue.NASDAQ])
    store = FakeStore({AAPL: corpus(calendar, AAPL)})

    history = build(Universe([AAPL]), calendar, store, survivorship_biased=True)

    assert history.survivorship_biased


# ── ScreenRules and LiquidityFloor validation ────────────────


@pytest.mark.parametrize("bad", [0, -1])
def test_a_lookback_under_one_session_is_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="at least one session"):
        ScreenRules(lookback_sessions=bad, floors=(USD_FLOOR,))


def test_a_screen_with_no_floors_is_rejected() -> None:
    with pytest.raises(ValueError, match="admits everything"):
        ScreenRules(lookback_sessions=3, floors=())


def test_two_floors_for_one_currency_are_rejected() -> None:
    with pytest.raises(ValueError, match="same currency"):
        ScreenRules(lookback_sessions=3, floors=(USD_FLOOR, USD_FLOOR))


@pytest.mark.parametrize("amount", ["0", "-1"])
def test_a_non_positive_dollar_volume_floor_is_rejected(amount: str) -> None:
    with pytest.raises(ValueError, match="dollar-volume floor must be positive"):
        LiquidityFloor(Money(Decimal(amount), Currency.USD), Price("5"))


def test_a_non_positive_price_floor_is_rejected_by_price_itself() -> None:
    """`LiquidityFloor` adds no check here — `Price` cannot be zero or negative,
    so a second guard would only be a second thing to keep in step."""
    with pytest.raises(ValueError, match="Price must be positive"):
        LiquidityFloor(Money("1000000", Currency.USD), Price("0"))


def test_a_floor_knows_its_own_currency() -> None:
    assert CAD_FLOOR.currency is Currency.CAD


# ── helpers ──────────────────────────────────────────────────


def _close_ns(calendar: FakeCalendar, day: date) -> int:
    session = calendar.session(Venue.NASDAQ, day)
    assert session is not None
    return session.close_ns


def _previous(calendar: FakeCalendar, day: date) -> date:
    return calendar.sessions(Venue.NASDAQ, FIRST, day - timedelta(days=1))[-1]
