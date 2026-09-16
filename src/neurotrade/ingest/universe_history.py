"""Point-in-time universe membership from the daily-bar corpus.

§12.1 stage 3, second half: "daily bars and universe history via yfinance".
Stage 3's first half filled `derived/daily/`; this turns that corpus into a
record of *who was tradable when*, so a backtest at date `t` sees the universe
as it was at `t` rather than as it is today.

**What the screen is, in plain terms.** An instrument is eligible for a session
if, over its previous `lookback_sessions` sessions, the typical day's traded
value (close price times shares, so "dollar volume") cleared a floor, and its
last close cleared a price floor. Both filters exist to keep out names that
cannot absorb an order without moving — a strategy that backtests beautifully on
a stock trading $40k a day has discovered nothing.

**The point-in-time rule is the whole job.** A bar is stamped at its session
close, so the bar for session `t` is *not* observable when the decision to
include `t` is made before the open. The window therefore ends at `t-1`. Getting
this wrong is a look-ahead leak of exactly the kind §17 names as the primary
risk, and it would flatter every backtest run afterwards.

**No currency conversion.** US and Canadian names are screened against their own
floors in their own currency. `Money` raises on cross-currency comparison, and
that is load-bearing here rather than incidental: a single "$10M" threshold
applied across both would silently mean two different things.

This module reads through `StoragePort` and knows nothing about Parquet or
Yahoo, in keeping with ingest depending on core alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.ports import CalendarPort, StoragePort
from neurotrade.core.types import Currency, Money, Price, Symbol
from neurotrade.core.universe import Universe, UniverseHistory, UniverseMembership

__all__ = ["LiquidityFloor", "ScreenRules", "screen_universe"]

_SLACK_DAYS = 14
"""Calendar days of headroom when reaching back for the first window. North
American venues trade about five days in seven, so twice the session count plus
a fortnight covers the longest holiday run. The arithmetic is checked rather
than trusted — see `_window_start`."""


@dataclass(frozen=True, slots=True)
class LiquidityFloor:
    """The bar an instrument must clear, in one currency.

    The currency lives on `min_median_dollar_volume` rather than as a separate
    field, so a floor cannot be built that disagrees with itself.
    """

    min_median_dollar_volume: Money  # median traded value over the window, at least this
    min_close: Price  # last close, at least this; in the same currency

    def __post_init__(self) -> None:
        if self.min_median_dollar_volume.amount <= 0:
            raise ValueError("a dollar-volume floor must be positive")
        # No check on min_close: `Price` is already positive by construction.

    @property
    def currency(self) -> Currency:
        """The currency this floor is denominated in."""
        return self.min_median_dollar_volume.currency


@dataclass(frozen=True, slots=True)
class ScreenRules:
    """The liquidity screen: one floor per currency, plus the window length.

    Example:
        >>> rules = ScreenRules(
        ...     lookback_sessions=20,
        ...     floors=(
        ...         LiquidityFloor(Money("20000000", Currency.USD), Price("5")),
        ...         LiquidityFloor(Money("5000000", Currency.CAD), Price("5")),
        ...     ),
        ... )
        >>> rules.floor_for(Currency.CAD).min_close
        Price(value=Decimal('5'))
    """

    lookback_sessions: int  # sessions in the trailing window, all strictly before the decision
    floors: tuple[LiquidityFloor, ...]  # one per currency; a currency with none is an error

    def __post_init__(self) -> None:
        if self.lookback_sessions < 1:
            raise ValueError("a lookback window needs at least one session")
        currencies = [floor.currency for floor in self.floors]
        if not currencies:
            raise ValueError("a screen with no floors admits everything")
        if len(set(currencies)) != len(currencies):
            raise ValueError("two floors given for the same currency")

    def floor_for(self, currency: Currency) -> LiquidityFloor:
        """The floor for one currency.

        Raises:
            KeyError: If no floor covers it. Passing an unscreened instrument
                would be the wrong default: silence here means a TSX name
                judged by a US threshold, or by none at all.
        """
        for floor in self.floors:
            if floor.currency is currency:
                return floor
        raise KeyError(f"no liquidity floor configured for {currency.value}")


def screen_universe(
    universe: Universe,
    calendar: CalendarPort,
    store: StoragePort,
    *,
    start: date,
    end: date,
    rules: ScreenRules,
    survivorship_biased: bool,
    interval: BarInterval = BarInterval.DAY_1,
) -> UniverseHistory:
    """Build a point-in-time membership history over the daily corpus.

    Args:
        universe: The candidates. Membership is a subset of this on every date;
            the screen narrows, it never discovers new names.
        calendar: Supplies each venue's sessions. A symbol is only considered on
            days its own venue traded, so a TSX holiday drops Canadian names
            from that date rather than carrying them forward.
        store: The daily corpus, read through the port.
        start: First session to decide membership for.
        end: Last session to decide membership for, inclusive.
        rules: Window length and per-currency floors.
        survivorship_biased: Whether the corpus behind `store` could only see
            names that still trade. Travels onto the result.
        interval: Bar size to screen on. Daily by design — the whole point is a
            cheap, long history.

    Returns:
        A `UniverseHistory` with one row per session date in range. Rows inside
        the first `lookback_sessions` of the corpus are empty by construction:
        no instrument has a full window yet, and a short window is not a
        smaller sample, it is a different screen.

    Raises:
        ValueError: If `end` precedes `start`, or if a bar's timestamp matches
            no session on its venue's calendar. The second means the corpus and
            the calendar disagree about when a venue traded, which makes every
            window boundary suspect — recompute `derived/`, do not paper over it.

    Example:
        >>> rules = ScreenRules(
        ...     lookback_sessions=1,
        ...     floors=(LiquidityFloor(Money("1", Currency.USD), Price("1")),),
        ... )
        >>> rules.lookback_sessions
        1
    """
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")

    window_start = _window_start(universe, calendar, start, rules.lookback_sessions)
    by_symbol = {
        symbol: _sessions_by_date(symbol, calendar, store, interval, window_start, end)
        for symbol in universe
    }
    # NYSE and TSX keep different holidays. A date is a decision point if ANY
    # venue traded on it, but an instrument is only eligible on dates its OWN
    # venue traded — Canada Day drops the TSX lines and keeps the US ones.
    venue_sessions = {
        venue: frozenset(calendar.sessions(venue, start, end)) for venue in universe.venues
    }
    evaluation_dates = sorted(
        {session_date for dates in venue_sessions.values() for session_date in dates}
    )

    rows = [
        UniverseMembership(
            session_date,
            [
                symbol
                for symbol in universe
                if session_date in venue_sessions[symbol.venue]
                and _passes(symbol, by_symbol[symbol], session_date, rules)
            ],
        )
        for session_date in evaluation_dates
    ]
    return UniverseHistory(rows, survivorship_biased=survivorship_biased)


def _window_start(universe: Universe, calendar: CalendarPort, start: date, lookback: int) -> date:
    """The earliest date the first trailing window can need.

    Reaching back a fixed number of calendar days is an estimate, so it is
    verified: every venue must really have `lookback` sessions between the date
    returned and `start`. Getting this wrong would not raise on its own — it
    would quietly fail the warm-up check on the first evaluation dates and
    exclude instruments that belonged in the universe.

    Raises:
        ValueError: If the reach-back proves too short for some venue.
    """
    reach_back = start - timedelta(days=lookback * 2 + _SLACK_DAYS)
    for venue in universe.venues:
        prior = calendar.sessions(venue, reach_back, start - timedelta(days=1))
        if len(prior) < lookback:
            raise ValueError(
                f"{venue.value} has only {len(prior)} sessions in the "
                f"{lookback * 2 + _SLACK_DAYS} days before {start}, short of the "
                f"{lookback} the window needs"
            )
    return reach_back


def _sessions_by_date(
    symbol: Symbol,
    calendar: CalendarPort,
    store: StoragePort,
    interval: BarInterval,
    window_start: date,
    end: date,
) -> tuple[tuple[date, Bar], ...]:
    """One symbol's bars, ascending, each paired with the session it closed.

    The pairing comes from the calendar rather than from the timestamp's UTC
    date. They agree for North American venues today — every close lands before
    midnight UTC — but that is a property of these venues' hours, not a rule,
    and a screen that silently shifted by a day on a venue that closes later
    would be very hard to see.
    """
    closes: dict[int, date] = {}
    opens: list[int] = []
    for session in calendar.sessions(symbol.venue, window_start, end):
        detail = calendar.session(symbol.venue, session)
        if detail is not None:
            closes[detail.close_ns] = session
            opens.append(detail.open_ns)
    if not opens:
        return ()

    paired: list[tuple[date, Bar]] = []
    # Bounded at both ends by the window's own sessions. A daily bar is stamped
    # at its close, so this admits every bar in the window and nothing outside
    # it — the corpus normally runs past `end`, and those bars belong to
    # sessions this build was not asked about.
    for bar in store.read_bars(symbol, interval, min(opens), max(closes) + 1):
        session_date = closes.get(bar.ts_event)
        if session_date is None:
            raise ValueError(
                f"{symbol} has a bar at {bar.ts_event} matching no {symbol.venue.value} "
                f"session close; the corpus and the calendar disagree"
            )
        paired.append((session_date, bar))
    paired.sort(key=lambda pair: pair[0])
    return tuple(paired)


def _passes(
    symbol: Symbol,
    history: Sequence[tuple[date, Bar]],
    session_date: date,
    rules: ScreenRules,
) -> bool:
    """Whether one instrument clears the screen for one session.

    The `<` is the point-in-time boundary: the session being decided contributes
    nothing to the decision.
    """
    window = [bar for day, bar in history if day < session_date][-rules.lookback_sessions :]
    if len(window) < rules.lookback_sessions:
        return False  # warming up; a short window is a different screen, not a weaker one

    floor = rules.floor_for(symbol.currency)
    if window[-1].close < floor.min_close:
        return False

    # Money comparison enforces the currency match. A CAD median tested against
    # a USD floor raises here rather than quietly ranking the two together.
    traded = [Money.of(bar.close, bar.volume, symbol.currency).amount for bar in window]
    typical = Money(median(traded), symbol.currency)
    return typical >= floor.min_median_dollar_volume
