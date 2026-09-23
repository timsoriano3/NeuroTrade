"""What a strategy is allowed to see at one instant, assembled for real.

`StrategyContext` is the narrow window a strategy gets onto the world: the
instrument, the moment, the venue phase, the regime, and its declared feature
values. This module is what fills it in. Until it existed the lab ran on a stub
that granted nothing, which is why a backtest with a real strategy in it came
back empty.

Three answers are assembled here, and each is deliberately smaller than it
could be:

**The venue phase comes from the calendar, and is `REGULAR` or `CLOSED`.**
`core.calendar` models the regular session only — pre- and post-market bounds
are "deliberately absent until a feed actually delivers pre/post bars", and the
corpus is fetched with `useRTH=1`, so it never does. A bar no session holds is
therefore `CLOSED`, which `MarketSession.is_tradable` already reports as
untradable, rather than being guessed into `PRE` or `POST` from bounds nothing
supplies. A `CLOSED` bar in an RTH corpus is a data fault, not a quiet edge
case, and it is `ingest/quality.py`'s job to say so.

**The regime is time-of-day only.** §5.7's classifier is an HMM and arrives in
Phase 5; fitting a provisional one now would put an unvalidated model in the
decision path of every Phase 2 result and spend trial budget on its thresholds
(§17). But one part of §5.7 needs no fitting at all: "The 12:00-14:00 ET
liquidity lull is treated as a distinct regime and is a default no-trade
window." That is a clock fact. It is classified here and everything else is
`UNKNOWN`, which grants nothing.

The lull is derived as an offset from the session open rather than as a
wall-clock time, which gets three things right for free: daylight saving (the
calendar's open is a real UTC instant for that date), venues that open at the
same local time in a different zone, and half days, where the window is clipped
by an early close instead of running past it.

**Feature values are resolved per strategy, from what it declared.** The union
of every declared feature is computed once per bar, then projected down to each
strategy's own declarations — so the work is shared but the visibility is not.
A strategy still cannot read a feature it did not declare, which is the property
`StrategyContext.feature` exists to enforce, and two strategies pinning
different versions of one feature each get their own (§10.2).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Final

from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import Nanos, to_datetime
from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.ports import CalendarPort
from neurotrade.core.types import Symbol, Venue
from neurotrade.features.registry import FeatureRegistry, FeatureSpec
from neurotrade.features.resolver import FeatureResolver
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

__all__ = [
    "LULL_ENDS_AFTER_OPEN_NS",
    "LULL_STARTS_AFTER_OPEN_NS",
    "MarketContext",
    "RegimeSource",
    "market_phase",
    "time_of_day_regime",
]

_MINUTE_NS: Final = 60 * 1_000_000_000

#: 12:00 ET for an 09:30 open. Expressed as an offset so that DST, venue time
#: zones and early closes come out of the calendar rather than out of a table.
LULL_STARTS_AFTER_OPEN_NS: Final = 150 * _MINUTE_NS

#: 14:00 ET for an 09:30 open. Clipped by the close on a half day.
LULL_ENDS_AFTER_OPEN_NS: Final = 270 * _MINUTE_NS

#: Classifies the regime for one moment, given the session that holds it.
#: Phase 5 replaces the default with the HMM of §5.7; the seam is here so that
#: swapping it is a wiring change rather than an edit to the engine.
RegimeSource = Callable[[TradingSession | None, Nanos], Regime]


def market_phase(session: TradingSession | None, ts: Nanos) -> MarketSession:
    """The venue phase a bar closing at `ts` falls in.

    Args:
        session: The session holding that bar, or `None` if none does.
        ts: The bar's close, UTC nanoseconds. Unused today, and taken anyway so
            that the signature does not change when extended-hours bounds land.

    Returns:
        `REGULAR` when a session holds the bar, `CLOSED` otherwise. Never `PRE`
        or `POST` — see the module docstring.

    Example:
        >>> market_phase(None, 0).is_tradable
        False
    """
    del ts  # part of the contract, not yet part of the answer
    return MarketSession.REGULAR if session is not None else MarketSession.CLOSED


def time_of_day_regime(session: TradingSession | None, ts: Nanos) -> Regime:
    """Classify the one regime that needs no fitted model.

    Args:
        session: The session holding the bar, or `None`.
        ts: The bar's close, UTC nanoseconds.

    Returns:
        `LIQUIDITY_LULL` inside §5.7's midday window, `UNKNOWN` everywhere else
        — including outside a session, where there is nothing to classify.

    Example:
        >>> from datetime import UTC, datetime
        >>> from neurotrade.core.clock import to_nanos
        >>> session = TradingSession(
        ...     venue=Venue.NYSE,
        ...     session_date=date(2024, 7, 8),
        ...     open_ns=to_nanos(datetime(2024, 7, 8, 13, 30, tzinfo=UTC)),
        ...     close_ns=to_nanos(datetime(2024, 7, 8, 20, 0, tzinfo=UTC)),
        ...     is_early_close=False,
        ... )
        >>> noon = to_nanos(datetime(2024, 7, 8, 16, 1, tzinfo=UTC))  # 12:01 ET
        >>> time_of_day_regime(session, noon).value
        'LIQUIDITY_LULL'
    """
    if session is None:
        return Regime.UNKNOWN
    # Open-exclusive and end-inclusive, matching `TradingSession.holds_bar`: a
    # bar stamped 12:00 covers 11:59-12:00 and is not yet in the lull, while one
    # stamped 14:00 covers 13:59-14:00 and still is.
    start = session.open_ns + LULL_STARTS_AFTER_OPEN_NS
    end = min(session.close_ns, session.open_ns + LULL_ENDS_AFTER_OPEN_NS)
    return Regime.LIQUIDITY_LULL if start < ts <= end else Regime.UNKNOWN


class MarketContext:
    """Builds each strategy's view of one bar: phase, regime and features.

    Stateful, and owns the bar history every feature is computed from. Bars are
    fed in through `observe` exactly once each — the engine subscribes it to
    the bus ahead of any strategy, so the history advances whether or not
    anything was eligible to trade on it.

    Example:
        >>> from neurotrade.features.indicators import indicators
        >>> class NoSessions:                       # a venue that never opened
        ...     def sessions(self, venue, start, end): return ()
        ...     def session(self, venue, session_date): return None
        >>> class Quiet(Strategy):
        ...     name, version = "quiet", "1.0.0"
        >>> context = MarketContext(features=indicators, calendar=NoSessions())
        >>> context.declare(Quiet())
        >>> context.observe(a_bar)
        >>> view = context(a_bar, Quiet())
        >>> (view.session.value, view.regime.value)
        ('CLOSED', 'UNKNOWN')
    """

    __slots__ = (
        "_calendar",
        "_features",
        "_interval",
        "_key",
        "_regime_source",
        "_resolver",
        "_sessions",
        "_specs",
        "_started",
        "_values",
    )

    def __init__(
        self,
        *,
        features: FeatureRegistry,
        calendar: CalendarPort,
        interval: BarInterval = BarInterval.MIN_1,
        regime_source: RegimeSource = time_of_day_regime,
    ) -> None:
        """Wire the context to a feature library and a calendar.

        Args:
            features: Library the declared features are looked up in.
            calendar: Source of session bounds, by venue and date.
            interval: Bar size this context is fed. Declared features must
                match it.
            regime_source: Classifier. Defaults to the time-of-day rule; Phase
                5 passes the HMM here instead.
        """
        self._features = features
        self._calendar = calendar
        self._interval = interval
        self._regime_source = regime_source
        self._specs: dict[str, FeatureSpec] = {}
        self._resolver = FeatureResolver({}, interval=interval)
        self._sessions: dict[tuple[Venue, date], TradingSession | None] = {}
        self._key: tuple[Symbol, Nanos] | None = None
        self._values: dict[str, float | None] = {}
        self._started = False

    def declare(self, strategy: Strategy) -> None:
        """Register a strategy's feature dependencies before the run starts.

        Args:
            strategy: The strategy that will be asked for contexts. Declaring
                the same one twice is harmless.

        Raises:
            KeyError: If it declares a feature the library does not hold.
            ValueError: If a declared feature is for a different bar size, or
                if the run has already started — adding a feature mid-run
                would give it a window that begins in the middle of the data.

        Example:
            >>> from neurotrade.features.indicators import indicators
            >>> from neurotrade.strategies.base import FeatureRef
            >>> class NoSessions:
            ...     def sessions(self, venue, start, end): return ()
            ...     def session(self, venue, session_date): return None
            >>> class Trend(Strategy):
            ...     name, version = "trend", "1.0.0"
            ...     features = (FeatureRef("atr"),)
            >>> context = MarketContext(features=indicators, calendar=NoSessions())
            >>> context.declare(Trend())
            >>> context.lookback
            15
        """
        if self._started:
            raise ValueError(f"{strategy.qualified_name} was declared after the run began")
        for ref in strategy.features:
            self._specs[str(ref)] = self._features.get(ref.name, ref.version)
        self._resolver = FeatureResolver(self._specs, interval=self._interval)

    @property
    def lookback(self) -> int:
        """Bars of history needed before every declared feature is warm.

        What `BacktestEngine.run` turns into a warm-up span, so that a strategy
        trading the open is not silently cold for its first `lookback` bars.
        """
        return self._resolver.lookback

    def observe(self, bar: Bar) -> None:
        """Advance the history and resolve every declared feature for this bar.

        Called once per bar, before any strategy is asked for its view of it.
        Re-observing the same instrument and instant is a no-op, so a duplicate
        bar in the corpus cannot enter a feature window twice.

        Args:
            bar: The bar that just closed.

        Raises:
            ValueError: If the bar is not at this context's interval.
        """
        key = (bar.symbol, bar.ts_event)
        if key == self._key:
            return
        self._started = True
        self._resolver.observe(bar)
        self._values = self._resolver.resolve(bar)
        self._key = key

    def __call__(self, bar: Bar, strategy: Strategy) -> StrategyContext:
        """The view of `bar` that `strategy` is entitled to.

        Args:
            bar: The bar being evaluated. Must be the one last observed.
            strategy: The strategy asking. Only its declared features appear.

        Returns:
            A context stamped at the bar's close, carrying the venue phase, the
            regime, and that strategy's features and no others.

        Raises:
            ValueError: If the bar was never observed, or if the strategy was
                never declared — both mean the engine was wired wrong, and a
                context built from stale history is the kind of defect that
                shows up as an unreproducible backtest months later.
        """
        if self._key != (bar.symbol, bar.ts_event):
            raise ValueError(f"{bar.symbol} at {bar.ts_event} was not observed before use")
        values: dict[str, float | None] = {}
        for ref in strategy.features:
            try:
                values[ref.name] = self._values[str(ref)]
            except KeyError:
                raise ValueError(
                    f"{strategy.qualified_name} was never declared to this context"
                ) from None
        session = self._session_for(bar)
        return StrategyContext(
            symbol=bar.symbol,
            # The close, never the open: a context stamped earlier would let a
            # feature computed from this bar be read as if it predated it.
            as_of=bar.ts_event,
            session=market_phase(session, bar.ts_event),
            regime=self._regime_source(session, bar.ts_event),
            values=values,
        )

    def _session_for(self, bar: Bar) -> TradingSession | None:
        """The trading session holding a bar, or `None` if none does.

        Two dates are tried because `ts_event` is UTC and a session is labelled
        in venue-local terms: a US session closing at 16:00 ET in November ends
        at 21:00 UTC on the same date, but anything later in that evening
        belongs to a session the UTC date has already left behind.
        """
        day = to_datetime(bar.ts_event).date()
        for candidate in (day, day - timedelta(days=1)):
            session = self._session(bar.symbol.venue, candidate)
            if session is not None and session.holds_bar(bar.ts_event):
                return session
        return None

    def _session(self, venue: Venue, day: date) -> TradingSession | None:
        """One session, memoised. A backtest asks for the same day 390 times."""
        key = (venue, day)
        if key not in self._sessions:
            self._sessions[key] = self._calendar.session(venue, day)
        return self._sessions[key]

    def __repr__(self) -> str:
        return f"MarketContext({len(self._specs)} features, lookback={self.lookback})"
