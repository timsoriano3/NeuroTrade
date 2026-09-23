"""Session-anchored reference levels — §5.3's "reference level library".

**Why these are not registered features.** A `FeatureSpec` declares a fixed
lookback and is handed exactly that many bars. These levels are anchored to an
*event* instead: the session open, the first fifteen minutes, yesterday's
close. How many bars separate that anchor from now changes every minute of the
session, so there is no lookback to declare. Forcing them into the registry
would mean either lying about the window or recomputing from a window large
enough for the worst case, and both are worse than a plain function that takes
the bars it actually needs.

§5.3 already treats them as a separate thing — "shared infrastructure feeding
every other strategy" — so this follows the spec rather than working around it.

**Every function here takes bars from one session and assumes they are
complete.** They cannot check: a session missing its first ten minutes still
produces an opening range, and that range is simply wrong. The corpus quality
gate (`ingest/quality.py`) is what establishes completeness, which is why it
comes before any of this is trusted.

**Point-in-time is the caller's responsibility here, not enforced.** Unlike
`FeatureSpec.evaluate`, nothing in these signatures carries an `as_of` to check
against. Pass bars up to the decision moment and no further. This is the one
place in the feature library where lookahead is not structurally prevented, so
it is the one place to be careful.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Final

from neurotrade.core.actions import tidy_decimal
from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar
from neurotrade.core.types import Price, Symbol

_PRICE_PLACES = 8
"""Decimal places a computed price keeps — the scale of `PRICE_TYPE` in the
corpus schema, and finer than any North American tick."""

__all__ = [
    "OPENING_WINDOWS",
    "OpeningRange",
    "SessionLevelTracker",
    "SessionLevels",
    "opening_range",
    "session_vwap",
    "vwap_distance",
]

OPENING_WINDOWS: Final = (5, 15, 30, 60)
"""Opening-range windows tracked live, in minutes — the four §5.2 names.

Fixed rather than configurable because each one is a **separate trial**: ORB at
5, 15, 30 and 60 minutes is four hypotheses, not one, and the trial ledger has
to count them that way. A window that is not in this tuple is not tracked, and
asking for it raises rather than returning a quiet `None`."""


@dataclass(frozen=True, slots=True)
class OpeningRange:
    """The high and low of the first minutes of a session.

    The reference ORB trades against (§5.2). Kept as `Price` rather than
    `float` because a breakout level becomes an order's limit price, and an
    order price is money.

    Example:
        >>> rng = OpeningRange(high=Price("101"), low=Price("99"), bar_count=15)
        >>> str(rng.width)
        '2'
    """

    high: Price  # highest high in the opening window
    low: Price  # lowest low in the opening window
    bar_count: int  # bars the range was built from; fewer than asked means a short session

    def __post_init__(self) -> None:
        """Validate the range.

        Raises:
            ValueError: If the high is below the low, which would make every
                breakout test invert silently.
        """
        if self.high < self.low:
            raise ValueError(f"opening range high {self.high} is below low {self.low}")

    @property
    def width(self) -> Decimal:
        """High minus low, in price units."""
        return self.high.value - self.low.value

    def breaks_up(self, price: Price) -> bool:
        """Whether a price has broken above the range.

        Strictly above, not at: a trade *at* the high is the high, and treating
        it as a breakout fires the signal on the bar that set the level.
        """
        return price > self.high

    def breaks_down(self, price: Price) -> bool:
        """Whether a price has broken below the range."""
        return price < self.low


def opening_range(bars: Sequence[Bar], *, minutes: int) -> OpeningRange | None:
    """High and low of the session's first `minutes` bars.

    Args:
        bars: One session's 1-minute bars, oldest first, starting at the open.
        minutes: How many bars to include — 5, 15, 30 and 60 are the windows
            §5.2 names.

    Returns:
        The range, or `None` when the session holds fewer bars than asked for.
        `None` rather than a partial range on purpose: a 15-minute range built
        from 4 bars is a different statistic wearing the same name, and a
        strategy that treated it as the real thing would trade a level nobody
        else is watching.

    Raises:
        ValueError: If `minutes` is below 1.

    Example:
        >>> from neurotrade.core.events import Bar, BarInterval
        >>> from neurotrade.core.types import Quantity, Symbol, Venue
        >>> def b(high, low):
        ...     return Bar(symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=1, ts_init=1,
        ...                interval=BarInterval.MIN_1, open=Price(low), high=Price(high),
        ...                low=Price(low), close=Price(high), volume=Quantity(1))
        >>> found = opening_range([b("101", "99"), b("102", "100")], minutes=2)
        >>> (str(found.high), str(found.low))
        ('102', '99')
        >>> opening_range([b("101", "99")], minutes=2) is None
        True
    """
    if minutes < 1:
        raise ValueError(f"minutes {minutes} must be at least 1")
    if len(bars) < minutes:
        return None
    window = bars[:minutes]
    return OpeningRange(
        high=max(bar.high for bar in window),
        low=min(bar.low for bar in window),
        bar_count=len(window),
    )


def session_vwap(bars: Sequence[Bar], *, up_to: Nanos | None = None) -> Price | None:
    """Volume-weighted average price since the session open.

    *VWAP* is the average price paid across the session, weighting each trade
    by its size — so it reflects where the volume actually changed hands rather
    than where the price happened to close. It is the benchmark institutional
    orders are measured against, which is why price reacts around it (§5.3).

    Each bar contributes its own `vwap` when the feed supplies one and its
    typical price `(high + low + close) / 3` otherwise. Mixing the two is
    deliberate: the feed's own figure is computed from every print inside the
    bar and is strictly better, so use it where it exists rather than
    discarding it for uniformity.

    Args:
        bars: One session's bars, oldest first, starting at the open.
        up_to: Include only bars closing at or before this instant. The
            point-in-time bound — pass the decision moment.

    Returns:
        The VWAP, or `None` when no bar in range carried any volume. `None`
        rather than zero: a session with no volume has no average price paid,
        and zero would read as a price.

        Quantized to eight decimals, the corpus price scale. The raw division
        runs to `Decimal`'s context precision — a real session produced
        `330.2556630065255193527176013` — and a `Price` is an order's limit
        price, so it has to be a number a venue and `decimal128(18, 8)` will
        both accept.

    Example:
        >>> from neurotrade.core.events import Bar, BarInterval
        >>> from neurotrade.core.types import Quantity, Symbol, Venue
        >>> def b(price, volume):
        ...     p = Price(price)
        ...     return Bar(symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=1, ts_init=1,
        ...                interval=BarInterval.MIN_1, open=p, high=p, low=p, close=p,
        ...                volume=Quantity(volume))
        >>> str(session_vwap([b("10", 100), b("20", 300)]))
        '17.5'
    """
    considered = bars if up_to is None else [bar for bar in bars if bar.ts_event <= up_to]
    total_value = Decimal(0)
    total_volume = Decimal(0)
    for bar in considered:
        value, volume = vwap_contribution(bar)
        total_value += value
        total_volume += volume
    return vwap_of(total_value, total_volume)


def vwap_contribution(bar: Bar) -> tuple[Decimal, Decimal]:
    """What one bar adds to a VWAP: weighted value, and the weight.

    Factored out so that the running tracker and the batch function cannot
    disagree about which price a bar contributes — the mixing rule below is the
    kind of detail that gets reimplemented slightly differently and then shows
    up as a backtest that does not match live.

    Args:
        bar: The bar to weigh.

    Returns:
        `(price * volume, volume)`, or `(0, 0)` for a bar with no volume, which
        must not move the average.

    Example:
        >>> value, volume = vwap_contribution(a_bar)   # typical price 100.166..., 1000 shares
        >>> (str(volume), str(vwap_of(value, volume)))
        ('1000', '100.16666667')
    """
    volume = bar.volume.value
    if volume <= 0:
        return Decimal(0), Decimal(0)
    price = (
        bar.vwap.value
        if bar.vwap is not None
        else (bar.high.value + bar.low.value + bar.close.value) / 3
    )
    return price * volume, volume


def vwap_of(total_value: Decimal, total_volume: Decimal) -> Price | None:
    """Divide accumulated value by accumulated volume, at the corpus scale.

    Args:
        total_value: Sum of `price * volume`.
        total_volume: Sum of `volume`.

    Returns:
        The VWAP, or `None` when nothing traded — see `session_vwap`.

    Example:
        >>> str(vwap_of(Decimal(3500), Decimal(200)))
        '17.5'
    """
    if total_volume == 0:
        return None
    return Price(tidy_decimal(total_value / total_volume, _PRICE_PLACES))


def vwap_distance(price: Price, vwap: Price) -> float:
    """How far a price sits from VWAP, as a fraction of VWAP.

    A fraction rather than an absolute distance so the number compares across
    instruments: being 50c above VWAP means something different on a $8 name
    than on a $400 one, and a cross-sectional ranking has to hold both.

    Args:
        price: The price to measure.
        vwap: The session VWAP.

    Returns:
        `(price - vwap) / vwap`. Positive above, negative below.

    No zero-division guard, deliberately: `Price` refuses a non-positive value
    at construction ("Price must be positive"), so a zero VWAP cannot be
    represented and a check for one would be unreachable code pretending to be
    a safeguard.

    Example:
        >>> round(vwap_distance(Price("102"), Price("100")), 4)
        0.02
    """
    return float((price.value - vwap.value) / vwap.value)


@dataclass(frozen=True, slots=True)
class SessionLevels:
    """Where a session has been so far, as of one instant inside it.

    The anchors §5.3 calls shared infrastructure, in one bundle: where the
    session opened, how far it has travelled, what the volume paid on average,
    the opening ranges that have completed, and where yesterday finished.

    Prices are `Price` rather than `float` because every one of them can become
    an order's limit or stop. `gap` is a fraction and therefore a float, like
    any other derived feature.

    Example:
        >>> levels = SessionLevels(
        ...     session_date=date(2024, 7, 8), open_ns=0, close_ns=23_400_000_000_000,
        ...     session_open=Price("100"), high=Price("102"), low=Price("99"),
        ...     close=Price("101"), vwap=Price("100.5"), bar_count=30,
        ...     prior_close=Price("98"),
        ... )
        >>> round(levels.gap, 6)
        0.020408
    """

    session_date: date  # the session these belong to; they reset when it changes
    open_ns: Nanos  # the session's open, UTC nanoseconds
    close_ns: Nanos  # its scheduled close — what "how long is left" is measured against
    session_open: Price  # open of the session's first bar
    high: Price  # highest high since the open
    low: Price  # lowest low since the open
    close: Price  # close of the most recent bar
    vwap: Price | None  # volume-weighted average since the open; None if nothing traded
    bar_count: int  # bars seen since the open, so a caller can tell warm from cold
    prior_close: Price | None  # previous session's final close; None on the first session seen
    opening_ranges: Mapping[int, OpeningRange] = field(default_factory=dict)
    """Completed opening ranges by window, for the windows in `OPENING_WINDOWS`.
    A window appears only once it has filled — see `opening_range`."""

    @property
    def gap(self) -> float | None:
        """Overnight gap as a fraction of the prior close, or `None` without one.

        A fraction rather than an absolute move so it compares across
        instruments, and signed: positive means the session opened above
        yesterday's close.

        Example:
            >>> SessionLevels(
            ...     session_date=date(2024, 7, 8), open_ns=0, close_ns=1,
            ...     session_open=Price("99"), high=Price("99"), low=Price("99"),
            ...     close=Price("99"), vwap=None, bar_count=1, prior_close=Price("100"),
            ... ).gap
            -0.01
        """
        if self.prior_close is None:
            return None
        return float((self.session_open.value - self.prior_close.value) / self.prior_close.value)

    @property
    def range_width(self) -> Decimal:
        """Session high minus session low, in price units."""
        return self.high.value - self.low.value

    def opening_range(self, minutes: int) -> OpeningRange | None:
        """The opening range for one window, once it has completed.

        Args:
            minutes: Window length. Must be one of `OPENING_WINDOWS` — each is
                a separate trial, and a window nobody declared is not tracked.

        Returns:
            The range, or `None` while the session is younger than the window.
            `None` rather than a partial range, for the reason the module-level
            `opening_range` gives: a 15-minute range built from 4 bars is a
            different statistic wearing the same name.

        Raises:
            ValueError: If the window is not tracked.

        Example:
            >>> levels = SessionLevels(
            ...     session_date=date(2024, 7, 8), open_ns=0, close_ns=1,
            ...     session_open=Price("100"), high=Price("102"), low=Price("99"),
            ...     close=Price("101"), vwap=None, bar_count=3, prior_close=None,
            ... )
            >>> levels.opening_range(5) is None          # three bars in
            True
        """
        if minutes not in OPENING_WINDOWS:
            raise ValueError(f"opening range {minutes} is not tracked; {OPENING_WINDOWS} are")
        return self.opening_ranges.get(minutes)


class SessionLevelTracker:
    """Keeps every symbol's session levels current, one bar at a time.

    Streaming rather than batch, and that is the point: the levels are computed
    from bars already seen, so there is no window to pass and no way to pass one
    that reaches past the decision moment. The module docstring's warning that
    point-in-time is the caller's responsibility applies to the plain functions;
    it does not apply here, because nothing downstream ever hands this object a
    bar from the future.

    A session boundary resets everything but the prior close, which is exactly
    what carries across it.

    Example:
        >>> tracker = SessionLevelTracker()
        >>> tracker.levels(AAPL) is None            # nothing observed yet
        True
    """

    __slots__ = ("_levels", "_opening_bars", "_prior_close", "_session", "_value", "_volume")

    def __init__(self) -> None:
        self._levels: dict[Symbol, SessionLevels] = {}
        self._session: dict[Symbol, Nanos] = {}
        self._prior_close: dict[Symbol, Price] = {}
        self._value: dict[Symbol, Decimal] = {}
        self._volume: dict[Symbol, Decimal] = {}
        self._opening_bars: dict[Symbol, list[Bar]] = {}

    def update(self, bar: Bar, session: TradingSession | None) -> None:
        """Fold one bar into its symbol's levels.

        Args:
            bar: The bar that just closed.
            session: The session holding it, or `None`. A bar no session holds
                is ignored rather than folded in: the corpus is fetched with
                `useRTH=1`, so one is a data fault for `ingest/quality.py` to
                report, and accumulating it would put after-hours prints into a
                session VWAP.

        Example:
            >>> from neurotrade.core.calendar import TradingSession
            >>> tracker = SessionLevelTracker()
            >>> tracker.update(a_bar, None)          # outside any session
            >>> tracker.levels(AAPL) is None
            True
        """
        if session is None:
            return
        symbol = bar.symbol
        if self._session.get(symbol) != session.open_ns:
            self._roll(symbol, session)
        value, volume = vwap_contribution(bar)
        self._value[symbol] += value
        self._volume[symbol] += volume
        opening = self._opening_bars[symbol]
        if len(opening) < OPENING_WINDOWS[-1]:
            opening.append(bar)
        previous = self._levels.get(symbol)
        count = previous.bar_count + 1 if previous is not None else 1
        ranges = dict(previous.opening_ranges) if previous is not None else {}
        # A window is computed once, on the bar that completes it, and never
        # again: the range is a property of the first `minutes` bars, so
        # recomputing it every bar would be the same answer at 390 times the
        # cost. `opening` holds exactly `count` bars here, so it cannot be short.
        if count in OPENING_WINDOWS:
            completed = opening_range(opening, minutes=count)
            assert completed is not None
            ranges[count] = completed
        self._levels[symbol] = SessionLevels(
            session_date=session.session_date,
            open_ns=session.open_ns,
            close_ns=session.close_ns,
            session_open=previous.session_open if previous is not None else bar.open,
            high=max(previous.high, bar.high) if previous is not None else bar.high,
            low=min(previous.low, bar.low) if previous is not None else bar.low,
            close=bar.close,
            vwap=vwap_of(self._value[symbol], self._volume[symbol]),
            bar_count=count,
            prior_close=self._prior_close.get(symbol),
            opening_ranges=ranges,
        )

    def levels(self, symbol: Symbol) -> SessionLevels | None:
        """The current levels for one symbol, or `None` before its first bar.

        Args:
            symbol: The instrument.

        Returns:
            The snapshot as of the last bar observed. Immutable, so a strategy
            holding one cannot be surprised by it changing underneath.
        """
        return self._levels.get(symbol)

    def _roll(self, symbol: Symbol, session: TradingSession) -> None:
        """Start a new session, carrying only what survives the bell.

        No snapshot exists between the roll and the session's first bar, which
        is why `levels` can return `None` for a symbol that traded yesterday.
        A snapshot with zero bars in it would have to invent an open, a high and
        a low, and every one of those would be a number a strategy could read.
        """
        finished = self._levels.pop(symbol, None)
        if finished is not None:
            self._prior_close[symbol] = finished.close
        self._session[symbol] = session.open_ns
        self._value[symbol] = Decimal(0)
        self._volume[symbol] = Decimal(0)
        self._opening_bars[symbol] = []

    def __repr__(self) -> str:
        return f"SessionLevelTracker({len(self._levels)} symbols)"
