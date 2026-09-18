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

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from neurotrade.core.actions import tidy_decimal
from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar
from neurotrade.core.types import Price

_PRICE_PLACES = 8
"""Decimal places a computed price keeps — the scale of `PRICE_TYPE` in the
corpus schema, and finer than any North American tick."""

__all__ = [
    "OpeningRange",
    "opening_range",
    "session_vwap",
    "vwap_distance",
]


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
        volume = bar.volume.value
        if volume <= 0:
            continue
        price = (
            bar.vwap.value
            if bar.vwap is not None
            else (bar.high.value + bar.low.value + bar.close.value) / 3
        )
        total_value += price * volume
        total_volume += volume
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
