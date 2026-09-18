"""The first tranche of registered features (§13 Phase 1).

Every feature here is a pure function of a **fixed trailing window** of bars,
which is what `FeatureSpec` gives it: exactly `lookback` bars, oldest first,
none stamped after the moment being modelled. That shape is deliberate and it
is also a constraint — anything anchored to the session open rather than to a
count of bars does not fit it, and lives in `features/levels.py` instead.

**Chosen to serve the first two strategies, not to be comprehensive.** §20.5
names ORB on Stocks-in-Play and beta-adjusted relative strength as the first
strategies to build, so the set here is what those two need — relative volume
to rank the universe, ATR to normalise a stop, realised volatility for sizing,
returns as the base of everything, an EMA for pullback logic — plus fractional
differentiation, which §8 names as the stationarity treatment.

**Everything returns `float`.** Prices and money are exact `Decimal` (§ the
precision invariant), but an indicator is an estimate: a rounding error in the
twelfth decimal of an ATR cannot cost anything, and the models downstream take
floats regardless. The conversion happens once, here, at the boundary.

## A note on what these do not handle

None of these know about sessions. A 20-bar EMA at 09:35 reaches back into
yesterday's close, because twenty bars is twenty bars. For an intraday strategy
that is usually wrong — the overnight gap is not a price move the strategy
could have traded — and the fix is to hand the feature a window that starts at
the session open, which is the caller's job and the calendar's. Nothing here
can enforce it, so it is written down instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Final

from neurotrade.core.events import Bar
from neurotrade.features.registry import FeatureRegistry

__all__ = [
    "FRACDIFF_D",
    "atr",
    "ema",
    "fracdiff_close",
    "fracdiff_weights",
    "indicators",
    "log_return",
    "realised_volatility",
    "relative_volume",
    "true_range",
]

indicators: Final = FeatureRegistry()
"""The registry these features register into.

Module-level rather than global: a caller that wants a different set — a test,
or a discovery run exploring variants — builds its own `FeatureRegistry` rather
than mutating a shared one. Nothing here reaches for a singleton.
"""

FRACDIFF_D: Final = 0.4
"""Differencing order for `fracdiff`.

Between 0 (raw prices, non-stationary) and 1 (plain returns, stationary but
memoryless). §8 wants "stationary without destroying memory", which is the
whole point of a fractional order: 0.4 is the conventional starting point in
the literature for financial series, and it is a **starting point, not a
result**. The right value is the smallest `d` that passes a stationarity test
on this corpus, and that test is not built yet — so this is a default to be
replaced, and the version string moves when it is.
"""

_ANNUALISATION_MINUTES: Final = 98_280.0
"""1-minute bars in a trading year: 390 per US session x 252 sessions.

Used to annualise a realised-volatility estimate so the number is comparable to
the volatilities everything else is quoted in. The 252 is the US convention;
the TSX differs by a day or two most years, which is far inside the estimate's
own error and not worth a per-venue constant.
"""


def true_range(previous: Bar, current: Bar) -> float:
    """The greater of the bar's own range and its gap from the previous close.

    Plain high-minus-low understates a bar that opened away from yesterday: a
    name that gaps 5% and then trades in a tight range had a violent day, and a
    stop sized off the tight range would be hit immediately.

    Args:
        previous: The bar before `current`.
        current: The bar to measure.

    Returns:
        The true range, in price units.

    Example:
        >>> from neurotrade.core.events import Bar, BarInterval
        >>> from neurotrade.core.types import Price, Quantity, Symbol, Venue
        >>> def b(o, h, low, c):
        ...     return Bar(symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=1, ts_init=1,
        ...                interval=BarInterval.MIN_1, open=Price(o), high=Price(h),
        ...                low=Price(low), close=Price(c), volume=Quantity(1))
        >>> true_range(b("100", "100", "100", "100"), b("105", "106", "104", "105"))
        6.0
    """
    high = float(current.high.value)
    low = float(current.low.value)
    close = float(previous.close.value)
    return max(high - low, abs(high - close), abs(low - close))


def fracdiff_weights(d: float, width: int) -> tuple[float, ...]:
    """Binomial weights for fractional differencing of order `d`.

    The weights come from expanding `(1 - B)^d`, where `B` is the backshift
    operator. Each is derived from the last: `w[k] = -w[k-1] * (d - k + 1) / k`.
    They alternate in sign and decay toward zero, which is what lets a *fixed*
    window approximate an infinite series.

    Args:
        d: Differencing order. 0 leaves the series untouched; 1 is a plain
            first difference.
        width: How many weights to produce, newest first.

    Returns:
        Weights ordered newest-first, so `weights[0]` multiplies the most
        recent observation.

    Raises:
        ValueError: If `width` is below 1.

    Example:
        >>> fracdiff_weights(1.0, 3)
        (1.0, -1.0, 0.0)
    """
    if width < 1:
        raise ValueError(f"width {width} must be at least 1")
    weights = [1.0]
    for k in range(1, width):
        weights.append(-weights[-1] * (d - k + 1) / k)
    return tuple(weights)


@indicators.feature(
    "log_return",
    "1.0.0",
    lookback=2,
    description="Log return from the previous close to the latest close.",
)
def log_return(bars: Sequence[Bar]) -> float:
    """Log return over the window.

    Logs rather than simple returns because they add across time — the return
    over two bars is the sum of the two one-bar returns — which makes every
    aggregation downstream correct by construction rather than approximately
    right for small moves.

    Args:
        bars: Exactly two bars, oldest first.

    Returns:
        `ln(last_close / first_close)`.

    Example:
        >>> round(math.log(101 / 100), 6)
        0.00995
    """
    first = float(bars[0].close.value)
    last = float(bars[-1].close.value)
    return math.log(last / first)


@indicators.feature(
    "atr",
    "1.0.0",
    lookback=15,
    description="Average true range over 14 bars; the unit stops and sizes are quoted in.",
)
def atr(bars: Sequence[Bar]) -> float:
    """Average true range over the window.

    A simple mean rather than Wilder's smoothing. Wilder's is an EMA with a
    long memory, which means the value depends on where the series started —
    fine live, wrong in a backtest that slices windows arbitrarily, because the
    same bar would get a different ATR depending on how much history the slice
    happened to include. A plain mean over a declared window is reproducible
    from the window alone, which is what `FeatureSpec` promises.

    Args:
        bars: 15 bars, oldest first. One more than the 14 ranges, because a
            true range needs the previous close.

    Returns:
        Mean true range in price units.
    """
    ranges = [true_range(previous, current) for previous, current in pairwise(bars)]
    return sum(ranges) / len(ranges)


@indicators.feature(
    "realised_vol",
    "1.0.0",
    lookback=21,
    description="Annualised realised volatility from 20 one-bar log returns.",
)
def realised_volatility(bars: Sequence[Bar]) -> float:
    """Annualised standard deviation of log returns over the window.

    Population standard deviation, not sample: the window is the whole
    population being described, and there is no inference to a wider one here.

    Args:
        bars: 21 bars, oldest first — 20 returns.

    Returns:
        Annualised volatility as a fraction, so 0.35 is 35%.
    """
    returns = [
        math.log(float(current.close.value) / float(previous.close.value))
        for previous, current in pairwise(bars)
    ]
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return math.sqrt(variance) * math.sqrt(_ANNUALISATION_MINUTES)


@indicators.feature(
    "rvol",
    "1.0.0",
    lookback=21,
    description="Latest bar's volume over the mean of the 20 before it.",
)
def relative_volume(bars: Sequence[Bar]) -> float:
    """How unusual the latest bar's volume is.

    The ranking signal behind "Stocks in Play" (§5.2): a name trading three
    times its normal volume is where the day's participation is, and ORB is
    only documented to work on such names.

    Returns `0.0` when the baseline is zero — a window in which nothing traded
    at all. That is a real state for an illiquid name, and the alternative is
    a division by zero or an infinity that propagates into every model input
    downstream.

    Args:
        bars: 21 bars, oldest first.

    Returns:
        A ratio. 1.0 is an ordinary bar; 3.0 is three times normal.
    """
    baseline = [float(bar.volume.value) for bar in bars[:-1]]
    average = sum(baseline) / len(baseline)
    if average == 0.0:
        return 0.0
    return float(bars[-1].volume.value) / average


@indicators.feature(
    "ema",
    "1.0.0",
    lookback=20,
    description="20-bar exponential moving average of the close, seeded from the window.",
)
def ema(bars: Sequence[Bar]) -> float:
    """Exponential moving average over the window.

    Seeded with the first close in the window rather than carried across
    windows, for the reason `atr` gives: a value that depends on unseen history
    is not reproducible from its declared lookback, and the backtest and the
    live engine would then disagree about the same bar. The cost is that the
    first few bars of the window weigh more than they would in a continuously
    maintained EMA; the benefit is that the number means one thing.

    Args:
        bars: 20 bars, oldest first.

    Returns:
        The EMA of the closes, in price units.
    """
    span = len(bars)
    alpha = 2.0 / (span + 1.0)
    value = float(bars[0].close.value)
    for bar in bars[1:]:
        value = alpha * float(bar.close.value) + (1.0 - alpha) * value
    return value


@indicators.feature(
    "fracdiff_close",
    "1.0.0",
    lookback=32,
    description="Fractionally differenced close (d=0.4) — stationary, memory retained (§8).",
)
def fracdiff_close(bars: Sequence[Bar]) -> float:
    """Fractionally differenced close over the window.

    Prices are not stationary, and most models assume stationarity. Taking
    plain returns fixes that and throws away the level — after differencing,
    "$400, falling" and "$40, falling" are the same series. Fractional
    differencing takes the smallest difference that achieves stationarity, so
    the memory of the level survives (§8).

    The window is finite, so this is the fixed-width approximation: weights
    beyond the lookback are dropped. They are small by then — that is why the
    truncation is tolerable — but it does mean the value depends on the window
    width as well as on `d`, and both belong to the feature's version.

    Args:
        bars: 32 bars, oldest first.

    Returns:
        The differenced value at the latest bar. Not a price, and not
        comparable across instruments without further scaling.
    """
    weights = fracdiff_weights(FRACDIFF_D, len(bars))
    # weights[0] multiplies the newest observation, so walk the window backwards.
    closes = [float(bar.close.value) for bar in reversed(bars)]
    return sum(weight * close for weight, close in zip(weights, closes, strict=True))
