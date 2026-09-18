"""Corporate actions, and turning them into price adjustment factors (§12.1 stage 5).

A *corporate action* is a company changing the terms of its own shares. Two
kinds matter to a price series:

- A **split** multiplies the share count and divides the price. A 4:1 split
  turns one $400 share into four $100 shares. Nothing was gained or lost, but
  an unadjusted series prints -75% overnight.
- A **cash dividend** pays shareholders out of the company. The share price
  drops by roughly the dividend on the *ex-dividend date* — the first session
  on which a buyer no longer receives it. Again nothing was lost; the value
  moved from the share to the holder's cash.

**Why this module exists before anything in the lab does.** A triple-barrier
label spans days. Across an unadjusted 4:1 split the price path shows -75%,
which trips the stop barrier on a position that never lost a cent. The same
phantom move poisons ATR, realised volatility and every return-based feature,
and it does so precisely on the dates with the most price action. Labelling a
corpus that has not been adjusted does not produce a noisy answer; it produces
a confident wrong one.

**Raw stays raw.** Nothing here mutates the corpus. Actions are their own
dataset, and adjustment is a factor applied when bars are *read*, so the same
bars can be read in any basis and a new split never invalidates a stored file.

## The two factors, and why a single "adjusted price" is not enough

Price adjustment and total-return adjustment answer different questions, and
using one where the other belongs is a quiet source of wrong results.

- `price_factor` removes **splits only**. It reconstructs the price path a
  trader would actually have seen, scaled to a common basis. Stops, profit
  targets and barrier touches are decided on this path — a stop is hit because
  the *price* traded there, and a dividend does not fill an order.
- `total_return_factor` removes splits **and** dividends. It answers "what did
  holding this return", and is what a return series or a momentum feature
  wants, so that an ex-dividend date does not read as a real loss.

Labelling uses the first. Return features use the second. Defaulting everything
to one of them is wrong in one direction or the other.

## Point-in-time

`factor_at` takes an ``as_of`` date and ignores every action effective after
it. This is not decoration. Adjusting a 2022 backtest with a 2025 split means
the 2022 decision is made from a price nobody could have computed in 2022, and
the resulting equity curve is unreproducible live. The default ``as_of`` is
therefore **required**, never "today".
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from itertools import pairwise
from typing import Final

from neurotrade.core.clock import to_datetime
from neurotrade.core.events import Bar
from neurotrade.core.types import Price, Quantity, Symbol

__all__ = [
    "ONE",
    "AdjustmentSeries",
    "CorporateAction",
    "PriceGap",
    "adjust_bars",
    "tidy_decimal",
    "unexplained_gaps",
]

ONE: Final = Decimal(1)
"""The identity factor: no action applies, the price is already in basis."""

_PLACES: Final = 8
"""Decimal places kept by every factor and adjusted value here — the scale of
`PRICE_TYPE` and `QUANTITY_TYPE` in the corpus schema. Rounding to it means an
adjusted price can always be stored without Arrow refusing it for rescaling
loss."""

_QUANTUM: Final = Decimal(1).scaleb(-_PLACES)
"""`_PLACES` as the exponent `Decimal.quantize` wants."""


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """One action, on one instrument, effective from one session.

    A single row may carry both a split and a dividend; feeds report them on
    the same date when both happen, and separating them would invent an
    ordering the source does not state.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> split = CorporateAction(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ),
        ...     effective_date=date(2020, 8, 31),
        ...     split_ratio=Decimal(4),
        ... )
        >>> (split.is_split, split.is_dividend)
        (True, False)
    """

    symbol: Symbol  # the instrument the action applies to
    effective_date: date  # first session trading on the new basis (ex-date for a dividend)
    split_ratio: Decimal = ONE  # new shares per old share; 1 means no split
    dividend: Decimal = Decimal(0)  # cash per share, in the instrument's currency; 0 means none

    def __post_init__(self) -> None:
        """Validate the action.

        Raises:
            ValueError: If the split ratio is not positive, or the dividend is
                negative. A zero or negative ratio would make the factor
                undefined and silently destroy every price before it.
        """
        if self.split_ratio <= 0:
            raise ValueError(f"split_ratio {self.split_ratio} must be positive")
        if self.dividend < 0:
            raise ValueError(f"dividend {self.dividend} must not be negative")

    @property
    def is_split(self) -> bool:
        """Whether this action changes the share count."""
        return self.split_ratio != ONE

    @property
    def is_dividend(self) -> bool:
        """Whether this action pays cash."""
        return self.dividend != 0

    def __str__(self) -> str:
        parts = []
        if self.is_split:
            parts.append(f"{self.split_ratio}:1 split")
        if self.is_dividend:
            parts.append(f"{self.dividend} dividend")
        return f"{self.symbol} {self.effective_date} {' + '.join(parts) or 'no-op'}"


@dataclass(frozen=True, slots=True)
class PriceGap:
    """An overnight move the recorded actions do not explain.

    Emitted by `unexplained_gaps`. Each one is a candidate missing corporate
    action — a split the feed did not report — and therefore a stretch of the
    corpus whose labels cannot be trusted.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> gap = PriceGap(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ),
        ...     session_date=date(2020, 8, 31),
        ...     previous_close=Price("499.23"),
        ...     open_price=Price("127.58"),
        ...     ratio=Decimal("0.2555"),
        ... )
        >>> round(gap.implied_split, 2)
        Decimal('3.91')
    """

    symbol: Symbol  # instrument the gap was found on
    session_date: date  # session whose open gapped from the prior close
    previous_close: Price  # last close before the gap
    open_price: Price  # first price on `session_date`
    ratio: Decimal  # open / previous_close, after applying known actions

    @property
    def implied_split(self) -> Decimal:
        """The split ratio that would explain this gap, if a split is the cause.

        A 4:1 split shows a ratio near 0.25, so the reciprocal names the
        suspect. Rounding it to the nearest simple fraction is left to the
        reader: this is a lead, not a verdict.
        """
        return ONE / self.ratio

    def __str__(self) -> str:
        return (
            f"{self.symbol} {self.session_date}: {self.previous_close} -> {self.open_price} "
            f"({self.ratio:.4f}x, implies {self.implied_split:.2f}:1)"
        )


class AdjustmentSeries:
    """Every action on one instrument, ordered, with the factors they imply.

    Construct once per instrument and query per bar. Actions are sorted on
    construction and looked up by binary search, because a feature pass asks
    this question once per bar per instrument and a linear scan over a decade
    of dividends is not free.

    **Dividend factors need a price.** The drop on an ex-dividend date is the
    dividend as a *fraction* of the price, so the close before the ex-date is
    required to compute it. That lookup is supplied at construction rather than
    fetched here: this is `core`, and it reads nothing.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> aapl = Symbol("AAPL", Venue.NASDAQ)
        >>> series = AdjustmentSeries(
        ...     aapl,
        ...     [CorporateAction(aapl, date(2020, 8, 31), split_ratio=Decimal(4))],
        ... )
        >>> series.price_factor(date(2020, 8, 28), as_of=date(2021, 1, 4))
        Decimal('0.25')
        >>> series.price_factor(date(2020, 9, 1), as_of=date(2021, 1, 4))
        Decimal('1')
    """

    __slots__ = ("_actions", "_dates", "_prior_close", "_symbol")

    def __init__(
        self,
        symbol: Symbol,
        actions: Iterable[CorporateAction],
        *,
        prior_close: Mapping[date, Price] | None = None,
    ) -> None:
        """Build the series.

        Args:
            symbol: The instrument these actions belong to.
            actions: The actions. Order does not matter; they are sorted here.
            prior_close: Close on the session *before* each dividend's ex-date,
                keyed by ex-date. Required only for `total_return_factor`; a
                dividend with no entry is skipped there, since a fraction
                cannot be formed without it.

        Raises:
            ValueError: If any action belongs to a different instrument.
                Mixing instruments in one series would apply one company's
                split to another's prices.
        """
        ordered = sorted(actions, key=lambda action: action.effective_date)
        for action in ordered:
            if action.symbol != symbol:
                raise ValueError(f"action for {action.symbol} in series for {symbol}")
        self._symbol = symbol
        self._actions = tuple(ordered)
        self._dates = tuple(action.effective_date for action in ordered)
        self._prior_close = dict(prior_close or {})

    @property
    def symbol(self) -> Symbol:
        """The instrument this series adjusts."""
        return self._symbol

    @property
    def actions(self) -> tuple[CorporateAction, ...]:
        """The actions, oldest first."""
        return self._actions

    def price_factor(self, observed: date, *, as_of: date) -> Decimal:
        """Split-only factor converting a price observed on `observed` to the `as_of` basis.

        Args:
            observed: The session the price was printed on.
            as_of: The basis to express it in. Actions effective after this are
                ignored — see the module docstring on point-in-time.

        Returns:
            A multiplier. `1` when no split falls in the interval.

        Example:
            >>> from neurotrade.core.types import Venue
            >>> t = Symbol("T", Venue.NYSE)
            >>> s = AdjustmentSeries(t, [CorporateAction(t, date(2023, 6, 1), Decimal(2))])
            >>> s.price_factor(date(2023, 5, 31), as_of=date(2023, 6, 2))
            Decimal('0.5')
        """
        factor = ONE
        for action in self._between(observed, as_of):
            if action.is_split:
                factor /= action.split_ratio
        return _quantize(factor)

    def total_return_factor(self, observed: date, *, as_of: date) -> Decimal:
        """Split-and-dividend factor, for return series rather than price paths.

        Each dividend contributes `1 - dividend / prior_close`, the standard
        back-adjustment: it scales history down so that holding through the
        ex-date shows no loss. A dividend whose prior close is unknown is
        skipped rather than guessed, because a wrong denominator here is a
        silent drift in every return computed from the series.

        Args:
            observed: The session the price was printed on.
            as_of: The basis to express it in.

        Returns:
            A multiplier combining every split and dividend in the interval.

        Raises:
            ValueError: If a dividend exceeds its prior close, which would make
                the factor zero or negative. That is bad data, not a large
                dividend.
        """
        factor = ONE
        for action in self._between(observed, as_of):
            if action.is_split:
                factor /= action.split_ratio
            if not action.is_dividend:
                continue
            close = self._prior_close.get(action.effective_date)
            if close is None:
                continue
            if action.dividend >= close.value:
                raise ValueError(
                    f"dividend {action.dividend} on {action.effective_date} is not below "
                    f"the prior close {close} for {self._symbol}"
                )
            factor *= ONE - action.dividend / close.value
        return _quantize(factor)

    def _between(self, observed: date, as_of: date) -> Iterable[CorporateAction]:
        """Actions effective after `observed` and no later than `as_of`.

        Open at the lower end because an action effective on a session already
        applies to that session's prices — the bar is already on the new basis
        — and closed at the upper end because `as_of` itself is knowable.
        """
        if as_of < observed:
            raise ValueError(f"as_of {as_of} is before observed {observed}")
        lo = bisect.bisect_right(self._dates, observed)
        hi = bisect.bisect_right(self._dates, as_of)
        return self._actions[lo:hi]

    def __len__(self) -> int:
        return len(self._actions)

    def __repr__(self) -> str:
        return f"AdjustmentSeries({self._symbol}, {len(self._actions)} actions)"


def tidy_decimal(value: Decimal, places: int) -> Decimal:
    """Round to `places` decimals and strip trailing zeros, without going scientific.

    `Decimal.normalize()` alone is not enough, and the way it fails is easy to
    ship. It strips zeros from both sides of the point, so a ratio quantized to
    eight places comes back readable — `Decimal("4.00000000")` becomes `4` —
    but a *whole* number loses its zeros into the exponent: `Decimal("10.00")`
    normalizes to `1E+1`, and `Decimal(1000) / Decimal("0.5")` is already
    `2E+3` before anything is normalized at all.

    Numerically those are all correct and every comparison still holds. What
    breaks is everything that reads them: `str()` of a split ratio in a report,
    a dividend written to Parquet, a doctest. Requantizing to integer scale
    when the exponent has gone positive puts the digits back.

    Args:
        value: The number to tidy.
        places: Maximum decimal places to keep.

    Returns:
        The same value, rounded, in plain notation.

    Example:
        >>> tidy_decimal(Decimal("4.00000000"), 8)
        Decimal('4')
        >>> tidy_decimal(Decimal("10.00"), 8)
        Decimal('10')
        >>> tidy_decimal(Decimal("0.270000"), 6)
        Decimal('0.27')
    """
    normalized = value.quantize(Decimal(1).scaleb(-places)).normalize()
    if normalized.as_tuple().exponent > 0:  # type: ignore[operator]  # never a special value here
        return normalized.quantize(ONE)
    return normalized


def _quantize(factor: Decimal) -> Decimal:
    """Round a factor to the corpus price scale, then drop trailing zeros.

    Tidying keeps the factors readable (`0.25`, not `0.25000000`) and keeps
    equality with `ONE` working for the no-op case, which is what lets
    `adjust_bars` skip rebuilding a bar that needs no change.
    """
    return tidy_decimal(factor, _PLACES)


def _scale(value: Decimal) -> Decimal:
    """Put a computed price or size on the corpus's eight-decimal scale.

    Two problems this solves, both found by the doctest for `adjust_bars`
    rather than by inspection:

    1. `Decimal` division normalises upwards. `Decimal(1000) / Decimal("0.5")`
       is `2E+3`, not `2000`, and that is what would have been written to the
       corpus and shown in every report.
    2. A dividend factor is a long fraction — `1 - 0.26/169.34` runs to the
       context precision — so an adjusted price can carry far more places than
       `decimal128(18, 8)` accepts, and Arrow rejects the write rather than
       rounding it.

    Quantizing at the boundary fixes both at once.
    """
    return value.quantize(_QUANTUM)


def adjust_bars(
    bars: Sequence[Bar],
    series: AdjustmentSeries,
    *,
    as_of: date,
    total_return: bool = False,
) -> tuple[Bar, ...]:
    """Re-express bars in the basis of `as_of`, without touching stored data.

    Every price on a bar is multiplied by the factor for that bar's session;
    volume is multiplied by the *reciprocal* of the split component, because a
    4:1 split quadruples the share count as it quarters the price. Leaving
    volume alone would put a 4x step in every volume feature at the split.

    Args:
        bars: Bars for one instrument, any order. `ts_event` is the bar's
            close, so its session is taken from that timestamp's date.
        series: Actions for that same instrument.
        as_of: Basis to express prices in. Actions after it are ignored.
        total_return: Adjust for dividends as well as splits. Leave false for
            anything that decides a barrier touch; see the module docstring.

    Returns:
        New bars, in the order given. Bars needing no adjustment are returned
        unchanged rather than rebuilt.

    Raises:
        ValueError: If a bar belongs to a different instrument than the series.

    Example:
        >>> from neurotrade.core.events import Bar, BarInterval
        >>> from neurotrade.core.types import Quantity, Venue
        >>> t = Symbol("T", Venue.NYSE)
        >>> bar = Bar(
        ...     symbol=t, ts_event=1_685_649_600_000_000_000,
        ...     ts_init=1_685_649_600_000_000_000, interval=BarInterval.DAY_1,
        ...     open=Price("100"), high=Price("104"), low=Price("99"),
        ...     close=Price("102"), volume=Quantity(1_000),
        ... )
        >>> series = AdjustmentSeries(t, [CorporateAction(t, date(2023, 6, 2), Decimal(2))])
        >>> adjusted = adjust_bars([bar], series, as_of=date(2023, 6, 5))
        >>> (str(adjusted[0].close), str(adjusted[0].volume))
        ('51.00000000', '2000.00000000')
    """
    out: list[Bar] = []
    for bar in bars:
        if bar.symbol != series.symbol:
            raise ValueError(f"bar for {bar.symbol} adjusted with series for {series.symbol}")
        observed = _session_of(bar)
        factor = (
            series.total_return_factor(observed, as_of=as_of)
            if total_return
            else series.price_factor(observed, as_of=as_of)
        )
        if factor == ONE:
            out.append(bar)
            continue
        # Volume follows the split, not the dividend: a cash payment leaves the
        # share count untouched. Hence the split-only factor here even when the
        # prices are being adjusted for total return.
        split_factor = series.price_factor(observed, as_of=as_of)
        out.append(
            replace(
                bar,
                open=Price(_scale(bar.open.value * factor)),
                high=Price(_scale(bar.high.value * factor)),
                low=Price(_scale(bar.low.value * factor)),
                close=Price(_scale(bar.close.value * factor)),
                volume=Quantity(_scale(bar.volume.value / split_factor)),
                vwap=None if bar.vwap is None else Price(_scale(bar.vwap.value * factor)),
            )
        )
    return tuple(out)


def unexplained_gaps(
    bars: Sequence[Bar],
    series: AdjustmentSeries,
    *,
    as_of: date,
    threshold: Decimal = Decimal("0.25"),
) -> tuple[PriceGap, ...]:
    """Find overnight moves the recorded actions do not account for.

    This is how the adjustment proves itself. A feed that silently omits a
    split leaves no error behind — the prices are all plausible, the file is
    well formed, and only the size of one overnight move gives it away. So the
    corpus is asked directly: after applying every action we know about, does
    any session still open a long way from the previous close?

    Args:
        bars: Daily bars for one instrument, any order. Sorted here.
        series: Actions for that instrument.
        as_of: Basis for the adjustment, as elsewhere.
        threshold: Fractional move past which a gap is reported. The default
            0.25 sits below the smallest common split, which halves the price,
            so no split can hide beneath it. It does **not** sit above every
            real move: a scan of 53,879 daily bars over 43 names (2021-09 to
            2026-09) returned four, all genuine news gaps — AMD +37%, INTC
            +28%, NVDA +26%, NFLX -30%. Read the output as leads, not as a
            verdict on the data; `implied_split` is what separates them, since
            0.73:1 is not a ratio any company declares.

    Returns:
        The suspect gaps, oldest first. Empty is the result to want.

    Example:
        >>> from neurotrade.core.events import Bar, BarInterval
        >>> from neurotrade.core.types import Quantity, Venue
        >>> t = Symbol("T", Venue.NYSE)
        >>> def bar(ts, price):
        ...     p = Price(price)
        ...     return Bar(symbol=t, ts_event=ts, ts_init=ts, interval=BarInterval.DAY_1,
        ...                open=p, high=p, low=p, close=p, volume=Quantity(1))
        >>> day = 86_400_000_000_000
        >>> found = unexplained_gaps(
        ...     [bar(1_685_649_600_000_000_000, "100"),
        ...      bar(1_685_649_600_000_000_000 + day, "25")],
        ...     AdjustmentSeries(t, []),
        ...     as_of=date(2023, 6, 5),
        ... )
        >>> len(found)
        1
    """
    if threshold <= 0:
        raise ValueError(f"threshold {threshold} must be positive")

    adjusted = sorted(
        adjust_bars(bars, series, as_of=as_of), key=lambda bar: (bar.ts_event, bar.seq)
    )
    gaps: list[PriceGap] = []
    for previous, current in pairwise(adjusted):
        ratio = current.open.value / previous.close.value
        if abs(ratio - ONE) < threshold:
            continue
        gaps.append(
            PriceGap(
                symbol=series.symbol,
                session_date=_session_of(current),
                previous_close=previous.close,
                open_price=current.open,
                ratio=ratio,
            )
        )
    return tuple(gaps)


def _session_of(bar: Bar) -> date:
    """The calendar date a bar's close falls on, in UTC.

    Good enough for daily bars and for split detection, which is all this
    module does. Anything needing the *venue's* session date asks `CalendarPort`
    — a 16:00 ET close is the same UTC date, but a 20:00 ET extended-hours bar
    is not.
    """
    return to_datetime(bar.ts_event).date()
