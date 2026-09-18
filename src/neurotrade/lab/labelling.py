"""Triple-barrier labelling and sample-uniqueness weighting (§8).

A *label* is the answer to "what happened next" for a decision made at some
bar. The naive answer — the return over the next N bars — is not how a position
actually ends, because a real position has a stop and a target and exits at
whichever it touches first. Labelling on fixed-horizon returns therefore trains
a model on outcomes the trading system could never have experienced.

**The triple barrier** is three exits, whichever comes first:

1. an upper barrier, the profit target;
2. a lower barrier, the stop;
3. a vertical barrier, a time limit — the position is closed at the bar's close
   if neither price barrier is touched.

The label is which one was hit. That matches how positions really end, which is
the whole point.

## The decisions that make this honest, and the ones that cannot be made honest

**Barriers are tested against the price path, not the close.** A stop is hit
intrabar, when the low trades through it, not at the end of a bar that happened
to close lower. Testing closes understates how often stops are hit — always in
the flattering direction.

**When both barriers fall inside the same bar, the stop wins.** From OHLC alone
the order of touches within a bar is unknowable: a bar whose high cleared the
target and whose low cleared the stop could have gone either way. Choosing the
stop is not a guess about which happened; it is the conservative reading, and
the only alternative that does not quietly inflate every result. It does bias
labels pessimistically — that is the intended direction, and it is why
`BarrierTouch.ambiguous` exists, so the share of such labels can be measured
rather than forgotten.

**Costs are inside the label.** §3.3 forbids subtracting them afterwards, so
`realised_return` is net: the entry pays, the exit pays, and the label's sign
can flip because of it. A label that says "+1, profitable" on a gross basis and
loses money net is the single most direct way to train a model to lose money.

**Prices must be split-adjusted before they arrive here.** A 4:1 split is a 75%
gap in an unadjusted series, which trips every stop it touches. `core/actions.py`
is what fixes it, and nothing here can detect that it was not done.

## Uniqueness weighting

Labels overlap. A label opened at 09:31 and running for thirty bars shares
twenty-nine of them with one opened at 09:32, so the two are nearly the same
observation. Treating them as independent inflates the effective sample size,
which inflates every significance test computed from it — §8 asks for
uniqueness weighting precisely because of this, and CPCV's purging depends on
knowing the same spans.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum
from typing import Final

from neurotrade.core.clock import Nanos
from neurotrade.core.costs import CostModel
from neurotrade.core.events import Bar
from neurotrade.core.types import Price, Quantity, Side

__all__ = [
    "BarrierTouch",
    "Label",
    "average_uniqueness",
    "concurrency",
    "label_barriers",
    "triple_barrier",
]

_RETURN_PLACES: Final = Decimal("0.00000001")
"""Scale a realised return is quantized to, matching the corpus decimal scale."""


class Label(IntEnum):
    """Which barrier a position touched first.

    Integer-valued so it can go straight into a model without a mapping step,
    and signed so that multiplying by the side gives the outcome from the
    position's point of view.

    Example:
        >>> (Label.PROFIT, Label.STOP, Label.TIMEOUT)
        (<Label.PROFIT: 1>, <Label.STOP: -1>, <Label.TIMEOUT: 0>)
    """

    PROFIT = 1  # the profit target was touched first
    TIMEOUT = 0  # neither price barrier was touched before the time limit
    STOP = -1  # the stop was touched first


@dataclass(frozen=True, slots=True)
class BarrierTouch:
    """The outcome of one labelled decision.

    Example:
        >>> touch = BarrierTouch(
        ...     label=Label.PROFIT, touched_at=1_000, bars_held=5,
        ...     entry=Price("100"), exit=Price("102"),
        ...     realised_return=Decimal("0.018"), ambiguous=False,
        ... )
        >>> touch.is_win
        True
    """

    label: Label  # which barrier was hit
    touched_at: Nanos  # ts_event of the bar that ended the position
    bars_held: int  # bars from entry to exit, inclusive of the exit bar
    entry: Price  # reference price the barriers were measured from
    exit: Price  # price the position was closed at
    realised_return: Decimal  # signed fractional return NET of modelled costs
    ambiguous: bool  # both barriers fell inside the exit bar; the stop was assumed

    @property
    def is_win(self) -> bool:
        """Whether the position made money after costs.

        Deliberately not the same as `label is Label.PROFIT`: a position can
        touch its profit target and still lose money once the spread and
        commission are paid, which is exactly the case a cost-blind label
        hides.
        """
        return self.realised_return > 0


def triple_barrier(
    bars: Sequence[Bar],
    entry_index: int,
    *,
    side: Side,
    profit_target: Decimal,
    stop_loss: Decimal,
    max_bars: int,
    costs: CostModel,
    quantity: Quantity,
    volatility: float | None = None,
) -> BarrierTouch | None:
    """Label one decision by walking forward to whichever barrier it touches.

    Args:
        bars: The instrument's bars, oldest first, **split-adjusted**. Must
            extend past `entry_index`.
        entry_index: Index of the bar whose close is the entry reference. The
            walk starts at the *next* bar — a decision made on a bar's close
            cannot be filled inside that same bar.
        side: Direction of the position. For a short, the profit barrier is
            below the entry and the stop above.
        profit_target: Distance to the profit barrier as a fraction of entry,
            e.g. `0.02` for 2%. Usually set from ATR rather than fixed.
        stop_loss: Distance to the stop barrier as a fraction of entry,
            positive.
        max_bars: The vertical barrier, in bars.
        costs: Applied at entry and exit, inside the label (§3.3).
        quantity: Position size; costs are not linear in it.
        volatility: Recent realised volatility, passed through to the cost
            model.

    Returns:
        The outcome, or `None` when there are no bars after `entry_index` to
        walk — the end of the corpus, where a label cannot be formed. `None`
        rather than a timeout: an unlabelled decision and a decision that
        timed out are different, and conflating them puts fabricated outcomes
        in the training set at exactly the most recent dates.

    Raises:
        ValueError: If `entry_index` is out of range, either barrier distance
            is not positive, or `max_bars` is below 1.

    Example:
        >>> from neurotrade.core.costs import CostModel, FeeSchedule, FlooredSpread
        >>> from neurotrade.core.events import Bar, BarInterval
        >>> from neurotrade.core.types import Quantity, Symbol, Venue
        >>> def b(i, high, low, close):
        ...     return Bar(symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=i, ts_init=i,
        ...                interval=BarInterval.MIN_1, open=Price(close), high=Price(high),
        ...                low=Price(low), close=Price(close), volume=Quantity(1000))
        >>> bars = [b(0, "100", "100", "100"), b(1, "103", "100", "103")]
        >>> model = CostModel(spreads=FlooredSpread(), fees=FeeSchedule())
        >>> touch = triple_barrier(bars, 0, side=Side.BUY, profit_target=Decimal("0.02"),
        ...                        stop_loss=Decimal("0.02"), max_bars=5, costs=model,
        ...                        quantity=Quantity(1000))
        >>> touch.label
        <Label.PROFIT: 1>
    """
    if not 0 <= entry_index < len(bars):
        raise ValueError(f"entry_index {entry_index} is outside the {len(bars)} bars given")
    if profit_target <= 0:
        raise ValueError(f"profit_target {profit_target} must be positive")
    if stop_loss <= 0:
        raise ValueError(f"stop_loss {stop_loss} must be positive")
    if max_bars < 1:
        raise ValueError(f"max_bars {max_bars} must be at least 1")

    entry = bars[entry_index].close
    forward = bars[entry_index + 1 : entry_index + 1 + max_bars]
    if not forward:
        return None

    direction = Decimal(side.sign)
    upper = entry.value * (1 + direction * profit_target)
    lower = entry.value * (1 - direction * stop_loss)
    # For a short, "upper" is below entry and "lower" above; name them by role
    # rather than by position on the chart so the comparisons stay readable.
    profit_level = upper
    stop_level = lower

    for offset, bar in enumerate(forward, start=1):
        if side is Side.BUY:
            hit_profit = bar.high.value >= profit_level
            hit_stop = bar.low.value <= stop_level
        else:
            hit_profit = bar.low.value <= profit_level
            hit_stop = bar.high.value >= stop_level

        if not (hit_profit or hit_stop):
            continue

        # Both inside one bar: the order of touches is unknowable from OHLC, so
        # take the stop. See the module docstring — this biases labels
        # pessimistically on purpose, and `ambiguous` makes the share countable.
        ambiguous = hit_profit and hit_stop
        label = Label.STOP if hit_stop else Label.PROFIT
        exit_price = Price(stop_level if hit_stop else profit_level)
        return _touch(
            label=label,
            bar=bar,
            bars_held=offset,
            entry=entry,
            exit_price=exit_price,
            side=side,
            costs=costs,
            quantity=quantity,
            volatility=volatility,
            ambiguous=ambiguous,
        )

    last = forward[-1]
    return _touch(
        label=Label.TIMEOUT,
        bar=last,
        bars_held=len(forward),
        entry=entry,
        exit_price=last.close,
        side=side,
        costs=costs,
        quantity=quantity,
        volatility=volatility,
        ambiguous=False,
    )


def _touch(
    *,
    label: Label,
    bar: Bar,
    bars_held: int,
    entry: Price,
    exit_price: Price,
    side: Side,
    costs: CostModel,
    quantity: Quantity,
    volatility: float | None,
    ambiguous: bool,
) -> BarrierTouch:
    """Build a `BarrierTouch`, applying costs to both legs.

    Costs are charged twice — once to open, once to close — because a round
    trip crosses the spread twice and pays commission twice. Charging one leg
    halves the modelled cost of every trade in the corpus.
    """
    entry_cost = costs.cost(side, quantity, entry, volatility=volatility)
    exit_cost = costs.cost(side.opposite, quantity, exit_price, volatility=volatility)
    gross = (exit_price.value - entry.value) * Decimal(side.sign)
    notional = entry.value * quantity.value
    net = gross * quantity.value - entry_cost.total.amount - exit_cost.total.amount
    return BarrierTouch(
        label=label,
        touched_at=bar.ts_event,
        bars_held=bars_held,
        entry=entry,
        exit=exit_price,
        realised_return=(net / notional).quantize(_RETURN_PLACES),
        ambiguous=ambiguous,
    )


def label_barriers(
    bars: Sequence[Bar],
    entries: Sequence[int],
    *,
    side: Side,
    profit_target: Decimal,
    stop_loss: Decimal,
    max_bars: int,
    costs: CostModel,
    quantity: Quantity,
) -> tuple[tuple[int, BarrierTouch], ...]:
    """Label many decisions over one instrument's bars.

    Args:
        bars: The instrument's bars, oldest first, split-adjusted.
        entries: Indices of the bars decisions were made on.
        side: Direction, shared by every entry here.
        profit_target: Profit barrier distance as a fraction of entry.
        stop_loss: Stop barrier distance as a fraction of entry.
        max_bars: Vertical barrier.
        costs: The cost model.
        quantity: Position size.

    Returns:
        `(entry_index, touch)` pairs, in the order given, skipping entries too
        close to the end of the corpus to label.
    """
    labelled: list[tuple[int, BarrierTouch]] = []
    for index in entries:
        touch = triple_barrier(
            bars,
            index,
            side=side,
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_bars=max_bars,
            costs=costs,
            quantity=quantity,
        )
        if touch is not None:
            labelled.append((index, touch))
    return tuple(labelled)


def concurrency(spans: Sequence[tuple[int, int]], length: int) -> tuple[int, ...]:
    """How many labels are live at each bar.

    Args:
        spans: `(start, end)` bar indices per label, both inclusive.
        length: Number of bars in the series.

    Returns:
        A count per bar index.

    Raises:
        ValueError: If a span ends before it starts, or falls outside the
            series.

    Example:
        >>> concurrency([(0, 2), (1, 3)], length=5)
        (1, 2, 2, 1, 0)
    """
    counts = [0] * length
    for start, end in spans:
        if end < start:
            raise ValueError(f"span ({start}, {end}) ends before it starts")
        if start < 0 or end >= length:
            raise ValueError(f"span ({start}, {end}) falls outside {length} bars")
        for index in range(start, end + 1):
            counts[index] += 1
    return tuple(counts)


def average_uniqueness(spans: Sequence[tuple[int, int]], length: int) -> tuple[float, ...]:
    """How independent each label is, as a weight in `(0, 1]`.

    A label's uniqueness at one bar is `1 / concurrency` there; its average
    uniqueness is the mean of that across its own span. A label sharing every
    bar with one other scores 0.5; a label alone in time scores 1.0.

    **This is what stops overlapping labels from inflating significance.** Ten
    labels that are really one observation carry ten times the weight in any
    unweighted statistic, and every Sharpe and p-value computed downstream is
    overstated accordingly (§8).

    Args:
        spans: `(start, end)` bar indices per label, both inclusive.
        length: Number of bars in the series.

    Returns:
        One weight per span, in the order given.

    Example:
        >>> average_uniqueness([(0, 1), (0, 1)], length=2)
        (0.5, 0.5)
        >>> average_uniqueness([(0, 0), (1, 1)], length=2)
        (1.0, 1.0)
    """
    counts = concurrency(spans, length)
    weights: list[float] = []
    for start, end in spans:
        shares = [1.0 / counts[index] for index in range(start, end + 1)]
        weights.append(sum(shares) / len(shares))
    return tuple(weights)
