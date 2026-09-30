"""Momentum ignition: trade the footprint of a burst that moved the book — §5.4, Family C.

**The idea in one line.** A single bar that trades several times its normal
volume *and* covers several times its normal range is not noise being noisy: it
is someone who had to get done, and the imbalance usually has more to run.

**The name is borrowed, and the mechanism is not what we can see.** In the
microstructure literature "momentum ignition" names a *manipulative* sequence —
aggressive orders fired to trip resting stops and pull momentum traders in, so
the initiator can exit into the move. Establishing that requires order-by-order
data: who cancelled, who crossed, who was resting. This corpus is one-minute
OHLCV, so what is implemented here is the **footprint** the sequence leaves,
not the sequence. Whether the footprint has an edge is exactly the open
question; whether it was ignition or a real buyer is not answerable from bars,
and the strategy makes no claim either way.

**Both conditions, never either.** Volume alone is an auction print, an index
rebalance or a block crossed away from the tape, none of which moves price.
Range alone is a thin bar in a name nobody is trading. The published
descriptions of the pattern are always the conjunction, and separating them
would be two hypotheses wearing one name.

**Measured against the instrument's own history, never in absolutes.** `rvol`
is this bar's volume over its recent mean and `atr` is the average true range
over fourteen bars, so "three times normal" means the same on SPY and on a $9
name. An absolute threshold in shares or cents would be a different strategy on
every instrument in the universe and could not be pooled.

**The stop is the igniting bar's own far end.** If the burst was real, price does
not come back through where it started; if it does, the burst was a fill being
worked and is finished. That also keeps R proportional to the bar that caused
the signal, so a violent ignition buys a small position rather than a large risk
(§6.1).

**Tier 2 in the plan, but it needs no breadth.** `00-phase-2.plan.md` groups it
with the cross-sectional strategies because it was expected to want a
stocks-in-play filter. It does not: the rule is entirely within one instrument,
and `rvol` is the per-instrument form of the same idea a universe ranking would
supply. It is therefore measurable on today's corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import ClassVar, Final, Self

from neurotrade.core.events import Bar, MarketSession
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.types import Side, Symbol
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import FeatureRef, Regime, Strategy, StrategyContext

__all__ = ["MomentumIgnition"]

_NO_TRADE: Final[tuple[Intent, ...]] = ()

DEFAULT_MIN_RVOL: Final = 3.0
"""Times normal volume a bar must trade before it counts as a burst.

Three because that is where the practitioner descriptions of the pattern put it
and because it is comfortably outside the noise: minute volume is right-skewed,
so twice normal happens several times a session on any liquid name while three
times does not."""

DEFAULT_MIN_RANGE_ATRS: Final = 2.0
"""Times the average true range a bar must cover to count as a burst.

Two ATRs in one minute is a move the instrument normally takes four minutes to
make. Paired with the volume test rather than used alone — see the module
docstring."""

DEFAULT_TARGET_R: Final = Decimal(2)
"""Profit barrier, in multiples of risk.

Two because the stop is tight by construction (one bar's range) and a
continuation that does not reach twice that distance is inside the noise it was
supposed to have left. Not swept: the axis with published evidence behind it is
the burst threshold, and a target tried beside it would raise the deflation
hurdle for the whole family for nothing (§17)."""

SWEEP_MIN_RVOL: Final = (2.0, 3.0, 4.0)
"""The burst thresholds measured, and the whole declared search.

Three trials on the one axis the pattern is ever described in terms of. The
range multiple moves with it rather than being swept independently: a 3x3 grid
is nine trials for two axes nobody has separated evidence on, and the ledger
would count all nine."""

MIN_SESSION_BARS: Final = 15
"""Bars a session needs before a burst is believed.

`atr` has a fifteen-bar lookback, so before that the feature is warm only on
history carried across the bell — and the first minutes of a session are its
widest and heaviest by construction. A rule that fires hardest at 09:31 every
day would be measuring the open, not an imbalance."""


@arsenal.strategy
class MomentumIgnition(Strategy):
    """Trade the direction of a bar that is abnormal in volume and in range at once.

    Example:
        >>> MomentumIgnition().qualified_name
        'momentum_ignition@1.0.0'
    """

    name, version = "momentum_ignition", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (
        Regime.TREND_UP,
        Regime.TREND_DOWN,
        Regime.HIGH_VOLATILITY,
    )
    """A burst that continues is a trend day's mechanism. Never CHOP or REVERSAL,
    where the same bar is the extreme of the day rather than the start of a move,
    and never the midday lull, where three times a thin mean is still thin."""

    features: ClassVar[tuple[FeatureRef, ...]] = (FeatureRef("rvol"), FeatureRef("atr"))
    """Both declared, because both are read. `rvol` supplies the volume test and
    `atr` the range test; the session anchors come off `context.levels` and are
    §5.3 shared infrastructure, so they are not declared."""

    cost_sensitivity: ClassVar[float] = 1.5
    """Higher than a gap or a noise-area trade. The stop is one bar's range, so
    the gross move this strategy is playing for is small in absolute terms and
    the spread is a larger share of it (§5.9)."""

    min_rvol: float = DEFAULT_MIN_RVOL
    """Times normal volume required. Each value is its own trial; see `sweep`.
    Not a `ClassVar` — an instance carries its own so one run measures several."""

    min_range_atrs: float = DEFAULT_MIN_RANGE_ATRS
    """Times the average true range the bar must cover."""

    target_r: ClassVar[Decimal] = DEFAULT_TARGET_R
    """Profit barrier as a multiple of risk. Fixed, not swept."""

    def __init__(
        self,
        *,
        min_rvol: float = DEFAULT_MIN_RVOL,
        min_range_atrs: float = DEFAULT_MIN_RANGE_ATRS,
    ) -> None:
        """Start with nothing proposed on any instrument.

        Args:
            min_rvol: Times normal volume a bar must trade.
            min_range_atrs: Times the average true range it must cover.

        Example:
            >>> MomentumIgnition(min_rvol=4.0).min_rvol
            4.0
        """
        self.min_rvol = min_rvol
        self.min_range_atrs = min_range_atrs
        # Per symbol: the host subscribes one instance to the whole universe, so
        # "already fired today" has to be keyed by instrument as well as by date.
        self._fired: dict[Symbol, date] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per burst threshold in `SWEEP_MIN_RVOL`.

        Example:
            >>> [label for label, _ in MomentumIgnition.sweep()]
            ['rvol>=2', 'rvol>=3', 'rvol>=4']
        """
        return tuple((f"rvol>={value:g}", cls(min_rvol=value)) for value in SWEEP_MIN_RVOL)

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Propose an entry in the direction of an abnormal bar.

        Args:
            bar: The bar that just closed. Its own volume and range are the
                signal, so acting on its close is not lookahead.
            context: Its view of the world. `rvol` and `atr` are declared;
                `levels` supplies the session and the bell.

        Returns:
            At most one proposal, and nothing on the overwhelming majority of
            bars.

        Example:
            >>> MomentumIgnition().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.TREND_UP, values={"rvol": None, "atr": None},
            ... ))
            ()
        """
        levels = context.levels
        if levels is None or context.session is not MarketSession.REGULAR:
            return _NO_TRADE
        if bar.ts_event >= levels.close_ns or levels.bar_count < MIN_SESSION_BARS:
            return _NO_TRADE
        if self._fired.get(bar.symbol) == levels.session_date:
            return _NO_TRADE
        if not context.requires("rvol", "atr"):
            return _NO_TRADE

        rvol = context.feature("rvol")
        atr = context.feature("atr")
        assert rvol is not None and atr is not None  # `requires` above
        if rvol < self.min_rvol or atr <= 0.0:
            return _NO_TRADE
        # Derived comparison, so float is the right unit on both sides — the
        # prices below stay `Price` because a stop is an order level.
        if float(bar.high.value - bar.low.value) < self.min_range_atrs * atr:
            return _NO_TRADE

        # The body, not the close against the previous bar: a burst that opened
        # low and closed high is a buyer, whatever yesterday did. A doji-bodied
        # burst names no direction and is declined rather than guessed.
        if bar.close > bar.open:
            side, invalidation = Side.BUY, bar.low
        elif bar.close < bar.open:
            side, invalidation = Side.SELL, bar.high
        else:
            return _NO_TRADE

        # One per instrument per session. A burst is a property of the moment,
        # and re-firing on the next abnormal bar of the same episode would let
        # one violent session supply a backtest's whole sample.
        self._fired[bar.symbol] = levels.session_date

        return (
            Intent(
                symbol=bar.symbol,
                ts_event=bar.ts_event,
                ts_init=bar.ts_event,
                side=side,
                entry=EntryTrigger.MARKET,
                entry_price=None,
                invalidation=invalidation,
                target_r=self.target_r,
                # Flat at the bell: the day engine's rule, and an ignition that
                # has not resolved by the close was not one (Phase 2 plan,
                # decision 2).
                horizon_ns=levels.close_ns - bar.ts_event,
                strategy=self.name,
                strategy_version=self.version,
                rationale=(
                    f"rvol {rvol:.1f} and range "
                    f"{float(bar.high.value - bar.low.value) / atr:.1f} atr "
                    f"at {bar.close}, stop {invalidation}"
                ),
            ),
        )
