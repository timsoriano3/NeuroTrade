"""Intraday short-term reversal: fade the last half hour's extremes — §5.5, Family B.

**The idea in one line.** Over thirty to sixty minutes the names that have run
furthest are the ones a liquidity demander pushed, not the ones news repriced,
so the crowd's own ranking says which way the next half hour leans.

**Cross-sectional, and that is the claim.** The hypothesis is not "this name
went up so it will go down" — that is a statement about one instrument and it is
false often enough to be useless. It is "this name went up *more than everyone
else did*", which strips the market move out by construction: on a morning the
whole tape rallies, every name's trailing return is positive and only the
ordering carries information. That is why this reads `context.cross_section`
rather than a per-instrument feature, and why it declines until at least
`MIN_RANKED` names are rankable.

**The window is 30 minutes and is not a parameter here.** §5.5 states the family
at 30 to 60 minutes. The measurement window lives on `CrossSectionTracker`,
shared by every cross-sectional strategy in a run, so varying it from inside one
strategy would silently change what the others saw. The declared search moves
the *extremity* instead — decile versus quintile — which is the axis the
cross-sectional literature actually reports.

**Strictly a Tier 2 strategy: it needs breadth, and 58 names is thin for it.**
A decile of 58 is five or six names, against the thousands a published
short-term-reversal portfolio is formed from. Expect a wide confidence interval
and read `Measurement.concentration` before believing any pooled number.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import ClassVar, Final, Self

from neurotrade.core.events import Bar
from neurotrade.core.intent import Intent
from neurotrade.core.types import Symbol
from neurotrade.strategies._cross_sectional import (
    DEFAULT_EXTREME_NAMES,
    DEFAULT_STOP_FRACTION,
    SWEEP_EXTREME_NAMES,
    cross_sectional_intent,
)
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

__all__ = ["IntradayReversal"]

DEFAULT_TARGET_R: Final = Decimal(1)
"""Profit barrier, in multiples of risk.

One, because the trade is symmetric: the stop is half the move given further
and the target is half the move given back. A 2R target on a reversion would be
asking the move to reverse twice as far as it is allowed to extend, which is a
different and much stronger claim than §5.5's."""


@arsenal.strategy
class IntradayReversal(Strategy):
    """Short the biggest trailing gainers and buy the biggest losers.

    Example:
        >>> IntradayReversal().qualified_name
        'intraday_reversal@1.0.0'
    """

    name, version = "intraday_reversal", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (Regime.CHOP, Regime.REVERSAL)
    """Reversion days. Never a trend day, where the extremes of a half-hour
    ranking are the names that keep going — which is `relative_strength`'s
    hypothesis, and the host is what stops the two being live together."""

    needs_cross_section: ClassVar[bool] = True
    """The ranking is the strategy; without it there is nothing to be extreme in."""

    cost_sensitivity: ClassVar[float] = 2.0
    """The highest bar in the arsenal. A half-hour hold on a move of a fraction
    of a percent pays the spread against a small gross edge, and §5.9 says that
    is exactly when the multiple has to rise. `00-phase-2.plan.md` rejected
    daily/weekly short-term reversal at 0.29% annualised *net* for this reason."""

    extreme_names: int = DEFAULT_EXTREME_NAMES
    """Names traded at each end. Each value is a separate trial; see `sweep`."""

    stop_fraction: float = DEFAULT_STOP_FRACTION
    target_r: ClassVar[Decimal] = DEFAULT_TARGET_R

    def __init__(
        self,
        *,
        extreme_names: int = DEFAULT_EXTREME_NAMES,
        stop_fraction: float = DEFAULT_STOP_FRACTION,
    ) -> None:
        """Start with nothing traded on any instrument.

        Example:
            >>> IntradayReversal(extreme_names=3).extreme_names
            3
        """
        self.extreme_names = extreme_names
        self.stop_fraction = stop_fraction
        self._traded: dict[Symbol, date] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per extremity in `SWEEP_EXTREME_NAMES`.

        Example:
            >>> [label for label, _ in IntradayReversal.sweep()]
            ['top/bottom 3', 'top/bottom 5', 'top/bottom 8']
        """
        return tuple(
            (f"top/bottom {value}", cls(extreme_names=value)) for value in SWEEP_EXTREME_NAMES
        )

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Fade this instrument if its trailing return is an extreme of the universe.

        Example:
            >>> from neurotrade.core.events import MarketSession
            >>> IntradayReversal().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.CHOP,
            ... ))
            ()
        """
        return cross_sectional_intent(
            bar,
            context,
            measure=lambda row: row.trailing_return,
            # Buy the bottom: this is the reversion side of the pair.
            long_end_is_top=False,
            extreme_names=self.extreme_names,
            stop_fraction=self.stop_fraction,
            target_r=self.target_r,
            traded=self._traded,
            strategy=self.name,
            version=self.version,
            measure_name="trailing return",
        )
