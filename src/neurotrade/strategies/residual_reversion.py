"""Index-relative residual reversion: fade what has outrun its sector — §5.5, Family B.

**The idea in one line.** The same residual `relative_strength` buys, sold. A
name that has pulled away from its sector with no sector move behind it has
usually been pushed by one participant getting done, and the push unwinds.

**The same number, the opposite end, and never both at once.** This and
`relative_strength` rank the universe on exactly the same quantity —
`session_return - beta * benchmark_return`, against the instrument's sector
proxy — and take opposite sides of it. Which reading is right is a property of
the day, not of the strategy, which is precisely what §5.7's regime classifier
decides: this declares CHOP and REVERSAL, that declares the trend regimes, and
the host enforces the exclusion. Running both ungated, as Phase 2 must, means
they take opposite positions in the same name on the same bar. That is expected
and is not a bug; it is also why neither of their Phase 2 numbers is the number
a live gate would produce, and why `BacktestResult.regime_gated` is stamped on
every run.

**Reversion, not contrarianism.** The rule is not "this went up so sell it". It
is "this went up *relative to the sector that explains most of its variance*,
and nothing explains the rest" — which is a statement about the residual's
mean-reversion, and only definable once the sector leg has been taken out. See
`features/cross_section.py` for the residual and `core/sectors.py` for where
the sector comes from.

**A name with no beta is excluded, not faded.** Without a beta there is no
residual, so an unresidualised name never enters the ranking. That matters more
here than for the continuation side: the largest *raw* movers on any day are
mostly high-beta names on a day the market moved, and a ranking that let them in
would be shorting beta and calling it reversion.
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

__all__ = ["ResidualReversion"]

DEFAULT_TARGET_R: Final = Decimal(1)
"""Profit barrier, in multiples of risk.

One, matching `intraday_reversal` and for the same reason: the stop is half the
residual given further and the target is half of it given back, so the trade is
symmetric. Asking a reversion for 2R would be asking the residual to cross zero
and keep going, which nothing in §5.5 claims."""


@arsenal.strategy
class ResidualReversion(Strategy):
    """Short the largest sector-adjusted residuals and buy the smallest.

    Example:
        >>> ResidualReversion().qualified_name
        'residual_reversion@1.0.0'
    """

    name, version = "residual_reversion", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (Regime.CHOP, Regime.REVERSAL)
    """The exact complement of `relative_strength`'s trend regimes, so the two
    can never be live together however the classifier is wired."""

    needs_cross_section: ClassVar[bool] = True

    cost_sensitivity: ClassVar[float] = 2.0
    """As high as `intraday_reversal`'s. Both fade a small move over a short
    hold, which §5.9 says is where the spread does the most damage."""

    extreme_names: int = DEFAULT_EXTREME_NAMES
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
            >>> ResidualReversion(extreme_names=3).extreme_names
            3
        """
        self.extreme_names = extreme_names
        self.stop_fraction = stop_fraction
        self._traded: dict[Symbol, date] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per extremity in `SWEEP_EXTREME_NAMES`.

        Example:
            >>> [label for label, _ in ResidualReversion.sweep()]
            ['top/bottom 3', 'top/bottom 5', 'top/bottom 8']
        """
        return tuple(
            (f"top/bottom {value}", cls(extreme_names=value)) for value in SWEEP_EXTREME_NAMES
        )

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Fade this instrument's residual, if it is an extreme of the universe.

        Example:
            >>> from neurotrade.core.events import MarketSession
            >>> ResidualReversion().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.CHOP,
            ... ))
            ()
        """
        return cross_sectional_intent(
            bar,
            context,
            measure=lambda row: row.residual,
            # Buy the bottom: the reversion side of the pair.
            long_end_is_top=False,
            extreme_names=self.extreme_names,
            stop_fraction=self.stop_fraction,
            target_r=self.target_r,
            traded=self._traded,
            strategy=self.name,
            version=self.version,
            measure_name="residual",
        )
