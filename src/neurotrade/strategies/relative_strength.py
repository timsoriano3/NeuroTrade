"""Beta-adjusted relative strength: buy what is outrunning its own sector — §5.4, Family C.

**The idea in one line.** A name up 2% on a day its sector is up 2% has done
nothing; a name up 2% on a day its sector is flat has been bought by someone who
wanted *it*, and that buyer is usually not finished.

**The residual is the whole point, and the sector leg is what makes it one.**
`session_return - beta * benchmark_return`, with the benchmark taken from
`core.sectors.SectorMap` — the instrument's own SPDR where it has one, the
venue's market leg otherwise. Without the sector leg the residual still carries
sector beta, so on a morning energy runs every energy name reads as alpha at
once and the ranking is a sector bet wearing a stock-picking name. That is the
reason `config/sectors.yaml` exists.

**Beta is daily, estimated over `BETA_MEMORY` completed sessions.** A beta
fitted on one-minute returns is dominated by microstructure and pulled toward
zero. A name without enough history has **no** beta, is not residualised, and is
excluded from the ranking — never given a beta of one, which would assume the
answer for exactly the names whose answer is missing.

**The exact complement of `residual_reversion`.** Same number, opposite end.
This one buys the top of the residual ranking, that one sells it; this declares
TREND_UP / TREND_DOWN / HIGH_VOLATILITY and that declares CHOP / REVERSAL, so
§5.7's classifier decides which reading applies and the two can never be live
together. Until Phase 5 that decision does not exist, which is why both are
measured `ungated` and neither number is the one a live gate would produce.
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

__all__ = ["RelativeStrength"]

DEFAULT_TARGET_R: Final = Decimal(2)
"""Profit barrier, in multiples of risk.

Two rather than the reversion pair's one: a continuation is asking the move to
keep going, and the asymmetry is the claim. Not swept — the declared search is
the extremity, and a second axis would triple the trials for no published
evidence separating the values (§17)."""


@arsenal.strategy
class RelativeStrength(Strategy):
    """Buy the largest sector-adjusted residuals and short the smallest.

    Example:
        >>> RelativeStrength().qualified_name
        'relative_strength@1.0.0'
    """

    name, version = "relative_strength", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (
        Regime.TREND_UP,
        Regime.TREND_DOWN,
        Regime.HIGH_VOLATILITY,
    )
    """Trend days, where a residual persists. The exact complement of
    `residual_reversion`'s CHOP and REVERSAL."""

    needs_cross_section: ClassVar[bool] = True
    """The residual is computed by `CrossSectionTracker`, which needs the
    benchmark's return at the same tick — not something a per-instrument feature
    can produce."""

    cost_sensitivity: ClassVar[float] = 1.5
    """A residual is a fraction of an already-small intraday move, so the spread
    is a larger share of the gross edge than a session-long trend trade's."""

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
            >>> RelativeStrength(extreme_names=8).extreme_names
            8
        """
        self.extreme_names = extreme_names
        self.stop_fraction = stop_fraction
        self._traded: dict[Symbol, date] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per extremity in `SWEEP_EXTREME_NAMES`.

        Example:
            >>> [label for label, _ in RelativeStrength.sweep()]
            ['top/bottom 3', 'top/bottom 5', 'top/bottom 8']
        """
        return tuple(
            (f"top/bottom {value}", cls(extreme_names=value)) for value in SWEEP_EXTREME_NAMES
        )

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Trade the continuation of this instrument's residual, if it is extreme.

        Example:
            >>> from neurotrade.core.events import MarketSession
            >>> RelativeStrength().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.TREND_UP,
            ... ))
            ()
        """
        return cross_sectional_intent(
            bar,
            context,
            measure=lambda row: row.residual,
            # Buy the top: this is the continuation side of the pair.
            long_end_is_top=True,
            extreme_names=self.extreme_names,
            stop_fraction=self.stop_fraction,
            target_r=self.target_r,
            traded=self._traded,
            strategy=self.name,
            version=self.version,
            measure_name="residual",
        )
