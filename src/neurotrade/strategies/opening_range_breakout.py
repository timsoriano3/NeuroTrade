"""ORB on stocks in play: trade the breakout, but only where the crowd is — §5.2, Family A.

**The selection is the hypothesis; the breakout is how it is expressed.**
`00-phase-2.plan.md` is explicit, and it is the most important thing to know
about this strategy: the published 2.4-2.8 Sharpe comes from RVOL-ranking some
7,000 names, and every single-instrument replication fails. QQQ at 2 cents a
share gives Sharpe 0.23 with break-even at 2.2 cents and 76% of the filtered PnL
from 2022 alone; a 14-signal MNQ falsification study had ORB long at T=0.88 with
nothing passing. Writing the breakout alone would be writing the half already
known not to work, so this strategy declines on every bar where
`strategies/selection.stocks_in_play` does not name the instrument.

**And on this universe it will decline almost always.** 58 mega-caps are the
most consistently traded names on either exchange; their relative volume rarely
reaches what a scan of thousands finds every morning. A near-empty result is the
correct reading of the corpus rather than a threshold to lower — see
`selection.py`. Expect this to be the sparsest strategy in the arsenal, and
expect `measure_strategy` to refuse a verdict on it more often than not.

**The stop is the other side of the range, not a fraction of it.** A breakout
is wrong when price is back through the opposite edge, which is where the fade
(`orb_fade.py`) would be taking the other side. That also makes R the range's
own width, so a wide opening range buys a small position rather than a large
risk (§6.1).

**The exact complement of `orb_fade`.** It declares CHOP and REVERSAL; this
declares TREND_UP, TREND_DOWN and HIGH_VOLATILITY. The host is what keeps them
from ever being live together (§5.7), and until the Phase 5 classifier exists
neither classification is produced — which is why both are measured `ungated`
and neither number is the one a live gate would produce.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import ClassVar, Final, Self

from neurotrade.core.events import Bar, MarketSession
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.types import Side, Symbol
from neurotrade.features.levels import OPENING_WINDOWS
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, Strategy, StrategyContext
from neurotrade.strategies.selection import (
    DEFAULT_SELECTION_FLOOR,
    DEFAULT_SELECTION_SIZE,
    stocks_in_play,
)

__all__ = ["OpeningRangeBreakout"]

_NO_TRADE: Final[tuple[Intent, ...]] = ()

DEFAULT_OPENING_MINUTES: Final = 30
"""Opening-range window traded, in minutes. Middle of the three swept."""

SWEEP_OPENING_MINUTES: Final = (15, 30, 60)
"""The windows measured, and the whole declared search.

Three of `OPENING_WINDOWS`' four, matching `orb_fade` so the pair is measured on
the same axis. Five minutes is excluded for the reason given there: a range
built from five one-minute bars is dominated by the opening auction's spread.
The selection size and floor are **not** swept — they come from
`strategies/selection.py` and are shared with the nightly selector, so a value
tuned here would silently retune that too."""

DEFAULT_TARGET_R: Final = Decimal(2)
"""Profit barrier, in multiples of risk. Two, and not swept: the axis with
published evidence is the window, and `00-phase-2.plan.md` already prices a
target sweep as trials spent for no evidence (§17)."""


@arsenal.strategy
class OpeningRangeBreakout(Strategy):
    """Trade a break of the opening range, on the names trading unusually heavily.

    Declares no features: the opening ranges come off `context.levels` as §5.3
    shared infrastructure, and the relative volume comes off the cross-section,
    which is declared through `needs_cross_section` rather than as a feature.

    Example:
        >>> OpeningRangeBreakout().qualified_name
        'opening_range_breakout@1.0.0'
    """

    name, version = "opening_range_breakout", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (
        Regime.TREND_UP,
        Regime.TREND_DOWN,
        Regime.HIGH_VOLATILITY,
    )
    """The exact complement of `orb_fade`'s CHOP and REVERSAL. Never the lull —
    a breakout on thin midday volume is the definition of a false one."""

    needs_cross_section: ClassVar[bool] = True
    """The selection is the hypothesis, so the universe view is not optional.
    Without it `context.cross_section` is `None` and the strategy declines
    everywhere, which is the safe direction to fail."""

    cost_sensitivity: ClassVar[float] = 1.5
    """The plan rejects the single-instrument version on cost — break-even at
    2.2 cents a share — so this family's bar is explicitly above a hold-to-the-
    bell trend trade's (§5.9)."""

    opening_minutes: int = DEFAULT_OPENING_MINUTES
    """Window the breakout is measured against. Each value is a separate trial."""

    selection_size: int = DEFAULT_SELECTION_SIZE
    """Names the selector may return. Shared with the nightly selector, not swept."""

    selection_floor: float = DEFAULT_SELECTION_FLOOR
    """Minimum relative volume to be selected at all. Shared, not swept."""

    target_r: ClassVar[Decimal] = DEFAULT_TARGET_R

    def __init__(
        self,
        *,
        opening_minutes: int = DEFAULT_OPENING_MINUTES,
        selection_size: int = DEFAULT_SELECTION_SIZE,
        selection_floor: float = DEFAULT_SELECTION_FLOOR,
    ) -> None:
        """Start with nothing traded on any instrument.

        Raises:
            ValueError: If the window is not one of `OPENING_WINDOWS`. Caught
                here rather than mid-run, where `levels.opening_range` raises
                and kills the measurement instead of declining the trade.

        Example:
            >>> OpeningRangeBreakout(opening_minutes=15).opening_minutes
            15
        """
        if opening_minutes not in OPENING_WINDOWS:
            raise ValueError(
                f"opening range {opening_minutes} is not tracked; {OPENING_WINDOWS} are"
            )
        self.opening_minutes = opening_minutes
        self.selection_size = selection_size
        self.selection_floor = selection_floor
        self._traded: dict[Symbol, date] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per window in `SWEEP_OPENING_MINUTES`.

        Example:
            >>> [label for label, _ in OpeningRangeBreakout.sweep()]
            ['orb 15m', 'orb 30m', 'orb 60m']
        """
        return tuple(
            (f"orb {value}m", cls(opening_minutes=value)) for value in SWEEP_OPENING_MINUTES
        )

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Propose a breakout entry on a selected instrument.

        Returns:
            At most one proposal, and nothing at all on an instrument the
            selector did not name — which on this universe is nearly every bar.

        Example:
            >>> OpeningRangeBreakout().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.TREND_UP,
            ... ))
            ()
        """
        levels = context.levels
        section = context.cross_section
        if levels is None or section is None:
            return _NO_TRADE
        if context.session is not MarketSession.REGULAR or bar.ts_event >= levels.close_ns:
            return _NO_TRADE
        if self._traded.get(bar.symbol) == levels.session_date:
            return _NO_TRADE
        opening = levels.opening_range(self.opening_minutes)
        if opening is None or opening.width <= 0:
            return _NO_TRADE

        # The selection first, so a breakout on a name nobody is trading never
        # reaches the rest of the rule. Checked per bar rather than once at the
        # open, because relative volume is time-of-day dependent and a name can
        # come into play at 11:00 — which is the case the published scan catches.
        if bar.symbol not in stocks_in_play(
            section, size=self.selection_size, floor=self.selection_floor
        ):
            return _NO_TRADE

        if opening.breaks_up(bar.close):
            side, invalidation = Side.BUY, opening.low
        elif opening.breaks_down(bar.close):
            side, invalidation = Side.SELL, opening.high
        else:
            return _NO_TRADE

        self._traded[bar.symbol] = levels.session_date
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
                horizon_ns=levels.close_ns - bar.ts_event,
                strategy=self.name,
                strategy_version=self.version,
                rationale=(
                    f"{self.opening_minutes}m range [{opening.low}, {opening.high}] "
                    f"broken at {bar.close}, in play at "
                    f"rvol {levels.relative_volume_from_open:.1f}"
                    if levels.relative_volume_from_open is not None
                    else f"{self.opening_minutes}m range broken at {bar.close}"
                ),
            ),
        )
