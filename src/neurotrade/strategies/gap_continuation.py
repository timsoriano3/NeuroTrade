"""Gap continuation, banded by recent daily volatility — §5.2, Family A.

**The idea in one line.** A price that opens far from yesterday's close usually
keeps going; a price that opens close to it usually drifts back. So trade *with*
a wide gap and leave a narrow one alone.

**Why wide and not narrow.** The practitioner evidence behind this is a fill
frequency: a gap under about 0.3 daily volatility units closes the same session
roughly three quarters of the time, while one over about 1.2 closes less than a
tenth of the time. The narrow case is the mean-reversion trade, and Phase 2's
literature sweep rejected gap *fade* as a primary strategy — it fails at every
entry time tested. The wide case is what survives, and this is that half.

**"Daily volatility units" is `SessionLevels.gap_in_ranges`, not ATR.** The
published thresholds are quoted against a daily average true range. The corpus
feeds one-minute bars, and the registered `atr` is therefore a one-minute
statistic — an overnight gap measured in those units is larger by two orders of
magnitude and no published threshold applies to it. The mean high-low range of
the last `RANGE_MEMORY` sessions is the same scale the thresholds were measured
on, and the level tracker already keeps it. Named `gap_continuation` rather than
anything mentioning ATR so the deviation is not hidden behind a familiar word.

**What proves it wrong.** The session open. A gap trade is a bet that the
opening level holds; price back through it means the fill is under way and the
premise is gone. That makes R large on a large gap, which is correct rather than
convenient — the risk engine sizes from R (§6.1), so a wide stop buys a small
position rather than a large risk.

**One trade per instrument per session.** The signal is a property of the open,
not of the bar that happened to trigger it, so re-firing at 11:00 would be the
same hypothesis counted twice — and would let one session dominate a backtest.
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
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

__all__ = ["GapContinuation"]

_NO_TRADE: Final[tuple[Intent, ...]] = ()

DEFAULT_MIN_GAP_RANGES: Final = 1.2
"""Gap width, in mean recent session ranges, below which nothing fires."""

DEFAULT_ENTRY_WINDOW_BARS: Final = 30
"""Bars after the open during which an entry is still taken."""

SWEEP_MIN_GAP_RANGES: Final = (1.0, 1.2, 1.5)
"""The band widths measured, and the whole declared search.

Three trials on the **published** axis: the practitioner evidence is a fill
frequency against gap width, 1.2 is where it puts the boundary, and 1.0 and 1.5
bracket it on either side. `entry_window_bars` is deliberately not swept — no
published result distinguishes 15 bars from 30, and a value tried anyway would
raise the deflation hurdle for every strategy in the family in exchange for
nothing (§17).
"""


@arsenal.strategy
class GapContinuation(Strategy):
    """Trade the direction of a gap too wide to expect a fill.

    Reads only session levels, so it declares no features: the open, the prior
    close and the recent range history are §5.3 shared infrastructure and arrive
    on every context.

    Example:
        >>> GapContinuation().qualified_name
        'gap_continuation@1.0.0'
    """

    name, version = "gap_continuation", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (
        Regime.TREND_UP,
        Regime.TREND_DOWN,
        Regime.HIGH_VOLATILITY,
    )
    """A gap that runs is a trend day. Never CHOP or REVERSAL — those are the
    days gaps fill, which is the opposite trade — and never the midday lull."""

    cost_sensitivity: ClassVar[float] = 1.0
    """One entry and one exit per session, held for hours: the spread is paid
    twice against a move measured in whole daily ranges."""

    min_gap_ranges: float = DEFAULT_MIN_GAP_RANGES
    """Gap width, in mean recent session ranges, below which nothing fires.
    Each distinct value is a separate trial — 1.0 and 1.2 are two hypotheses,
    and the ledger counts them that way, which is what `sweep` declares. Not a
    `ClassVar`: an instance carries its own, so one run can measure several."""

    entry_window_bars: int = DEFAULT_ENTRY_WINDOW_BARS
    """Bars after the open during which an entry is still taken. A gap that has
    held for two hours is no longer the event this strategy trades, and entering
    late pays the same spread for a fraction of the remaining move."""

    target_r: ClassVar[Decimal] = Decimal(2)
    """Profit barrier as a multiple of risk."""

    def __init__(
        self,
        *,
        min_gap_ranges: float = DEFAULT_MIN_GAP_RANGES,
        entry_window_bars: int = DEFAULT_ENTRY_WINDOW_BARS,
    ) -> None:
        """Start with no session traded on any instrument.

        Args:
            min_gap_ranges: Band width, in mean recent session ranges. Each
                value is its own trial; see `sweep`.
            entry_window_bars: Bars after the open during which an entry is
                still taken.

        Example:
            >>> GapContinuation(min_gap_ranges=1.5).min_gap_ranges
            1.5
        """
        self.min_gap_ranges = min_gap_ranges
        self.entry_window_bars = entry_window_bars
        # Per symbol, because the host subscribes one instance to the whole
        # universe. Keyed on the session date rather than a bar count so that a
        # gap in the corpus cannot reopen a session that was already traded.
        self._traded: dict[Symbol, date] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per band width in `SWEEP_MIN_GAP_RANGES`.

        Example:
            >>> [label for label, _ in GapContinuation.sweep()]
            ['gap>=1 ranges', 'gap>=1.2 ranges', 'gap>=1.5 ranges']
        """
        return tuple(
            (f"gap>={value:g} ranges", cls(min_gap_ranges=value)) for value in SWEEP_MIN_GAP_RANGES
        )

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Propose a gap-continuation entry, or nothing.

        Args:
            bar: The bar that just closed.
            context: Its view of the world; `levels` carries everything read.

        Returns:
            At most one proposal. Nothing is the overwhelmingly common case —
            most sessions do not gap, and a session that does is traded once.

        Example:
            >>> GapContinuation().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.TREND_UP,
            ... ))
            ()
        """
        levels = context.levels
        if levels is None or context.session is not MarketSession.REGULAR:
            return _NO_TRADE
        if self._traded.get(bar.symbol) == levels.session_date:
            return _NO_TRADE
        if levels.bar_count > self.entry_window_bars or bar.ts_event >= levels.close_ns:
            return _NO_TRADE

        gap = levels.gap_in_ranges
        if gap is None or abs(gap) < self.min_gap_ranges:
            return _NO_TRADE

        side = Side.BUY if gap > 0 else Side.SELL
        # The open must still be holding. A close back through it means the fill
        # is already under way, and it would also put the stop on the wrong side
        # of the entry, which `Intent` refuses outright.
        held = (
            bar.close > levels.session_open if side is Side.BUY else bar.close < levels.session_open
        )
        if not held:
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
                invalidation=levels.session_open,
                target_r=self.target_r,
                # Flat at the bell: the day engine holds nothing overnight, and
                # a stop cannot fill through a gap (Phase 2 plan, decision 2).
                horizon_ns=levels.close_ns - bar.ts_event,
                strategy=self.name,
                strategy_version=self.version,
                rationale=f"gap {gap:+.2f} session ranges, open holding",
            ),
        )
