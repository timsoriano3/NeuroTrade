"""ORB fade: trade a breakout that failed back into the range — §5.2, Family A.

**The idea in one line.** A break of the opening range that cannot hold is the
strongest evidence a session has that the range is the day's fair value, so sell
the failed high back toward the other side of it.

**This is `#11` of the twelve, and it is the counterpart the plan pairs with
`#7`.** An opening-range breakout and an opening-range fade are opposites and
must never be live at the same time. `strategies/base.py` names exactly this
case as why `regimes` is a permission rather than a preference: the breakout
declares TREND_UP / TREND_DOWN, this one declares CHOP and REVERSAL, and the
host enforces the exclusion centrally rather than each strategy remembering.
Until the Phase 5 classifier exists neither classification is produced, which is
why every Phase 2 number from either is stamped `ungated`.

**Two events, not one, and the order is the signal.** Price has to leave the
range *and then* close back inside it. A bar that merely trades inside the range
is not a failed breakout, it is a normal bar; the failure is only defined
relative to an excursion that happened first. The strategy therefore carries one
piece of state per instrument per session — which side was broken, and how far
the excursion went.

**The excursion's extreme is the stop, and the far side of the range is the
target.** Both come out of the rule rather than out of a parameter. If price
takes out the failed high again, the breakout was not a failure after all and
the idea is dead; if the fade works, the range's own opposite edge is where the
next decision belongs. That makes `target_r` a *computed* multiple rather than a
constant, which is correct — the reward is a level, and the risk is a different
level, so their ratio is whatever the session made it.

**Why the practitioner evidence for the primary ORB does not transfer here.**
`00-phase-2.plan.md` rejects single-instrument ORB on replication: QQQ at
2 cents/share slippage gives Sharpe 0.23, break-even at 2.2 cents, and 76% of
the filtered PnL is 2022 alone. That is a finding about the *breakout*. It says
nothing about the fade, which is a different sign on a different subset of
sessions — but it does mean the fade arrives with no published number to compare
against, so its measurement is the only evidence it will have.

**Three windows, three trials.** `OPENING_WINDOWS` exists precisely because ORB
at 5, 15, 30 and 60 minutes is four hypotheses rather than one knob. The sweep
here declares three of them; the 5-minute window is left out because a range
built from five one-minute bars at the open is mostly the opening auction's
spread, and a fade of it is a fade of the tick size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import ClassVar, Final, Self

from neurotrade.core.events import Bar, MarketSession
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.types import Price, Side, Symbol
from neurotrade.features.levels import OPENING_WINDOWS, OpeningRange
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

__all__ = ["Excursion", "OrbFade"]

_NO_TRADE: Final[tuple[Intent, ...]] = ()

DEFAULT_OPENING_MINUTES: Final = 30
"""Opening-range window the fade is measured against.

Thirty because it is the middle of the three swept and the one §5.2 states the
family in. Every value is a separate trial, so this is a default rather than a
tuned choice."""

SWEEP_OPENING_MINUTES: Final = (15, 30, 60)
"""The windows measured, and the whole declared search.

Three of `OPENING_WINDOWS`' four. Five minutes is excluded: a range built from
five one-minute bars is dominated by the opening auction's spread, so fading it
would mostly be fading the tick size — and a variant tried anyway raises the
deflation hurdle for the whole family (§17)."""

MIN_EXCURSION_FRACTION: Final = 0.25
"""How far outside the range price must travel before a return counts as failure.

A quarter of the range's own width. Without it every bar that pokes one tick
through the high and closes back inside is a signal, which is not a failed
breakout — it is the range's edge being the range's edge. Expressed as a
fraction of the width rather than in cents or ATRs so it means the same thing on
a wide session and a narrow one."""


@dataclass(frozen=True, slots=True)
class Excursion:
    """A break of the opening range that this instrument has made today.

    Carried per symbol so that "price is back inside" has something to be a
    failure *of*. Frozen because it is replaced on a new extreme rather than
    mutated — a strategy's per-symbol state is the one place an accidental
    shared reference is hardest to see.

    Example:
        >>> Excursion(session_date=date(2024, 7, 8), side=Side.BUY,
        ...           extreme=Price("101.5")).faded_side
        <Side.SELL: 'SELL'>
    """

    session_date: date  # the session it happened in; a new session starts clean
    side: Side  # BUY when the range was broken upward
    extreme: Price  # furthest price reached outside the range, so far

    @property
    def faded_side(self) -> Side:
        """The direction a fade of this excursion trades."""
        return Side.SELL if self.side is Side.BUY else Side.BUY


@arsenal.strategy
class OrbFade(Strategy):
    """Fade a break of the opening range once price closes back inside it.

    Declares no features: the opening ranges are §5.3 shared infrastructure and
    arrive on every context.

    Example:
        >>> OrbFade().qualified_name
        'orb_fade@1.0.0'
    """

    name, version = "orb_fade", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (Regime.CHOP, Regime.REVERSAL)
    """The days a breakout fails, and the exact complement of what an ORB
    strategy declares — never TREND_UP, TREND_DOWN or HIGH_VOLATILITY, and never
    the midday lull. This is the pair `strategies/base.py` names when it explains
    why regime gating is enforced by the host."""

    cost_sensitivity: ClassVar[float] = 1.5
    """A mean-reversion trade inside one session's opening range is a small gross
    move, so the spread is a larger share of it than a trend trade's (§5.9)."""

    opening_minutes: int = DEFAULT_OPENING_MINUTES
    """Window the range is taken from. Each value is a separate trial; see
    `sweep`. Not a `ClassVar` — an instance carries its own so one run can
    measure several."""

    def __init__(self, *, opening_minutes: int = DEFAULT_OPENING_MINUTES) -> None:
        """Start with no excursion recorded on any instrument.

        Args:
            opening_minutes: Opening-range window, one of `OPENING_WINDOWS`.

        Raises:
            ValueError: If the window is not tracked. A window nobody tracks
                would make `levels.opening_range` raise mid-run, which kills the
                measurement rather than declining the trade.

        Example:
            >>> OrbFade(opening_minutes=60).opening_minutes
            60
        """
        if opening_minutes not in OPENING_WINDOWS:
            raise ValueError(
                f"opening range {opening_minutes} is not tracked; {OPENING_WINDOWS} are"
            )
        self.opening_minutes = opening_minutes
        # Per symbol, because one instance sees the whole universe.
        self._excursions: dict[Symbol, Excursion] = {}
        self._faded: dict[Symbol, date] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per window in `SWEEP_OPENING_MINUTES`.

        Example:
            >>> [label for label, _ in OrbFade.sweep()]
            ['orb 15m', 'orb 30m', 'orb 60m']
        """
        return tuple(
            (f"orb {value}m", cls(opening_minutes=value)) for value in SWEEP_OPENING_MINUTES
        )

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Track an excursion outside the opening range, and fade its return.

        Args:
            bar: The bar that just closed.
            context: Its view of the world; `levels.opening_range` is what is
                read.

        Returns:
            At most one proposal — on the bar that closes back inside the range
            after a real excursion, and on no other.

        Example:
            >>> OrbFade().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.CHOP,
            ... ))
            ()
        """
        levels = context.levels
        if levels is None or context.session is not MarketSession.REGULAR:
            return _NO_TRADE
        if bar.ts_event >= levels.close_ns:
            return _NO_TRADE
        opening = levels.opening_range(self.opening_minutes)
        if opening is None or opening.width <= 0:
            return _NO_TRADE

        held = self._excursions.get(bar.symbol)
        if held is not None and held.session_date != levels.session_date:
            held = None  # a new session; yesterday's excursion is not evidence
        held = self._record(bar, opening, held, levels.session_date)
        if held is None:
            # Nothing outstanding — and the entry is *dropped*, not replaced by a
            # placeholder. An earlier draft stored a sentinel `Excursion` to carry
            # the session date, and the next bar read it back as a real excursion
            # and faded against an extreme of 1e-8. Caught by
            # `test_an_outside_bar_names_no_failed_side`.
            self._excursions.pop(bar.symbol, None)
            return _NO_TRADE
        self._excursions[bar.symbol] = held

        # The failure: price closed back inside the range after having left it.
        if opening.breaks_up(bar.close) or opening.breaks_down(bar.close):
            return _NO_TRADE
        if not self._travelled_far_enough(held, opening):
            return _NO_TRADE
        if self._faded.get(bar.symbol) == levels.session_date:
            return _NO_TRADE

        side = held.faded_side
        # The excursion's extreme is where the idea is wrong: price back through
        # it means the breakout was real and merely paused.
        invalidation = held.extreme
        # The far side of the range is where the fade is done. Computed rather
        # than declared, because both legs are levels the session produced and
        # their ratio is not a parameter anyone chose.
        target = opening.low if side is Side.SELL else opening.high
        risk = abs(bar.close.value - invalidation.value)
        reward = abs(bar.close.value - target.value)
        if risk <= 0 or reward <= 0:
            # Entry sitting exactly on the stop or exactly on the target. Both
            # are degenerate rather than rare — a bar closing precisely on a
            # range edge is a normal print — and `Intent` refuses either.
            return _NO_TRADE

        self._faded[bar.symbol] = levels.session_date
        return (
            Intent(
                symbol=bar.symbol,
                ts_event=bar.ts_event,
                ts_init=bar.ts_event,
                side=side,
                entry=EntryTrigger.MARKET,
                entry_price=None,
                invalidation=invalidation,
                target_r=reward / risk,
                horizon_ns=levels.close_ns - bar.ts_event,
                strategy=self.name,
                strategy_version=self.version,
                rationale=(
                    f"{self.opening_minutes}m range [{opening.low}, {opening.high}] "
                    f"broken to {held.extreme} and reclaimed at {bar.close}"
                ),
            ),
        )

    def _record(
        self,
        bar: Bar,
        opening: OpeningRange,
        held: Excursion | None,
        session_date: date,
    ) -> Excursion | None:
        """Extend or open the excursion this bar implies, or leave it alone.

        Reads the bar's **high and low**, not its close: a breakout that spiked
        through the range and closed back inside within one minute is still an
        excursion, and reading the close only would miss exactly the fastest
        failures the strategy exists to trade.
        """
        broke_up = opening.breaks_up(bar.high)
        broke_down = opening.breaks_down(bar.low)
        if broke_up and broke_down:
            # An outside bar: the range was left in both directions inside one
            # minute, so neither side is the failure. Declined rather than
            # resolved by a tie-break nobody could justify.
            return held
        if broke_up:
            extreme = (
                bar.high
                if held is None or held.side is not Side.BUY
                else max(held.extreme, bar.high)
            )
            return Excursion(session_date=session_date, side=Side.BUY, extreme=extreme)
        if broke_down:
            extreme = (
                bar.low
                if held is None or held.side is not Side.SELL
                else min(held.extreme, bar.low)
            )
            return Excursion(session_date=session_date, side=Side.SELL, extreme=extreme)
        return held

    def _travelled_far_enough(self, held: Excursion, opening: OpeningRange) -> bool:
        """Whether the excursion cleared `MIN_EXCURSION_FRACTION` of the range's width."""
        edge = opening.high if held.side is Side.BUY else opening.low
        travelled = abs(held.extreme.value - edge.value)
        return travelled >= opening.width * Decimal(str(MIN_EXCURSION_FRACTION))
