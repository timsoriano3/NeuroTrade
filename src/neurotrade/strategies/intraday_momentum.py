"""Intraday momentum: trade a move that has left the day's noise area — §5.4, Family C.

**The idea in one line.** Most of what a session does is noise; a move that is
already larger than the move this instrument usually makes *by this hour* is
evidence of a real demand/supply imbalance, so trade with it and hold it to the
bell.

**The noise area, as published.** Zarattini, Aziz and Barbon (SSRN 4824172,
section 3) define an equilibrium zone from the instrument's own recent history.
For each of the previous 14 sessions they measure the absolute move from that
session's open to the *same time of day*, average those into `sigma`, and place
the boundaries at:

    UpperBound = max(open, prior_close) x (1 + VM x sigma)
    LowerBound = min(open, prior_close) x (1 - VM x sigma)

Two features of that are the whole point. The band is **time-of-day dependent**
— the move required to signal an imbalance at 10:00 is about half the move
required at 15:30, because the average distance travelled from the open grows
through the session — and the anchors are **gap-adjusted**, because an overnight
gap is itself an imbalance and should not be spent proving one. `VM` is the
volatility multiplier of section 4.4: above 1 is more conservative, below 1 more
aggressive, and 1.5 is where the authors measure the best risk-adjusted return.

**Checkpoints, not ticks.** The published rule only looks at HH:00 and HH:30,
"to mitigate the risk of overtrading caused by short-term market fluctuations".
A strategy reading every minute bar would enter on the spike that a
semi-hourly rule waits out — their Figure 2 shows the entry taken at 10:30 on a
move that left the area minutes earlier.

**What is deliberately not the published model.** The paper's headline (Sharpe
1.33, +1,985% over 2007-2024) is its *third* variant: a trailing stop at
`max(UpperBound, VWAP)` that ratchets through the session, plus daily
volatility-targeted sizing capped at 4x leverage. Neither is expressible here.
`Intent` carries one static invalidation level, and sizing is the risk engine's
job (§6.1), not a strategy's. What this module implements is the paper's **base
model** — stop at the opposite boundary, flat at the close — whose published
result is a Sharpe of 0.61 net of $0.0035/share commission and $0.001/share
slippage, on SPY, with 100% notional. That is the number to compare a
measurement against; anything quoting 1.33 for this code is quoting a different
strategy. A ratcheting stop is Phase 3 work, once fills are simulated.

**No profit target, so the target barrier is out of reach on purpose.** The
published exit is the stop or the closing bell and nothing else. `target_r` is
required and must be positive, so `NO_TARGET_R` sits at ten times a risk that is
already most of a session's typical travel — a barrier a session would have to
move some 8% to reach. It is not a tuned parameter and is not swept.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import ClassVar, Final, Self

from neurotrade.core.events import Bar, MarketSession
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.types import CORPUS_PLACES, Price, Side, Symbol, tidy_decimal
from neurotrade.features.levels import SessionLevels
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

__all__ = ["IntradayMomentum", "noise_bounds"]

_NO_TRADE: Final[tuple[Intent, ...]] = ()

_NANOS_PER_MINUTE: Final = 60_000_000_000

DEFAULT_VOLATILITY_MULTIPLIER: Final = 1.0
"""Width of the noise area, in units of the mean move from the open.

One is the paper's own setting: at that value the band is exactly the average
move recorded at this time of day, so "outside the area" means "further than
usual" with nothing added on top."""

DEFAULT_CHECKPOINT_MINUTES: Final = 30
"""Minutes between the only moments an entry is considered.

Semi-hourly is the published rule, and it is a filter rather than a convenience:
it is what separates a sustained imbalance from a spike that has already
reverted by the time the half hour closes."""

SWEEP_VOLATILITY_MULTIPLIERS: Final = (1.0, 1.5)
"""The band widths measured, and the whole declared search.

Two trials, both published: 1.0 is the value every headline result in the paper
uses, and 1.5 is where its section 4.4 sweep puts the best risk-adjusted return.
Nothing else is swept — the 14-session lookback and the 30-minute checkpoint are
the published construction, and a value tried beside them would raise the
deflation hurdle for the whole family in exchange for no evidence (§17)."""

NO_TARGET_R: Final = Decimal(10)
"""Profit barrier, in multiples of risk, standing in for "there is no target".

The published model exits at the stop or at the bell. Risk here is the distance
to the far side of the noise area, typically most of a session's usual travel,
so ten times it is a barrier a normal session cannot reach. Deliberately not a
parameter: it is not swept, and it exists only because `Intent.target_r` must be
positive."""


def noise_bounds(levels: SessionLevels, *, multiplier: float) -> tuple[Price, Price] | None:
    """The upper and lower boundary of the noise area, as of these levels.

    Args:
        levels: The session so far. `mean_abs_move_from_open` supplies `sigma`
            and is already time-of-day dependent, so nothing here has to be.
        multiplier: The volatility multiplier — how many mean moves wide the
            half-band is.

    Returns:
        `(upper, lower)`, or `None` while the band cannot be stated: before the
        move history is full, or when the resulting half-width is not a fraction
        strictly between 0 and 1. The second case is a corpus fault rather than a
        market — a typical move of a whole 100% would put the lower boundary at
        or below zero, which no `Price` can hold — and a strategy that raises
        kills the run it was being measured in.

    Example:
        A session that opened at 100 after closing at 99, with a typical move of
        1% by now: the boundaries sit 1% outside the wider pair of anchors.

        >>> from neurotrade.features.levels import SessionLevels
        >>> bounds = noise_bounds(SessionLevels(
        ...     session_date=date(2024, 7, 8), open_ns=0, close_ns=1,
        ...     session_open=Price("100"), high=Price("100"), low=Price("100"),
        ...     close=Price("100"), vwap=None, bar_count=30,
        ...     prior_close=Price("99"), mean_abs_move_from_open=0.01,
        ... ), multiplier=1.0)
        >>> for bound in bounds:
        ...     print(bound)
        101
        98.01
    """
    sigma = levels.mean_abs_move_from_open
    if sigma is None:
        return None
    half_width = multiplier * sigma
    if not 0.0 < half_width < 1.0:
        return None
    # Via `str`: the half-width is a float feature, and `Decimal(float)` would
    # carry the binary artefact of it into a price that a stop is read from.
    band = Decimal(str(half_width))
    high_anchor, low_anchor = levels.session_open, levels.session_open
    # The gap is part of the imbalance, so it widens the area rather than being
    # traded twice: after a gap down the upper boundary is lifted by the gap.
    if levels.prior_close is not None:
        high_anchor = max(high_anchor, levels.prior_close)
        low_anchor = min(low_anchor, levels.prior_close)
    return (
        Price(tidy_decimal(high_anchor.value * (1 + band), CORPUS_PLACES)),
        Price(tidy_decimal(low_anchor.value * (1 - band), CORPUS_PLACES)),
    )


@arsenal.strategy
class IntradayMomentum(Strategy):
    """Trade a half-hourly close outside the day's noise area, flat at the bell.

    Reads only session levels, so it declares no features: the open, the prior
    close and the mean move from the open are §5.3 shared infrastructure and
    arrive on every context.

    Example:
        >>> IntradayMomentum().qualified_name
        'intraday_momentum@1.0.0'
    """

    name, version = "intraday_momentum", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (
        Regime.TREND_UP,
        Regime.TREND_DOWN,
        Regime.HIGH_VOLATILITY,
    )
    """A sustained imbalance is a trend day, and the paper's section 4.1 puts the
    strategy's Sharpe at 1.50 with VIX above 6 and 3.50 above 40 — the edge is
    concentrated in volatile sessions. Never CHOP or REVERSAL, which are the days
    a move back inside the area is the trade, and never the midday lull."""

    cost_sensitivity: ClassVar[float] = 1.0
    """One or two round trips a session held for hours — the paper's base model
    averages 1.3 trades a day. Its section 4.6 is a whole section on commission
    and slippage, so the bar is not lower than a gap trade's even at that
    frequency."""

    volatility_multiplier: float = DEFAULT_VOLATILITY_MULTIPLIER
    """Half-width of the noise area, in mean moves from the open. Each distinct
    value is a separate trial, which is what `sweep` declares. Not a `ClassVar`:
    an instance carries its own, so one run can measure several."""

    checkpoint_minutes: int = DEFAULT_CHECKPOINT_MINUTES
    """Minutes between the moments an entry is considered, counted from the
    session open."""

    target_r: ClassVar[Decimal] = NO_TARGET_R
    """Profit barrier as a multiple of risk. See `NO_TARGET_R` — the published
    model has no target, and this stands in for one."""

    def __init__(
        self,
        *,
        volatility_multiplier: float = DEFAULT_VOLATILITY_MULTIPLIER,
        checkpoint_minutes: int = DEFAULT_CHECKPOINT_MINUTES,
    ) -> None:
        """Start with nothing proposed on any instrument.

        Args:
            volatility_multiplier: Half-width of the noise area, in mean moves
                from the open. Each value is its own trial; see `sweep`.
            checkpoint_minutes: Minutes between the moments an entry is
                considered.

        Example:
            >>> IntradayMomentum(volatility_multiplier=1.5).volatility_multiplier
            1.5
        """
        self.volatility_multiplier = volatility_multiplier
        self.checkpoint_minutes = checkpoint_minutes
        # Per symbol, because the host subscribes one instance to the whole
        # universe. The date is carried so a new session starts clean without
        # anyone having to notice the bell.
        self._proposed: dict[Symbol, tuple[date, Side]] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per band width in `SWEEP_VOLATILITY_MULTIPLIERS`.

        Example:
            >>> [label for label, _ in IntradayMomentum.sweep()]
            ['VM 1', 'VM 1.5']
        """
        return tuple(
            (f"VM {value:g}", cls(volatility_multiplier=value))
            for value in SWEEP_VOLATILITY_MULTIPLIERS
        )

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Propose an entry in the direction of a move outside the noise area.

        Args:
            bar: The bar that just closed.
            context: Its view of the world; `levels` carries everything read.

        Returns:
            At most one proposal, and nothing on the 29 bars out of 30 that are
            not a checkpoint.

        Example:
            >>> IntradayMomentum().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.TREND_UP,
            ... ))
            ()
        """
        levels = context.levels
        if levels is None or context.session is not MarketSession.REGULAR:
            return _NO_TRADE
        if bar.ts_event >= levels.close_ns:
            return _NO_TRADE
        # The published rule acts at HH:00 and HH:30 — with a 09:30 open, every
        # thirtieth minute of the session. Minute zero is the open itself, where
        # no move from the open exists to measure.
        minute = (bar.ts_event - levels.open_ns) // _NANOS_PER_MINUTE
        if minute <= 0 or minute % self.checkpoint_minutes != 0:
            return _NO_TRADE

        bounds = noise_bounds(levels, multiplier=self.volatility_multiplier)
        if bounds is None:
            return _NO_TRADE
        upper, lower = bounds
        if bar.close > upper:
            # The stop is the far side of the area: a long is wrong once price
            # has crossed the boundary that would have made it a short.
            side, invalidation = Side.BUY, lower
        elif bar.close < lower:
            side, invalidation = Side.SELL, upper
        else:
            return _NO_TRADE

        # One proposal per direction per session. The published rule re-enters
        # on a crossover to the *opposite* boundary and that is kept, because a
        # flip is new evidence; a second long at 11:00 is the same evidence
        # counted twice and would let one session dominate a backtest.
        if self._proposed.get(bar.symbol) == (levels.session_date, side):
            return _NO_TRADE
        self._proposed[bar.symbol] = (levels.session_date, side)

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
                # Flat at the bell, which is the published exit and the day
                # engine's rule besides (Phase 2 plan, decision 2).
                horizon_ns=levels.close_ns - bar.ts_event,
                strategy=self.name,
                strategy_version=self.version,
                rationale=(
                    f"{minute}m in, close {bar.close} outside noise area "
                    f"[{lower}, {upper}] at VM {self.volatility_multiplier:g}"
                ),
            ),
        )
