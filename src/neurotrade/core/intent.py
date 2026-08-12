"""The output of a strategy: a proposal to trade.

An ``Intent`` is what a strategy plugin produces. It is **a proposal, never an
order** (§5.1). It says *what* and *which way* and *where the idea is wrong*. It
deliberately does not say *how much* — position size is decided downstream by
the risk engine from the model's calibrated probability, and giving a strategy
any say in sizing would let a rule-based component overrule a calibrated one.

There is no ``quantity`` field on this class, and that absence is the design.

**What "R" means.** R is the amount risked on a trade, per share: the distance
from the entry price to the invalidation (stop) price. Buy at 100 with a stop at
99 and 1R is 1.00 per share. A 2R target then means 102. Expressing everything
in R makes results comparable across a $5 stock and a $500 one, and §6.1 makes
it the unit of all bookkeeping — a strategy's record is read as "+2R, -1R, -1R,
+3R", not in dollars, because dollars depend on how much capital happened to be
deployed that day.

**An Intent is a triple-barrier specification.** §8 labels every training
example by which of three barriers a trade hit first: profit target, stop, or
time limit. Those are exactly ``target_r``, ``invalidation`` and ``horizon_ns``.
This is not a coincidence to be noted later — it is what makes a live trade and
a training label the same shape, so the meta-labeller learns from the thing the
system actually does rather than an approximation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from neurotrade.core.clock import Nanos
from neurotrade.core.events import Event
from neurotrade.core.types import Price, Side, Symbol

__all__ = [
    "EntryTrigger",
    "Intent",
]


class EntryTrigger(StrEnum):
    """How the entry level should be interpreted.

    The strategy states its intent about *where*; the execution engine decides
    the actual order type and placement (§5.9), which may differ — a LIMIT
    intent can be worked passively then escalated to aggressive.

    Example:
        >>> EntryTrigger.MARKET.value
        'MARKET'
    """

    MARKET = "MARKET"  # take whatever the book offers now; always fills, costs the spread
    LIMIT = "LIMIT"  # enter at entry_price or better; may never fill
    STOP = "STOP"  # enter once price trades through entry_price, confirming momentum


@dataclass(frozen=True, slots=True, kw_only=True)
class Intent(Event):
    """A strategy's proposal to take a position.

    Inherits ``Event`` rather than ``MarketEvent``: it is scoped to one
    instrument but it is not an *observation* of the market, it is an output of
    the system. That distinction matters because point-in-time rules constrain
    what a strategy may read, not what it may emit.

    Example:
        `demo_intent` is a long at 100 with a stop at 99, so 1R is 1.00 and the
        2R target is 102.

        >>> demo_intent.risk_per_share(Price("100"))
        Decimal('1')
        >>> demo_intent.target_price(Price("100"))
        Price(value=Decimal('102'))
    """

    symbol: Symbol  # instrument being proposed
    side: Side  # BUY to go long, SELL to go short

    entry: EntryTrigger  # how to read entry_price
    entry_price: Price | None  # reference level; None iff entry is MARKET

    invalidation: Price  # stop level — where the idea is proven wrong; defines 1R
    target_r: Decimal  # profit barrier, in multiples of R (2 means "twice what we risk")
    horizon_ns: Nanos  # time barrier; the position closes on expiry regardless of P&L

    strategy: str  # registered plugin name that produced this
    strategy_version: str  # its semantic version, so a retune is distinguishable
    rationale: str  # human-readable reason, shown on the dashboard (§14) and journal (§6.3)

    def __post_init__(self) -> None:
        """Validate that the proposal is internally coherent.

        Raises:
            ValueError: If entry_price disagrees with the trigger type, if
                target_r or horizon_ns are not positive, if the audit fields are
                blank, or if the stop sits on the wrong side of the entry (a
                long whose stop is above its entry would be stopped out
                immediately).
        """
        Event.__post_init__(self)

        if (self.entry is EntryTrigger.MARKET) != (self.entry_price is None):
            raise ValueError(
                f"entry_price must be None for MARKET and set otherwise; "
                f"got entry={self.entry.value} entry_price={self.entry_price}"
            )
        if self.target_r <= 0:
            raise ValueError(f"target_r must be positive: {self.target_r}")
        if self.horizon_ns <= 0:
            raise ValueError(f"horizon_ns must be positive: {self.horizon_ns}")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty — it is an audit record")
        if not self.strategy or not self.strategy_version:
            raise ValueError("strategy and strategy_version identify the author")

        # Direction sanity. Only checkable when the entry level is known: a
        # MARKET intent has no reference price until it fills.
        if self.entry_price is not None:
            wrong_side = (
                self.invalidation >= self.entry_price
                if self.side is Side.BUY
                else self.invalidation <= self.entry_price
            )
            if wrong_side:
                raise ValueError(
                    f"invalidation {self.invalidation} is on the wrong side of entry "
                    f"{self.entry_price} for a {self.side.value}"
                )

    def risk_per_share(self, entry_price: Price) -> Decimal:
        """1R — the loss per share if the invalidation level is reached.

        Takes the entry explicitly rather than reading ``self.entry_price``,
        because for a MARKET intent the real entry is the fill price, and R must
        be computed from what was actually paid rather than what was hoped for.
        Reading the proposed price would understate risk exactly when the fill
        was bad.

        Args:
            entry_price: The realised entry — the fill price, not the proposal.

        Returns:
            Distance from entry to the stop, always positive.

        Raises:
            ValueError: If entry equals the invalidation, leaving the trade with
                no defined risk and R undefined.

        Example:
            >>> demo_intent.risk_per_share(Price("100"))       # filled as proposed
            Decimal('1')
            >>> demo_intent.risk_per_share(Price("100.50"))    # slipped 0.50
            Decimal('1.50')
        """
        risk = abs(entry_price - self.invalidation)
        if risk == 0:
            raise ValueError(
                f"entry {entry_price} equals invalidation — the trade has no defined risk"
            )
        return risk

    def target_price(self, entry_price: Price) -> Price:
        """The profit barrier as a price, given a realised entry.

        Args:
            entry_price: The realised entry — the fill price.

        Returns:
            Entry plus `target_r` times 1R, in the direction of the trade. A
            worse fill widens R and pushes the target further out, which keeps
            the reward-to-risk ratio honest instead of silently degrading it.

        Example:
            >>> demo_intent.target_price(Price("100"))         # 2R above a 1.00 risk
            Price(value=Decimal('102'))
            >>> demo_intent.target_price(Price("100.50"))      # worse fill, wider R
            Price(value=Decimal('103.50'))
        """
        move = self.risk_per_share(entry_price) * self.target_r
        return Price(entry_price.value + move * self.side.sign)

    def expires_at(self) -> Nanos:
        """Absolute time of the time barrier.

        Returns:
            The instant after which the position is closed regardless of P&L.

        Example:
            >>> demo_intent.expires_at()          # ts_event 1_000 + one hour
            3600000001000
        """
        return self.ts_event + self.horizon_ns

    @property
    def reward_to_risk(self) -> Decimal:
        """Gross reward-to-risk, before costs.

        Gross is the operative word: §3.3 requires a signal to beat spread, fees
        and modelled slippage, and none of those are visible here. A 2R intent
        on an instrument with a wide spread can still be negative expectancy
        once the cost model is applied.
        """
        return self.target_r
