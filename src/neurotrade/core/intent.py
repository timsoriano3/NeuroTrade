"""The output of a strategy: a proposal to trade.

An ``Intent`` is what a strategy plugin produces. It is **a proposal, never an
order** (§5.1). It says *what* and *which way* and *where the idea is wrong*. It
deliberately does not say *how much* — position size is decided downstream by
the risk engine from the model's calibrated probability, and giving a strategy
any say in sizing would let a rule-based component overrule a calibrated one.

There is no ``quantity`` field on this class, and that absence is the design.

**An Intent is a triple-barrier specification.** §8 labels every training
example by which of three barriers a trade hit first: profit target, stop, or
time limit. Those are exactly ``target_r``, ``invalidation`` and ``horizon_ns``.
This is not a coincidence to be noted later — it is what makes a live trade and
a training label the same shape, so the meta-labeller learns from the thing the
system actually does rather than an approximation of it.

Everything is expressed in **R** — multiples of the risk taken on the trade,
where 1R is the distance from entry to invalidation. A strategy that says
"target 2R" is portable across instruments and price levels in a way that
"target $1.50" is not, and §6.1 makes R the unit of all bookkeeping.
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
    """

    MARKET = "MARKET"
    """Enter now, at whatever the book offers."""

    LIMIT = "LIMIT"
    """Enter at ``entry_price`` or better. May never fill."""

    STOP = "STOP"
    """Enter once price trades through ``entry_price``, confirming momentum."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Intent(Event):
    """A strategy's proposal to take a position.

    Inherits ``Event`` rather than ``MarketEvent``: it is scoped to one
    instrument but it is not an *observation* of the market, it is an output of
    the system. That distinction matters because point-in-time rules constrain
    what a strategy may read, not what it may emit.
    """

    symbol: Symbol
    side: Side

    entry: EntryTrigger
    entry_price: Price | None
    """The reference level. ``None`` if and only if ``entry`` is MARKET."""

    invalidation: Price
    """Where the idea is proven wrong. The stop barrier, and the definition of 1R."""

    target_r: Decimal
    """Profit barrier, in multiples of R."""

    horizon_ns: Nanos
    """Time barrier. The position is closed on expiry regardless of P&L."""

    strategy: str
    strategy_version: str
    rationale: str
    """Human-readable reason, surfaced on the dashboard (§14) and journal (§6.3)."""

    def __post_init__(self) -> None:
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
        because for a MARKET intent the real entry is the fill price and R must
        be computed from what was actually paid, not from what was hoped for.
        """
        risk = abs(entry_price - self.invalidation)
        if risk == 0:
            raise ValueError(
                f"entry {entry_price} equals invalidation — the trade has no defined risk"
            )
        return risk

    def target_price(self, entry_price: Price) -> Price:
        """The profit barrier as a price, given a realised entry."""
        move = self.risk_per_share(entry_price) * self.target_r
        return Price(entry_price.value + move * self.side.sign)

    def expires_at(self) -> Nanos:
        """Absolute time of the time barrier."""
        return self.ts_event + self.horizon_ns

    @property
    def reward_to_risk(self) -> Decimal:
        """Gross reward-to-risk, before costs.

        Gross is the operative word: §3.3 requires a signal to beat spread, fees
        and modelled slippage, and none of those are visible here. A 2R intent
        on an instrument with a wide spread can still be negative expectancy.
        """
        return self.target_r
