"""Orders and fills — what execution does with a proposal.

An ``Order`` is the immutable record of a placement, and a ``Fill`` the
immutable record of one execution against it. Neither mutates. An order that
carried a ``status`` field which changed over time would destroy the append-only
property §4.1 depends on: replaying the log would give you the *final* state of
each order rather than the state it had at each moment, and "what did the system
know at 09:47" becomes unanswerable.

Current order state is therefore a **projection** — folded from the placement
record plus its subsequent events — and lives in ``execution/``, not here.

``Fill`` is where the cost model gets its ground truth. §8 requires spread,
commission and slippage to be applied *inside* the backtest and recalibrated
nightly from real fills, which is only possible if every fill records what was
actually paid and what price we expected when we decided to trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from neurotrade.core.events import Event
from neurotrade.core.ids import FillId, IntentId, OrderId
from neurotrade.core.types import Money, Price, Quantity, Side, Symbol

__all__ = [
    "Fill",
    "LiquidityFlag",
    "Order",
    "OrderType",
    "TimeInForce",
]


class OrderType(StrEnum):
    """How the venue should treat the order's prices."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

    @property
    def needs_limit_price(self) -> bool:
        return self in (OrderType.LIMIT, OrderType.STOP_LIMIT)

    @property
    def needs_stop_price(self) -> bool:
        return self in (OrderType.STOP, OrderType.STOP_LIMIT)


class TimeInForce(StrEnum):
    """How long the order stays live.

    DAY is the default for a day-trading system, and GTC is deliberately
    available but rare: an order surviving overnight contradicts the premise
    that positions are closed within the session.
    """

    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    """Immediate-or-cancel. Fill what you can now, cancel the rest."""

    FOK = "FOK"
    """Fill-or-kill. All of it now, or none."""


class LiquidityFlag(StrEnum):
    """Whether this execution added or removed liquidity.

    Drives fee versus rebate, and is the measurement §5.9's adaptive limit
    placement optimises. A strategy whose fills are 90% TAKER is paying the
    spread every time, and that shows up here before it shows up in the P&L.
    """

    MAKER = "MAKER"
    TAKER = "TAKER"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True, kw_only=True)
class Order(Event):
    """The immutable record of one order placement.

    Carries ``intent_id`` so every order traces back to the proposal that
    justified it, and ``config_hash`` so §6.3's requirement — that any trade be
    reconstructable months later — is satisfied by the order record itself
    rather than by hoping the configuration was written down somewhere.
    """

    id: OrderId
    intent_id: IntentId
    symbol: Symbol
    side: Side
    quantity: Quantity
    order_type: OrderType
    limit_price: Price | None = None
    stop_price: Price | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    config_hash: str = ""

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        if self.quantity.is_zero:
            raise ValueError("an order must have non-zero quantity")

        if self.order_type.needs_limit_price and self.limit_price is None:
            raise ValueError(f"{self.order_type.value} requires a limit_price")
        if not self.order_type.needs_limit_price and self.limit_price is not None:
            raise ValueError(f"{self.order_type.value} must not carry a limit_price")

        if self.order_type.needs_stop_price and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} requires a stop_price")
        if not self.order_type.needs_stop_price and self.stop_price is not None:
            raise ValueError(f"{self.order_type.value} must not carry a stop_price")

    @property
    def is_marketable(self) -> bool:
        """True if this order can execute immediately without a price move."""
        return self.order_type in (OrderType.MARKET,) or self.time_in_force in (
            TimeInForce.IOC,
            TimeInForce.FOK,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Fill(Event):
    """One execution against an order.

    Carries both what was paid and what was expected, because the difference is
    the only honest measure of execution quality. ``reference_price`` is the
    price the decision was made against — the microprice at decision time (§5.6)
    — and without it slippage can only be estimated from a model rather than
    measured from reality.
    """

    id: FillId
    order_id: OrderId
    symbol: Symbol
    side: Side
    price: Price
    quantity: Quantity
    commission: Money
    liquidity: LiquidityFlag = LiquidityFlag.UNKNOWN
    reference_price: Price | None = None
    broker_exec_id: str = ""

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        if self.quantity.is_zero:
            raise ValueError("a fill must have non-zero quantity")
        if self.commission.currency is not self.symbol.currency:
            raise ValueError(
                f"commission is {self.commission.currency.value} but "
                f"{self.symbol} settles {self.symbol.currency.value}"
            )

    @property
    def notional(self) -> Money:
        """Value of the shares exchanged, before costs."""
        return Money.of(self.price, self.quantity, self.symbol.currency)

    @property
    def slippage_per_share(self) -> Decimal | None:
        """Signed cost per share versus the decision price. Positive is worse.

        A buy filled above the reference paid up; a sell filled below the
        reference gave up edge. Both are positive here, so slippage always reads
        as a cost regardless of direction. ``None`` when no reference was
        recorded, which is honest — an unmeasured cost must not read as zero.
        """
        if self.reference_price is None:
            return None
        return (self.price - self.reference_price) * self.side.sign

    @property
    def slippage_cost(self) -> Money | None:
        """Total slippage on this execution, in the instrument's currency."""
        per_share = self.slippage_per_share
        if per_share is None:
            return None
        return Money(per_share * self.quantity.value, self.symbol.currency)

    @property
    def total_cost(self) -> Money | None:
        """Commission plus slippage — what the execution actually cost.

        This is the quantity §3.3 requires a signal to beat, and the one the
        nightly cost-model recalibration fits against.
        """
        slippage = self.slippage_cost
        if slippage is None:
            return None
        return self.commission + slippage
