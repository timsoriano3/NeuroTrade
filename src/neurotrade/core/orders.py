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

**Terms used here**, for readers coming from outside markets:

- *Slippage* — the gap between the price you decided to trade at and the price
  you actually got. Usually a cost, occasionally a gain, and almost always the
  difference between a backtest that works and a live system that does not.
- *Maker / taker* — a *maker* posts a resting order and waits, adding liquidity
  to the book, and is typically charged less or paid a rebate. A *taker* crosses
  the spread to execute immediately and pays for the privilege.
- *Partial fill* — an order for 1,000 shares may execute as several smaller
  trades at different prices. Each is a separate `Fill`.
- *Commission* — the broker's per-trade charge, separate from slippage.
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
    """How the venue should treat the order's prices.

    Each member declares which prices it requires, so `Order` validation is
    derived from the enum rather than a hand-maintained matrix that can drift.

    Example:
        >>> OrderType.STOP_LIMIT.needs_limit_price, OrderType.STOP_LIMIT.needs_stop_price
        (True, True)
    """

    MARKET = "MARKET"  # execute now at whatever the book offers
    LIMIT = "LIMIT"  # execute only at limit_price or better; may never fill
    STOP = "STOP"  # becomes a market order once price trades through stop_price
    STOP_LIMIT = "STOP_LIMIT"  # becomes a limit order once stop_price is touched

    @property
    def needs_limit_price(self) -> bool:
        """True if this type is meaningless without a `limit_price`."""
        return self in (OrderType.LIMIT, OrderType.STOP_LIMIT)

    @property
    def needs_stop_price(self) -> bool:
        """True if this type is meaningless without a `stop_price`."""
        return self in (OrderType.STOP, OrderType.STOP_LIMIT)


class TimeInForce(StrEnum):
    """How long the order stays live.

    DAY is the default for a day-trading system, and GTC is deliberately
    available but rare: an order surviving overnight contradicts the premise
    that positions are closed within the session.

    Example:
        >>> TimeInForce.DAY.value
        'DAY'
    """

    DAY = "DAY"  # cancelled at the close if unfilled
    GTC = "GTC"  # good till cancelled — survives overnight; rare here by design
    IOC = "IOC"  # immediate-or-cancel: fill what you can now, cancel the rest
    FOK = "FOK"  # fill-or-kill: all of it now, or none of it


class LiquidityFlag(StrEnum):
    """Whether this execution added or removed liquidity.

    Drives fee versus rebate, and is the measurement §5.9's adaptive limit
    placement optimises. A strategy whose fills are 90% TAKER is paying the
    spread every time, and that shows up here before it shows up in the P&L.

    Example:
        >>> LiquidityFlag.UNKNOWN.value
        'UNKNOWN'
    """

    MAKER = "MAKER"  # posted and waited — added liquidity, lower fee or a rebate
    TAKER = "TAKER"  # crossed the spread — removed liquidity, pays the fee
    UNKNOWN = "UNKNOWN"  # broker did not report; guessing would bias the cost model


@dataclass(frozen=True, slots=True, kw_only=True)
class Order(Event):
    """The immutable record of one order placement.

    Carries ``intent_id`` so every order traces back to the proposal that
    justified it, and ``config_hash`` so §6.3's requirement — that any trade be
    reconstructable months later — is satisfied by the order record itself
    rather than by hoping the configuration was written down somewhere.
    """

    id: OrderId  # derived from intent_id + timestamp + attempt
    intent_id: IntentId  # the proposal this order implements
    symbol: Symbol  # instrument being traded
    side: Side  # BUY or SELL — the action, not the resulting exposure
    quantity: Quantity  # shares requested; never zero
    order_type: OrderType  # determines which of the two prices below are required
    limit_price: Price | None = None  # worst acceptable price; LIMIT and STOP_LIMIT only
    stop_price: Price | None = None  # trigger level; STOP and STOP_LIMIT only
    time_in_force: TimeInForce = TimeInForce.DAY  # how long it stays live
    config_hash: str = ""  # hash of the config that produced it, for §6.3 reconstruction

    def __post_init__(self) -> None:
        """Validate size and price/type consistency.

        Raises:
            ValueError: If quantity is zero, or if the prices present do not
                match what `order_type` requires. A LIMIT order with no limit
                price would be silently submitted as something else.
        """
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
        """True if this order can execute immediately without a price move.

        A rough classification used to decide whether to expect a fill at all.
        It will need refining once execution logic is real — a limit order
        priced through the spread is also marketable, and that is not detectable
        without the current book.
        """
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

    id: FillId  # derived from order_id + broker_exec_id, so re-sent reports collapse
    order_id: OrderId  # the order this executed against
    symbol: Symbol  # instrument traded
    side: Side  # direction of this execution
    price: Price  # what we actually paid or received, per share
    quantity: Quantity  # shares in this execution; a partial fill is smaller than the order
    commission: Money  # broker charge for this execution, in the instrument's currency
    liquidity: LiquidityFlag = LiquidityFlag.UNKNOWN  # maker or taker, if reported
    reference_price: Price | None = None  # price we decided against; None means unmeasured
    broker_exec_id: str = ""  # the venue's own execution identifier

    def __post_init__(self) -> None:
        """Validate size and currency.

        Raises:
            ValueError: If quantity is zero, or the commission currency differs
                from the instrument's settlement currency — booking USD fees
                against a TSX fill corrupts the equity curve in a way that looks
                like a small persistent edge.
        """
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
        """Cash value of the shares exchanged, before any costs.

        Example:
            >>> str(demo_fill.notional)               # 100 shares at 100.05
            '10005.00 USD'
        """
        return Money.of(self.price, self.quantity, self.symbol.currency)

    @property
    def slippage_per_share(self) -> Decimal | None:
        """Signed cost per share versus the decision price. Positive is worse.

        Multiplying by `side.sign` is what makes both directions read as costs:
        a buy filled *above* the reference paid up, and a sell filled *below*
        the reference gave up edge. Without the sign flip those two cancel, and
        the nightly cost-model recalibration would conclude execution is free.

        Returns:
            Cost per share, positive when worse than the decision price and
            negative on price improvement. `None` when no reference was
            recorded — an unmeasured cost must not read as zero.

        Example:
            >>> demo_fill.slippage_per_share          # bought 0.05 above 100.00
            Decimal('0.05')
        """
        if self.reference_price is None:
            return None
        return (self.price - self.reference_price) * self.side.sign

    @property
    def slippage_cost(self) -> Money | None:
        """Total slippage on this execution, in the instrument's currency.

        Returns:
            `slippage_per_share` scaled by the shares filled, or `None` if no
            reference price was recorded.

        Example:
            >>> str(demo_fill.slippage_cost)          # 0.05 x 100 shares
            '5.00 USD'
        """
        per_share = self.slippage_per_share
        if per_share is None:
            return None
        return Money(per_share * self.quantity.value, self.symbol.currency)

    @property
    def total_cost(self) -> Money | None:
        """Commission plus slippage — what the execution actually cost.

        This is the quantity §3.3 requires a signal to beat, and the one the
        nightly cost-model recalibration fits against.

        Returns:
            Commission plus slippage, or `None` when slippage is unmeasurable.

        Example:
            >>> str(demo_fill.total_cost)             # 1.00 commission + 5.00 slippage
            '6.00 USD'
        """
        slippage = self.slippage_cost
        if slippage is None:
            return None
        return self.commission + slippage
