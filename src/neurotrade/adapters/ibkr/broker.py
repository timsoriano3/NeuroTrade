"""Placing orders with IBKR, and turning executions into domain events.

This is the only component in the system that can move money, so most of what
follows is about making the wrong thing hard rather than about placing orders,
which is the easy part.

**The live-order guard is structural.** `allow_live_orders` is false in every
profile but `live`. Combined with the port — 4001 and 7496 are live, 4002 and
7497 are paper — it means a research or paper process connected to a live
Gateway *cannot* submit, even if a bug routes an order here. §6.2 wants risk
limits that cannot be talked around; the cheapest version of that is a process
that is unable to trade rather than merely configured not to.

**Submitting is not filling.** `submit` returns once IBKR has accepted the
order. Fills arrive afterwards, possibly several of them, possibly minutes
later, possibly never. They are delivered through the `on_fill` callback rather
than returned, because a return value would be a lie for every order that does
not fill instantly.

**Fills are idempotent.** IBKR re-sends execution reports on reconnect. `FillId`
derives from the broker's own `execId`, so the same report seen twice produces
the same id and books one fill rather than doubling a position.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Protocol, cast

from ib_async import Contract, LimitOrder, MarketOrder, Stock, StopLimitOrder, StopOrder
from ib_async import Order as IbOrder

from neurotrade.adapters.ibkr.connection import IbkrConnection
from neurotrade.adapters.ibkr.market_data import IBKR_EXCHANGE
from neurotrade.core.clock import Clock
from neurotrade.core.ids import FillId, OrderId
from neurotrade.core.orders import Fill, LiquidityFlag, Order, OrderType
from neurotrade.core.types import Currency, Money, Price, Quantity, Side, Symbol

__all__ = ["FillHandler", "IbkrBroker", "LiveOrdersNotPermitted", "OrderRejected"]

logging.getLogger("ib_async").setLevel(logging.WARNING)

type FillHandler = Callable[[Fill], None]
"""Called with each execution. Returns nothing: whatever wants the fill
publishes it, so the emission is itself an event on the log rather than a value
passed privately between two components."""

_SIDE_FROM_IBKR = {"BOT": Side.BUY, "SLD": Side.SELL}
"""IBKR reports executions as BOT/SLD rather than BUY/SELL."""

_LIQUIDITY = {1: LiquidityFlag.MAKER, 2: LiquidityFlag.TAKER}
"""`lastLiquidity`: 1 added liquidity, 2 removed it. Anything else — including
0, which IBKR uses for "not applicable" — stays UNKNOWN rather than guessing,
because a wrong maker/taker flag biases the cost model in the direction that
makes execution look cheaper than it is."""


class LiveOrdersNotPermitted(RuntimeError):
    """Raised when a process that may not trade live tries to.

    Not a configuration error to be worked around: reaching this means an order
    was routed to a live broker by something that should not have been able to.
    """


class OrderRejected(RuntimeError):
    """Raised when IBKR refuses an order.

    Rejection is normal — insufficient margin, a locked symbol, a broker-side
    risk limit — and must not be treated as a system fault. §6.2 expects the
    caller to handle it, not to crash.
    """


class _Execution(Protocol):
    """The execution report fields we read."""

    execId: str  # IBKR's own id; what makes booking a fill idempotent
    time: datetime
    side: str  # "BOT" or "SLD"
    shares: float
    price: float
    lastLiquidity: int


class _CommissionReport(Protocol):
    """The commission fields we read."""

    commission: float
    currency: str


class _IbFill(Protocol):
    """One execution as `ib_async` reports it."""

    execution: _Execution
    commissionReport: _CommissionReport


class _Event(Protocol):
    """`ib_async` events subscribe with `+=` rather than a method."""

    def __iadd__(self, handler: Callable[[object, _IbFill], None]) -> _Event: ...


class _OrderStatus(Protocol):
    """The order-state field the gate G2 probe watches."""

    status: str


class _Trade(Protocol):
    """What `placeOrder` returns: the order, and a stream of its fills."""

    order: object
    orderStatus: _OrderStatus
    fillEvent: _Event


class IbkrBroker:
    """Sends orders to IBKR. Satisfies `BrokerPort`.

    Example:
        >>> from neurotrade.config import IbkrSettings
        >>> from neurotrade.core.clock import LiveClock
        >>> broker = IbkrBroker(
        ...     IbkrConnection(IbkrSettings()), LiveClock(), allow_live_orders=False
        ... )
        >>> broker.may_trade_live
        False
    """

    __slots__ = ("_allow_live_orders", "_clock", "_connection", "_on_fill", "_working")

    def __init__(
        self,
        connection: IbkrConnection,
        clock: Clock,
        *,
        allow_live_orders: bool,
        on_fill: FillHandler | None = None,
    ) -> None:
        """Create the broker.

        Args:
            connection: An `IbkrConnection`. Opened on first use.
            clock: Stamps `ts_init` on fills — when we learned of them, as
                distinct from when they happened at the venue.
            allow_live_orders: From `Settings`. False everywhere but the live
                profile, and required rather than defaulted: a caller must state
                what it is permitting.
            on_fill: Called with each execution. Fills are dropped if omitted,
                which is correct for a probe and wrong for a trading session.
        """
        self._connection = connection
        self._clock = clock
        self._allow_live_orders = allow_live_orders
        self._on_fill = on_fill
        self._working: dict[OrderId, _Trade] = {}

    @property
    def may_trade_live(self) -> bool:
        """Whether this broker is permitted to move real money."""
        return self._allow_live_orders

    async def is_connected(self) -> bool:
        """Whether the broker session is live.

        A disconnect mid-session is a risk event: open positions cannot be
        managed and stops cannot be honoured, which is why §6.2 specifies a
        flat-on-disconnect policy.
        """
        return self._connection.is_connected

    async def submit(self, order: Order) -> None:
        """Send an order to the venue.

        Returns once IBKR has **accepted** the order, which is not the same as
        it having executed. Fills arrive later through `on_fill`.

        Args:
            order: The order to place.

        Raises:
            LiveOrdersNotPermitted: If the connection is to a live port and this
                process may not trade live.
            OrderRejected: If IBKR refuses it.
        """
        self._guard_live(order)
        await self._connection.connect()

        contract = self._contract(order.symbol)
        try:
            trade = cast(_Trade, self._connection.ib.placeOrder(contract, self._ib_order(order)))
        except Exception as error:
            raise OrderRejected(f"IBKR refused {order.id}: {error}") from error

        self._working[order.id] = trade
        trade.fillEvent += lambda _trade, fill: self._handle_fill(order, fill)

    def working(self, order_id: OrderId) -> _Trade | None:
        """The live trade for an order, if this broker still has one.

        Exposed for the gate G2 probe, which has to watch an order reach an
        acknowledged state. Ordinary code should not need it: order state is a
        projection folded from events, not something to poll.
        """
        return self._working.get(order_id)

    async def cancel(self, order_id: OrderId) -> None:
        """Request cancellation of a working order.

        Cancellation is a request, not a guarantee: an order can fill in the gap
        between deciding to cancel and the venue receiving it, so callers must
        handle a fill arriving after a successful cancel.

        Args:
            order_id: The order to cancel. Unknown ids are ignored — an order
                already filled or cancelled is not an error to cancel again.
        """
        trade = self._working.get(order_id)
        if trade is None:
            return
        self._connection.ib.cancelOrder(trade.order)

    def _guard_live(self, order: Order) -> None:
        """Refuse a live order from a process that may not place one.

        The check is on the *port*, not the profile name: what makes an order
        real is which Gateway answered, and a profile mislabelled `paper` while
        pointed at 4001 would otherwise trade real money.
        """
        if self._connection.settings.is_paper_port or self._allow_live_orders:
            return
        raise LiveOrdersNotPermitted(
            f"refusing to submit {order.id} for {order.symbol}: connected to live port "
            f"{self._connection.settings.port} but allow_live_orders is false"
        )

    def _contract(self, symbol: Symbol) -> Contract:
        """Build an IBKR contract for an instrument."""
        exchange = IBKR_EXCHANGE.get(symbol.venue)
        if exchange is None:
            raise OrderRejected(f"no IBKR exchange mapped for {symbol.venue.value}")
        return Stock(symbol.ticker, "SMART", symbol.currency.value, primaryExchange=exchange)

    @staticmethod
    def _ib_order(order: Order) -> IbOrder:
        """Translate a domain order into IBKR's.

        Prices go out as `float`, which is the one place precision is
        deliberately given up: IBKR's wire protocol has no decimal type. The
        order record keeps the exact value, so the audit trail stays exact even
        though the wire is not.
        """
        action = "BUY" if order.side is Side.BUY else "SELL"
        quantity = float(order.quantity.value)
        tif = order.time_in_force.value

        ib_order: IbOrder
        match order.order_type:
            case OrderType.MARKET:
                ib_order = MarketOrder(action, quantity)
            case OrderType.LIMIT:
                assert order.limit_price is not None  # guaranteed by Order
                ib_order = LimitOrder(action, quantity, float(order.limit_price.value))
            case OrderType.STOP:
                assert order.stop_price is not None
                ib_order = StopOrder(action, quantity, float(order.stop_price.value))
            case OrderType.STOP_LIMIT:
                assert order.limit_price is not None
                assert order.stop_price is not None
                ib_order = StopLimitOrder(
                    action,
                    quantity,
                    float(order.limit_price.value),
                    float(order.stop_price.value),
                )

        ib_order.tif = tif
        # Our own id travels with the order, so an execution report can be tied
        # back to the intent that caused it without a lookup table.
        ib_order.orderRef = order.id.value
        return ib_order

    def _handle_fill(self, order: Order, ib_fill: _IbFill) -> None:
        """Convert an execution report and hand it to the caller."""
        if self._on_fill is None:
            return
        self._on_fill(self._to_fill(order, ib_fill, self._clock.now_ns()))

    @staticmethod
    def _to_fill(order: Order, ib_fill: _IbFill, received_at: int) -> Fill:
        """Build a domain `Fill` from an IBKR execution report.

        `FillId` derives from the broker's `execId`, so a report re-sent on
        reconnect produces the same id and books one fill rather than two.
        """
        execution = ib_fill.execution
        report = ib_fill.commissionReport

        commission = (
            Money(Decimal(repr(float(report.commission))), Currency(report.currency))
            if report is not None and report.currency
            else Money.zero(order.symbol.currency)
        )

        return Fill(
            id=FillId.derive(order_id=order.id, broker_exec_id=execution.execId),
            order_id=order.id,
            symbol=order.symbol,
            ts_event=int(execution.time.timestamp() * 1_000_000_000),
            ts_init=received_at,
            side=_SIDE_FROM_IBKR.get(execution.side, order.side),
            price=Price.from_float(execution.price),
            quantity=Quantity.from_float(execution.shares),
            commission=commission,
            liquidity=_LIQUIDITY.get(execution.lastLiquidity, LiquidityFlag.UNKNOWN),
            broker_exec_id=execution.execId,
        )

    def __repr__(self) -> str:
        permission = "live" if self._allow_live_orders else "paper only"
        return f"IbkrBroker({self._connection!r}, {permission})"
