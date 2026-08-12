"""Position — what the fills add up to.

A ``Position`` is a **projection**, not a record: it is what you get by folding
a sequence of fills, and it holds no information that is not derivable from
them. It is immutable, and ``apply`` returns a new position rather than mutating
in place, so the position as of any point in the log can be reconstructed by
replaying up to that point. That is what makes "what were we holding at 09:47"
answerable months later (§6.3).

Realised P&L is kept **gross of fees**, with fees accumulated separately.
Netting them as you go makes it impossible to answer "was the strategy right and
the execution expensive, or was the strategy simply wrong" — which is the first
question worth asking about a losing week, and the one §5.9 exists to act on.

Nothing here knows about R. Converting P&L into R-multiples needs the risk the
trade was taken with, which lives on the ``Intent``, so the R helpers take it as
an argument rather than the position storing a copy that could drift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from neurotrade.core.orders import Fill
from neurotrade.core.types import Currency, Money, Price, Quantity, Side, Symbol

__all__ = ["Position"]

_NO_SHARES = Quantity(0)
"""Shared default. Safe because Quantity is frozen — the usual objection to
calling a constructor in a dataclass default is mutable state, which cannot
arise here, and one shared instance avoids re-allocating on every fold step."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Position:
    """Net exposure in one instrument, folded from fills.

    ``quantity`` is always non-negative; direction is carried by ``side``, which
    is ``None`` exactly when the position is flat.
    """

    symbol: Symbol
    side: Side | None = None
    quantity: Quantity = _NO_SHARES
    average_price: Price | None = None
    realised_pnl: Money
    fees: Money
    fill_count: int = 0

    def __post_init__(self) -> None:
        flat = self.quantity.is_zero
        if flat != (self.side is None):
            raise ValueError(
                f"side must be None exactly when flat; got side={self.side} "
                f"quantity={self.quantity}"
            )
        if flat != (self.average_price is None):
            raise ValueError("average_price must be None exactly when flat")
        for money, name in ((self.realised_pnl, "realised_pnl"), (self.fees, "fees")):
            if money.currency is not self.symbol.currency:
                raise ValueError(
                    f"{name} is {money.currency.value} but {self.symbol} "
                    f"settles {self.symbol.currency.value}"
                )

    @classmethod
    def flat(cls, symbol: Symbol) -> Position:
        """An empty position, before any fill."""
        currency: Currency = symbol.currency
        return cls(
            symbol=symbol,
            realised_pnl=Money.zero(currency),
            fees=Money.zero(currency),
        )

    # ── State ────────────────────────────────────────────────

    @property
    def is_flat(self) -> bool:
        return self.side is None

    @property
    def is_long(self) -> bool:
        return self.side is Side.BUY

    @property
    def is_short(self) -> bool:
        return self.side is Side.SELL

    @property
    def signed_quantity(self) -> Decimal:
        """Positive when long, negative when short, zero when flat."""
        return Decimal(0) if self.side is None else self.quantity.value * self.side.sign

    # ── Folding in a fill ────────────────────────────────────

    def apply(self, fill: Fill) -> Position:
        """Return the position that results from this fill.

        Handles the four cases a fill can produce: opening from flat, adding to
        an existing position, reducing it, and reversing straight through zero.
        """
        if fill.symbol != self.symbol:
            raise ValueError(f"fill is for {fill.symbol}, position is {self.symbol}")

        accrued_fees = self.fees + fill.commission

        if self.side is None:
            return self._opened(fill, accrued_fees)
        if fill.side is self.side:
            return self._increased(fill, accrued_fees)
        return self._reduced_or_reversed(fill, accrued_fees)

    def _opened(self, fill: Fill, fees: Money) -> Position:
        return replace(
            self,
            side=fill.side,
            quantity=fill.quantity,
            average_price=fill.price,
            fees=fees,
            fill_count=self.fill_count + 1,
        )

    def _increased(self, fill: Fill, fees: Money) -> Position:
        assert self.average_price is not None  # guaranteed by __post_init__
        total = self.quantity + fill.quantity
        weighted = (
            self.average_price.value * self.quantity.value + fill.price.value * fill.quantity.value
        ) / total.value
        return replace(
            self,
            quantity=total,
            average_price=Price(weighted),
            fees=fees,
            fill_count=self.fill_count + 1,
        )

    def _reduced_or_reversed(self, fill: Fill, fees: Money) -> Position:
        assert self.side is not None
        assert self.average_price is not None

        closing = min(self.quantity.value, fill.quantity.value)
        gain_per_share = (fill.price - self.average_price) * self.side.sign
        realised = self.realised_pnl + Money(gain_per_share * closing, self.symbol.currency)
        remaining = self.quantity.value - closing

        if remaining > 0:
            # Partial close. The average entry price is unchanged: closing part
            # of a position does not alter what the rest of it cost.
            return replace(
                self,
                quantity=Quantity(remaining),
                realised_pnl=realised,
                fees=fees,
                fill_count=self.fill_count + 1,
            )

        flipped = fill.quantity.value - closing
        if flipped == 0:
            return replace(
                self,
                side=None,
                quantity=_NO_SHARES,
                average_price=None,
                realised_pnl=realised,
                fees=fees,
                fill_count=self.fill_count + 1,
            )

        # Reversal: the excess opens a new position on the other side, priced at
        # this fill. The closed leg is fully realised first, so a flip is never
        # silently netted into a single averaged position.
        return replace(
            self,
            side=fill.side,
            quantity=Quantity(flipped),
            average_price=fill.price,
            realised_pnl=realised,
            fees=fees,
            fill_count=self.fill_count + 1,
        )

    # ── Valuation ────────────────────────────────────────────

    def unrealised_pnl(self, mark: Price) -> Money:
        """Open P&L at ``mark``, gross of fees. Zero when flat."""
        if self.side is None or self.average_price is None:
            return Money.zero(self.symbol.currency)
        gain_per_share = (mark - self.average_price) * self.side.sign
        return Money(gain_per_share * self.quantity.value, self.symbol.currency)

    def market_value(self, mark: Price) -> Money:
        """Signed notional of the open position."""
        return Money(mark.value * self.signed_quantity, self.symbol.currency)

    @property
    def net_realised_pnl(self) -> Money:
        """Realised P&L after fees — what actually reached the account."""
        return self.realised_pnl - self.fees

    def total_pnl(self, mark: Price) -> Money:
        """Realised plus unrealised, after fees."""
        return self.net_realised_pnl + self.unrealised_pnl(mark)

    # ── R-multiples (§6.1) ───────────────────────────────────

    def realised_r(self, risk_per_share: Decimal, size: Quantity) -> Decimal:
        """Realised P&L expressed in R, net of fees.

        ``risk_per_share`` comes from the originating ``Intent`` and ``size``
        is the quantity the risk was sized against. Both are supplied by the
        caller: a position holding its own copy could drift from the intent it
        came from, and then the R-multiples in the journal would be fiction.
        """
        risk = risk_per_share * size.value
        if risk <= 0:
            raise ValueError(f"risk must be positive to express P&L in R: {risk}")
        return self.net_realised_pnl.amount / risk
