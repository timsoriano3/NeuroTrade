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

**Terms used here**, for readers coming from outside markets:

- *Long / short* — long means owning shares and profiting when price rises;
  short means having sold shares you do not own, profiting when price falls.
- *Average price* — when a position is built from several fills at different
  prices, the size-weighted average of what was paid. It is the break-even
  level, and closing part of a position does not change it.
- *Realised vs unrealised* — realised P&L is locked in by closing; unrealised is
  the paper gain or loss on what is still open, and moves with every tick.
- *Reversal* — selling more than you hold flips a long into a short in one
  action. The old position must be fully closed and booked first.
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

    symbol: Symbol  # the instrument held
    side: Side | None = None  # BUY when long, SELL when short, None when flat
    quantity: Quantity = _NO_SHARES  # shares held, always non-negative
    average_price: Price | None = None  # size-weighted cost basis; None when flat
    realised_pnl: Money  # locked in by closing, GROSS of fees
    fees: Money  # commissions accumulated across every fill
    fill_count: int = 0  # how many fills have been folded in

    def __post_init__(self) -> None:
        """Validate that the flat/non-flat fields agree.

        Raises:
            ValueError: If side or average_price disagree with quantity, or if
                the money fields are not in the instrument's currency. These are
                internal-consistency faults — reaching one means the fold is
                wrong, not the input.
        """
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
        """An empty position — the starting point for the fold.

        Args:
            symbol: The instrument. Its venue determines the currency that P&L
                and fees are booked in.

        Example:
            >>> Position.flat(AAPL).is_flat
            True
        """
        currency: Currency = symbol.currency
        return cls(
            symbol=symbol,
            realised_pnl=Money.zero(currency),
            fees=Money.zero(currency),
        )

    # ── State ────────────────────────────────────────────────

    @property
    def is_flat(self) -> bool:
        """True when nothing is held."""
        return self.side is None

    @property
    def is_long(self) -> bool:
        """True when holding shares that profit if price rises."""
        return self.side is Side.BUY

    @property
    def is_short(self) -> bool:
        """True when holding borrowed shares that profit if price falls."""
        return self.side is Side.SELL

    @property
    def signed_quantity(self) -> Decimal:
        """Exposure as one number: positive long, negative short, zero flat.

        Convenient for portfolio-level sums, where adding a long and a short in
        the same name should net out.

        Example:
            >>> Position.flat(AAPL).apply(demo_fill).signed_quantity
            Decimal('100')
        """
        return Decimal(0) if self.side is None else self.quantity.value * self.side.sign

    # ── Folding in a fill ────────────────────────────────────

    def apply(self, fill: Fill) -> Position:
        """Return the position that results from this fill.

        Handles the four cases a fill can produce: opening from flat, adding to
        an existing position, reducing it, and reversing straight through zero.

        Args:
            fill: The execution to fold in. Must be for this position's symbol.

        Returns:
            A new `Position`. The receiver is unchanged, so the state before the
            fill remains available.

        Raises:
            ValueError: If the fill is for a different instrument.

        Example:
            >>> opened = Position.flat(AAPL).apply(demo_fill)
            >>> (opened.is_long, str(opened.average_price), str(opened.fees))
            (True, '100.05', '1.00 USD')
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
        """Open from flat: the fill sets side, size and cost basis outright."""
        return replace(
            self,
            side=fill.side,
            quantity=fill.quantity,
            average_price=fill.price,
            fees=fees,
            fill_count=self.fill_count + 1,
        )

    def _increased(self, fill: Fill, fees: Money) -> Position:
        """Add to an existing position, re-weighting the cost basis.

        Nothing is realised: adding does not close anything.
        """
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
        """Close some or all of the position, and reopen the other way if asked.

        The closed portion realises P&L at the fill price. Any excess beyond the
        held quantity opens a fresh position on the opposite side.
        """
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
        """Paper gain or loss on the open position, gross of fees.

        Args:
            mark: The price to value the position at — usually the last trade or
                the microprice.

        Returns:
            Gain if the market has moved in the position's favour, loss if not.
            Zero when flat.

        Example:
            >>> held = Position.flat(AAPL).apply(demo_fill)   # long 100 @ 100.05
            >>> str(held.unrealised_pnl(Price("101.05")))
            '100.00 USD'
        """
        if self.side is None or self.average_price is None:
            return Money.zero(self.symbol.currency)
        gain_per_share = (mark - self.average_price) * self.side.sign
        return Money(gain_per_share * self.quantity.value, self.symbol.currency)

    def market_value(self, mark: Price) -> Money:
        """Signed cash value of the open position.

        Negative for a short, so exposures across a portfolio sum correctly.

        Example:
            >>> held = Position.flat(AAPL).apply(demo_fill)
            >>> str(held.market_value(Price("101.00")))
            '10100.00 USD'
        """
        return Money(mark.value * self.signed_quantity, self.symbol.currency)

    @property
    def net_realised_pnl(self) -> Money:
        """Realised P&L after fees — what actually reached the account.

        The gross figure and the fees are kept apart so this stays derivable
        rather than being the only number available.
        """
        return self.realised_pnl - self.fees

    def total_pnl(self, mark: Price) -> Money:
        """Everything the position is worth so far: realised plus open, net of fees.

        Args:
            mark: Price to value the open portion at.

        Example:
            >>> held = Position.flat(AAPL).apply(demo_fill)
            >>> str(held.total_pnl(Price("101.05")))   # +100 open, -1.00 fees
            '99.00 USD'
        """
        return self.net_realised_pnl + self.unrealised_pnl(mark)

    # ── R-multiples (§6.1) ───────────────────────────────────

    def realised_r(self, risk_per_share: Decimal, size: Quantity) -> Decimal:
        """Realised P&L expressed in R, net of fees.

        R is the amount risked per share when the trade was taken, so dividing
        P&L by total risk gives a figure comparable across instruments and
        position sizes: "+2R" means the trade made twice what it risked.

        Args:
            risk_per_share: 1R, from `Intent.risk_per_share` on the actual fill.
            size: The quantity the risk was sized against.

        Returns:
            Net realised P&L divided by total risk. A full stop-out is `-1`.

        Raises:
            ValueError: If total risk is not positive, leaving R undefined.

        Example:
            `demo_round_trip` risked 1.00 per share on 100 shares and made
            198.00 net, so just under 2R:

            >>> demo_round_trip.realised_r(Decimal("1.00"), Quantity(100))
            Decimal('1.98')
        """
        risk = risk_per_share * size.value
        if risk <= 0:
            raise ValueError(f"risk must be positive to express P&L in R: {risk}")
        return self.net_realised_pnl.amount / risk
