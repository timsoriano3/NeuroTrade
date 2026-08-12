"""Value objects the rest of the domain is built from.

Two rules govern everything here.

**Money and prices are exact.** They are backed by `Decimal`, never `float`.
Binary floating point cannot represent 0.01, so a float P&L accumulates error
that shows up as a drifting equity curve and, worse, as a replay digest that
differs across machines. Gate G1 requires bit-for-bit reproducible replays, and
that is impossible on floats. Derived features — moving averages, z-scores,
indicator values — are a different matter and stay `float`, where speed matters
and a rounding error in the twelfth decimal cannot cost money.

**Constructing from `float` is an error.** `Decimal(0.1)` is
`0.1000000000000000055511151231257827`, which silently poisons every downstream
calculation. Callers pass strings, ints, or `Decimal`. Feeds that genuinely hold
floats must go through `from_float`, which routes via `repr` and is explicit at
the call site about where precision was last trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Self

__all__ = [
    "Currency",
    "Money",
    "Price",
    "Quantity",
    "Side",
    "Symbol",
    "Venue",
]

Numeric = Decimal | int | str
"""What may become a `Decimal` without losing precision. `float` is deliberately
absent — see the module docstring."""


def _to_decimal(value: Numeric, field: str) -> Decimal:
    """Convert a caller-supplied value to `Decimal`, refusing lossy inputs.

    Args:
        value: A `Decimal`, `int` or `str`. Passing a `float` is an error.
        field: Name of the field being built, used in the error message.

    Returns:
        The exact `Decimal` equivalent of `value`.

    Raises:
        TypeError: If `value` is a `float`, which cannot convert exactly.
        ValueError: If `value` is a string that is not a valid decimal.

    Example:
        >>> _to_decimal("0.1", "Price")
        Decimal('0.1')
    """
    if isinstance(value, float):  # pragma: no cover - unreachable via typing
        raise TypeError(
            f"{field} must not be built from float: {value!r} is not exactly "
            f"representable. Pass a str or Decimal, or use from_float()."
        )
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not a valid decimal: {value!r}") from exc


class Currency(StrEnum):
    """Currencies the system can hold.

    US-listed equities settle in USD and TSX-listed ones in CAD, so both are live
    at once and must never be added together without an explicit rate.

    Example:
        >>> Currency.USD
        <Currency.USD: 'USD'>
    """

    USD = "USD"  # US-listed equities
    CAD = "CAD"  # Canadian-listed equities


class Side(StrEnum):
    """Direction of an order or fill.

    Names the *action*, not the resulting exposure: closing a long position is a
    SELL, and opening a short is also a SELL. Exposure direction lives on
    `Position.side` instead.

    Example:
        >>> Side.BUY.opposite
        <Side.SELL: 'SELL'>
    """

    BUY = "BUY"  # acquiring shares — opens a long or closes a short
    SELL = "SELL"  # disposing of shares — closes a long or opens a short

    @property
    def opposite(self) -> Side:
        """The side that reverses this one, used to build closing orders.

        Example:
            >>> Side.SELL.opposite
            <Side.BUY: 'BUY'>
        """
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """`+1` for BUY, `-1` for SELL.

        Lets one expression cover both directions in P&L and slippage
        arithmetic, instead of branching on side everywhere.

        Example:
            >>> (Side.BUY.sign, Side.SELL.sign)
            (1, -1)
        """
        return 1 if self is Side.BUY else -1


class Venue(StrEnum):
    """Listing venues in scope at launch (§1: US and Canadian equities).

    A venue is the exchange an instrument is *listed* on. SMART is the exception:
    it is IBKR's order router, valid as an execution destination but never as a
    listing, which is why `Symbol` rejects it.

    Example:
        >>> Venue.TSX.currency
        <Currency.CAD: 'CAD'>
    """

    NASDAQ = "NASDAQ"  # US
    NYSE = "NYSE"  # US
    AMEX = "AMEX"  # US
    ARCA = "ARCA"  # US, mostly ETFs
    BATS = "BATS"  # US
    TSX = "TSX"  # Toronto Stock Exchange
    TSXV = "TSXV"  # TSX Venture — smaller Canadian listings
    SMART = "SMART"  # IBKR's router, not a listing venue

    @property
    def currency(self) -> Currency:
        """The currency instruments on this venue settle in.

        Example:
            >>> (Venue.NASDAQ.currency, Venue.TSXV.currency)
            (<Currency.USD: 'USD'>, <Currency.CAD: 'CAD'>)
        """
        return Currency.CAD if self in (Venue.TSX, Venue.TSXV) else Currency.USD


@dataclass(frozen=True, slots=True, order=True)
class Symbol:
    """A tradable instrument, qualified by listing venue.

    The venue is not decoration. Tickers collide across countries — a bare "TD"
    is Toronto-Dominion on TSX and Tandem Diabetes on NASDAQ — so a universe
    spanning both would otherwise merge two unrelated companies into one.

    Example:
        >>> str(Symbol("AAPL", Venue.NASDAQ))
        'AAPL.NASDAQ'
    """

    ticker: str  # exchange ticker, upper case, no whitespace
    venue: Venue  # where it is listed; determines settlement currency

    def __post_init__(self) -> None:
        """Reject malformed tickers and non-listing venues.

        Raises:
            ValueError: If the ticker is empty, padded or not upper case, or if
                the venue is SMART.
        """
        if not self.ticker:
            raise ValueError("ticker must not be empty")
        if self.ticker != self.ticker.strip():
            raise ValueError(f"ticker has surrounding whitespace: {self.ticker!r}")
        if self.ticker != self.ticker.upper():
            raise ValueError(f"ticker must be upper case: {self.ticker!r}")
        if self.venue is Venue.SMART:
            raise ValueError("SMART is an order route, not a listing venue")

    @property
    def currency(self) -> Currency:
        """Settlement currency, derived from the venue rather than supplied.

        Example:
            >>> Symbol("SHOP", Venue.TSX).currency
            <Currency.CAD: 'CAD'>
        """
        return self.venue.currency

    def __str__(self) -> str:
        return f"{self.ticker}.{self.venue.value}"


@dataclass(frozen=True, slots=True, order=True)
class Price:
    """A price level. Strictly positive.

    Zero is rejected rather than tolerated: feeds emit `0.0` to mean "no data",
    and letting that through as a price is how a stop ends up at zero. Callers
    with a possibly-absent price use `Price | None`.

    Subtracting two prices gives a `Decimal`, not a `Price` — a difference
    between levels (a spread, a stop distance) can be negative or zero, which no
    price level may be.

    Example:
        >>> Price("10.50") - Price("10.25")
        Decimal('0.25')
    """

    value: Decimal  # the level itself, exact

    def __init__(self, value: Numeric) -> None:
        """Build a price from an exact numeric value.

        Args:
            value: A `Decimal`, `int` or decimal string. Not a `float`.

        Raises:
            TypeError: If given a `float` — use `from_float` at feed boundaries.
            ValueError: If the value is not finite or is not positive.

        Example:
            >>> Price("142.60").value
            Decimal('142.60')
        """
        decimal = _to_decimal(value, "Price")
        if not decimal.is_finite():
            raise ValueError(f"Price must be finite: {value!r}")
        if decimal <= 0:
            raise ValueError(f"Price must be positive: {value!r}")
        object.__setattr__(self, "value", decimal)

    @classmethod
    def from_float(cls, value: float) -> Self:
        """Build from a float, going via `repr` to avoid binary artefacts.

        For feed boundaries only. Every call marks a place where precision was
        last trusted, which is exactly what makes those places auditable.

        Args:
            value: A float from an external source.

        Returns:
            The price matching the float's shortest round-tripping decimal form.

        Example:
            >>> Price.from_float(0.1).value          # not 0.1000000000000000055…
            Decimal('0.1')
        """
        return cls(repr(value))

    def __sub__(self, other: Price) -> Decimal:
        return self.value - other.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True, order=True)
class Quantity:
    """A number of shares. Non-negative; direction is carried by `Side`.

    `Decimal` rather than `int` because IBKR supports fractional shares, and
    because position sizing off a Kelly fraction rarely lands on a round lot.

    Example:
        >>> (Quantity(100) + Quantity(50)).value
        Decimal('150')
    """

    value: Decimal  # share count; may be fractional, never negative

    def __init__(self, value: Numeric) -> None:
        """Build a share count from an exact numeric value.

        Args:
            value: A `Decimal`, `int` or decimal string. Not a `float`.

        Raises:
            TypeError: If given a `float`.
            ValueError: If the value is not finite or is negative.

        Example:
            >>> Quantity("0.5").value                # fractional shares are legal
            Decimal('0.5')
        """
        decimal = _to_decimal(value, "Quantity")
        if not decimal.is_finite():
            raise ValueError(f"Quantity must be finite: {value!r}")
        if decimal < 0:
            raise ValueError(f"Quantity must not be negative: {value!r}")
        object.__setattr__(self, "value", decimal)

    @classmethod
    def from_float(cls, value: float) -> Self:
        """Build from a float via `repr`. See `Price.from_float`.

        Example:
            >>> Quantity.from_float(1.5).value
            Decimal('1.5')
        """
        return cls(repr(value))

    @property
    def is_zero(self) -> bool:
        """True when nothing is held — the flat case.

        Example:
            >>> Quantity(0).is_zero
            True
        """
        return self.value == 0

    def __add__(self, other: Quantity) -> Quantity:
        return Quantity(self.value + other.value)

    def __sub__(self, other: Quantity) -> Quantity:
        """Reduce a share count.

        Raises:
            ValueError: If the result would be negative. Closing more than is
                held is a bug, not a short — opening a short is a separate
                action with its own `Side`.

        Example:
            >>> (Quantity(100) - Quantity(40)).value
            Decimal('60')
        """
        return Quantity(self.value - other.value)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class Money:
    """An amount in a specific currency. May be negative — losses are money.

    Cross-currency arithmetic raises. Trading US and Canadian names means both
    currencies are live simultaneously, and silently adding CAD to USD would
    produce a plausible-looking equity curve that is simply wrong. Converting
    requires an explicit rate, which is not this layer's concern.

    Example:
        >>> str(Money("100.50", Currency.USD) + Money("10", Currency.USD))
        '110.50 USD'
    """

    amount: Decimal  # signed; negative means a loss or a debit
    currency: Currency  # never mixed — see _check

    def __init__(self, amount: Numeric, currency: Currency) -> None:
        """Build an amount of money.

        Args:
            amount: A `Decimal`, `int` or decimal string. Not a `float`.
            currency: The currency the amount is denominated in.

        Raises:
            TypeError: If `amount` is a `float`.
            ValueError: If `amount` is not finite.

        Example:
            >>> Money("-25.00", Currency.CAD).amount
            Decimal('-25.00')
        """
        decimal = _to_decimal(amount, "Money")
        if not decimal.is_finite():
            raise ValueError(f"Money must be finite: {amount!r}")
        object.__setattr__(self, "amount", decimal)
        object.__setattr__(self, "currency", currency)

    @classmethod
    def zero(cls, currency: Currency) -> Self:
        """A zero balance, used as the starting point for accumulations.

        Example:
            >>> str(Money.zero(Currency.USD))
            '0 USD'
        """
        return cls(0, currency)

    @classmethod
    def of(cls, price: Price, quantity: Quantity, currency: Currency) -> Self:
        """Notional value of `quantity` shares at `price`.

        Args:
            price: Price per share.
            quantity: Number of shares.
            currency: Settlement currency, normally `symbol.currency`.

        Returns:
            The gross value of the shares, before any costs.

        Example:
            >>> str(Money.of(Price("100.25"), Quantity(4), Currency.USD))
            '401.00 USD'
        """
        return cls(price.value * quantity.value, currency)

    def _check(self, other: Money) -> None:
        """Refuse to combine two different currencies.

        Raises:
            ValueError: If the currencies differ. The caller must convert with
                an explicit rate first.
        """
        if self.currency is not other.currency:
            raise ValueError(
                f"cannot combine {self.currency.value} and {other.currency.value}; "
                f"convert explicitly with a rate"
            )

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.amount - other.amount, self.currency)

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def __mul__(self, factor: Numeric) -> Money:
        """Scale an amount, e.g. by a share count or a risk fraction.

        Example:
            >>> str(Money("10", Currency.USD) * 3)
            '30 USD'
        """
        return Money(self.amount * _to_decimal(factor, "factor"), self.currency)

    def __lt__(self, other: Money) -> bool:
        self._check(other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        self._check(other)
        return self.amount <= other.amount

    def __gt__(self, other: Money) -> bool:
        self._check(other)
        return self.amount > other.amount

    def __ge__(self, other: Money) -> bool:
        self._check(other)
        return self.amount >= other.amount

    def __str__(self) -> str:
        return f"{self.amount} {self.currency.value}"
