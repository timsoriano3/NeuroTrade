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

# What may be turned into a Decimal without losing precision. `float` is
# deliberately absent.
Numeric = Decimal | int | str


def _to_decimal(value: Numeric, field: str) -> Decimal:
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
    """Currencies we can hold. US equities settle USD, TSX-listed names CAD."""

    USD = "USD"
    CAD = "CAD"


class Side(StrEnum):
    """Direction of an order or fill.

    Deliberately not LONG/SHORT: this describes the action, not the resulting
    exposure. Closing a long is a SELL.
    """

    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL. For signed quantity and P&L arithmetic."""
        return 1 if self is Side.BUY else -1


class Venue(StrEnum):
    """Listing venues in scope at launch (§1: US and Canadian equities).

    SMART is IBKR's order router rather than a listing venue; it is valid for
    execution but never as a symbol's primary listing.
    """

    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    AMEX = "AMEX"
    ARCA = "ARCA"
    BATS = "BATS"
    TSX = "TSX"
    TSXV = "TSXV"
    SMART = "SMART"

    @property
    def currency(self) -> Currency:
        return Currency.CAD if self in (Venue.TSX, Venue.TSXV) else Currency.USD


@dataclass(frozen=True, slots=True, order=True)
class Symbol:
    """A tradable instrument, qualified by listing venue.

    The venue is not decoration. Tickers collide across countries — a bare
    "TD" is Toronto-Dominion on TSX and Tandem Diabetes on NASDAQ — and a
    universe spanning both would otherwise merge two instruments into one.
    """

    ticker: str
    venue: Venue

    def __post_init__(self) -> None:
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
        return self.venue.currency

    def __str__(self) -> str:
        return f"{self.ticker}.{self.venue.value}"


@dataclass(frozen=True, slots=True, order=True)
class Price:
    """A price level. Strictly positive.

    Zero is rejected rather than tolerated: feeds emit 0.0 to mean "no data",
    and letting that through as a price is how a stop ends up at zero. Callers
    with possibly-absent prices use `Price | None`.

    Differences between prices are not prices — subtraction returns Decimal.
    """

    value: Decimal

    def __init__(self, value: Numeric) -> None:
        decimal = _to_decimal(value, "Price")
        if not decimal.is_finite():
            raise ValueError(f"Price must be finite: {value!r}")
        if decimal <= 0:
            raise ValueError(f"Price must be positive: {value!r}")
        object.__setattr__(self, "value", decimal)

    @classmethod
    def from_float(cls, value: float) -> Self:
        """Build from a float, going via `repr` to avoid binary artefacts.

        For feed boundaries only. `Price.from_float(0.1)` is `Decimal("0.1")`,
        not `Decimal("0.1000000000000000055511151231257827")`.
        """
        return cls(repr(value))

    def __sub__(self, other: Price) -> Decimal:
        return self.value - other.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True, order=True)
class Quantity:
    """A number of shares. Non-negative; direction is carried by `Side`.

    Decimal rather than int because IBKR supports fractional shares, and
    because position sizing off a Kelly fraction rarely lands on a round lot.
    """

    value: Decimal

    def __init__(self, value: Numeric) -> None:
        decimal = _to_decimal(value, "Quantity")
        if not decimal.is_finite():
            raise ValueError(f"Quantity must be finite: {value!r}")
        if decimal < 0:
            raise ValueError(f"Quantity must not be negative: {value!r}")
        object.__setattr__(self, "value", decimal)

    @classmethod
    def from_float(cls, value: float) -> Self:
        return cls(repr(value))

    @property
    def is_zero(self) -> bool:
        return self.value == 0

    def __add__(self, other: Quantity) -> Quantity:
        return Quantity(self.value + other.value)

    def __sub__(self, other: Quantity) -> Quantity:
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
    """

    amount: Decimal
    currency: Currency

    def __init__(self, amount: Numeric, currency: Currency) -> None:
        decimal = _to_decimal(amount, "Money")
        if not decimal.is_finite():
            raise ValueError(f"Money must be finite: {amount!r}")
        object.__setattr__(self, "amount", decimal)
        object.__setattr__(self, "currency", currency)

    @classmethod
    def zero(cls, currency: Currency) -> Self:
        return cls(0, currency)

    @classmethod
    def of(cls, price: Price, quantity: Quantity, currency: Currency) -> Self:
        """Notional value of `quantity` shares at `price`."""
        return cls(price.value * quantity.value, currency)

    def _check(self, other: Money) -> None:
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
