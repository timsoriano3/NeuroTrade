"""Identifiers linking an intent to its orders to their fills.

**Identifiers are derived, not generated.** They are a hash of the facts that
make the thing unique, so replaying the same session produces the same
identifiers — every time, on every machine.

A UUID would break gate G1 outright. Replay the same session twice with
``uuid4`` and every order gets a different id, so the journal differs, so the
run digest differs, and a replay that was supposed to prove determinism proves
nothing. ``uuid1`` is worse: it embeds the wall clock and the MAC address.

Python's built-in ``hash()`` is also unusable here. It is randomly salted per
process for strings and bytes unless ``PYTHONHASHSEED`` is pinned, so the same
input hashes differently between runs of the same program. BLAKE2b has no salt
and is specified byte-for-byte, which is why these ids do not depend on any
environment variable being set correctly.

The derivation inputs are always fields already stored on the record, so an id
can be recomputed from the journal months later and checked against the one
recorded at the time. An identifier that cannot be re-derived is a number that
has to be trusted; one that can is a checksum.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar, Self

__all__ = [
    "FillId",
    "IntentId",
    "OrderId",
]

_DIGEST_CHARS = 16
"""64 bits of digest. At the volume this system trades — thousands of orders a
day, not billions — collision probability stays negligible while the id remains
short enough to read in a log line."""

_SEPARATOR = b"\x1f"
"""ASCII unit separator. Joining parts with a printable character would let
("AB", "C") and ("A", "BC") derive the same id; \\x1f cannot appear in the
string forms of the fields we hash."""


def _derive(prefix: str, parts: tuple[object, ...]) -> str:
    """Hash the given parts into a prefixed identifier string.

    Args:
        prefix: Three-letter kind marker, e.g. `"int"`.
        parts: The facts that make this thing unique. Each is stringified, so
            every part must have a stable `__str__` — which is why `Symbol`
            renders as `TICKER.VENUE` rather than a default object repr.

    Returns:
        A string of the form `"<prefix>_<16 hex chars>"`.

    Raises:
        ValueError: If `parts` is empty, which would give every record of that
            kind the same identifier.

    Example:
        >>> _derive("int", ("a", "b")) == _derive("int", ("a", "b"))
        True
    """
    if not parts:
        raise ValueError("cannot derive an identifier from no parts")
    payload = _SEPARATOR.join(str(part).encode("utf-8") for part in parts)
    digest = hashlib.blake2b(payload, digest_size=32).hexdigest()[:_DIGEST_CHARS]
    return f"{prefix}_{digest}"


@dataclass(frozen=True, slots=True, order=True)
class _DerivedId:
    """Base for content-derived identifiers.

    Subclasses differ only by prefix. They are distinct types on purpose: an
    ``OrderId`` passed where a ``FillId`` belongs is a mistake mypy should
    catch, and with a bare ``str`` alias it could not.
    """

    value: str  # the full "<prefix>_<digest>" string

    PREFIX: ClassVar[str] = ""  # overridden per subclass; identifies the kind

    def __post_init__(self) -> None:
        """Validate the shape of an identifier built from a raw string.

        Raises:
            ValueError: If the prefix is wrong or the length is not exactly
                prefix plus digest. This is the runtime counterpart to the
                static type distinction — it catches ids read back from storage
                or from a broker message.
        """
        expected = f"{self.PREFIX}_"
        if not self.value.startswith(expected):
            raise ValueError(f"{type(self).__name__} must start with {expected!r}: {self.value!r}")
        if len(self.value) != len(expected) + _DIGEST_CHARS:
            raise ValueError(f"malformed {type(self).__name__}: {self.value!r}")

    @classmethod
    def _from_parts(cls, *parts: object) -> Self:
        """Derive an identifier of this kind from its defining facts."""
        return cls(_derive(cls.PREFIX, parts))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class IntentId(_DerivedId):
    """Identifies one strategy proposal.

    Derived from the author, the instrument and the moment. Two strategies
    firing on the same bar produce different ids; the same strategy firing twice
    on the same bar produces the same id, which is correct — that is one
    proposal, not two.
    """

    PREFIX: ClassVar[str] = "int"

    @classmethod
    def derive(
        cls, *, strategy: str, strategy_version: str, symbol: object, ts_event: int, seq: int
    ) -> Self:
        """Derive the id for a strategy's proposal.

        Args:
            strategy: Registered plugin name, e.g. `"orb_stocks_in_play"`.
            strategy_version: Semantic version of that plugin. Included so a
                retuned strategy cannot be confused with its predecessor.
            symbol: The instrument, normally a `Symbol`.
            ts_event: Venue timestamp of the bar or tick that triggered it.
            seq: Tiebreaker within that timestamp.

        Returns:
            A stable `IntentId` for exactly this proposal.

        Example:
            >>> IntentId.derive(
            ...     strategy="orb", strategy_version="1.0.0",
            ...     symbol="AAPL.NASDAQ", ts_event=1_773_495_000_000_000_000, seq=0,
            ... )
            IntentId(value='int_4a2256397fa1b828')
        """
        return cls._from_parts(strategy, strategy_version, symbol, ts_event, seq)


@dataclass(frozen=True, slots=True, order=True)
class OrderId(_DerivedId):
    """Identifies one order.

    Includes ``attempt`` because a single intent can legitimately produce
    several orders: a passive limit that is cancelled and re-placed higher is a
    new order against the same intent (§5.9's escalation ladder). Without it the
    replacement would collide with the order it replaces.
    """

    PREFIX: ClassVar[str] = "ord"

    @classmethod
    def derive(cls, *, intent_id: IntentId, ts_event: int, attempt: int = 0) -> Self:
        """Derive the id for an order placed against an intent.

        Args:
            intent_id: The proposal this order implements.
            ts_event: When the order was created.
            attempt: Which placement this is for that intent. Increment when
                cancelling and re-placing, or the new order collides with the
                one it replaces.

        Returns:
            A stable `OrderId`.

        Example:
            >>> intent = IntentId("int_1e0d31ab35f1caf1")
            >>> first = OrderId.derive(intent_id=intent, ts_event=1, attempt=0)
            >>> retry = OrderId.derive(intent_id=intent, ts_event=1, attempt=1)
            >>> first == retry
            False
        """
        return cls._from_parts(intent_id, ts_event, attempt)


@dataclass(frozen=True, slots=True, order=True)
class FillId(_DerivedId):
    """Identifies one execution against an order.

    ``broker_exec_id`` is the venue's own execution identifier. Deriving from it
    makes the id idempotent: brokers re-send execution reports on reconnect, and
    the same report must not book the same fill twice.
    """

    PREFIX: ClassVar[str] = "fil"

    @classmethod
    def derive(cls, *, order_id: OrderId, broker_exec_id: str) -> Self:
        """Derive the id for one execution report.

        Args:
            order_id: The order that executed.
            broker_exec_id: The broker's execution identifier for this specific
                partial or complete fill.

        Returns:
            A stable `FillId`, identical for a re-sent copy of the same report.

        Raises:
            ValueError: If `broker_exec_id` is empty. Without it two separate
                partial fills of the same order would derive the same id and
                one would be silently discarded.

        Example:
            >>> order = OrderId("ord_0123456789abcdef")
            >>> a = FillId.derive(order_id=order, broker_exec_id="0000e0d5.01")
            >>> b = FillId.derive(order_id=order, broker_exec_id="0000e0d5.01")
            >>> a == b                               # re-sent report, one fill
            True
        """
        if not broker_exec_id:
            raise ValueError("broker_exec_id is required — it is what makes fills idempotent")
        return cls._from_parts(order_id, broker_exec_id)
