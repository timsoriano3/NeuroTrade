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

    value: str

    PREFIX: ClassVar[str] = ""

    def __post_init__(self) -> None:
        expected = f"{self.PREFIX}_"
        if not self.value.startswith(expected):
            raise ValueError(f"{type(self).__name__} must start with {expected!r}: {self.value!r}")
        if len(self.value) != len(expected) + _DIGEST_CHARS:
            raise ValueError(f"malformed {type(self).__name__}: {self.value!r}")

    @classmethod
    def _from_parts(cls, *parts: object) -> Self:
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
        if not broker_exec_id:
            raise ValueError("broker_exec_id is required — it is what makes fills idempotent")
        return cls._from_parts(order_id, broker_exec_id)
