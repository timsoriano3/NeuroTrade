"""The set of instruments the system is allowed to consider.

**Not in `TRADER_PLAN.md` as a Phase 0 artefact.** §12.1 stage 1 says "start the
IBKR backfill crawler … for our exact tradable universe" and stage 3 says the
universe history arrives via yfinance — but stage 1 runs first, and a crawler
with no list of symbols has no work queue. This is the smallest thing that
unblocks it: an explicit, ordered, deduplicated set of `Symbol`, supplied as
data rather than derived.

Later phases replace how it is *populated* — §5's Universe Selector ranks
candidates nightly by relative volume, catalyst tags and liquidity — without
replacing what a universe *is*. That is why this lives in `core` while the file
it is loaded from does not.

**A universe is keyed on `Symbol`, never on ticker.** "TD" is Toronto-Dominion
on both NYSE (in USD) and TSE (in CAD), at different prices. Collapsing those
into one entry would average two currencies into one series, which is the exact
failure `Symbol` carries a venue to prevent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from neurotrade.core.types import Symbol, Venue

__all__ = ["Universe"]

_DIGEST_CHARS = 16
"""64 bits, matching `core.ids`. Long enough that two different universes will
not collide in this system's lifetime, short enough to sit in a log line."""


@dataclass(frozen=True, slots=True)
class Universe:
    """An immutable set of instruments, in a stable order.

    Construction sorts and deduplicates, so two universes built from the same
    symbols in different orders are equal and share a digest. That is what lets
    `digest` identify *what* was crawled rather than what order a YAML file
    happened to list.

    Example:
        >>> universe = Universe([Symbol("MSFT", Venue.NASDAQ), Symbol("AAPL", Venue.NASDAQ)])
        >>> [str(symbol) for symbol in universe]
        ['AAPL.NASDAQ', 'MSFT.NASDAQ']
    """

    symbols: tuple[Symbol, ...]  # sorted by (ticker, venue), no duplicates

    def __init__(self, symbols: Iterable[Symbol]) -> None:
        """Build a universe from any iterable of symbols.

        Args:
            symbols: The instruments in scope. Duplicates are collapsed; a
                ticker listed on two venues is two distinct instruments and
                both are kept.

        Raises:
            ValueError: If no symbols are given. An empty universe is a
                configuration mistake that would otherwise read downstream as
                "nothing to fetch" — the crawler would report a complete corpus
                having done nothing, which is the same silent-success failure
                the venue calendar refuses to produce for an unknown date.

        Example:
            >>> Universe([Symbol("TD", Venue.NYSE), Symbol("TD", Venue.TSX)]).symbols
            (Symbol(ticker='TD', venue=<Venue.NYSE: 'NYSE'>), Symbol(ticker='TD', venue=<Venue.TSX: 'TSX'>))
        """  # noqa: E501
        # sorted() of a set is deterministic; iterating the set itself is not,
        # and the determinism invariant forbids depending on that order.
        ordered = tuple(sorted(set(symbols)))
        if not ordered:
            raise ValueError("a universe must hold at least one symbol")
        object.__setattr__(self, "symbols", ordered)

    @property
    def venues(self) -> tuple[Venue, ...]:
        """The venues represented, sorted.

        The crawler asks the calendar for sessions per venue rather than per
        symbol, since every instrument on a venue shares its trading days.

        Example:
            >>> Universe([Symbol("SHOP", Venue.TSX), Symbol("AAPL", Venue.NASDAQ)]).venues
            (<Venue.NASDAQ: 'NASDAQ'>, <Venue.TSX: 'TSX'>)
        """
        return tuple(sorted({symbol.venue for symbol in self.symbols}))

    def by_venue(self, venue: Venue) -> tuple[Symbol, ...]:
        """The symbols listed on one venue, in universe order.

        Returns a tuple rather than a `Universe` because the result may be
        empty, and an empty `Universe` is rejected by construction.

        Args:
            venue: The listing venue to filter on.

        Example:
            >>> universe = Universe([Symbol("TD", Venue.NYSE), Symbol("TD", Venue.TSX)])
            >>> [str(symbol) for symbol in universe.by_venue(Venue.TSX)]
            ['TD.TSX']
        """
        return tuple(symbol for symbol in self.symbols if symbol.venue is venue)

    @property
    def digest(self) -> str:
        """A stable fingerprint of the membership.

        The universe is data, not configuration: it grows from a few dozen
        hand-listed names to a few thousand discovered ones, and folding that
        into the config hash would churn the hash recorded on every trade
        whenever a ticker was added. A run records this instead, so what was
        crawled stays answerable later.

        BLAKE2b for the same reason `core.ids` uses it — `hash()` is salted per
        process and would differ between runs of the same program.

        Example:
            >>> forwards = Universe([Symbol("AAPL", Venue.NASDAQ), Symbol("MSFT", Venue.NASDAQ)])
            >>> backwards = Universe([Symbol("MSFT", Venue.NASDAQ), Symbol("AAPL", Venue.NASDAQ)])
            >>> forwards.digest == backwards.digest
            True
            >>> len(forwards.digest)
            16
        """
        payload = "\n".join(str(symbol) for symbol in self.symbols).encode()
        return hashlib.blake2b(payload, digest_size=32).hexdigest()[:_DIGEST_CHARS]

    def __len__(self) -> int:
        return len(self.symbols)

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self.symbols)

    def __contains__(self, item: object) -> bool:
        return item in self.symbols

    def __repr__(self) -> str:
        venues = "+".join(venue.value for venue in self.venues)
        return f"Universe({len(self.symbols)} symbols, {venues}, digest={self.digest})"
