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

import bisect
import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

from neurotrade.core.types import Symbol, Venue

__all__ = ["Universe", "UniverseHistory", "UniverseMembership"]

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


@dataclass(frozen=True, slots=True)
class UniverseMembership:
    """Who was eligible on one session date.

    Symbols rather than a `Universe` because a screen may legitimately admit
    nobody — a price floor on a market-wide gap down, or a corpus with a hole in
    it — and `Universe` rejects an empty set. An empty membership is a real
    answer here: it says "nothing qualified", which is different from "we never
    looked".
    """

    session_date: date  # the session this membership governs, decided before its open
    symbols: tuple[Symbol, ...]  # sorted by (ticker, venue), no duplicates; may be empty

    def __init__(self, session_date: date, symbols: Iterable[Symbol]) -> None:
        object.__setattr__(self, "session_date", session_date)
        object.__setattr__(self, "symbols", tuple(sorted(set(symbols))))

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, item: object) -> bool:
        return item in self.symbols


@dataclass(frozen=True, slots=True)
class UniverseHistory:
    """Point-in-time membership over a span of sessions.

    What makes this worth having over a flat `Universe` is that a backtest at
    date `t` must see the universe as it was *at* `t`, not as it is now. A
    strategy screened against today's membership has been told which names
    survived, which is the survivorship leak §17 names as a primary risk.

    Carrying `survivorship_biased` on the object is deliberate: the flag travels
    with the data instead of living in a README nobody reads at the point of
    use. A history built from a source that only lists names still trading is
    biased no matter how carefully the screen was applied to it, and the
    honest thing is to say so where a caller can assert on it.

    Example:
        >>> history = UniverseHistory(
        ...     [
        ...         UniverseMembership(date(2024, 7, 1), [Symbol("AAPL", Venue.NASDAQ)]),
        ...         UniverseMembership(date(2024, 7, 2), [Symbol("MSFT", Venue.NASDAQ)]),
        ...     ],
        ...     survivorship_biased=True,
        ... )
        >>> [str(symbol) for symbol in history.as_of(date(2024, 7, 2))]
        ['MSFT.NASDAQ']
    """

    rows: tuple[UniverseMembership, ...]  # ascending by session_date, one per date
    survivorship_biased: bool  # True when the source lists only surviving names

    def __init__(self, rows: Iterable[UniverseMembership], *, survivorship_biased: bool) -> None:
        """Build a history from its rows.

        Args:
            rows: One membership per session date, in any order. Sorted here.
            survivorship_biased: Whether the source could only see names that
                still exist. Required rather than defaulted: a caller has to
                decide, and the safe answer is rarely the convenient one.

        Raises:
            ValueError: If two rows claim the same session date. Which one is
                right is not something to guess at.
        """
        ordered = tuple(sorted(rows, key=lambda row: row.session_date))
        dates = [row.session_date for row in ordered]
        if len(set(dates)) != len(dates):
            raise ValueError("two memberships claim the same session date")
        object.__setattr__(self, "rows", ordered)
        object.__setattr__(self, "survivorship_biased", survivorship_biased)

    @property
    def dates(self) -> tuple[date, ...]:
        """Every session date the history covers, ascending."""
        return tuple(row.session_date for row in self.rows)

    def as_of(self, day: date) -> tuple[Symbol, ...]:
        """Membership governing `day`: the latest row at or before it.

        Carrying the last known membership forward is what makes a monthly
        rebalance and a daily one the same object to a caller.

        Args:
            day: The session being traded.

        Returns:
            The symbols eligible on `day`, sorted. Empty when the screen
            admitted nobody.

        Raises:
            LookupError: If `day` precedes the history. Answering "nobody" there
                would read as a screen that rejected everyone, and a backtest
                would quietly trade nothing instead of failing.

        Example:
            >>> history = UniverseHistory(
            ...     [UniverseMembership(date(2024, 7, 1), [Symbol("AAPL", Venue.NASDAQ)])],
            ...     survivorship_biased=False,
            ... )
            >>> len(history.as_of(date(2024, 7, 9)))  # carried forward
            1
        """
        index = bisect.bisect_right(self.dates, day)
        if index == 0:
            first = self.rows[0].session_date if self.rows else None
            raise LookupError(f"no membership on or before {day}; history starts {first}")
        return self.rows[index - 1].symbols

    @property
    def digest(self) -> str:
        """A stable fingerprint of the whole history, bias flag included.

        A run records this the way it records `Universe.digest`, so which
        membership a backtest used stays answerable after the screen's
        thresholds have moved on.

        Example:
            >>> rows = [UniverseMembership(date(2024, 7, 1), [Symbol("AAPL", Venue.NASDAQ)])]
            >>> biased = UniverseHistory(rows, survivorship_biased=True)
            >>> unbiased = UniverseHistory(rows, survivorship_biased=False)
            >>> biased.digest == unbiased.digest
            False
        """
        lines = [f"survivorship_biased={self.survivorship_biased}"]
        lines += [
            f"{row.session_date.isoformat()}|" + ",".join(str(symbol) for symbol in row.symbols)
            for row in self.rows
        ]
        payload = "\n".join(lines).encode()
        return hashlib.blake2b(payload, digest_size=32).hexdigest()[:_DIGEST_CHARS]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[UniverseMembership]:
        return iter(self.rows)

    def __repr__(self) -> str:
        span = (
            f"{self.rows[0].session_date}..{self.rows[-1].session_date}" if self.rows else "empty"
        )
        bias = ", survivorship-biased" if self.survivorship_biased else ""
        return f"UniverseHistory({len(self.rows)} sessions, {span}{bias}, digest={self.digest})"
