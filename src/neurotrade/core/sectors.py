"""Which sector an instrument belonged to, on a given day.

§5.4's beta-adjusted relative strength and §5.5's index-relative residual
reversion both regress a name's intraday return on a market leg and a sector
leg. Without the sector leg the residual still carries sector beta — so a day
when energy runs reads as alpha on every energy name at once, which is the exact
thing those strategies exist to strip out. Nothing in the system said that AAPL
belongs with MSFT rather than with XOM until this module.

**Sectors are named by their tradable proxy, not by a label.** A sector here is
an `XL*` SPDR, which is an instrument in `config/universe.yaml` with bars in the
corpus. Two things fall out. The residual is *hedgeable* — a position in the
name against a position in the sector ETF is a trade someone can put on — and
the sector leg of the regression is a real return series rather than an index
level nobody can buy. `00-phase-2.plan.md`'s decision 3 chose this over per-name
GICS membership, which is licensed and whose point-in-time history would have to
be reconstructed from present-day membership.

**Point-in-time, and the shape is not decoration.** `sector_of` takes a date and
returns what was true *then*. The reason it has to is in the file already: GICS
moved the payment networks out of Information Technology and into Financials in
its 2023 structure change, so V and MA sit in XLK at the start of the seed
window and in XLF by the end of the crawled one. A map that answered "XLF"
for a session in October 2022 would be residualising those names against a
sector they were not in, and no test downstream could detect it.

**The honest limitation.** Every assignment whose start we do not know is
recorded from `UNKNOWN_SINCE`, which means "at least as far back as any session
this project holds". That is present-day classification applied backwards, and
for a reclassification nobody recorded it is wrong in exactly the direction that
flatters a backtest — the name is residualised against the sector it ended up
in, which is the sector it correlated with most by the end of the sample. The
mitigation is that the universe is 58 mega-caps over four years with one known
reclassification in it, and that the dated shape means recording another one is
a data change rather than a schema change. It is a real bias, it is stated, and
it is not measured.

**A benchmark has no sector.** SPY, QQQ, the SPDRs themselves and VXX are
regressors or instruments in their own right, never members. `sector_of`
returns `None` for them rather than inventing a self-membership, because a
sector ETF regressed on itself has a residual of exactly zero and would enter a
cross-sectional ranking as the most average name in the universe.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from neurotrade.core.types import Symbol, Venue

__all__ = ["UNKNOWN_SINCE", "SectorAssignment", "SectorMap"]

UNKNOWN_SINCE: date = date(1900, 1, 1)
"""Start date for an assignment whose real start nobody recorded.

Not a lie dressed as a date: it reads as "before anything this project has data
for", and `SectorMap.unknown_starts` counts how many assignments carry it so the
bias the module docstring describes can be quoted rather than guessed at."""

_DIGEST_CHARS = 16
"""64 bits, matching `core.universe` and `core.ids`."""


@dataclass(frozen=True, slots=True)
class SectorAssignment:
    """One instrument's membership of one sector, over one span of time.

    Half-open: `effective` is the first session it holds and `until` is the
    first session it does not. Half-open because a reclassification takes effect
    before an open — GICS 2023 moved the payment networks before the open on
    2023-03-17 — so the two spans must meet at that date without overlapping on
    it.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> was = SectorAssignment(
        ...     symbol=Symbol("V", Venue.NYSE), sector=Symbol("XLK", Venue.ARCA),
        ...     effective=UNKNOWN_SINCE, until=date(2023, 3, 17),
        ... )
        >>> (was.holds(date(2023, 3, 16)), was.holds(date(2023, 3, 17)))
        (True, False)
    """

    symbol: Symbol  # the instrument classified
    sector: Symbol  # the sector's tradable proxy, an XL* SPDR
    effective: date  # first session this membership holds, inclusive
    until: date | None  # first session it does not, exclusive; None means current

    def __post_init__(self) -> None:
        """Validate the span.

        Raises:
            ValueError: If the span is empty or inverted, or if an instrument is
                assigned to itself. A self-assignment would give a sector ETF a
                residual of exactly zero against its own regressor.
        """
        if self.until is not None and self.until <= self.effective:
            raise ValueError(
                f"{self.symbol} in {self.sector}: until {self.until} "
                f"is not after effective {self.effective}"
            )
        if self.symbol == self.sector:
            raise ValueError(f"{self.symbol} cannot be its own sector")

    def holds(self, day: date) -> bool:
        """Whether this membership was in force on a session."""
        return self.effective <= day and (self.until is None or day < self.until)

    @property
    def start_is_known(self) -> bool:
        """Whether the start date is real, or the `UNKNOWN_SINCE` stand-in."""
        return self.effective != UNKNOWN_SINCE

    def __str__(self) -> str:
        span = "onwards" if self.until is None else f"until {self.until}"
        start = "always" if not self.start_is_known else f"from {self.effective}"
        return f"{self.symbol.ticker} in {self.sector.ticker} {start} {span}"


@dataclass(frozen=True, slots=True)
class SectorMap:
    """Point-in-time sector membership, plus the market leg for each venue.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> xlk, spy = Symbol("XLK", Venue.ARCA), Symbol("SPY", Venue.ARCA)
        >>> aapl = Symbol("AAPL", Venue.NASDAQ)
        >>> sectors = SectorMap(
        ...     assignments=[SectorAssignment(aapl, xlk, UNKNOWN_SINCE, None)],
        ...     markets={Venue.NASDAQ: spy},
        ... )
        >>> sectors.sector_of(aapl, on=date(2024, 7, 8)).ticker
        'XLK'
        >>> sectors.market_of(aapl).ticker
        'SPY'
    """

    assignments: tuple[SectorAssignment, ...]  # sorted by (symbol, effective)
    markets: Mapping[Venue, Symbol]  # the market leg a venue's names regress on

    def __init__(
        self,
        assignments: Iterable[SectorAssignment],
        markets: Mapping[Venue, Symbol],
    ) -> None:
        """Build a map, checking that no instrument is in two sectors at once.

        Args:
            assignments: The memberships. Sorted on the way in, so the caller's
                ordering cannot move `digest` or any lookup.
            markets: Market leg per listing venue. A venue absent from this has
                no market regressor, which `market_of` reports as `None` rather
                than substituting one from another country.

        Raises:
            ValueError: If two assignments for one instrument overlap in time.
                That is the failure mode the dated shape exists to prevent, and
                a silent overlap would make `sector_of` depend on iteration
                order — which the determinism invariant forbids outright.
        """
        ordered = tuple(sorted(assignments, key=lambda row: (str(row.symbol), row.effective)))
        _refuse_overlaps(ordered)
        object.__setattr__(self, "assignments", ordered)
        object.__setattr__(self, "markets", dict(markets))

    def sector_of(self, symbol: Symbol, *, on: date) -> Symbol | None:
        """The sector proxy an instrument belonged to on a session.

        Args:
            symbol: The instrument.
            on: The session. Required and keyword-only, because every caller
                has one and a default of "today" would silently look ahead in a
                backtest.

        Returns:
            The sector's SPDR, or `None` for an instrument with no membership on
            that date — a benchmark, or a name whose assignment starts later.

        Example:
            >>> from neurotrade.core.types import Venue
            >>> spy = Symbol("SPY", Venue.ARCA)
            >>> SectorMap([], {}).sector_of(spy, on=date(2024, 7, 8)) is None
            True
        """
        for row in self.assignments:
            if row.symbol == symbol and row.holds(on):
                return row.sector
        return None

    def market_of(self, symbol: Symbol) -> Symbol | None:
        """The market leg for an instrument's listing venue.

        Per venue rather than per instrument because the market factor is a
        property of where a name trades: SPY does not price TSX listings, so the
        fourteen Canadian names regress on XIU instead. A venue with no entry
        returns `None`, never a substitute from another country — a Canadian
        name residualised against a US index would carry the whole
        currency-and-country move as alpha.
        """
        return self.markets.get(symbol.venue)

    def members(self, sector: Symbol, *, on: date) -> tuple[Symbol, ...]:
        """Every instrument in one sector on a session, in universe order."""
        return tuple(
            row.symbol for row in self.assignments if row.sector == sector and row.holds(on)
        )

    def sectors(self, *, on: date) -> tuple[Symbol, ...]:
        """Every sector with at least one member on a session, sorted."""
        return tuple(sorted({row.sector for row in self.assignments if row.holds(on)}))

    @property
    def classified(self) -> tuple[Symbol, ...]:
        """Every instrument this map has an assignment for, at any date, sorted."""
        return tuple(sorted({row.symbol for row in self.assignments}))

    @property
    def unknown_starts(self) -> int:
        """Assignments whose start date is the `UNKNOWN_SINCE` stand-in.

        The size of the bias the module docstring describes, as a number a report
        can print. Every one of these is present-day classification applied
        backwards.
        """
        return sum(1 for row in self.assignments if not row.start_is_known)

    @property
    def digest(self) -> str:
        """A stable fingerprint of the membership, like `Universe.digest`.

        The map is data rather than configuration — it grows as names are added
        and as reclassifications are recorded — so a run records this instead of
        folding it into the config hash and churning the fingerprint stamped on
        every trade.
        """
        fingerprint = hashlib.blake2b(digest_size=_DIGEST_CHARS // 2)
        for row in self.assignments:
            fingerprint.update(f"{row.symbol}|{row.sector}|{row.effective}|{row.until}\n".encode())
        for venue in sorted(self.markets):
            fingerprint.update(f"market|{venue.value}|{self.markets[venue]}\n".encode())
        return fingerprint.hexdigest()

    def __len__(self) -> int:
        return len(self.assignments)

    def __repr__(self) -> str:
        return (
            f"SectorMap({len(self.classified)} instruments, "
            f"{len(self.assignments)} assignments, {len(self.markets)} markets)"
        )


def _refuse_overlaps(ordered: tuple[SectorAssignment, ...]) -> None:
    """Raise if any instrument holds two memberships on one day.

    Checked on the sorted sequence, so only adjacent pairs can overlap: the rows
    for one instrument are contiguous and ascending by `effective`, and a later
    row starting before an earlier one ends is the only way to double-book.
    """
    for earlier, later in pairwise(ordered):
        if earlier.symbol != later.symbol:
            continue
        if earlier.until is None or later.effective < earlier.until:
            raise ValueError(
                f"{earlier.symbol} is in two sectors at once: {earlier} overlaps {later}"
            )
