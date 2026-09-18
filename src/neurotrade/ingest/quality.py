"""The corpus quality gate (§12.1 stage 5), minus the adjustment half.

Adjustment is in `ingest/actions.py`, because it has to be fixed before a label
can be trusted. The rest lives here: missing sessions, short sessions, holes
inside a session, duplicate prints, halted-looking sessions, and a survivorship
audit. None of it blocks the feature library — it reports, and reports do not
gate.

**Why report rather than repair.** Every fault here has at least two causes and
the corpus cannot tell them apart. A missing session is a failed fetch or a day
the listing did not trade. A hole is a halt or a dropped request. A flat session
is a halt or a genuinely illiquid name. Repairing on a guess would write a
plausible number over a known unknown, which is worse than the gap: the gap is
visible and the guess is not.

**On survivorship, this can only measure one side.** The universe comes from a
source that lists what still trades, so a name that delisted in 2023 is absent
from the corpus entirely and no scan of that corpus can miss it — there is
nothing to find. What *is* measurable is the visible half: how many names have
no history before some date. That is a floor on the bias, never the bias
itself, and `SurvivorshipAudit` says so rather than implying a number it cannot
justify.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from neurotrade.core.events import BarInterval
from neurotrade.core.ports import CalendarPort, CatalogPort, CorpusQualityPort
from neurotrade.core.quality import Duplicate, Gap, SuspectSession
from neurotrade.core.types import Symbol
from neurotrade.core.universe import Universe
from neurotrade.logs import get_logger

__all__ = [
    "CorpusReport",
    "MissingSession",
    "ShortSession",
    "SurvivorshipAudit",
    "audit_corpus",
]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MissingSession:
    """A session the venue held that the corpus has no bars for.

    Expected in bulk while a backfill is still crawling, which is why the
    report truncates this class before the others: a count in the thousands is
    progress, not a fault, and printing it all buries the duplicates that are.

    Example:
        >>> from neurotrade.core.types import Symbol, Venue
        >>> str(MissingSession(Symbol("JPM", Venue.NYSE), date(2026, 9, 11)))
        'JPM.NYSE 2026-09-11'
    """

    symbol: Symbol  # instrument
    session_date: date  # trading day the venue held and the corpus lacks

    def __str__(self) -> str:
        return f"{self.symbol} {self.session_date}"


@dataclass(frozen=True, slots=True)
class ShortSession:
    """A session held, but with fewer bars than the calendar expects.

    Distinct from a missing session, and worse in one specific way: a missing
    session is obvious to any consumer, while a short one looks complete. A
    day holding 370 of its 390 minutes still produces a closing price, an ATR
    and a label — all of them computed from a session that quietly ended early.

    Example:
        >>> from neurotrade.core.types import Symbol, Venue
        >>> ShortSession(Symbol("AAPL", Venue.NASDAQ), date(2026, 3, 14), 370, 390).missing
        20
    """

    symbol: Symbol  # instrument
    session_date: date  # trading day
    held: int  # bars actually in the corpus
    expected: int  # bars the venue calendar says the session should hold

    @property
    def missing(self) -> int:
        """How many bars short the session is."""
        return self.expected - self.held

    def __str__(self) -> str:
        return f"{self.symbol} {self.session_date}: {self.held}/{self.expected} bars"


@dataclass(frozen=True, slots=True)
class SurvivorshipAudit:
    """What can and cannot be said about survivorship bias in this corpus.

    Example:
        >>> audit = SurvivorshipAudit(
        ...     candidates=43, with_history=43, late_listings=(), source_lists_survivors_only=True
        ... )
        >>> audit.measurable
        False
    """

    candidates: int  # names in the universe file
    with_history: int  # names holding at least one bar
    late_listings: tuple[tuple[Symbol, date], ...]  # names whose history starts after the range
    source_lists_survivors_only: bool  # whether the feed can report delisted names at all

    @property
    def measurable(self) -> bool:
        """Whether the bias can be quantified from this corpus at all.

        False whenever the source lists survivors only, which is the case for
        every free source in use. A delisted name is not under-represented in
        the corpus; it is absent, and absence leaves no trace to count.
        """
        return not self.source_lists_survivors_only

    def describe(self) -> str:
        """One line for the terminal."""
        if not self.measurable:
            return (
                f"{self.with_history}/{self.candidates} names have history; "
                f"bias NOT measurable — source lists survivors only"
            )
        return f"{self.with_history}/{self.candidates} names have history"


@dataclass(frozen=True, slots=True)
class CorpusReport:
    """Everything the gate found.

    Example:
        >>> CorpusReport((), (), (), (), (), None, BarInterval.DAY_1).clean
        True
    """

    missing_sessions: tuple[MissingSession, ...]  # calendar expected, corpus has nothing
    short_sessions: tuple[ShortSession, ...]  # held but under-filled
    gaps: tuple[Gap, ...]  # holes inside an otherwise present session
    duplicates: tuple[Duplicate, ...]  # the same instant held twice; always a fault
    suspect: tuple[SuspectSession, ...]  # zero volume, or no price movement
    survivorship: SurvivorshipAudit | None  # None when no universe history was supplied
    interval: BarInterval  # bar size the whole report was computed against

    @property
    def clean(self) -> bool:
        """Whether nothing was found.

        Survivorship is excluded deliberately: it is a standing property of the
        data source, not a fault that can be fixed by re-fetching, so letting it
        turn the gate red would make the gate permanently red and therefore
        ignored.
        """
        return not (
            self.missing_sessions
            or self.short_sessions
            or self.gaps
            or self.duplicates
            or self.suspect
        )

    def describe(self) -> str:
        """One line per fault class, for the terminal."""
        return (
            f"{len(self.missing_sessions)} missing sessions, "
            f"{len(self.short_sessions)} short, "
            f"{len(self.gaps)} gaps, "
            f"{len(self.duplicates)} duplicates, "
            f"{len(self.suspect)} suspect"
        )


def audit_corpus(
    universe: Universe,
    calendar: CalendarPort,
    catalog: CatalogPort,
    quality: CorpusQualityPort,
    *,
    start: date,
    end: date,
    interval: BarInterval = BarInterval.MIN_1,
    survivors_only: bool = True,
) -> CorpusReport:
    """Run every stage-5 check that does not need corporate actions.

    Args:
        universe: Instruments to audit.
        calendar: Which sessions each venue actually held.
        catalog: Bar counts per session, for the missing/short checks.
        quality: The audit queries — gaps, duplicates, suspect sessions.
        start: First session to audit.
        end: Last session to audit, inclusive.
        interval: Bar size. Minute bars are the corpus this is really for.
            Daily bars hold one bar per session, so the short-session and
            intra-session gap checks are vacuous for them and the gap query is
            skipped — see the note at the call site.
        survivors_only: Whether the universe's source can report delisted
            names. True for every free source in use.

    Returns:
        A `CorpusReport`. `clean` is the result to want.

    Raises:
        ValueError: If `end` precedes `start`.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")

    missing: list[MissingSession] = []
    short: list[ShortSession] = []
    gaps: list[Gap] = []
    late: list[tuple[Symbol, date]] = []
    with_history = 0

    for symbol in universe:
        counts = catalog.bar_counts(symbol, interval)
        sessions = calendar.sessions(symbol.venue, start, end)
        if counts:
            with_history += 1
            first_held = min(counts)
            # A name whose history starts inside the range is either a late
            # listing or an incomplete crawl. Either way it is not evidence of
            # survivorship; it is the one thing a survivor-only source CAN show.
            if sessions and first_held > sessions[0]:
                late.append((symbol, first_held))

        for session_date in sessions:
            held = counts.get(session_date, 0)
            if held == 0:
                missing.append(MissingSession(symbol, session_date))
                continue
            session = calendar.session(symbol.venue, session_date)
            if session is None:
                continue
            expected = session.expected_bars(interval)
            if held < expected:
                short.append(ShortSession(symbol, session_date, held, expected))
            # Only sessions that hold something can have a hole inside them;
            # a missing session is already reported above.
            #
            # Skipped entirely for daily bars, for two reasons. It is vacuous —
            # a daily session holds one bar, so there is no "inside" to have a
            # hole in — and it is expensive: `gaps` is one query per
            # instrument-session, which over five years of 43 names is fifty
            # thousand queries against the corpus to prove nothing. The first
            # real run took longer than ten minutes before being killed.
            if interval is not BarInterval.DAY_1:
                gaps.extend(quality.gaps(symbol, session_date, interval))

    duplicates = tuple(
        dup for dup in quality.duplicate_timestamps(interval) if start <= dup.session_date <= end
    )
    suspect = tuple(
        item for item in quality.suspect_sessions(interval) if start <= item.session_date <= end
    )

    audit = SurvivorshipAudit(
        candidates=len(universe),
        with_history=with_history,
        late_listings=tuple(late),
        source_lists_survivors_only=survivors_only,
    )
    report = CorpusReport(
        missing_sessions=tuple(missing),
        short_sessions=tuple(short),
        gaps=tuple(gaps),
        duplicates=duplicates,
        suspect=suspect,
        survivorship=audit,
        interval=interval,
    )
    log.info(
        "corpus_audit",
        interval=interval.value,
        missing=len(report.missing_sessions),
        short=len(report.short_sessions),
        gaps=len(report.gaps),
        duplicates=len(report.duplicates),
        suspect=len(report.suspect),
        clean=report.clean,
    )
    return report


def summarise(items: Sequence[object], limit: int = 20) -> tuple[list[str], int]:
    """First `limit` items as strings, plus how many were withheld.

    A corpus audit can find tens of thousands of missing sessions during a
    backfill that is simply not finished. Printing all of them buries the
    duplicates — which are always faults — under noise that is expected.

    Example:
        >>> summarise([1, 2, 3], limit=2)
        (['1', '2'], 1)
    """
    shown = [str(item) for item in items[:limit]]
    return shown, max(0, len(items) - limit)
