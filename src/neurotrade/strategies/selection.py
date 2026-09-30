"""The universe selector: which names are worth trading today — §5, §5.2.

§5 promises a selector that "ranks candidates nightly by relative volume,
catalyst tags and liquidity". This is the part of it that today's data supports:
relative volume, measured intraday against each instrument's own profile.
Catalyst tags need a news feed the data plan does not ingest, and liquidity
needs a book; both are Phase 4 and neither is approximated here.

**It exists because the published ORB edge is the selection, not the breakout.**
`00-phase-2.plan.md` is blunt about this. The 2.4-2.8 Sharpe comes from
RVOL-ranking 7,000 stocks; the single-instrument replications fail — QQQ at
2 cents/share gives Sharpe 0.23, and a 14-signal MNQ falsification study had ORB
long at T=0.88 with nothing passing. So the ranking is the hypothesis, and the
breakout is the way the ranking is expressed. Writing the breakout without the
ranking would be writing the part already known not to work.

**One implementation, two callers (§3.6).** `stocks_in_play` is what
`strategies/opening_range_breakout.py` asks on every bar, and it is what a
nightly selector will ask once there is a host to ask it. A selector that
disagreed with the filter a strategy applies would make the research and the
live universe two different universes, which is the project's primary failure
mode restated one level up.

**The honest caveat about this universe.** 58 mega-caps produce almost no
genuine stocks in play: these are the most consistently traded names on either
exchange, so their relative volume rarely reaches what a 7,000-name scan finds
every morning. `selection_floor` will therefore reject most sessions outright,
and that is the correct behaviour rather than a threshold to lower — a ranking
of 58 always has a top five, and calling them "in play" because they are the top
five is how the published edge gets quoted for a universe that cannot produce it.
"""

from __future__ import annotations

from typing import Final

from neurotrade.core.types import Symbol
from neurotrade.features.cross_section import CrossSection, InstrumentSnapshot

__all__ = [
    "DEFAULT_SELECTION_FLOOR",
    "DEFAULT_SELECTION_SIZE",
    "stocks_in_play",
]

DEFAULT_SELECTION_SIZE: Final = 5
"""How many names a session's selection holds at most.

Five of 58 is the top nine percent, which is roughly where an RVOL scan of a
few thousand names cuts. Held fixed rather than as a fraction of the universe so
that adding instruments does not silently widen what "in play" means."""

DEFAULT_SELECTION_FLOOR: Final = 2.0
"""Times normal volume a name must trade before it can be selected at all.

The floor is what stops the ranking from always returning something. Without it
the top five of a quiet morning are "in play" by definition, and the strategy
measures the universe's alphabetical ordering rather than unusual activity.
Two because that is the lower bound practitioner RVOL scans are set at; names
above it on a 58-name mega-cap universe are genuinely rare."""


def stocks_in_play(
    section: CrossSection,
    *,
    size: int = DEFAULT_SELECTION_SIZE,
    floor: float = DEFAULT_SELECTION_FLOOR,
) -> tuple[Symbol, ...]:
    """The instruments trading unusually heavily right now, heaviest first.

    Args:
        section: The universe as of the last closed tick.
        size: Maximum names returned. Fewer come back when fewer clear `floor`,
            which is the common case on this universe.
        floor: Minimum relative volume. A name below it is not selected however
            high it ranks — a ranking always has a top, and "top of a quiet
            morning" is not the hypothesis.

    Returns:
        Up to `size` symbols, ordered by relative volume descending with ties
        broken on the symbol. Empty when nothing clears the floor, which is a
        finding about the session rather than a failure.

    Raises:
        ValueError: If `size` is below one, or `floor` is not positive. A floor
            of zero admits everything and would make the ranking the whole rule.

    Example:
        >>> from neurotrade.core.types import Symbol
        >>> MSFT = Symbol("MSFT", AAPL.venue)
        >>> section = CrossSection(as_of=0, rows={
        ...     AAPL: InstrumentSnapshot(AAPL, 0.01, None, 3.4, None),
        ...     MSFT: InstrumentSnapshot(MSFT, 0.01, None, 1.1, None),
        ... })
        >>> [symbol.ticker for symbol in stocks_in_play(section)]
        ['AAPL']
    """
    if size < 1:
        raise ValueError(f"size {size} must be at least 1")
    if floor <= 0:
        raise ValueError(f"floor {floor} must be positive")

    def heaviness(row: InstrumentSnapshot) -> float | None:
        # `None` for a cold instrument and for one under the floor, so both are
        # excluded by the same mechanism `rank_by` already applies — rather than
        # ranked and then sliced off, which would let a cold name displace a
        # warm one out of the top `size`.
        if row.relative_volume is None or row.relative_volume < floor:
            return None
        return row.relative_volume

    return section.rank_by(heaviness)[:size]
