"""Point-in-time sector membership.

The dated shape is the whole design, so the tests are about time: a membership
that ended must not answer for a later session, two memberships for one name
must never overlap, and a benchmark must have none at all. A map that got any of
those wrong would residualise a name against the wrong sector and report the
sector's move as the name's alpha — which no test downstream could detect.
"""

from __future__ import annotations

from datetime import date

import pytest

from neurotrade.core.sectors import UNKNOWN_SINCE, SectorAssignment, SectorMap
from neurotrade.core.types import Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
V = Symbol("V", Venue.NYSE)
RY = Symbol("RY", Venue.TSX)
XLK = Symbol("XLK", Venue.ARCA)
XLF = Symbol("XLF", Venue.ARCA)
SPY = Symbol("SPY", Venue.ARCA)
XIU = Symbol("XIU", Venue.TSX)

SWITCH = date(2023, 3, 17)
BEFORE = date(2023, 3, 16)
LATER = date(2024, 7, 8)


def always(symbol: Symbol, sector: Symbol) -> SectorAssignment:
    return SectorAssignment(symbol=symbol, sector=sector, effective=UNKNOWN_SINCE, until=None)


def reclassified() -> tuple[SectorAssignment, SectorAssignment]:
    """The GICS 2023 move of the payment networks, as two half-open spans."""
    return (
        SectorAssignment(symbol=V, sector=XLK, effective=UNKNOWN_SINCE, until=SWITCH),
        SectorAssignment(symbol=V, sector=XLF, effective=SWITCH, until=None),
    )


# ── One assignment ──────────────────────────────────────────────────────


def test_a_span_is_half_open_so_two_spans_can_meet_without_overlapping() -> None:
    was, now = reclassified()
    assert was.holds(BEFORE) and not was.holds(SWITCH)
    assert now.holds(SWITCH) and not now.holds(BEFORE)


def test_an_inverted_span_is_refused() -> None:
    with pytest.raises(ValueError, match="is not after effective"):
        SectorAssignment(symbol=V, sector=XLK, effective=SWITCH, until=BEFORE)


def test_an_empty_span_is_refused() -> None:
    with pytest.raises(ValueError, match="is not after effective"):
        SectorAssignment(symbol=V, sector=XLK, effective=SWITCH, until=SWITCH)


def test_an_instrument_cannot_be_its_own_sector() -> None:
    """A sector ETF regressed on itself has a residual of exactly zero."""
    with pytest.raises(ValueError, match="cannot be its own sector"):
        always(XLK, XLK)


def test_an_unknown_start_is_visible_on_the_assignment() -> None:
    assert not always(AAPL, XLK).start_is_known
    assert SectorAssignment(V, XLF, SWITCH, None).start_is_known


# ── The map, through time ───────────────────────────────────────────────


def test_a_reclassification_answers_differently_on_each_side_of_its_date() -> None:
    """The case that justifies the dated shape: GICS 2023 moved V out of tech."""
    sectors = SectorMap(reclassified(), {Venue.NYSE: SPY})
    assert sectors.sector_of(V, on=BEFORE) == XLK
    assert sectors.sector_of(V, on=LATER) == XLF


def test_two_memberships_that_overlap_are_refused_at_construction() -> None:
    """A silent overlap would make `sector_of` depend on iteration order."""
    with pytest.raises(ValueError, match="is in two sectors at once"):
        SectorMap(
            [
                SectorAssignment(V, XLK, UNKNOWN_SINCE, date(2023, 4, 1)),
                SectorAssignment(V, XLF, SWITCH, None),
            ],
            {},
        )


def test_an_open_ended_span_followed_by_another_is_an_overlap() -> None:
    """`until=None` means forever, so nothing may start after it."""
    with pytest.raises(ValueError, match="is in two sectors at once"):
        SectorMap([always(V, XLK), SectorAssignment(V, XLF, SWITCH, None)], {})


def test_an_instrument_with_no_membership_has_no_sector() -> None:
    """A benchmark, rather than a name we forgot."""
    assert SectorMap([always(AAPL, XLK)], {}).sector_of(SPY, on=LATER) is None


def test_a_membership_that_has_not_started_yet_does_not_answer() -> None:
    sectors = SectorMap([SectorAssignment(V, XLF, SWITCH, None)], {})
    assert sectors.sector_of(V, on=BEFORE) is None


# ── Members, sectors, and the market leg ────────────────────────────────


def test_members_are_point_in_time_too() -> None:
    sectors = SectorMap([always(AAPL, XLK), *reclassified()], {})
    assert sectors.members(XLK, on=BEFORE) == (AAPL, V)
    assert sectors.members(XLK, on=LATER) == (AAPL,)
    assert sectors.members(XLF, on=LATER) == (V,)


def test_sectors_lists_only_those_with_a_member_that_day() -> None:
    sectors = SectorMap(reclassified(), {})
    assert sectors.sectors(on=BEFORE) == (XLK,)
    assert sectors.sectors(on=LATER) == (XLF,)


def test_the_market_leg_is_per_venue_not_per_instrument() -> None:
    """SPY does not price TSX listings, so the Canadian names regress on XIU."""
    sectors = SectorMap(
        [always(AAPL, XLK), always(RY, XLF)],
        {Venue.NASDAQ: SPY, Venue.TSX: XIU},
    )
    assert sectors.market_of(AAPL) == SPY
    assert sectors.market_of(RY) == XIU


def test_a_venue_with_no_market_leg_gets_none_not_a_substitute() -> None:
    """A Canadian name against a US index carries the whole country move as alpha."""
    sectors = SectorMap([always(RY, XLF)], {Venue.NASDAQ: SPY})
    assert sectors.market_of(RY) is None


# ── Determinism and the recorded bias ───────────────────────────────────


def test_the_digest_does_not_depend_on_the_order_given() -> None:
    """A universe digest identifies what was crawled, not how the file was typed."""
    rows = [always(AAPL, XLK), always(MSFT, XLK), *reclassified()]
    assert (
        SectorMap(rows, {Venue.NYSE: SPY}).digest
        == SectorMap(list(reversed(rows)), {Venue.NYSE: SPY}).digest
    )


def test_the_digest_moves_when_a_market_leg_changes() -> None:
    rows = [always(AAPL, XLK)]
    assert (
        SectorMap(rows, {Venue.NASDAQ: SPY}).digest != SectorMap(rows, {Venue.NASDAQ: XIU}).digest
    )


def test_the_map_counts_how_many_assignments_are_backdated_guesses() -> None:
    """The size of the survivorship-flavoured bias, as a number a report can print."""
    sectors = SectorMap([always(AAPL, XLK), *reclassified()], {})
    # AAPL and the pre-2023 leg of V are both `UNKNOWN_SINCE`; the post leg is dated.
    assert sectors.unknown_starts == 2
    assert len(sectors) == 3
    assert sectors.classified == (AAPL, V)
