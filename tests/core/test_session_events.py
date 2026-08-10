"""Tests for venue-scoped session events and per-instrument halts.

The scoping distinction is the point: a session boundary is a fact about a
venue, a halt is a fact about one instrument, and conflating them produces
either thousands of duplicate events or a halt that stops the whole market.
"""

from __future__ import annotations

import pytest

from neurotrade.core.events import (
    Bar,
    BarInterval,
    Event,
    HaltReason,
    MarketEvent,
    MarketSession,
    SessionBoundary,
    TradingHalt,
    TradingResumed,
    VenueEvent,
)
from neurotrade.core.types import Price, Quantity, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
OPEN_NS = 1_773_495_000_000_000_000  # 2026-03-14 13:30:00 UTC


def make_boundary(**overrides: object) -> SessionBoundary:
    defaults: dict[str, object] = {
        "venue": Venue.NASDAQ,
        "ts_event": OPEN_NS,
        "ts_init": OPEN_NS,
        "session": MarketSession.REGULAR,
    }
    return SessionBoundary(**{**defaults, **overrides})  # type: ignore[arg-type]


# ── Scoping ──────────────────────────────────────────────────


def test_session_boundary_is_venue_scoped_not_symbol_scoped() -> None:
    """One boundary covers every instrument on the venue."""
    boundary = make_boundary()
    assert boundary.venue is Venue.NASDAQ
    assert not hasattr(boundary, "symbol")


def test_halt_is_symbol_scoped() -> None:
    """A LULD halt stops one instrument, not the market."""
    halt = TradingHalt(symbol=AAPL, ts_event=OPEN_NS, ts_init=OPEN_NS)
    assert halt.symbol == AAPL
    assert not hasattr(halt, "venue")


def test_both_scopes_share_the_ordering_contract() -> None:
    """Venue and market events interleave in one replay stream."""
    boundary = make_boundary(ts_event=OPEN_NS, seq=0)
    halt = TradingHalt(symbol=AAPL, ts_event=OPEN_NS, ts_init=OPEN_NS, seq=1)
    assert isinstance(boundary, Event)
    assert isinstance(halt, Event)
    assert sorted([halt, boundary], key=lambda e: e.sort_key) == [boundary, halt]


def test_class_hierarchy() -> None:
    assert issubclass(MarketEvent, Event)
    assert issubclass(VenueEvent, Event)
    assert not issubclass(VenueEvent, MarketEvent)


def test_venue_events_inherit_base_validation() -> None:
    with pytest.raises(ValueError, match="seq"):
        make_boundary(seq=-1)


# ── Sessions ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("session", "tradable"),
    [
        (MarketSession.PRE, True),
        (MarketSession.REGULAR, True),
        (MarketSession.POST, True),
        (MarketSession.CLOSED, False),
    ],
)
def test_session_tradability(session: MarketSession, tradable: bool) -> None:
    assert session.is_tradable is tradable


def test_session_field_is_the_phase_being_entered() -> None:
    assert make_boundary(session=MarketSession.REGULAR).opens_regular_trading
    assert not make_boundary(session=MarketSession.POST).opens_regular_trading


def test_a_trading_day_is_a_sequence_of_boundaries() -> None:
    """Reference levels reset on these, so the sequence must be replayable."""
    phases = [
        MarketSession.PRE,
        MarketSession.REGULAR,
        MarketSession.POST,
        MarketSession.CLOSED,
    ]
    day = [
        make_boundary(session=phase, ts_event=OPEN_NS + i, seq=i) for i, phase in enumerate(phases)
    ]
    assert [b.session for b in sorted(day, key=lambda e: e.sort_key)] == phases


def test_midday_lull_is_not_a_session() -> None:
    """It is a regime (§5.7) that happens inside REGULAR, not a venue phase."""
    assert "LULL" not in {s.name for s in MarketSession}


# ── Halts ────────────────────────────────────────────────────


def test_halt_reason_defaults_to_unknown() -> None:
    """Feeds do not always say why; UNKNOWN is honest, LULD would be a guess."""
    halt = TradingHalt(symbol=AAPL, ts_event=OPEN_NS, ts_init=OPEN_NS)
    assert halt.reason is HaltReason.UNKNOWN


def test_halt_reason_is_recorded_when_known() -> None:
    halt = TradingHalt(symbol=AAPL, ts_event=OPEN_NS, ts_init=OPEN_NS, reason=HaltReason.LULD)
    assert halt.reason is HaltReason.LULD


def test_resumption_auction_price_is_optional() -> None:
    """Not every venue publishes a reopening auction print."""
    assert TradingResumed(symbol=AAPL, ts_event=OPEN_NS, ts_init=OPEN_NS).auction_price is None


def test_resumption_carries_the_auction_price_when_published() -> None:
    resumed = TradingResumed(
        symbol=AAPL,
        ts_event=OPEN_NS,
        ts_init=OPEN_NS,
        auction_price=Price("187.40"),
    )
    assert resumed.auction_price == Price("187.40")


def test_halt_and_resumption_bracket_a_gap_in_the_bar_stream() -> None:
    """The shape the data-audit gate looks for: no bars between the two."""
    halt = TradingHalt(symbol=AAPL, ts_event=OPEN_NS, ts_init=OPEN_NS, seq=0)
    resumed = TradingResumed(
        symbol=AAPL,
        ts_event=OPEN_NS + 600_000_000_000,
        ts_init=OPEN_NS + 600_000_000_000,
        seq=1,
    )
    assert resumed.ts_event > halt.ts_event
    assert (resumed.ts_event - halt.ts_event) == 600_000_000_000  # 10 minutes


def test_halts_are_immutable() -> None:
    halt = TradingHalt(symbol=AAPL, ts_event=OPEN_NS, ts_init=OPEN_NS)
    with pytest.raises(AttributeError):
        halt.reason = HaltReason.LULD  # type: ignore[misc]


def test_events_of_every_scope_interleave_deterministically() -> None:
    """A replay stream mixes bars, halts and session boundaries."""
    stream: list[Event] = [
        make_boundary(session=MarketSession.REGULAR, ts_event=OPEN_NS, seq=0),
        Bar(
            symbol=AAPL,
            ts_event=OPEN_NS + 60_000_000_000,
            ts_init=OPEN_NS + 60_000_000_000,
            seq=1,
            interval=BarInterval.MIN_1,
            open=Price("100"),
            high=Price("101"),
            low=Price("99"),
            close=Price("100.5"),
            volume=Quantity(1_000),
        ),
        TradingHalt(
            symbol=AAPL,
            ts_event=OPEN_NS + 120_000_000_000,
            ts_init=OPEN_NS + 120_000_000_000,
            seq=2,
        ),
    ]
    assert [e.seq for e in sorted(stream, key=lambda e: e.sort_key)] == [0, 1, 2]
