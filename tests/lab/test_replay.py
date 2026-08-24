"""Tests for the replay engine — gate G1.

The gate is one sentence: replay a session and get the identical result. Most of
what follows tries to break that, because a determinism guarantee is only worth
what its counterexamples are.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from neurotrade.adapters.storage.event_store import EventStore
from neurotrade.bus import EventBus, HandlerFailed
from neurotrade.core.clock import SimClock
from neurotrade.core.events import (
    Bar,
    BarInterval,
    Event,
    MarketSession,
    Quote,
    SessionBoundary,
    TradingHalt,
)
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.lab.replay import ReplayEngine, ReplayResult, RunDigest

AAPL = Symbol("AAPL", Venue.NASDAQ)
MINUTE = 60_000_000_000
OPEN_NS = 1_773_495_000_000_000_000
FOREVER = OPEN_NS + 10_000 * MINUTE


def bar(index: int, *, seq: int = 0, close: str = "100") -> Bar:
    ts = OPEN_NS + index * MINUTE
    return Bar(
        symbol=AAPL,
        ts_event=ts,
        ts_init=ts,
        seq=seq,
        interval=BarInterval.MIN_1,
        open=Price(close),
        high=Price(close),
        low=Price(close),
        close=Price(close),
        volume=Quantity(1_000),
    )


class ListLog:
    """An in-memory `EventStorePort`. Replay needs no file to be exercised."""

    def __init__(self, events: Sequence[Event], *, sort: bool = True) -> None:
        self._events = sorted(events, key=lambda e: e.sort_key) if sort else list(events)

    def append(self, event: Event) -> None:
        self._events.append(event)

    def stream(self, start: int, end: int) -> Iterator[Event]:
        return iter([e for e in self._events if start <= e.ts_event < end])


def a_session() -> list[Event]:
    """A small but heterogeneous session: a boundary, bars, and a halt."""
    events: list[Event] = [
        SessionBoundary(
            venue=Venue.NASDAQ,
            ts_event=OPEN_NS,
            ts_init=OPEN_NS,
            session=MarketSession.REGULAR,
        )
    ]
    events += [bar(i, close=str(100 + i)) for i in range(1, 30)]
    events.append(
        TradingHalt(symbol=AAPL, ts_event=OPEN_NS + 30 * MINUTE, ts_init=OPEN_NS + 30 * MINUTE)
    )
    return events


def replay(events: list[Event]) -> ReplayResult:
    return ReplayEngine(ListLog(events), SimClock(0)).run(0, FOREVER)


# ── Gate G1 ──────────────────────────────────────────────────


def test_two_replays_of_one_session_agree() -> None:
    """The gate itself."""
    session = a_session()
    assert replay(session).digest == replay(session).digest


def test_the_digest_is_stable_across_many_runs() -> None:
    session = a_session()
    assert len({replay(session).digest for _ in range(10)}) == 1


def test_replaying_from_a_real_log_file_reproduces(tmp_path: Path) -> None:
    """End to end: record to disk, then replay it twice."""
    log = tmp_path / "session.jsonl"
    with EventStore(log) as store:
        for event in a_session():
            store.append(event)

    def run() -> ReplayResult:
        return ReplayEngine(EventStore(log), SimClock(0)).run(0, FOREVER)

    first, second = run(), run()
    assert first.digest == second.digest
    assert first.events_read == len(a_session())


def test_input_order_does_not_change_the_digest() -> None:
    """The store sorts, so a log written out of order still replays identically."""
    session = a_session()
    assert replay(session).digest == replay(list(reversed(session))).digest


# ── The digest must actually discriminate ────────────────────


def test_a_changed_price_changes_the_digest() -> None:
    """A digest that ignored content would pass every test above and be useless."""
    session = a_session()
    altered = [*session[:5], bar(5, close="999"), *session[6:]]
    assert replay(session).digest != replay(altered).digest


def test_a_changed_order_changes_the_digest() -> None:
    """Same events, different sequence — a different session."""
    ordered = [bar(0, seq=0), bar(0, seq=1)]
    swapped = [bar(0, seq=1), bar(0, seq=0)]
    unsorted = ReplayEngine(ListLog(ordered, sort=False), SimClock(0)).run(0, FOREVER)
    reversed_run = ReplayEngine(ListLog(swapped, sort=False), SimClock(0)).run(0, FOREVER)
    assert unsorted.digest != reversed_run.digest


def test_a_missing_event_changes_the_digest() -> None:
    session = a_session()
    assert replay(session).digest != replay(session[:-1]).digest


def test_an_extra_event_changes_the_digest() -> None:
    session = a_session()
    assert replay(session).digest != replay([*session, bar(99)]).digest


def test_different_sessions_differ() -> None:
    assert replay([bar(0)]).digest != replay([bar(1)]).digest


# ── Outputs are digested, not just inputs ────────────────────


def test_a_reacting_subscriber_changes_the_digest() -> None:
    """The property that makes this a behaviour digest rather than a data checksum.

    Once strategies exist, a strategy that starts deciding differently changes
    the digest even though the input data is untouched.
    """
    session = a_session()
    quiet = ReplayEngine(ListLog(session), SimClock(0))
    quiet_digest = quiet.run(0, FOREVER).digest

    noisy = ReplayEngine(ListLog(session), SimClock(0))
    reactions: list[Event] = []

    def react(event: Event) -> None:
        if isinstance(event, Bar) and not reactions:
            extra = bar(500)
            reactions.append(extra)
            noisy.bus.publish(extra)

    noisy.bus.subscribe(Bar, react)
    assert noisy.run(0, FOREVER).digest != quiet_digest


def test_dispatched_counts_reactions_but_read_does_not() -> None:
    engine = ReplayEngine(ListLog([bar(0)]), SimClock(0))
    published: list[Event] = []

    def react(event: Event) -> None:
        if not published:
            published.append(event)
            engine.bus.publish(bar(900))

    engine.bus.subscribe(Bar, react)
    result = engine.run(0, FOREVER)
    assert result.events_read == 1
    assert result.events_dispatched == 2


# ── The clock leads each event ───────────────────────────────


def test_handlers_see_the_moment_being_modelled() -> None:
    """Not the moment the replay happens to be running."""
    clock = SimClock(0)
    engine = ReplayEngine(ListLog([bar(0), bar(1), bar(2)]), clock)
    seen: list[tuple[int, int]] = []
    engine.bus.subscribe(Bar, lambda event: seen.append((clock.now_ns(), event.ts_event)))
    engine.run(0, FOREVER)
    assert all(now == ts for now, ts in seen)


def test_the_clock_ends_at_the_last_event() -> None:
    clock = SimClock(0)
    ReplayEngine(ListLog([bar(0), bar(5)]), clock).run(0, FOREVER)
    assert clock.now_ns() == OPEN_NS + 5 * MINUTE


def test_an_out_of_order_store_fails_loudly() -> None:
    """A mis-ordered source must not quietly produce a different session."""
    engine = ReplayEngine(ListLog([bar(5), bar(0)], sort=False), SimClock(0))
    with pytest.raises(ValueError, match="backwards"):
        engine.run(0, FOREVER)


# ── Result reporting ─────────────────────────────────────────


def test_result_reports_the_span() -> None:
    result = replay([bar(0), bar(10)])
    assert result.first_ts == OPEN_NS
    assert result.last_ts == OPEN_NS + 10 * MINUTE
    assert result.span_ns == 10 * MINUTE
    assert not result.is_empty


def test_an_empty_replay_is_not_an_error() -> None:
    """A session that produced nothing is an ordinary outcome."""
    result = replay([])
    assert result.is_empty
    assert (result.first_ts, result.last_ts, result.span_ns) == (None, None, 0)


def test_the_range_is_half_open() -> None:
    result = ReplayEngine(ListLog([bar(i) for i in range(10)]), SimClock(0)).run(
        OPEN_NS + 2 * MINUTE, OPEN_NS + 5 * MINUTE
    )
    assert result.events_read == 3


def test_an_empty_replay_still_has_a_digest() -> None:
    """Comparable against another empty run, which is the useful property."""
    assert replay([]).digest == replay([]).digest
    assert len(replay([]).digest) == 32


def test_no_timing_is_recorded_in_the_result() -> None:
    """Wall-clock duration varies for reasons unrelated to behaviour; a digest
    including it would never match."""
    from dataclasses import fields

    names = {field.name for field in fields(ReplayResult)}
    assert not names & {"duration", "elapsed", "started_at", "wall_time"}


# ── Failure handling ─────────────────────────────────────────


def test_a_failing_subscriber_stops_the_replay() -> None:
    """A partial dispatch would report a digest for something that never
    fully happened."""
    engine = ReplayEngine(ListLog([bar(0), bar(1)]), SimClock(0))

    def explode(_event: Event) -> None:
        raise ValueError("strategy blew up")

    engine.bus.subscribe(Bar, explode)
    with pytest.raises(HandlerFailed):
        engine.run(0, FOREVER)


# ── RunDigest on its own ─────────────────────────────────────


def test_the_digest_is_order_sensitive() -> None:
    forwards, backwards = RunDigest(), RunDigest()
    events = [bar(0), bar(1)]
    for event in events:
        forwards(event)
    for event in reversed(events):
        backwards(event)
    assert forwards.hexdigest != backwards.hexdigest


def test_the_digest_counts_what_it_folded() -> None:
    digest = RunDigest()
    for index in range(5):
        digest(bar(index))
    assert digest.count == 5


def test_an_empty_digest_is_not_the_hash_of_nothing() -> None:
    """It is seeded with the codec version, so digests from different encodings
    are never mistaken for each other."""
    import hashlib

    assert RunDigest().hexdigest != hashlib.blake2b(digest_size=16).hexdigest()


def test_a_shared_bus_is_not_reused_between_engines() -> None:
    """Two engines on one bus would digest each other's events."""
    shared = EventBus()
    first = ReplayEngine(ListLog([bar(0)]), SimClock(0), bus=shared)
    second = ReplayEngine(ListLog([bar(0)]), SimClock(0), bus=shared)
    # Both digests subscribed to the same bus; the second run feeds both.
    second.run(0, FOREVER)
    assert first.run(0, FOREVER).events_dispatched > 1


# ── The committed fixture ────────────────────────────────────
#
# Everything above builds events in Python. These replay a real file, so they
# exercise decode as well as dispatch — and the file stays fixed while the code
# around it changes, which is what makes a digest regression visible.

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "session.jsonl"


def replay_fixture() -> ReplayResult:
    return ReplayEngine(EventStore(FIXTURE), SimClock(0)).run(0, 2**63 - 1)


def test_the_fixture_replays_deterministically() -> None:
    """Gate G1 against a recorded session on disk."""
    assert replay_fixture().digest == replay_fixture().digest


def test_the_fixture_digest_is_pinned() -> None:
    """A change in encoding, ordering or event content moves this value.

    Pinned deliberately. If a refactor changes it, that is the question worth
    asking — either behaviour changed, or the fixture needs regenerating with
    `scripts/make_replay_fixture.py` and the new digest recorded here.
    """
    assert replay_fixture().digest == "4f58fe2c99cd26dc7cbb8faf033a39d1"


def test_the_fixture_holds_a_full_session() -> None:
    result = replay_fixture()
    assert result.events_read == 357
    assert result.span_ns == 720 * 60_000_000_000  # pre-market through close


def test_the_fixture_exercises_what_synthetic_data_does_not() -> None:
    """Guards the fixture's own value.

    A regenerated fixture that lost these properties would still replay
    deterministically and would stop testing anything interesting.
    """
    events = list(EventStore(FIXTURE).stream(0, 2**63 - 1))
    bars = [event for event in events if isinstance(event, Bar)]

    assert {type(event).__name__ for event in events} >= {
        "Bar",
        "Quote",
        "TickTrade",
        "SessionBoundary",
        "TradingHalt",
        "TradingResumed",
    }
    # Real intrabar range on every bar, not a single-point ramp.
    assert all(bar.high > bar.low for bar in bars)
    # Optional fields present on some bars and absent on others.
    assert any(bar.vwap is not None for bar in bars)
    assert any(bar.vwap is None for bar in bars)
    assert any(bar.trade_count is None for bar in bars)
    # A ticker containing a dot, and an instrument settling in CAD.
    tickers = {bar.symbol.ticker for bar in bars}
    assert "BRK.B" in tickers
    assert any(bar.symbol.currency.value == "CAD" for bar in bars)
    # An illiquid minute really does have no volume.
    assert any(bar.volume.is_zero for bar in bars)


def test_the_fixture_contains_locked_and_crossed_books() -> None:
    """Both occur in real consolidated data and must survive a round trip."""
    quotes = [
        event for event in EventStore(FIXTURE).stream(0, 2**63 - 1) if isinstance(event, Quote)
    ]
    assert any(quote.is_locked for quote in quotes)
    assert any(quote.is_crossed for quote in quotes)
