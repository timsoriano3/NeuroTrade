"""Regenerate the committed replay fixture.

    uv run python scripts/make_replay_fixture.py

The fixture is the artifact; this script exists so it can be rebuilt when the
codec version changes, and so the properties it deliberately contains are
written down rather than inferred from the file.

**Why not generate it in the test.** A fixture built at test time proves the
generator and the replay agree, which is a weaker claim than it sounds — both
run in the same process against the same objects. A committed file has to be
*parsed*, so it exercises decode, and it stays fixed while the code around it
changes, which is what makes a digest regression visible.

**What this deliberately contains**, because a synthetic ramp has none of it:

- real intrabar range: open, high, low and close all differ
- `vwap` and `trade_count` populated on some bars and absent on others
- irregular spacing — a halt leaves a 25-minute hole
- three instruments, including `BRK.B` (a ticker containing a dot) and a
  Canadian listing that settles in CAD
- sub-penny prices, which US venues do quote
- a zero-volume minute, which illiquid names genuinely produce
- quotes that are locked and crossed, which real consolidated feeds contain
- trade prints with and without an inferred aggressor
- every session phase, and a halt with a reopening auction price

Randomness is seeded, so re-running produces the same file.
"""

from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path

from neurotrade.adapters.storage.event_store import EventStore
from neurotrade.core.events import (
    Bar,
    BarInterval,
    Event,
    HaltReason,
    MarketSession,
    Quote,
    SessionBoundary,
    TickTrade,
    TradingHalt,
    TradingResumed,
)
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "session.jsonl"

SEED = 20260316
MINUTE = 60_000_000_000
OPEN_NS = 1_773_495_000_000_000_000  # 2026-03-16 13:30 UTC — a US open

AAPL = Symbol("AAPL", Venue.NASDAQ)
BRK_B = Symbol("BRK.B", Venue.NYSE)  # the ticker-with-a-dot case
SHOP = Symbol("SHOP", Venue.TSX)  # settles CAD

HALT_FROM, HALT_UNTIL = 60, 85  # minutes into the session


def _walk(rng: random.Random, previous: Decimal) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """One bar of a random walk, with a real high-low range.

    Returns open, high, low and close such that the domain model accepts them:
    low <= open, close <= high.
    """
    open_ = previous
    close = open_ + Decimal(str(round(rng.uniform(-0.35, 0.35), 4)))
    if close <= 0:
        close = open_
    high = max(open_, close) + Decimal(str(round(rng.uniform(0, 0.22), 4)))
    low = min(open_, close) - Decimal(str(round(rng.uniform(0, 0.22), 4)))
    return open_, high, max(low, Decimal("0.0001")), close


def _session(rng: random.Random) -> list[Event]:
    events: list[Event] = []
    seq = 0

    def stamp(minute: int) -> int:
        return OPEN_NS + minute * MINUTE

    # Pre-market opens 90 minutes early; regular trading at minute 0.
    events.append(
        SessionBoundary(
            venue=Venue.NASDAQ, ts_event=stamp(-90), ts_init=stamp(-90), session=MarketSession.PRE
        )
    )
    events.append(
        SessionBoundary(
            venue=Venue.NASDAQ, ts_event=stamp(0), ts_init=stamp(0), session=MarketSession.REGULAR
        )
    )

    prices = {AAPL: Decimal("187.42"), BRK_B: Decimal("412.85"), SHOP: Decimal("94.31")}

    for minute in range(1, 120):
        for symbol in (AAPL, BRK_B, SHOP):
            # AAPL is halted for part of the session; the others keep trading.
            if symbol is AAPL and HALT_FROM <= minute < HALT_UNTIL:
                continue

            open_, high, low, close = _walk(rng, prices[symbol])
            prices[symbol] = close

            # One deliberately illiquid minute: no volume, and therefore no vwap.
            quiet = symbol is SHOP and minute == 47
            volume = Quantity(0) if quiet else Quantity(rng.randrange(800, 90_000))

            seq += 1
            events.append(
                Bar(
                    symbol=symbol,
                    ts_event=stamp(minute),
                    ts_init=stamp(minute) + rng.randrange(1_000, 4_000_000),
                    seq=seq,
                    interval=BarInterval.MIN_1,
                    open=Price(open_),
                    high=Price(high),
                    low=Price(low),
                    close=Price(close),
                    volume=volume,
                    # Absent on some bars: not every feed supplies them.
                    vwap=None if quiet or minute % 5 == 0 else Price((high + low) / 2),
                    trade_count=None if minute % 7 == 0 else rng.randrange(4, 900),
                )
            )

        # Quotes and prints, sparsely — enough to exercise their codecs.
        if minute % 11 == 0:
            mid = prices[AAPL]
            spread = Decimal("0.01") if minute % 22 else Decimal("0.0001")  # sub-penny
            seq += 1
            events.append(
                Quote(
                    symbol=AAPL,
                    ts_event=stamp(minute) + 30_000_000_000,
                    ts_init=stamp(minute) + 30_000_000_000,
                    seq=seq,
                    bid_price=Price(mid - spread),
                    bid_size=Quantity(rng.randrange(100, 5_000)),
                    ask_price=Price(mid + spread),
                    ask_size=Quantity(rng.randrange(100, 5_000)),
                )
            )
        if minute % 17 == 0:
            seq += 1
            events.append(
                TickTrade(
                    symbol=BRK_B,
                    ts_event=stamp(minute) + 15_000_000_000,
                    ts_init=stamp(minute) + 15_000_000_000,
                    seq=seq,
                    price=Price(prices[BRK_B]),
                    size=Quantity(rng.randrange(1, 400)),
                    # Unknown until something classifies it (§5.6).
                    aggressor=None if minute % 34 == 0 else Side.BUY,
                )
            )

    # A locked book and a crossed book — both occur in real consolidated data.
    seq += 1
    events.append(
        Quote(
            symbol=AAPL,
            ts_event=stamp(52),
            ts_init=stamp(52),
            seq=seq,
            bid_price=Price("188.20"),
            ask_price=Price("188.20"),  # locked
            bid_size=Quantity(300),
            ask_size=Quantity(300),
        )
    )
    seq += 1
    events.append(
        Quote(
            symbol=AAPL,
            ts_event=stamp(53),
            ts_init=stamp(53),
            seq=seq,
            bid_price=Price("188.31"),
            ask_price=Price("188.29"),  # crossed
            bid_size=Quantity(100),
            ask_size=Quantity(100),
        )
    )

    # The halt that produced the hole above, and its reopening auction.
    seq += 1
    events.append(
        TradingHalt(
            symbol=AAPL,
            ts_event=stamp(HALT_FROM),
            ts_init=stamp(HALT_FROM),
            seq=seq,
            reason=HaltReason.LULD,
        )
    )
    seq += 1
    events.append(
        TradingResumed(
            symbol=AAPL,
            ts_event=stamp(HALT_UNTIL),
            ts_init=stamp(HALT_UNTIL),
            seq=seq,
            auction_price=Price("189.7350"),  # sub-penny auction print
        )
    )

    for minute, phase in ((390, MarketSession.POST), (630, MarketSession.CLOSED)):
        events.append(
            SessionBoundary(
                venue=Venue.NASDAQ,
                ts_event=stamp(minute),
                ts_init=stamp(minute),
                session=phase,
            )
        )
    return events


def main() -> None:
    rng = random.Random(SEED)
    events = _session(rng)

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.unlink(missing_ok=True)  # rebuild, never append
    with EventStore(FIXTURE) as store:
        for event in sorted(events, key=lambda e: e.sort_key):
            store.append(event)

    print(f"{len(events)} events -> {FIXTURE} ({FIXTURE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
