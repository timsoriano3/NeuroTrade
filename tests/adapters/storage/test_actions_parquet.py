"""Tests for persisting corporate actions.

The round trip is the point: what comes back must be what went in, including
the distinction the schema exists to preserve — a symbol known to have no
actions is not the same as a symbol never fetched.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from neurotrade.adapters.storage.actions_parquet import SCHEMA_VERSION, ActionStore
from neurotrade.core.actions import CorporateAction
from neurotrade.core.clock import SimClock
from neurotrade.core.types import Price, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)

CLOCK = SimClock(1_700_000_000_000_000_000)
HASH = "cfg_test"


def store(tmp_path: Path) -> ActionStore:
    return ActionStore(tmp_path / "actions" / "yfinance")


def split(day: date, ratio: str, symbol: Symbol = AAPL) -> CorporateAction:
    return CorporateAction(symbol, day, split_ratio=Decimal(ratio))


# ── Round trip ───────────────────────────────────────────────────────────────


def test_actions_survive_a_round_trip(tmp_path: Path) -> None:
    written = {AAPL: (split(date(2020, 8, 31), "4"), split(date(2014, 6, 9), "7"))}
    subject = store(tmp_path)
    subject.write(written, clock=CLOCK, config_hash=HASH)
    assert subject.read()[AAPL] == tuple(sorted(written[AAPL], key=lambda a: a.effective_date))


def test_a_dividend_survives_at_full_precision(tmp_path: Path) -> None:
    action = CorporateAction(AAPL, date(2026, 8, 10), dividend=Decimal("0.27"))
    subject = store(tmp_path)
    subject.write({AAPL: (action,)}, clock=CLOCK, config_hash=HASH)
    (read_back,) = subject.read()[AAPL]
    assert read_back.dividend == Decimal("0.27")


def test_a_fractional_reverse_split_survives(tmp_path: Path) -> None:
    """`decimal128(18, 8)` must hold a ratio below one without rescaling."""
    subject = store(tmp_path)
    subject.write({AAPL: (split(date(2023, 6, 1), "0.1"),)}, clock=CLOCK, config_hash=HASH)
    (read_back,) = subject.read()[AAPL]
    assert read_back.split_ratio == Decimal("0.1")


def test_multiple_symbols_stay_separate(tmp_path: Path) -> None:
    subject = store(tmp_path)
    subject.write(
        {AAPL: (split(date(2020, 8, 31), "4"),), SHOP: (split(date(2022, 7, 4), "10", SHOP),)},
        clock=CLOCK,
        config_hash=HASH,
    )
    read_back = subject.read()
    assert read_back[AAPL][0].symbol == AAPL
    assert read_back[SHOP][0].symbol == SHOP


# ── The empty-versus-absent distinction ──────────────────────────────────────


def test_a_symbol_with_no_actions_is_recorded_as_fetched(tmp_path: Path) -> None:
    """Otherwise "never split" and "never asked" look identical."""
    subject = store(tmp_path)
    subject.write({MSFT: ()}, clock=CLOCK, config_hash=HASH)
    read_back = subject.read()
    assert MSFT in read_back
    assert read_back[MSFT] == ()


def test_a_symbol_never_written_is_absent(tmp_path: Path) -> None:
    subject = store(tmp_path)
    subject.write({AAPL: (split(date(2020, 8, 31), "4"),)}, clock=CLOCK, config_hash=HASH)
    assert MSFT not in subject.read()


# ── Rejection ────────────────────────────────────────────────────────────────


def test_reading_before_writing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no corporate actions"):
        store(tmp_path).read()


def test_an_action_filed_under_the_wrong_symbol_is_refused(tmp_path: Path) -> None:
    """The mapping key and the action's own symbol must agree."""
    with pytest.raises(ValueError, match="filed under"):
        store(tmp_path).write(
            {MSFT: (split(date(2020, 8, 31), "4", AAPL),)}, clock=CLOCK, config_hash=HASH
        )


def test_an_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    """A file from a future shape must not be read as if it were this one."""
    import pyarrow.parquet as pq

    subject = store(tmp_path)
    subject.write({AAPL: ()}, clock=CLOCK, config_hash=HASH)
    table = pq.read_table(subject.path)
    stale = table.schema.with_metadata({"schema_version": "999", "dataset": "corporate_actions"})
    pq.write_table(table.cast(stale), subject.path)

    with pytest.raises(ValueError, match="schema version 999"):
        subject.read()


def test_the_current_schema_version_is_recorded(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    subject = store(tmp_path)
    subject.write({AAPL: ()}, clock=CLOCK, config_hash=HASH)
    metadata = pq.read_table(subject.path).schema.metadata
    assert metadata[b"schema_version"].decode() == SCHEMA_VERSION
    assert metadata[b"config_hash"].decode() == HASH


# ── series() ─────────────────────────────────────────────────────────────────


def test_series_builds_from_the_stored_actions(tmp_path: Path) -> None:
    subject = store(tmp_path)
    subject.write({AAPL: (split(date(2020, 8, 31), "4"),)}, clock=CLOCK, config_hash=HASH)
    series = subject.series(AAPL)
    assert series.price_factor(date(2020, 8, 28), as_of=date(2021, 1, 4)) == Decimal("0.25")


def test_series_for_an_unknown_symbol_adjusts_nothing(tmp_path: Path) -> None:
    """Documented behaviour, and the reason the gap audit exists."""
    subject = store(tmp_path)
    subject.write({AAPL: ()}, clock=CLOCK, config_hash=HASH)
    assert len(subject.series(MSFT)) == 0


def test_series_carries_prior_closes_through_for_dividends(tmp_path: Path) -> None:
    subject = store(tmp_path)
    subject.write(
        {AAPL: (CorporateAction(AAPL, date(2023, 6, 1), dividend=Decimal("1")),)},
        clock=CLOCK,
        config_hash=HASH,
    )
    series = subject.series(AAPL, prior_close={date(2023, 6, 1): Price("100")})
    assert series.total_return_factor(date(2023, 5, 31), as_of=date(2023, 6, 30)) == Decimal("0.99")
