"""The trade journal on disk: round-trip, exactness, and refusing bad rows."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from neurotrade.adapters.storage.trade_journal import CorruptTradeJournal, TradeJournalStore
from neurotrade.core.ports import TradeJournalPort
from neurotrade.core.trades import TradeRecord
from neurotrade.core.types import Side


def record(**overrides: object) -> TradeRecord:
    fields: dict[str, object] = {
        "run_id": "run_a",
        "config_hash": "cfg_a",
        "recorded_ns": 5,
        "symbol": "AAPL.NASDAQ",
        "strategy": "gap_continuation",
        "strategy_version": "1.0.0",
        "variant": "gap>=1.5 ranges",
        "side": Side.BUY,
        "entry_ns": 1,
        "exit_ns": 4,
        "bars_held": 3,
        "entry": Decimal("336.33"),
        "exit": Decimal("342.12"),
        "profit_target": Decimal("0.02"),
        "stop_loss": Decimal("0.01"),
        "max_bars": 30,
        "label": 1,
        "ambiguous": False,
        "realised_return": Decimal("0.01694"),
        "gross_return": Decimal("0.01722"),
        "mfe": Decimal("0.02531"),
        "mae": Decimal("0.00412"),
        "spread_fraction": Decimal("0.0005"),
        "commission_per_share": Decimal("0.005"),
    }
    fields.update(overrides)
    return TradeRecord(**fields)  # type: ignore[arg-type]


# ── the port ────────────────────────────────────────────────────────────


def test_the_store_satisfies_the_port(tmp_path: Path) -> None:
    assert isinstance(TradeJournalStore(tmp_path / "trades.jsonl"), TradeJournalPort)


# ── round-trip ──────────────────────────────────────────────────────────


def test_a_record_round_trips_unchanged(tmp_path: Path) -> None:
    store = TradeJournalStore(tmp_path / "trades.jsonl")
    original = record()
    store.append(original)
    assert store.records() == (original,)


def test_decimals_survive_as_decimals_not_floats(tmp_path: Path) -> None:
    """A price written as a JSON float comes back as 336.32999999999998.

    Every one of these fields is eventually compared or summed, and `Decimal`
    is the project's price type precisely so that comparison is exact.
    """
    store = TradeJournalStore(tmp_path / "trades.jsonl")
    store.append(record(entry=Decimal("336.33"), mfe=Decimal("0.00001")))
    back = store.records()[0]
    assert back.entry == Decimal("336.33")
    assert back.mfe == Decimal("0.00001")
    assert isinstance(back.entry, Decimal)


def test_rows_come_back_in_the_order_they_were_appended(tmp_path: Path) -> None:
    store = TradeJournalStore(tmp_path / "trades.jsonl")
    for n in range(4):
        store.append(record(entry_ns=n, variant=f"v{n}"))
    assert [row.variant for row in store.records()] == ["v0", "v1", "v2", "v3"]


def test_a_short_keeps_its_side(tmp_path: Path) -> None:
    store = TradeJournalStore(tmp_path / "trades.jsonl")
    store.append(record(side=Side.SELL))
    assert store.records()[0].side is Side.SELL


def test_each_append_is_durable_on_its_own(tmp_path: Path) -> None:
    """The reason for JSONL over a batched columnar write: a run killed at hour
    four must leave a complete journal of the first four hours."""
    path = tmp_path / "trades.jsonl"
    store = TradeJournalStore(path)
    store.append(record(variant="first"))
    # No close, no flush, no finalise — a different reader sees it immediately.
    assert [row.variant for row in TradeJournalStore(path).records()] == ["first"]


# ── rejection ───────────────────────────────────────────────────────────


def test_an_absent_journal_reads_as_empty(tmp_path: Path) -> None:
    """A run that took no trades is an ordinary outcome, not an error."""
    assert TradeJournalStore(tmp_path / "never-written.jsonl").records() == ()


def test_the_parent_directory_is_created(tmp_path: Path) -> None:
    store = TradeJournalStore(tmp_path / "deep" / "nested" / "trades.jsonl")
    store.append(record())
    assert len(store.records()) == 1


@pytest.mark.parametrize(
    "line",
    [
        "not json at all",
        '{"run_id": "run_a"}',  # every other field missing
        '{"run_id":"r","config_hash":"c","recorded_ns":1,"symbol":"S","strategy":"s",'
        '"strategy_version":"1","variant":"v","side":"NOT_A_SIDE","entry_ns":1,"exit_ns":2,'
        '"bars_held":1,"max_bars":5,"label":1,"ambiguous":false,"entry":"1","exit":"1",'
        '"profit_target":"1","stop_loss":"1","realised_return":"0","gross_return":"0",'
        '"mfe":"0","mae":"0","spread_fraction":"0","commission_per_share":"0"}',
        '{"run_id":"r","config_hash":"c","recorded_ns":1,"symbol":"S","strategy":"s",'
        '"strategy_version":"1","variant":"v","side":"BUY","entry_ns":1,"exit_ns":2,'
        '"bars_held":1,"max_bars":5,"label":1,"ambiguous":false,"entry":"not-a-decimal",'
        '"exit":"1","profit_target":"1","stop_loss":"1","realised_return":"0",'
        '"gross_return":"0","mfe":"0","mae":"0","spread_fraction":"0",'
        '"commission_per_share":"0"}',
    ],
    ids=["garbage", "missing-fields", "bad-side", "bad-decimal"],
)
def test_an_unreadable_row_is_fatal_and_names_its_line(tmp_path: Path, line: str) -> None:
    """Skipping a bad row would give a capture ratio computed on an unknown
    subset — a number that looks plausible while being wrong."""
    path = tmp_path / "trades.jsonl"
    store = TradeJournalStore(path)
    store.append(record())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with pytest.raises(CorruptTradeJournal, match=r"trades\.jsonl:2"):
        store.records()


def test_blank_lines_are_not_an_error(tmp_path: Path) -> None:
    """A trailing newline from an editor is not corruption."""
    path = tmp_path / "trades.jsonl"
    store = TradeJournalStore(path)
    store.append(record())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert len(store.records()) == 1
