"""The trade journal on disk — newline-delimited JSON, one file per run.

Satisfies `TradeJournalPort`. The choices here are deliberately the trial
ledger's, for one reason it did not have to care about: a measurement run is
hours long, and the runs worth reading are often the ones that died.

**JSONL, appended per trade, not a batch written at the end.** A columnar
format would read faster, but a run that crashes at hour four would leave
nothing — which is precisely what happened to the September wave. Every
`append` opens, writes one line and closes, so a killed process leaves a
complete journal of everything up to the moment it died. DuckDB reads JSONL
directly (`read_json_auto`), so the analysis path costs nothing for this.

**One file per run, not one for the project's lifetime.** Unlike trials, trade
rows are never deflated against history, and concurrent runs sharing one file
is the append race that forced the September wave onto per-run ledgers in the
first place. A run's journal belongs to that run.

**Decimals are written as strings.** `json` would turn them into floats, and a
price that round-trips to 336.32999999999998 is a price that will eventually be
compared for equality and lose. Read back with `Decimal(str)`.

**Features are written as JSON numbers, and that is not an inconsistency.** A
derived feature is a `float` by Invariant, never a `Decimal`, and Python's JSON
encoder emits the shortest repr that round-trips — so a float survives the trip
exactly while a string would invite someone reading it back as a `Decimal` and
mixing precision domains. The snapshot becomes a nested object keyed on feature
name, which is also the shape `read_json_auto` gives a usable column for. A
`null` inside it means that feature was still warming up at the decision, never
that it read zero.

**A bad line is fatal, not skipped.** A journal that silently drops rows gives
a capture ratio and a continuation probability computed on an unknown subset,
and both are the kind of number that looks plausible while being wrong.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from neurotrade.core.snapshot import FeatureSnapshot
from neurotrade.core.trades import TradeRecord
from neurotrade.core.types import Side

__all__ = ["CorruptTradeJournal", "TradeJournalStore"]

# Fields that must survive as exact decimals rather than as JSON floats.
_DECIMAL_FIELDS = (
    "entry",
    "exit",
    "profit_target",
    "stop_loss",
    "realised_return",
    "gross_return",
    "mfe",
    "mae",
    "spread_fraction",
    "commission_per_share",
)


class CorruptTradeJournal(ValueError):
    """Raised when a line in the journal cannot be read back.

    Names the file and the line number. Not recovered from automatically — see
    the module docstring.
    """


class TradeJournalStore:
    """An append-only trade journal, on one JSONL file.

    Satisfies `TradeJournalPort` structurally. The parent directory is created
    on construction, so a caller does not have to pre-make a run directory that
    only this class knows the shape of.

    Example:
        >>> import tempfile
        >>> record = TradeRecord(
        ...     run_id="run_a", config_hash="cfg_a", recorded_ns=5,
        ...     symbol="AAPL.NASDAQ", strategy="gap_continuation",
        ...     strategy_version="1.0.0", variant="gap>=1.5 ranges",
        ...     side=Side.BUY, entry_ns=1, exit_ns=4, bars_held=3,
        ...     entry=Decimal("100"), exit=Decimal("102"),
        ...     profit_target=Decimal("0.02"), stop_loss=Decimal("0.01"),
        ...     max_bars=30, label=1, ambiguous=False,
        ...     realised_return=Decimal("0.0194"), gross_return=Decimal("0.02"),
        ...     mfe=Decimal("0.025"), mae=Decimal("0.004"),
        ...     spread_fraction=Decimal("0.0005"),
        ...     commission_per_share=Decimal("0.005"),
        ...     features=FeatureSnapshot.of({"gap_ranges": 1.6, "rvol": None}),
        ... )
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     store = TradeJournalStore(Path(directory) / "trades.jsonl")
        ...     store.append(record)
        ...     back = store.records()
        ...     (back[0].symbol, back[0].entry, back[0].features.get("gap_ranges"))
        ('AAPL.NASDAQ', Decimal('100'), 1.6)
    """

    __slots__ = ("_path",)

    def __init__(self, path: Path) -> None:
        """Point the journal at a file, creating its directory if needed.

        Args:
            path: The JSONL file to append to. Created on first write.
        """
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Where rows are being written, so a run can report it."""
        return self._path

    def append(self, record: TradeRecord) -> None:
        """Write one trade as a single line.

        One `open`/`write`/`close` per row: see the module docstring. At tens of
        thousands of rows against an hours-long run this is not the bottleneck,
        and it is what makes a killed run's journal complete.
        """
        payload: dict[str, object] = {
            "run_id": record.run_id,
            "config_hash": record.config_hash,
            "recorded_ns": record.recorded_ns,
            "symbol": record.symbol,
            "strategy": record.strategy,
            "strategy_version": record.strategy_version,
            "variant": record.variant,
            "side": record.side.value,
            "entry_ns": record.entry_ns,
            "exit_ns": record.exit_ns,
            "bars_held": record.bars_held,
            "max_bars": record.max_bars,
            "label": record.label,
            "ambiguous": record.ambiguous,
            # Already in sorted order (`FeatureSnapshot` guarantees it), so the
            # line's bytes do not depend on this encoder sorting nested keys.
            "features": record.features.as_dict(),
        }
        for field in _DECIMAL_FIELDS:
            payload[field] = str(getattr(record, field))
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def records(self) -> tuple[TradeRecord, ...]:
        """Read every trade back, in the order it was appended.

        Returns:
            Empty when the journal does not exist yet — a run that took no
            trades is an ordinary outcome, not an error.

        Raises:
            CorruptTradeJournal: If any line cannot be parsed. Named with the
                file and line number, and never skipped.
        """
        if not self._path.exists():
            return ()
        rows: list[TradeRecord] = []
        with self._path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    rows.append(
                        TradeRecord(
                            run_id=raw["run_id"],
                            config_hash=raw["config_hash"],
                            recorded_ns=raw["recorded_ns"],
                            symbol=raw["symbol"],
                            strategy=raw["strategy"],
                            strategy_version=raw["strategy_version"],
                            variant=raw["variant"],
                            side=Side(raw["side"]),
                            entry_ns=raw["entry_ns"],
                            exit_ns=raw["exit_ns"],
                            bars_held=raw["bars_held"],
                            max_bars=raw["max_bars"],
                            label=raw["label"],
                            ambiguous=raw["ambiguous"],
                            # `.get`, so a journal written before the snapshot
                            # existed still reads back — as "none recorded",
                            # which is what those rows truthfully hold.
                            features=FeatureSnapshot.of(raw.get("features") or {}),
                            **{field: Decimal(raw[field]) for field in _DECIMAL_FIELDS},
                        )
                    )
                # InvalidOperation subclasses ArithmeticError, not ValueError, so a
                # malformed price would otherwise escape as a raw decimal error
                # rather than as a CorruptTradeJournal naming the line.
                except (KeyError, ValueError, TypeError, InvalidOperation) as error:
                    raise CorruptTradeJournal(
                        f"{self._path}:{number} is not a readable trade record: {error}"
                    ) from error
        return tuple(rows)
