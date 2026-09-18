"""Persisting corporate actions as one Parquet file.

**Why this sits under `derived/` and not `raw/`.** An action is downloaded, not
computed, so `raw/` looks like its home. It goes to `derived/` anyway, for the
same reason the daily bars and the vendor seed samples do: `raw/` is immutable,
and a feed *revises* corporate actions. Yahoo corrects a mis-stated ratio, adds
a dividend it missed, restates an effective date. Re-fetching would then mean
overwriting `raw/`, which is exactly what the invariant forbids. Treating the
action set as recomputable — delete it, run the fetch again, get it back — keeps
the invariant intact and costs nothing, because the source is a free HTTP call.

**One file, not a partitioned dataset.** A couple of thousand names times a
few dozen actions each is a table of low hundreds of thousands of rows, and
every caller wants all of one symbol's history at once — a split from 2014
still applies to a 2013 bar, so there is no useful date pruning. The
file-per-session grain of the bar corpus exists because a crawler fills it one
session at a time; nothing fills this incrementally.

**A symbol that has no actions still gets a row**, with nulls in both value
columns. Without it, "we asked Yahoo and it has never split" is indistinguishable
from "we never asked", and a silent re-fetch gap would look like a clean answer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from neurotrade.adapters.storage.schemas import Source
from neurotrade.core.actions import AdjustmentSeries, CorporateAction
from neurotrade.core.clock import Clock
from neurotrade.core.types import Price, Symbol, Venue

__all__ = ["ACTION_SCHEMA", "SCHEMA_VERSION", "ActionStore"]

SCHEMA_VERSION = "1"
"""Bumped whenever the column set or a type changes, and written into the file
metadata — same contract as the bar and universe-history schemas."""

_RATIO_TYPE = pa.decimal128(18, 8)
"""Split ratios and dividends share a scale with prices. Eight places is far
more than either needs, and matching `PRICE_TYPE` means a factor never has to
be rescaled on the way into an adjusted price."""

ACTION_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        # Null on all three when the symbol has no actions at all. See the
        # module docstring: the row is the evidence the symbol was fetched.
        pa.field("effective_date", pa.date32(), nullable=True),
        pa.field("split_ratio", _RATIO_TYPE, nullable=True),
        pa.field("dividend", _RATIO_TYPE, nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("fetched_at", pa.int64(), nullable=False),
    ],
    metadata={"schema_version": SCHEMA_VERSION, "dataset": "corporate_actions"},
)
"""Canonical column layout for corporate actions."""

_FILENAME = "actions.parquet"


class ActionStore:
    """Reads and writes the corporate-action set under one root.

    Example:
        >>> store = ActionStore(Path("data/derived/actions/yfinance"))
        >>> store.path.name
        'actions.parquet'
    """

    __slots__ = ("_root", "_source")

    def __init__(self, root: Path, source: Source = Source.YFINANCE) -> None:
        """Create the store.

        Args:
            root: Directory holding the file. Created on write.
            source: Provenance stamped on every row written here.
        """
        self._root = root
        self._source = source

    @property
    def path(self) -> Path:
        """The file this store reads and writes."""
        return self._root / _FILENAME

    def write(
        self,
        actions: Mapping[Symbol, Sequence[CorporateAction]],
        *,
        clock: Clock,
        config_hash: str,
    ) -> Path:
        """Write the action set, replacing whatever was there.

        Args:
            actions: Every symbol fetched, mapped to its actions. A symbol with
                an empty sequence is recorded as fetched-and-empty, not
                omitted.
            clock: Stamps the fetch time. Taken as an argument rather than read
                from the wall, so replay stays deterministic.
            config_hash: Resolved settings the fetch ran under.

        Returns:
            The path written.

        Raises:
            ValueError: If an action is filed under a different symbol than its
                own, which would apply one company's split to another's prices.
        """
        fetched_at = clock.now_ns()
        rows: list[dict[str, object]] = []
        for symbol in sorted(actions, key=str):
            symbol_actions = actions[symbol]
            for action in symbol_actions:
                if action.symbol != symbol:
                    raise ValueError(f"action for {action.symbol} filed under {symbol}")
            if not symbol_actions:
                rows.append(_row(symbol, None, self._source, fetched_at))
                continue
            rows.extend(
                _row(symbol, action, self._source, fetched_at)
                for action in sorted(symbol_actions, key=lambda a: a.effective_date)
            )

        schema = ACTION_SCHEMA.with_metadata(
            {
                **{key.decode(): value.decode() for key, value in ACTION_SCHEMA.metadata.items()},
                "config_hash": config_hash,
                "source": self._source.value,
                "symbols": str(len(actions)),
            }
        )
        self._root.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), self.path)
        return self.path

    def read(self) -> dict[Symbol, tuple[CorporateAction, ...]]:
        """Read the action set back.

        Returns:
            Every symbol in the file, mapped to its actions oldest first. A
            symbol with no actions maps to an empty tuple — present, and known
            to have none.

        Raises:
            FileNotFoundError: If nothing has been written here.
            ValueError: If the file carries a schema version this code does not
                know.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"no corporate actions at {self.path}")

        table = pq.read_table(self.path)
        metadata = {
            key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()
        }
        version = metadata.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"corporate actions at {self.path} are schema version {version}")

        found: defaultdict[Symbol, list[CorporateAction]] = defaultdict(list)
        for row in table.to_pylist():
            symbol = Symbol(row["ticker"], Venue(row["venue"]))
            actions = found[symbol]  # defaultdict: the symbol exists even with no actions
            if row["effective_date"] is None:
                continue
            actions.append(
                CorporateAction(
                    symbol=symbol,
                    effective_date=row["effective_date"],
                    split_ratio=row["split_ratio"],
                    dividend=row["dividend"],
                )
            )
        return {
            symbol: tuple(sorted(actions, key=lambda a: a.effective_date))
            for symbol, actions in found.items()
        }

    def series(
        self, symbol: Symbol, *, prior_close: Mapping[date, Price] | None = None
    ) -> AdjustmentSeries:
        """Build the `AdjustmentSeries` for one symbol from the stored set.

        A symbol absent from the file gets an empty series rather than an
        error. That is deliberate but worth knowing: an empty series adjusts
        nothing, so a name that was never fetched is silently treated as a name
        that never split. `neurotrade actions check` is what catches that, by
        looking at the prices rather than at this file.

        Args:
            symbol: The instrument to build for.
            prior_close: Closes before each ex-dividend date, needed only for
                total-return adjustment.

        Returns:
            The series, ready to adjust bars.
        """
        return AdjustmentSeries(symbol, self.read().get(symbol, ()), prior_close=prior_close)

    def __repr__(self) -> str:
        return f"ActionStore({self._root}, {self._source.value})"


def _row(
    symbol: Symbol, action: CorporateAction | None, source: Source, fetched_at: int
) -> dict[str, object]:
    """One Parquet row, for an action or for a symbol known to have none."""
    return {
        "ticker": symbol.ticker,
        "venue": symbol.venue.value,
        "effective_date": None if action is None else action.effective_date,
        "split_ratio": None if action is None else _ratio(action.split_ratio),
        "dividend": None if action is None else _ratio(action.dividend),
        "source": source.value,
        "fetched_at": fetched_at,
    }


def _ratio(value: Decimal) -> Decimal:
    """Put a ratio on the column's exact scale.

    `decimal128(18, 8)` refuses a value carrying more places than it can hold —
    "Rescaling Decimal value would cause data loss" — rather than rounding it,
    and a dividend derived from a float can easily carry more.
    """
    return value.quantize(Decimal(1).scaleb(-8))
