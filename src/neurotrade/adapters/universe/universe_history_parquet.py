"""Persisting a `UniverseHistory` as one Parquet file.

The history is derived data: recomputable from `derived/daily/` at any time, so
it lives beside the corpus rather than in `config/`. It is written out anyway
because a backtest has to be able to say *which* membership it used, and
recomputing it later against moved thresholds would answer a different
question.

**One file, not a partitioned dataset.** A few thousand sessions times a few
thousand names is still a small table, and the whole point of the artifact is
to be read in one go. The daily corpus's file-per-session grain exists because
the crawler fills it a session at a time; nothing fills this incrementally.

**A date with no members still gets a row**, with a null ticker. Dropping it
would make an empty session indistinguishable from one never evaluated, and
`UniverseHistory.as_of` would then carry the previous membership across it.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from neurotrade.core.clock import Clock
from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import UniverseHistory, UniverseMembership

__all__ = ["HISTORY_SCHEMA", "SCHEMA_VERSION", "UniverseHistoryStore"]

SCHEMA_VERSION = "1"
"""Bumped whenever the column set or a type changes, and written into the file
metadata — same contract as the bar schema."""

HISTORY_SCHEMA = pa.schema(
    [
        pa.field("session_date", pa.date32(), nullable=False),
        # Null on both when the screen admitted nobody that session. See the
        # module docstring: the row is the evidence the date was evaluated.
        pa.field("ticker", pa.string(), nullable=True),
        pa.field("venue", pa.string(), nullable=True),
    ],
    metadata={"schema_version": SCHEMA_VERSION, "dataset": "universe_history"},
)

_FILENAME = "history.parquet"


class UniverseHistoryStore:
    """Reads and writes the membership artifact under one root.

    Example:
        >>> store = UniverseHistoryStore(Path("data/derived/universe/yfinance"))
        >>> store.path.name
        'history.parquet'
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        """Open a store over a directory.

        Args:
            root: Where `history.parquet` lives. Created on write, not here —
                constructing a reader must not leave directories behind.
        """
        self._root = root

    @property
    def path(self) -> Path:
        """The file this store reads and writes."""
        return self._root / _FILENAME

    def write(
        self,
        history: UniverseHistory,
        *,
        universe_digest: str,
        config_hash: str,
        clock: Clock,
    ) -> Path:
        """Write the history, stamped with what produced it.

        Args:
            history: The membership to persist.
            universe_digest: `Universe.digest` of the candidate set screened.
                The artifact is a subset of that universe on every date, and
                without this there is no way to know what it was a subset of.
            config_hash: The resolved settings the screen ran under, so a
                threshold change is visible as a different artifact.
            clock: Stamps when this was built. Taken as an argument rather than
                read from the wall so replay stays deterministic.

        Returns:
            The path written.
        """
        rows: list[dict[str, object]] = []
        for row in history:
            if not row.symbols:
                rows.append({"session_date": row.session_date, "ticker": None, "venue": None})
                continue
            rows.extend(
                {
                    "session_date": row.session_date,
                    "ticker": symbol.ticker,
                    "venue": symbol.venue.value,
                }
                for symbol in row.symbols
            )

        schema = HISTORY_SCHEMA.with_metadata(
            {
                **{key.decode(): value.decode() for key, value in HISTORY_SCHEMA.metadata.items()},
                "survivorship_biased": str(history.survivorship_biased).lower(),
                "history_digest": history.digest,
                "universe_digest": universe_digest,
                "config_hash": config_hash,
                "built_at_ns": str(clock.now_ns()),
            }
        )
        self._root.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), self.path)
        return self.path

    def read(self) -> UniverseHistory:
        """Read the history back, verifying it is the one that was written.

        The digest recorded at write time is recomputed from the rows and
        compared. A truncated or hand-edited artifact is worse than a missing
        one: it would quietly change which names a backtest was allowed to
        trade, and nothing downstream would notice.

        Raises:
            FileNotFoundError: If nothing has been written here.
            ValueError: If the recorded digest does not match the rows, or the
                file carries a schema version this code does not know.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"no universe history at {self.path}")

        table = pq.read_table(self.path)
        metadata = {
            key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()
        }
        version = metadata.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"universe history at {self.path} is schema version {version}")

        members: defaultdict[date, list[Symbol]] = defaultdict(list)
        for row in table.to_pylist():
            session_date = row["session_date"]
            symbols = members[session_date]  # defaultdict: the date exists even if empty
            if row["ticker"] is not None:
                symbols.append(Symbol(row["ticker"], Venue(row["venue"])))

        history = UniverseHistory(
            (UniverseMembership(day, symbols) for day, symbols in members.items()),
            survivorship_biased=metadata.get("survivorship_biased") == "true",
        )
        recorded = metadata.get("history_digest")
        if recorded != history.digest:
            raise ValueError(
                f"universe history at {self.path} digests {history.digest}, "
                f"but was written as {recorded}"
            )
        return history

    def __repr__(self) -> str:
        return f"UniverseHistoryStore({self._root})"
