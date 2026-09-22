"""The trial ledger on disk — newline-delimited JSON, append-only.

Satisfies `TrialLedgerPort`. The ledger is the permanent record of how many
hypotheses have been tested (§8), so the storage choices here are all about
surviving carelessness rather than about speed.

**JSONL, and one file for the project's lifetime.** Unlike the event log, which
is rotated per session, this file accumulates: a candidate tested today
deflates against searches run months ago. Volume is low — a trial is a whole
backtest, not a bar — and the file stays readable in a terminal, which matters
for something whose job is to be audited.

**Every write opens, appends and closes.** Trials arrive seconds or minutes
apart, so holding a handle open buys nothing and costs the guarantee that a
process dying mid-sweep leaves a complete ledger on disk. It also makes two
processes appending concurrently safe enough in practice: each record is a
single short line written in one `write` call under `O_APPEND`.

**A bad line is fatal, not skipped.** Skipping an unreadable record would
silently lower the trial count, which lowers the hurdle, which is the one
direction of error this entire subsystem exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

from neurotrade.core.ids import TrialId
from neurotrade.core.trials import Trial, TrialSource

__all__ = ["CorruptTrialLedger", "TrialLedgerStore"]


class CorruptTrialLedger(ValueError):
    """Raised when a line in the ledger cannot be read back.

    Names the file and line number. Not recovered from automatically: see the
    module docstring — a ledger that quietly drops records reports a smaller
    search than actually happened, and every deflation computed from it is too
    generous.
    """


class TrialLedgerStore:
    """An append-only trial ledger, on one JSONL file.

    Satisfies `TrialLedgerPort` structurally.

    Example:
        >>> import tempfile
        >>> from neurotrade.core.trials import Trial, TrialSource
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     store = TrialLedgerStore(Path(directory) / "trials.jsonl")
        ...     store.append(Trial(
        ...         trial_id=TrialId.derive(
        ...             hypothesis="orb 15m", config_hash="cfg_a", recorded_ns=1
        ...         ),
        ...         recorded_ns=1, hypothesis="orb 15m", family="opening-range",
        ...         source=TrialSource.MANUAL, config_hash="cfg_a", sharpe=0.04,
        ...         n_observations=1200, n_paths=5,
        ...     ))
        ...     [trial.family for trial in store.trials()]
        ['opening-range']
    """

    __slots__ = ("_path",)

    def __init__(self, path: Path) -> None:
        """Point the store at a ledger file.

        Args:
            path: The ledger. Parent directories are created on first append.
                Opening for append never truncates, so pointing a new process
                at an existing ledger continues it.
        """
        self._path = path

    @property
    def path(self) -> Path:
        """Where the ledger is written."""
        return self._path

    def append(self, trial: Trial) -> None:
        """Record one trial.

        Args:
            trial: The hypothesis and what it scored.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     store = TrialLedgerStore(Path(directory) / "t.jsonl")
            ...     for index in range(3):
            ...         store.append(Trial(
            ...             trial_id=TrialId.derive(
            ...                 hypothesis=f"h{index}", config_hash="c", recorded_ns=index
            ...             ),
            ...             recorded_ns=index, hypothesis=f"h{index}", family="f",
            ...             source=TrialSource.SWEEP, config_hash="c", sharpe=0.01,
            ...             n_observations=100, n_paths=0,
            ...         ))
            ...     len(store.trials("f"))
            3
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_encode(trial), sort_keys=True, separators=(",", ":"))
        # One open-write-close per record, and one `write` call: see the module
        # docstring. `sort_keys` keeps the file byte-identical for identical
        # trials, which is what lets a rebuilt ledger be diffed against the old.
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def trials(self, family: str | None = None) -> tuple[Trial, ...]:
        """Every trial recorded, oldest first.

        Args:
            family: Restrict to one search space; all of them when omitted.

        Returns:
            The matching trials in append order. Empty when the ledger does not
            exist yet — a project that has tested nothing is the ordinary
            starting state, not an error.

        Raises:
            CorruptTrialLedger: If any line cannot be decoded.
        """
        if not self._path.exists():
            return ()
        records: list[Trial] = []
        with self._path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    trial = _decode(json.loads(line))
                except (ValueError, KeyError, TypeError) as error:
                    raise CorruptTrialLedger(
                        f"{self._path}:{number} cannot be read back: {error}"
                    ) from error
                if family is None or trial.family == family:
                    records.append(trial)
        return tuple(records)


def _encode(trial: Trial) -> dict[str, object]:
    """Flatten a trial into JSON-safe primitives."""
    return {
        "trial_id": trial.trial_id.value,
        "recorded_ns": trial.recorded_ns,
        "hypothesis": trial.hypothesis,
        "family": trial.family,
        "source": trial.source.value,
        "config_hash": trial.config_hash,
        "sharpe": trial.sharpe,
        "n_observations": trial.n_observations,
        "n_paths": trial.n_paths,
    }


def _decode(payload: dict[str, object]) -> Trial:
    """Rebuild a trial from one decoded ledger line.

    Every field is required. A ledger line missing `sharpe` or `family` is not
    defaulted into something plausible — it is a corrupt record, and the caller
    is told so.
    """
    return Trial(
        trial_id=TrialId(str(payload["trial_id"])),
        recorded_ns=int(payload["recorded_ns"]),  # type: ignore[call-overload]
        hypothesis=str(payload["hypothesis"]),
        family=str(payload["family"]),
        source=TrialSource(str(payload["source"])),
        config_hash=str(payload["config_hash"]),
        sharpe=float(payload["sharpe"]),  # type: ignore[arg-type]
        n_observations=int(payload["n_observations"]),  # type: ignore[call-overload]
        n_paths=int(payload["n_paths"]),  # type: ignore[call-overload]
    )
