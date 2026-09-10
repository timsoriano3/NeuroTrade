"""A fake IB client, shared by the IBKR adapter tests.

One fake rather than one per test file. `IbkrClient` names the surface we depend
on, and every member added to it has to be added here too — which is the point:
the fake going stale is how a test suite drifts into passing against a client
that no longer resembles the real one.

Nothing here reaches a network. Tests that need a real Gateway are marked `ibkr`
and excluded by default.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

PAPER_ACCOUNT = "DUT108414"


@dataclass
class FakeBarData:
    """One historical bar as `ib_async` returns it.

    `date` is the bar's **open**, matching IBKR: a 390-bar US session runs
    13:30 to 19:59 UTC. The adapter shifts it to the close.
    """

    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    average: float  # VWAP; zero when the bar did not trade
    barCount: int


@dataclass
class FakeContract:
    """A qualified contract."""

    symbol: str = "AAPL"
    exchange: str = "SMART"
    currency: str = "USD"
    primaryExchange: str = "NASDAQ"
    conId: int = 265598


class FakeClient:
    """Stands in for `ib_async.IB.client`."""

    def __init__(self, server_version: int = 178) -> None:
        self._server_version = server_version

    def serverVersion(self) -> int:
        return self._server_version


class FakeIB:
    """A fake `ib_async.IB` covering exactly what `IbkrClient` declares."""

    def __init__(
        self,
        accounts: tuple[str, ...] = (PAPER_ACCOUNT,),
        *,
        fails_with: Exception | None = None,
        hangs: bool = False,
        bars: list[FakeBarData] | None = None,
        contracts: list[FakeContract] | None = None,
        historical_error: Exception | None = None,
    ) -> None:
        self.client: FakeClient = FakeClient()
        self._accounts = accounts
        self._connected = False
        self._fails_with = fails_with
        self._hangs = hangs
        self._bars = bars if bars is not None else []
        self._contracts = contracts if contracts is not None else [FakeContract()]
        self._historical_error = historical_error

        self.connect_calls = 0
        self.disconnect_calls = 0
        self.historical_calls: list[dict[str, Any]] = []

    # ── Connection ───────────────────────────────────────────

    async def connectAsync(self, *args: object, **kwargs: object) -> None:
        self.connect_calls += 1
        if self._hangs:
            await asyncio.sleep(60)
        if self._fails_with is not None:
            raise self._fails_with
        self._connected = True

    def isConnected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    def managedAccounts(self) -> list[str]:
        return list(self._accounts)

    # ── Market data ──────────────────────────────────────────

    async def qualifyContractsAsync(self, *contracts: object) -> list[Any]:
        return list(self._contracts)

    async def reqHistoricalDataAsync(self, *args: object, **kwargs: object) -> list[Any]:
        self.historical_calls.append(dict(kwargs))
        if self._historical_error is not None:
            raise self._historical_error
        return list(self._bars)


def a_bar(
    minute: int,
    *,
    close: float = 100.5,
    volume: float = 1000.0,
    average: float = 100.4,
    bar_count: int = 42,
) -> FakeBarData:
    """A bar stamped at its open, `minute` minutes after the US open.

    13:30 UTC is 09:30 ET, the regular-session open.
    """
    # High and low bracket both open and close, or the domain model rejects the
    # bar — which it should: a close outside the range is not a bar.
    open_ = close
    return FakeBarData(
        date=datetime(2026, 3, 16, 13, 30, tzinfo=UTC).replace(
            hour=13 + (30 + minute) // 60, minute=(30 + minute) % 60
        ),
        open=open_,
        high=max(open_, close) + 1.0,
        low=min(open_, close) - 1.0,
        close=close,
        volume=volume,
        average=average if average == 0.0 else close,
        barCount=bar_count,
    )
