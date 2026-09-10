"""Connecting to IB Gateway, and checking the connection is the one we meant.

**Connecting is not the same as being connected to the right thing.** The socket
answers whether Gateway is logged into a paper account or a live one, so a
successful connection proves almost nothing on its own. `probe` therefore checks
identity as well as reachability: which account answered, whether the port is a
paper port, and whether the account matches what configuration expected.

That matters because the failure mode is asymmetric. Connecting to paper when
you wanted live wastes a session; connecting to live when you wanted paper
spends money.

**One client id per connected process.** Gateway rejects a second connection
using an id already in use, so the crawler, the live engine and an interactive
session each need their own. The id is configuration rather than a constant for
that reason.

**Disconnects are routine, not exceptional.** Gateway restarts daily to reload
contract definitions, and re-authentication is needed after the weekly server
reset. Callers must treat a dropped connection as an ordinary event to recover
from, which is why this module exposes connection state rather than assuming it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Protocol, Self, cast

from ib_async import IB, Contract

from neurotrade.config import IbkrSettings

__all__ = [
    "ConnectionProbe",
    "IbkrClient",
    "IbkrConnection",
    "IbkrConnectionError",
]

# ib_async logs connection chatter at INFO through the standard library logger,
# which lands on stderr unstructured and interleaves with ours. Its warnings are
# worth keeping — "Market data farm connection is inactive" is a real signal —
# so the floor is WARNING rather than silence.
logging.getLogger("ib_async").setLevel(logging.WARNING)


class _ServerInfo(Protocol):
    """The one thing we read off `ib_async.IB.client`."""

    def serverVersion(self) -> int: ...


class IbkrClient(Protocol):
    """The slice of `ib_async.IB` this module actually uses.

    Naming it documents the dependency surface — five members out of a large
    class — so swapping the client library later has a checklist rather than a
    search, and tests can substitute a fake without inheriting from a
    third-party class. Same ports-and-adapters reasoning, one level down.

    Signatures name only the four arguments we pass and widen the return types,
    which is what a protocol should do: state what we *depend on*, not what the
    library happens to offer. `ib_async.disconnect` returns `str | None`, for
    instance, and we ignore it — declaring `object` says so.

    The real `IB` does not match this structurally, because its `connectAsync`
    takes eight parameters and mypy compares callables by position and name. The
    single `cast` below is that gap, deliberately confined to one line: the
    members are verified present by the `ibkr`-marked tests, which run against a
    live Gateway.
    """

    @property
    def client(self) -> _ServerInfo:
        """Read-only. A mutable attribute is invariant, so a fake whose client
        is a subtype would not satisfy this protocol."""
        ...

    def connectAsync(
        self,
        host: str = ...,
        port: int = ...,
        clientId: int = ...,
        readonly: bool = ...,
    ) -> Awaitable[object]: ...

    def isConnected(self) -> bool: ...

    def disconnect(self) -> object: ...

    def managedAccounts(self) -> list[str]: ...

    def qualifyContractsAsync(self, *contracts: Contract) -> Awaitable[list[Any]]: ...

    def placeOrder(self, contract: Contract, order: object) -> object: ...

    def cancelOrder(self, order: object) -> object: ...

    def reqHistoricalDataAsync(
        self,
        contract: Contract,
        endDateTime: datetime | str | None,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int = ...,
    ) -> Awaitable[list[Any]]: ...


class IbkrConnectionError(RuntimeError):
    """Raised when Gateway cannot be reached, or answers as the wrong account.

    Carries what was expected alongside what was found: "connection refused" is
    rarely the useful part, and "expected DUT108414, got U1234567" is.
    """


@dataclass(frozen=True, slots=True)
class ConnectionProbe:
    """What one health check found.

    Example:
        >>> probe = ConnectionProbe(
        ...     connected=True, server_version=178, accounts=("DUT108414",),
        ...     is_paper_port=True, expected_account="DUT108414",
        ... )
        >>> (probe.is_healthy, probe.account)
        (True, 'DUT108414')
    """

    connected: bool  # the socket answered and the handshake completed
    server_version: int | None  # Gateway's API protocol version; None if not connected
    accounts: tuple[str, ...]  # accounts this login can act on
    is_paper_port: bool  # whether the port dialled is a paper port
    expected_account: str  # from configuration; empty means no expectation

    @property
    def account(self) -> str | None:
        """The single account, or `None` when there are none or several.

        Several is possible — an advisor login manages many — and is not a
        failure, but it does mean an order needs an explicit account rather than
        an implied one.
        """
        return self.accounts[0] if len(self.accounts) == 1 else None

    @property
    def account_matches(self) -> bool:
        """Whether the account found is the one configuration expected.

        True when no expectation was set, so an unconfigured `account` does not
        make every probe fail.
        """
        return not self.expected_account or self.expected_account in self.accounts

    @property
    def is_healthy(self) -> bool:
        """Whether this connection is usable and is the one intended."""
        return self.connected and bool(self.accounts) and self.account_matches

    def describe(self) -> str:
        """One line per fact, for `neurotrade ibkr check`."""
        lines = [
            f"connected      {'yes' if self.connected else 'no'}",
            f"server version {self.server_version if self.server_version else '-'}",
            f"accounts       {', '.join(self.accounts) if self.accounts else '-'}",
            f"paper port     {'yes' if self.is_paper_port else 'NO — this is a live port'}",
        ]
        if self.expected_account:
            verdict = "matches" if self.account_matches else "MISMATCH"
            lines.append(f"expected       {self.expected_account} ({verdict})")
        return "\n".join(lines)


class IbkrConnection:
    """A connection to IB Gateway.

    Wraps `ib_async.IB` so that the rest of the system sees settings and domain
    errors rather than a third-party client. Use as an async context manager;
    the connection is closed even when the body raises, which matters because a
    leaked connection holds its client id and the next run cannot reuse it.

    Example:
        >>> settings = IbkrSettings(port=4002, client_id=7)
        >>> connection = IbkrConnection(settings)
        >>> connection.is_connected
        False
    """

    __slots__ = ("_ib", "_settings")

    def __init__(self, settings: IbkrSettings, ib: IbkrClient | None = None) -> None:
        """Prepare a connection. Nothing is opened until `connect`.

        Args:
            settings: Where Gateway is and which client id to use.
            ib: An existing client, for tests. A fresh one is created otherwise.
        """
        self._settings = settings
        # See IbkrClient's docstring for why the real client needs a cast.
        self._ib: IbkrClient = ib if ib is not None else cast(IbkrClient, IB())

    @property
    def is_connected(self) -> bool:
        """Whether the socket is currently up.

        Worth checking rather than assuming: Gateway restarts daily, so a
        long-running process will see this go false during normal operation.
        """
        return bool(self._ib.isConnected())

    @property
    def ib(self) -> IbkrClient:
        """The underlying client.

        Exposed for adapters in this package that need calls beyond connection
        management — the market data feed, and later the broker. Deliberately
        not re-exported outside `adapters.ibkr`: the rest of the system talks to
        ports, and a caller reaching this would be depending on `ib_async`.
        """
        return self._ib

    @property
    def settings(self) -> IbkrSettings:
        """The settings this connection was built with."""
        return self._settings

    async def connect(self) -> None:
        """Open the connection.

        Raises:
            IbkrConnectionError: If Gateway is unreachable, refuses the client
                id, or does not answer within the configured timeout. The
                message names the host, port and client id, because "connection
                refused" alone does not say which of the four ports was dialled.
        """
        if self.is_connected:
            return
        try:
            await asyncio.wait_for(
                self._ib.connectAsync(
                    self._settings.host,
                    self._settings.port,
                    clientId=self._settings.client_id,
                    readonly=False,
                ),
                timeout=self._settings.timeout_seconds,
            )
        except TimeoutError as error:
            raise IbkrConnectionError(
                f"IB Gateway at {self._settings.host}:{self._settings.port} did not answer "
                f"within {self._settings.timeout_seconds}s — is it logged in?"
            ) from error
        except Exception as error:
            raise IbkrConnectionError(
                f"could not connect to IB Gateway at {self._settings.host}:"
                f"{self._settings.port} as client {self._settings.client_id}: {error}"
            ) from error

    def disconnect(self) -> None:
        """Close the connection. Safe when already closed."""
        if self.is_connected:
            self._ib.disconnect()

    async def probe(self) -> ConnectionProbe:
        """Connect if needed and report what answered.

        Returns:
            A `ConnectionProbe` describing the connection, including whether the
            account matches configuration. Identity is reported rather than
            enforced: the caller decides whether a mismatch is fatal, because
            during setup it usually means "you have not filled this in yet".

        Raises:
            IbkrConnectionError: If the connection cannot be opened at all.
        """
        await self.connect()
        return ConnectionProbe(
            connected=self.is_connected,
            server_version=self._ib.client.serverVersion(),
            accounts=tuple(self._ib.managedAccounts()),
            is_paper_port=self._settings.is_paper_port,
            expected_account=self._settings.account,
        )

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        state = "connected" if self.is_connected else "disconnected"
        return f"IbkrConnection({self._settings.host}:{self._settings.port}, {state})"
