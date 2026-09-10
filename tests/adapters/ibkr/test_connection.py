"""Tests for the IB Gateway connection.

Split deliberately. Most of these use a fake client and run anywhere, because a
test suite that needs a broker logged in is a test suite that stops being run.
The handful marked `ibkr` talk to a real Gateway and are opt-in:

    uv run pytest -m ibkr

The property under test throughout is identity, not reachability: connecting
proves the socket answered, and says nothing about *which* account answered.
"""

from __future__ import annotations

import pytest

from neurotrade.adapters.ibkr.connection import (
    ConnectionProbe,
    IbkrConnection,
    IbkrConnectionError,
)
from neurotrade.config import IbkrSettings, Profile, load_settings
from tests.adapters.ibkr.conftest import PAPER_ACCOUNT, FakeIB


def a_connection(
    settings: IbkrSettings | None = None,
    *,
    accounts: tuple[str, ...] = (PAPER_ACCOUNT,),
    fails_with: Exception | None = None,
    hangs: bool = False,
) -> IbkrConnection:
    return IbkrConnection(
        settings or IbkrSettings(account=PAPER_ACCOUNT),
        ib=FakeIB(accounts, fails_with=fails_with, hangs=hangs),
    )


# ── Identity, not just reachability ──────────────────────────


async def test_probe_reports_which_account_answered() -> None:
    probe = await a_connection().probe()
    assert probe.connected
    assert probe.accounts == (PAPER_ACCOUNT,)
    assert probe.account == PAPER_ACCOUNT


async def test_a_different_account_is_reported_as_a_mismatch() -> None:
    """Gateway logged into the wrong account answers happily; only the account
    number reveals it."""
    probe = await a_connection(accounts=("U1234567",)).probe()
    assert probe.connected
    assert not probe.account_matches
    assert not probe.is_healthy


async def test_no_expectation_means_any_account_matches() -> None:
    """An unconfigured account must not make every probe fail."""
    connection = IbkrConnection(IbkrSettings(account=""), ib=FakeIB())
    probe = await connection.probe()
    assert probe.account_matches
    assert probe.is_healthy


async def test_a_connection_with_no_accounts_is_unhealthy() -> None:
    """The socket can answer before the login has finished."""
    probe = await a_connection(accounts=()).probe()
    assert probe.connected
    assert not probe.is_healthy
    assert probe.account is None


async def test_several_accounts_leaves_the_account_ambiguous() -> None:
    """An advisor login manages many; that is not a failure, but an order then
    needs an explicit account rather than an implied one."""
    probe = await a_connection(accounts=(PAPER_ACCOUNT, "DU999999")).probe()
    assert probe.account is None
    assert probe.is_healthy  # the expected account is among them


# ── Paper vs live ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("port", "is_paper"),
    [(4002, True), (7497, True), (4001, False), (7496, False)],
)
def test_paper_ports_are_distinguished_from_live(port: int, is_paper: bool) -> None:
    """4001/4002 and 7496/7497 differ by one digit and by real money."""
    assert IbkrSettings(port=port).is_paper_port is is_paper


async def test_the_probe_reports_a_live_port() -> None:
    """Connecting to live when paper was intended spends money; the asymmetry
    is why this is reported rather than inferred."""
    connection = IbkrConnection(IbkrSettings(port=4001), ib=FakeIB())
    probe = await connection.probe()
    assert not probe.is_paper_port
    assert "live port" in probe.describe()


# ── Failure modes ────────────────────────────────────────────


async def test_a_refused_connection_names_host_port_and_client() -> None:
    """ "Connection refused" alone does not say which of the four ports was dialled."""
    connection = IbkrConnection(
        IbkrSettings(port=4002, client_id=7),
        ib=FakeIB(fails_with=ConnectionRefusedError("nope")),
    )
    with pytest.raises(IbkrConnectionError, match=r"4002 as client 7"):
        await connection.connect()


async def test_a_timeout_suggests_the_likely_cause() -> None:
    """Gateway running but not logged in accepts the socket and never answers."""
    connection = IbkrConnection(IbkrSettings(timeout_seconds=0.05), ib=FakeIB(hangs=True))
    with pytest.raises(IbkrConnectionError, match="is it logged in"):
        await connection.connect()


# ── Lifecycle ────────────────────────────────────────────────


async def test_connecting_twice_is_a_no_op() -> None:
    fake = FakeIB()
    connection = IbkrConnection(IbkrSettings(), ib=fake)
    await connection.connect()
    await connection.connect()
    assert fake.connect_calls == 1


async def test_the_context_manager_disconnects_even_when_the_body_raises() -> None:
    """A leaked connection holds its client id, and the next run cannot reuse it."""
    fake = FakeIB()
    connection = IbkrConnection(IbkrSettings(), ib=fake)
    with pytest.raises(RuntimeError):
        async with connection:
            raise RuntimeError("boom")
    assert fake.disconnect_calls == 1
    assert not connection.is_connected


async def test_disconnecting_twice_is_safe() -> None:
    connection = a_connection()
    await connection.connect()
    connection.disconnect()
    connection.disconnect()
    assert not connection.is_connected


def test_nothing_is_opened_by_construction() -> None:
    """Building a connection must not dial; the CLI builds one to read settings."""
    fake = FakeIB()
    IbkrConnection(IbkrSettings(), ib=fake)
    assert fake.connect_calls == 0


# ── Reporting ────────────────────────────────────────────────


def test_describe_covers_every_fact_a_reader_needs() -> None:
    probe = ConnectionProbe(
        connected=True,
        server_version=178,
        accounts=(PAPER_ACCOUNT,),
        is_paper_port=True,
        expected_account=PAPER_ACCOUNT,
    )
    described = probe.describe()
    for expected in ("connected", "178", PAPER_ACCOUNT, "paper port", "matches"):
        assert expected in described


def test_describe_omits_the_expectation_when_none_is_set() -> None:
    probe = ConnectionProbe(
        connected=True,
        server_version=178,
        accounts=(PAPER_ACCOUNT,),
        is_paper_port=True,
        expected_account="",
    )
    assert "expected" not in probe.describe()


# ── Configuration ────────────────────────────────────────────


def test_the_default_port_is_paper() -> None:
    """A misconfiguration should fail to connect, not trade real money."""
    assert IbkrSettings().port == 4002
    assert IbkrSettings().is_paper_port


def test_the_paper_profile_names_the_real_account() -> None:
    """So a Gateway logged into the wrong account is caught by `ibkr check`
    rather than by a surprising fill."""
    ibkr = load_settings(Profile.PAPER).ibkr
    assert ibkr.port == 4002
    assert ibkr.account == PAPER_ACCOUNT


def test_broker_settings_do_not_affect_the_config_hash() -> None:
    """Which socket we dial does not change what the system decides to trade, so
    a trade must not look different for having been placed from another machine."""
    from neurotrade.config import behavioural_values

    assert "ibkr" not in behavioural_values(load_settings(Profile.PAPER))


# ── Against a real Gateway ───────────────────────────────────


@pytest.mark.ibkr
async def test_a_real_gateway_answers_as_the_configured_account() -> None:
    """Opt-in: `uv run pytest -m ibkr` with Gateway logged in."""
    settings = load_settings(Profile.PAPER).ibkr
    connection = IbkrConnection(IbkrSettings(**{**settings.model_dump(), "client_id": 77}))
    try:
        probe = await connection.probe()
    finally:
        connection.disconnect()

    assert probe.is_healthy
    assert probe.is_paper_port
    assert probe.account == settings.account
    assert probe.server_version is not None and probe.server_version > 100
