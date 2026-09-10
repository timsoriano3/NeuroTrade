"""Command-line entry point.

Every operational command routes through here, which is what makes startup have
exactly one shape: resolve the profile, configure logging, bind the run context,
then do the work. A command that built its own settings would be able to run
against a different configuration than the one its log lines claim.

**Output goes to stdout, logs go to stderr.** So ``neurotrade config hash`` can
be piped into another command without log noise, and redirecting logs never
swallows the answer you asked for.

``--profile`` is a global option, so it precedes the subcommand — the same shape
as ``git --no-pager log``::

    neurotrade --profile paper config show
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from neurotrade import __version__
from neurotrade.adapters.ibkr.broker import IbkrBroker
from neurotrade.adapters.ibkr.connection import IbkrConnection, IbkrConnectionError
from neurotrade.adapters.storage.event_store import EventStore
from neurotrade.config import Profile, Settings, config_hash, describe, load_settings
from neurotrade.core.clock import LiveClock, SimClock, to_datetime
from neurotrade.core.ids import IntentId, OrderId, RunId
from neurotrade.core.orders import Fill, Order, OrderType
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.lab.replay import ReplayEngine
from neurotrade.logs import configure, get_logger

__all__ = ["app"]

log = get_logger(__name__)

_FOREVER_NS = 2**63 - 1
"""Upper bound for a whole-session replay. The log holds one session, so the
range is only there to satisfy the port; bounding it by date would mean the CLI
needing a venue calendar to know when the session ended."""


@dataclass(frozen=True, slots=True)
class AppContext:
    """What every command needs, resolved once by the root callback.

    Reached through ``ctx.obj``. Frozen because a command that mutated the
    settings mid-run would invalidate the config hash already stamped on this
    run's log lines.
    """

    settings: Settings  # resolved configuration for this invocation
    run_id: RunId  # identifies this run in the logs


app = typer.Typer(
    name="neurotrade",
    help="Autonomous day-trading system for US and Canadian equities.",
    no_args_is_help=True,
    add_completion=False,
)

config_app = typer.Typer(help="Inspect resolved configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

ibkr_app = typer.Typer(help="Talk to Interactive Brokers.", no_args_is_help=True)
app.add_typer(ibkr_app, name="ibkr")


@app.callback()
def main(
    ctx: typer.Context,
    profile: Annotated[
        Profile | None,
        typer.Option("--profile", "-p", help="Which environment to run as."),
    ] = None,
) -> None:
    """Resolve configuration and start logging before any command runs.

    Args:
        ctx: Typer context. Receives the resolved `AppContext` as `ctx.obj`.
        profile: Overrides `NEUROTRADE_PROFILE`. Defaults to `research`, the
            only profile that cannot spend money.
    """
    settings = load_settings(profile)
    # LiveClock: a command invocation happens in real time. Replay commands
    # will build their own SimClock for the session they drive.
    run_id = configure(settings, LiveClock())
    ctx.obj = AppContext(settings=settings, run_id=run_id)


@app.command()
def version() -> None:
    """Print the package version.

    Example:
        $ neurotrade version
        0.0.0
    """
    typer.echo(__version__)


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Print the resolved configuration and its hash.

    Answers "what is this process actually running with" without attaching a
    debugger — which matters most when a profile file, a `.env` and an
    environment variable disagree.

    Example:
        $ neurotrade --profile paper config show
    """
    app_context: AppContext = ctx.obj
    typer.echo(describe(app_context.settings))
    typer.echo(f"{'config_hash':<20} {config_hash(app_context.settings)}")
    typer.echo(f"{'run_id':<20} {app_context.run_id}")


@config_app.command("hash")
def config_hash_command(ctx: typer.Context) -> None:
    """Print only the config hash, for scripting.

    The hash covers settings that change trading decisions, not where the
    process happens to run — so it is stable across machines and comparable
    against the hash recorded on any order (§6.3).

    Example:
        $ neurotrade --profile live config hash
        cfg_0355e7b4b9bef4d8
    """
    app_context: AppContext = ctx.obj
    typer.echo(config_hash(app_context.settings))


@app.command()
def replay(
    ctx: typer.Context,
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Trading day to replay, as YYYY-MM-DD."),
    ] = None,
    log_path: Annotated[
        Path | None,
        typer.Option("--log", "-l", help="Replay a specific log file instead."),
    ] = None,
) -> None:
    """Replay a recorded session and print its digest.

    The digest is a hash of every event dispatched, in order. Running this twice
    on the same log must print the same value — that is gate G1, and it is what
    makes "did that change alter behaviour?" answerable without reading logs.

    Give either `--session`, which looks under the configured data root, or
    `--log`, which takes a path directly. The second exists because a log worth
    replaying does not always live where this machine keeps its own: a shadow
    run, a session copied off the trading host, or the test fixture.

    The digest goes to stdout alone so it can be compared directly; the summary
    goes to stderr, which is why piping this gives one clean line.

    Example:
        $ neurotrade replay --session 2026-03-16
        $ neurotrade replay --log tests/fixtures/session.jsonl
    """
    app_context: AppContext = ctx.obj
    storage = app_context.settings.storage

    if (session is None) == (log_path is None):
        typer.echo("give exactly one of --session or --log", err=True)
        raise typer.Exit(code=2)

    path = log_path if log_path is not None else storage.session_log(str(session))

    if not path.exists():
        typer.echo(f"no recorded session at {path}", err=True)
        available = sorted(p.stem for p in storage.events_dir.glob("*.jsonl"))
        if available:
            typer.echo(f"recorded sessions: {', '.join(available)}", err=True)
        else:
            typer.echo(
                f"nothing recorded under {storage.events_dir} yet — "
                f"sessions are written by the live engine, which is Phase 3. "
                f"To try this now: --log tests/fixtures/session.jsonl",
                err=True,
            )
        raise typer.Exit(code=1)

    # A fresh SimClock per replay: its start seeds nothing here, but sharing one
    # across runs would let the first run's end position offset the second.
    result = ReplayEngine(EventStore(path), SimClock(0)).run(0, _FOREVER_NS)

    log.info(
        "replay_complete",
        source=str(path),
        digest=result.digest,
        events_read=result.events_read,
        events_dispatched=result.events_dispatched,
    )

    typer.echo(f"source    {path}", err=True)
    typer.echo(
        f"events    {result.events_read} read, {result.events_dispatched} dispatched", err=True
    )
    typer.echo(f"span      {result.span_ns / 60_000_000_000:.0f} minutes", err=True)
    typer.echo(result.digest)


@ibkr_app.command("check")
def ibkr_check(ctx: typer.Context) -> None:
    """Check IB Gateway is reachable and is the account we expect.

    Connecting proves less than it appears: the socket answers whether Gateway
    is logged into paper or live. This reports *which* account answered and
    whether the port dialled is a paper port, so a misconfiguration is visible
    before an order is placed rather than after.

    Exits non-zero when the connection is unusable or the account does not match
    configuration.

    Example:
        $ neurotrade --profile paper ibkr check
    """
    app_context: AppContext = ctx.obj
    settings = app_context.settings.ibkr

    async def run() -> None:
        connection = IbkrConnection(settings)
        try:
            probe = await connection.probe()
        finally:
            connection.disconnect()

        typer.echo(probe.describe(), err=True)
        log.info(
            "ibkr_probe",
            host=settings.host,
            port=settings.port,
            accounts=list(probe.accounts),
            server_version=probe.server_version,
            healthy=probe.is_healthy,
        )
        if not probe.is_healthy:
            typer.echo("unhealthy — see above", err=True)
            raise typer.Exit(code=1)
        typer.echo(probe.account or ",".join(probe.accounts))

    try:
        asyncio.run(run())
    except IbkrConnectionError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@ibkr_app.command("paper-smoke")
def ibkr_paper_smoke(ctx: typer.Context) -> None:
    """Gate G2: submit a paper order, see it acknowledged, cancel it.

    Places a buy limit far below the market on a liquid name, so it cannot fill.
    The point is to prove the order path works end to end — submission,
    acknowledgement, cancellation — not to acquire a position.

    Every event is written to the session log, so the run is replayable and the
    order record carries the config hash that produced it (§6.3).

    Refuses to run outside a paper port, which is the same structural guard the
    broker applies.

    Example:
        $ neurotrade --profile paper ibkr paper-smoke
    """
    app_context: AppContext = ctx.obj
    settings = app_context.settings

    if not settings.ibkr.is_paper_port:
        typer.echo(
            f"refusing: port {settings.ibkr.port} is a live port, and this places "
            f"a real order there",
            err=True,
        )
        raise typer.Exit(code=2)

    async def run() -> None:
        connection = IbkrConnection(settings.ibkr)
        clock = LiveClock()
        log_path = settings.storage.session_log(to_datetime(clock.now_ns()).strftime("%Y-%m-%d"))
        recorded: list[str] = []

        with EventStore(log_path) as store:

            def record(fill: Fill) -> None:
                """A fill is unexpected here — the limit cannot trade — but if
                one arrives it belongs in the log like any other event."""
                store.append(fill)
                recorded.append(fill.id.value)

            broker = IbkrBroker(
                connection,
                clock,
                allow_live_orders=settings.allow_live_orders,
                on_fill=record,
            )
            order = _smoke_order(clock.now_ns(), config_hash(settings))
            try:
                await broker.submit(order)
                store.append(order)
                typer.echo(f"submitted  {order.id}", err=True)

                status = await _await_status(
                    broker, order.id, {"Submitted", "PreSubmitted", "Cancelled"}
                )
                typer.echo(f"status     {status}", err=True)
                if status not in {"Submitted", "PreSubmitted"}:
                    typer.echo("not acknowledged", err=True)
                    raise typer.Exit(code=1)

                await broker.cancel(order.id)
                status = await _await_status(broker, order.id, {"Cancelled", "ApiCancelled"})
                typer.echo(f"cancelled  {status}", err=True)
            finally:
                connection.disconnect()

        typer.echo(f"recorded   {log_path}", err=True)
        log.info(
            "paper_smoke_complete",
            order_id=order.id.value,
            status=status,
            fills=len(recorded),
            log=str(log_path),
        )
        typer.echo(order.id.value)

    try:
        asyncio.run(run())
    except IbkrConnectionError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


def _smoke_order(now: int, fingerprint: str) -> Order:
    """One share of a liquid name, priced where it cannot fill."""
    symbol = Symbol("AAPL", Venue.NASDAQ)
    intent_id = IntentId.derive(
        strategy="paper_smoke",
        strategy_version="1.0.0",
        symbol=symbol,
        ts_event=now,
        seq=0,
    )
    return Order(
        id=OrderId.derive(intent_id=intent_id, ts_event=now),
        intent_id=intent_id,
        symbol=symbol,
        ts_event=now,
        ts_init=now,
        side=Side.BUY,
        quantity=Quantity(1),
        order_type=OrderType.LIMIT,
        limit_price=Price("1.00"),  # far below the market; cannot fill
        config_hash=fingerprint,
    )


async def _await_status(
    broker: IbkrBroker, order_id: OrderId, wanted: set[str], *, tries: int = 40
) -> str:
    """Poll until the order reaches one of `wanted`, or give up.

    Polling is right here and wrong in the trading loop: this is a one-shot
    probe with nothing else to do, where the live engine reacts to events.
    """
    trade = broker.working(order_id)
    if trade is None:
        return "unknown"
    for _ in range(tries):
        await asyncio.sleep(0.25)
        if trade.orderStatus.status in wanted:
            break
    return trade.orderStatus.status


if __name__ == "__main__":
    app()
