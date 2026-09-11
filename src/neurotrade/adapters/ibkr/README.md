# adapters/ibkr

Talks to Interactive Brokers through IB Gateway.

## Files

| File | What it does |
|---|---|
| `connection.py` | Opens the connection and checks it is the account we meant |
| `market_data.py` | Fetches historical bars and converts them to `Bar` events |
| `pacing.py` | Keeps historical requests under IBKR's rate limit |
| `broker.py` | Places and cancels orders; turns executions into `Fill` records |

The backfill crawler that drives `market_data.py` across a universe of symbols
is not here yet — this layer fetches what it is asked for, nothing decides what
to ask for.

## Connecting proves less than it looks

The socket answers whether Gateway is logged into a **paper** account or a
**live** one. So a successful connection says almost nothing on its own, and the
probe checks identity as well as reachability: which account answered, whether
the port is a paper port, and whether the account matches configuration.

The failure is asymmetric, which is why it is worth the extra check. Connecting
to paper when you wanted live wastes a session. Connecting to live when you
wanted paper spends money.

```bash
make ibkr-check PROFILE=paper
```

## Ports differ by one digit

| Port | Application | Mode |
|---|---|---|
| 4001 | Gateway | **live** |
| 4002 | Gateway | paper |
| 7496 | TWS | **live** |
| 7497 | TWS | paper |

The default in `IbkrSettings` is 4002, so a misconfiguration fails to connect
rather than trading real money.

## Gateway setup

In Gateway: **Configure → Settings → API → Settings**

- **Read-Only API** must be **off** — it is on by default and silently blocks
  order placement while everything else works
- **Socket port** 4002
- There is no "Enable ActiveX and Socket Clients" checkbox in Gateway; that is a
  TWS setting, because TWS is a GUI app where API access is opt-in. Gateway
  exists to serve the API, so the socket is always on.

**One session per username.** Logging into Client Portal with the username
Gateway is using stops Gateway reconnecting after the next server reset. A
paper username is already distinct from the live one, so this only bites when
the crawler runs against the live account.

**Disconnects are routine.** Gateway restarts daily to reload contract
definitions, and the weekly server reset needs re-authentication. Code here
treats a dropped connection as an ordinary event, not a fault.

## Testing

Most tests use a fake client and run anywhere — a suite that needs a broker
logged in is a suite that stops being run. The few that need a real Gateway are
marked and excluded by default:

```bash
uv run pytest -m ibkr     # opt in, with Gateway logged in
make paper-smoke          # gate G2: submit a paper order, acknowledge, cancel
```

`paper-smoke` is the end-to-end proof: the order it places is written to the
event log with its config hash, and that log replays deterministically.
