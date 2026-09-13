# adapters

Everything that touches the outside world.

Each adapter implements an interface from `core/ports.py`. The trading logic
depends on the interface, never on the adapter, so swapping the broker or moving
the data to object storage is a new file here rather than a change anywhere else.

Conformance is **structural** — an adapter satisfies a port by having the right
methods, with no import from `core.ports` and nothing to subclass. That keeps the
dependency arrow pointing one way: adapters know about core, core knows nothing
about adapters.

## Subfolders

| Folder | Status |
|---|---|
| `storage/` | Built. The market-data corpus and the event log |
| `calendar/` | Built. Which days each venue traded, and between which times |
| `universe/` | Built. Which instruments are in scope, read from `config/universe.yaml` |
| `ibkr/` | Built. Connection, historical bars, request pacing, order placement. The backfill crawler that drives it is still open |
| `feeds/` | Not built. yfinance and the free sample sources |
| `notify/` | Not built. Alerting |

The empty ones are absent on purpose: a folder appears when there is real code
to put in it.

## Testing adapters

Anything that reaches a network or a disk gets a fake in the tests, which is
straightforward precisely because the ports are small. See `tests/core/test_ports.py`
for in-memory implementations of every port.
