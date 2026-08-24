# core

The vocabulary everything else is written in.

`core` **imports nothing else in this repo** — not configuration, not storage,
not adapters. That is what lets the trading logic be tested with no broker, no
database and no network, and it is checked by a test rather than left to
goodwill.

## Files

| File | What it holds |
|---|---|
| `types.py` | Prices, quantities, money, instruments. Exact arithmetic — never floats |
| `clock.py` | The single source of time. Real in production, simulated in replay |
| `events.py` | What the market did: bars, quotes, trade prints, session changes, halts |
| `ids.py` | Identifiers, derived from content so a replay reproduces them |
| `codec.py` | Turning events into text and back, without losing precision |
| `intent.py` | What a strategy proposes: a side and where the idea is wrong |
| `orders.py` | What we sent to the broker, and what came back |
| `position.py` | What the fills add up to — size, cost basis, profit and loss |
| `ports.py` | The interfaces to the outside world. Adapters implement these |
| `registry.py` | Plugin registration by name and version |

## Reading order

Start with `types.py`, then `clock.py`. Those two explain most of the decisions
in the rest — why money is exact, and why nothing reads the system clock.

Then `events.py` for what comes in, `intent.py` → `orders.py` → `position.py`
for what goes out, and `ports.py` for the boundary between the two.

`codec.py` sits here rather than beside the storage code that writes it, because
the research lab needs the same encoding to compute replay digests — and a
shared need between two layers belongs below both.

## Rules that apply here

- **No wall clock.** Time comes from the `Clock` passed in. A test enforces it.
- **Money is exact**, features are floats. Building a `Price` from a float raises.
- **Everything is immutable.** Events are facts about the past; a component that
  could rewrite one could rewrite history between backtest and live.
