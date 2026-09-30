# config

Settings for the three environments the system runs as.

| Profile | Data | Fills | Money |
|---|---|---|---|
| `research` | historical | modelled | none |
| `paper` | live | simulated against the real book | none |
| `live` | live | real | real |

`base.yaml` holds what they share; each profile file holds only what makes it
different. `paper` and `live` are deliberately near-identical — paper exists to
be evidence about live, and every difference between them weakens that.

## One file here is data, not settings

`universe.yaml` lists the instruments the backfill crawler walks. It is read by
`adapters/universe`, not by `Settings`, and is deliberately **outside the config
hash** — it grows from a few dozen tickers to a few thousand, and folding it in
would churn the fingerprint stamped on every trade record each time a name was
added. A run records `Universe.digest` instead.

Quote any ticker YAML would read as something else: `ON`, `NO` and `OFF` parse
as booleans, and `ON` is a real NASDAQ listing.

`sectors.yaml` says which sector proxy each of those instruments belonged to,
**on a given date**, and which index its venue regresses on. Read by
`adapters/universe/sector_file.py` into `core.sectors.SectorMap`, and outside
the config hash for the same reason. It is what §5.4's relative strength and
§5.5's residual reversion residualise against; without it a residual still
carries sector beta, so a day energy runs reads as alpha on every energy name at
once.

Two things about it are worth knowing before quoting a result built on it. It is
**dated on purpose** — GICS moved the payment networks out of technology part
way through the seed window, and V and MA carry two rows each because of it. And
every assignment with no `from` is present-day classification applied backwards,
which is a real bias in the flattering direction; `SectorMap.unknown_starts`
counts them. The file's own header says both at length.

## Settings live in files; machine details live in the environment

Anything specific to one computer — where the data directory is, credentials —
belongs in an environment variable, not in these files. These are committed, so
they must be true for anyone who clones the repo.

Environment variables override files:

```bash
NEUROTRADE_LOG_LEVEL=DEBUG
NEUROTRADE_STORAGE__DATA_ROOT=/Volumes/nvme/neurotrade   # double underscore nests
```

That ordering is what makes moving to another machine a deployment change rather
than a code change. Copy `.env.example` to `.env` for local overrides.

## Seeing what is actually in effect

```bash
make show-config PROFILE=paper
```

Prints the resolved settings and a fingerprint of them. That fingerprint is
stamped on every order, so any trade can be traced back to the exact settings
that produced it.

## One setting is a safety mechanism

`allow_live_orders` is false everywhere except `live`. It is structural, not a
preference: a research process should be *unable* to place a real order even if
something is wired up wrong.
