# Config, logging, CLI

## `config.py` — twelve-factor settings

Pydantic-settings. Precedence, lowest to highest:

```
config/base.yaml → config/<profile>.yaml → .env → real environment variables
```

Env vars use `NEUROTRADE_` prefix and `__` for nesting:
`NEUROTRADE_STORAGE__DATA_ROOT`, `NEUROTRADE_IBKR__PORT`.

`Profile` is `research` | `paper` | `live`. **`research` is the default** — the only profile
that cannot spend money if the environment forgets to say which one it wants.

| Profile | Data | Fills | `allow_live_orders` | IBKR port |
|---|---|---|---|---|
| research | historical | modelled | false | — |
| paper | live | simulated against the real book | false | 4002 |
| live | live | real | true | 4001 |

`paper.yaml` and `live.yaml` are deliberately near-identical apart from `allow_live_orders`
and the port. A paper profile that diverges from live is a paper profile that proves nothing.
`paper.yaml` pins `account: DUT108414`, checked on every connect, so a Gateway logged into the
wrong account fails at `ibkr check` rather than at a surprising fill.

### The config hash

`config_hash(settings)` hashes only **behavioural** values. Fields marked
`ENVIRONMENTAL` — data root, log level, host — are excluded, because moving the corpus to
another disk must not make two otherwise identical runs look different. Every trade record
carries this hash, which is what makes a trade reconstructable months later.

`describe(settings)` renders the resolved config; `_canonical` gives the stable ordering the
hash depends on.

## `logs.py` — structured logging

structlog. JSON in `paper`/`live`, human-readable in `research`.
`configure()`, `bind(**values)`, `clear_context()`, `get_logger(name)`.

`_CurrentStderr` is a proxy object, not `sys.stderr` directly: structlog caches its logger
factory on first use, so binding the real stream at import time breaks any test or tool that
later replaces `sys.stderr`. See `08-gotchas`.

## `cli.py` — the operator surface

Typer. `AppContext` carries the resolved settings to every subcommand.

```
neurotrade --profile <p> version
                        config show          # resolved settings
                        config hash          # behavioural hash only
                        replay --log <path> | --session <YYYY-MM-DD>
                        ibkr check           # reachable? right account?
                        ibkr paper-smoke     # gate G2: submit, ack, cancel
```

Exposed as `[project.scripts] neurotrade`, and wrapped by `make show-config`, `make replay`,
`make verify-replay`, `make ibkr-check`, `make paper-smoke`.
