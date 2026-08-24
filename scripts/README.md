# scripts

One-off tools. Not part of the package, not imported by anything, and not
collected by the test runner.

| Script | What it does |
|---|---|
| `check_docs.py` | Checks the documentation still describes the code |
| `make_replay_fixture.py` | Rebuilds the committed replay fixture |

## check_docs.py

Run by `make docs-check`, which `make lint` includes, so stale documentation
fails the build like anything else.

It checks only what can be verified without judgement — whether a sentence is
*true* is not checkable, but whether the file it names exists is:

- a README naming a file that no longer exists, or was moved
- a documented `make` target that is not in the Makefile
- a package with code and no README
- a broken link between documents

All four had actually happened before it was written.

## make_replay_fixture.py

```bash
uv run python scripts/make_replay_fixture.py
```

Writes `tests/fixtures/session.jsonl` — a small recorded session that CI replays
to prove determinism against a real file rather than in-memory objects.

The **fixture is the artifact**; this script exists so it can be rebuilt if the
event encoding ever changes, and so the awkward cases it deliberately contains
are written down rather than inferred from the file: real intrabar range, a
ticker containing a dot, a Canadian listing, sub-penny prices, a zero-volume
minute, locked and crossed quotes, and a halt that leaves an irregular gap.

Randomness is seeded, so re-running produces byte-identical output. If you
regenerate it, the pinned digest in `tests/lab/test_replay.py` will need
updating — and that failing test is the point, not an inconvenience.
