# tests

Mirrors `src/neurotrade/`. Docstring examples in the source are also run as
tests, so an example that drifts out of date fails the build.

```bash
make test                              # everything
uv run pytest tests/core -q            # one area
uv run pytest path/to/test.py::name    # one test
```

## What is worth testing here

This codebase leans toward tests that pin down a **property**, rather than
checking a function returns what it returned yesterday. The ones that matter
most:

- **Replay determinism** — the same inputs produce byte-identical output.
- **Exactness** — prices survive a round trip through disk unchanged.
- **Refusal** — bad input raises instead of being quietly accepted. Most of the
  domain tests are about what is *rejected*.
- **Architecture** — `core` imports nothing else; nothing reads the system clock
  outside one file. These are enforced by tests that read the source.

Where a test protects against a specific mistake, its docstring says which
mistake. That is usually more useful than the assertion.

## Fakes, not mocks

Ports are small enough that in-memory implementations are a few lines each — see
`tests/core/test_ports.py`. Nothing in the suite touches a network or a real
broker.
