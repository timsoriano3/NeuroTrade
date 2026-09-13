---
name: work-journal
description: Use when a coherent body of work is complete and should be recorded in WORK/ before clearing context, or when the user asks to journal, checkpoint, or update WORK.
---

# Work journal

Journalling runs at the end of a session, which is its peak context — writing the doc in the
main loop re-reads all of it on every step. Dispatch the `work-journal` agent (sonnet) instead;
it reads git and the source itself.

## Dispatch

Send a brief of **≤25 lines** — only what git and the source cannot show:

- **Doc path** — `WORK/phase <#> - <title>/<subtitle>.doc.md`, as agreed with the user.
- **Commit range** — e.g. `abc123..HEAD`, or the files if uncommitted.
- **Why** — the decisions that shaped it and the alternatives rejected.
- **Deferred** — what is deliberately not done, and the seam left.
- **Traps** — anything that cost time this session, with its fix.
- **Status change** — what moves in the `INDEX.md` status table, if anything.

Do not paste code, diffs or doc text into the brief.

## Then

1. Relay the agent's ≤10-line return to the user.
2. Use the `commit-handoff` skill for the WORK/ change.
3. Stop; after the user commits, recommend `/clear`.
