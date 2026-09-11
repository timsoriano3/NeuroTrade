---
name: work-journal
description: Use when a coherent body of work is complete and should be recorded in WORK/ before clearing context, or when the user asks to journal, checkpoint, or update WORK.
---

# Work journal

`WORK/` is the project's session-context ledger. It exists so a future session can learn what
was built by reading ~40 lines instead of re-deriving it from 15k lines of source.

## Layout

```
WORK/INDEX.md                                  the map — injected at every session start
WORK/phase <#> - <title>/<subtitle>.doc.md     one coherent subsystem per file
```

Phase numbers and titles come from the roadmap table in `TRADER_PLAN.md` §13. Do not invent a
phase; grep for `## 13. Phased Roadmap` and read that range.

## Writing a doc

**Verify before you write.** Every claim in `WORK/` gets read by a future session as fact. Run
the command, read the file, check the API surface. A wrong doc is worse than no doc because it
is trusted.

Useful, cheap ways to establish ground truth:
```bash
git log --reverse --format='%h %ad %s' --date=short
grep -nE '^(class |def |    def [a-z])' <file> | grep -v 'def _'
make check          # and paste what it actually printed
```

Each `.doc.md` describes **what is true now**, not what changed. Git holds the history. When
something moves, rewrite the doc in place.

Content, in rough priority order:
1. **What exists** — modules, their responsibility, the public surface worth knowing.
2. **Why it is shaped that way** — the decision and the alternative rejected.
3. **What is deliberately not done yet**, and the seam left for it.
4. **Traps** — anything that cost time once. These belong in the phase's `08-gotchas.doc.md`
   equivalent and are the single highest-value thing in the whole folder.

Do not include: file listings without explanation, line counts, changelogs, restated code, or
anything a `grep` would answer faster.

Keep each doc under ~100 lines. If it needs more, it is two subsystems — split it.

## Finishing

1. Write or update the `.doc.md`.
2. Update `WORK/INDEX.md` — the status table and the doc table. **The index is injected into
   every session, so keep it under ~45 lines.**
3. Run `make docs-check`.
4. Write the commit handoff (see `CLAUDE.md`) and stop. The user commits.
5. Then suggest clearing context.
