---
name: work-journal
description: Use when a coherent body of work is complete and should be recorded in WORK/ before clearing context, or when the user asks to journal, checkpoint, or update WORK.
---

# Work journal

`WORK/` is the project's session-context ledger: a future session learns what was built by reading
~40 lines instead of re-deriving it from 15k lines of source. Journalling runs at the session's
peak context, so gather in slices and write from what you already know.

## Layout

```
WORK/INDEX.md                                  the map — injected at every session start
WORK/phase <#> - <title>/<subtitle>.doc.md     one coherent subsystem per file
```

Phase numbers and titles come from the roadmap table in `TRADER_PLAN.md` §13 — `grep -n` the
heading and `sed -n` that range; never read the spec whole.

## Gather — slices, not files

```bash
git log --reverse --format='%h %ad %s' --date=short <range>
git diff --stat <range>
grep -nE '^(class |def |    def [a-z])' <file> | grep -v 'def _'
```

Read the existing doc (if rewriting), `WORK/INDEX.md`, and the phase's `08-gotchas.doc.md`. Open
source only to confirm a public surface or a claim.

## Write

**Verify before you write.** Every claim in `WORK/` is read later as fact; a wrong doc is worse
than no doc because it is trusted. Never state a count, digest or test result not produced by a
command run this session.

Each `.doc.md` describes **what is true now**, not what changed. Rewrite in place; git holds the
history. In rough priority order:

1. **What exists** — modules, their responsibility, the public surface worth knowing.
2. **Why it is shaped that way** — the decision, and the alternative rejected.
3. **What is deliberately not done yet**, and the seam left for it.
4. **Traps** — anything that cost time once. These go in the phase's `08-gotchas.doc.md` and are
   the highest-value lines in the folder.

Leave out: file listings without explanation, line counts, changelogs, restated code, anything a
`grep` answers faster. Keep a doc under ~100 lines; more means two subsystems — split it.

## Finish

1. Write or update the `.doc.md`, and `08-gotchas.doc.md` if this commit's work produced traps.
2. Update `WORK/INDEX.md` — status table and doc table. **Injected into every session; keep it
   under ~45 lines.**
3. Run `make docs-check`; fix what it reports.
4. Tell the user in ≤10 lines which paths changed and the `docs-check` result, then use the
   `commit-handoff` skill covering code and `WORK/` together.
