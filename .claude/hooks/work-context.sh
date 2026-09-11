#!/usr/bin/env bash
# SessionStart: inject the WORK ledger index.
#
# This is the token-saving read side. Every session begins knowing what has been
# built and which doc to open, without re-deriving it from the source tree or
# reading all of WORK/. The index is deliberately kept short for this reason.

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
idx="$root/WORK/INDEX.md"
[[ -f $idx ]] || exit 0

echo "<work-ledger>"
echo "Project state from WORK/INDEX.md. Open a specific .doc.md only when the task needs it."
cat "$idx"
echo "</work-ledger>"

# Model choice is the largest single cost lever and is invisible unless raised.
# Kept to two lines because this text is re-read on every request of the session.
echo "<model-discipline>"
echo "Execution work (writing modules, wiring config, running gates, applying an agreed plan) belongs on sonnet. Opus is for design, ambiguous debugging and Phase 1 validation. If this session is execution, say so in your first reply and suggest /model sonnet."
echo "</model-discipline>"
