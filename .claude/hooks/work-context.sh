#!/usr/bin/env bash
# SessionStart: inject the WORK ledger index.
#
# This is the token-saving read side. Every session begins knowing what has been
# built and which doc to open, without re-deriving it from the source tree or
# reading all of WORK/. The index is deliberately kept short for this reason.

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
idx="$root/WORK/INDEX.mem.md"
[[ -f $idx ]] || exit 0

echo "<work-ledger>"
echo "Project state from WORK/INDEX.mem.md. Open a specific .doc.md / .mem.md only when the task needs it."
cat "$idx"
echo "</work-ledger>"

# Model choice is the largest single cost lever and is invisible unless raised.
# Kept to two lines because this text is re-read on every request of the session.
echo "<model-discipline>"
echo "Default is sonnet. If this task is design, ambiguous debugging or Phase 1 validation, say so in your first reply and suggest /model opus NOW — the prompt cache is per-model, so switching mid-session re-writes the whole window. A bounded methodology question goes to quant-methodology-reviewer (opus) instead of switching. Once a plan is agreed, suggest /model sonnet, or /clear and execute on sonnet."
echo "</model-discipline>"
