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
