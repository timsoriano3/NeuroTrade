#!/usr/bin/env bash
# Stop: notice when enough work has piled up to be worth journalling.
#
# The point is to catch the moment BEFORE the context window forces a compaction,
# because a compaction loses detail a WORK file would have preserved cheaply.
# Judgement is unreliable here, so the trigger is mechanical.
#
# WORK/ is gitignored, so "when was it last written" cannot come from git history
# any more — it comes from the newest mtime under WORK/. That is also more honest:
# it measures when the ledger was actually updated, not when a commit happened to
# carry it.
#
# Fires at most once per HEAD, so declining does not nag on every turn — it goes
# quiet until the next commit.

payload=$(cat)
[[ $(jq -r '.stop_hook_active // false' <<<"$payload") == true ]] && exit 0

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0
[[ -d WORK ]] || exit 0

# Newest mtime under WORK/. BSD stat (darwin); GNU falls back to -c %Y.
written=$(find WORK -type f -name '*.md' -exec stat -f %m {} + 2>/dev/null \
          || find WORK -type f -name '*.md' -exec stat -c %Y {} + 2>/dev/null)
written=$(printf '%s\n' "$written" | sort -rn | head -1)
[[ -n $written ]] || exit 0

head=$(git rev-parse HEAD 2>/dev/null) || exit 0
marker=".git/neurotrade-checkpoint-nag"
[[ -f $marker && $(cat "$marker" 2>/dev/null) == "$head" ]] && exit 0

commits=$(git rev-list --count --since="@$written" HEAD 2>/dev/null || echo 0)
(( commits == 0 )) && exit 0
lines=$(git log --since="@$written" --shortstat --format=%h 2>/dev/null \
        | grep -oE '[0-9]+ insertion' | grep -oE '[0-9]+' \
        | awk '{ total += $1 } END { print total + 0 }')
lines=${lines:-0}

# 4 commits or ~600 inserted lines. Roughly one coherent subsystem — the unit a
# .doc.md should describe.
if (( commits >= 4 || lines >= 600 )); then
  echo "$head" > "$marker"
  cat >&2 <<MSG
WORK checkpoint due: $commits commit(s) and $lines inserted line(s) since WORK/ was last written.

Working files are GATED (CLAUDE.md, Working files): do not write them now. Put the findings in
chat, then ask, verbatim:

  "Have you reviewed the code and want to update the working files?"

Write only on an explicit yes, to WORK/phase <#> - <title>/<subtitle>.doc.md plus the index.
The exemption still stands: a trap that would otherwise be lost goes in the phase's
08-gotchas.mem.md without asking — say in one line that you did.
MSG
  exit 2
fi
exit 0
