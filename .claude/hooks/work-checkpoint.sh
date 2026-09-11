#!/usr/bin/env bash
# Stop: notice when enough work has piled up to be worth journalling.
#
# The point is to catch the moment BEFORE the context window forces a compaction,
# because a compaction loses detail that a WORK doc would have preserved cheaply.
# Judgement is unreliable here, so the trigger is mechanical: commits and lines
# landed since WORK/ was last written.
#
# Fires at most once per HEAD, so declining does not produce a nag on every turn —
# it goes quiet until the next commit.

payload=$(cat)
[[ $(jq -r '.stop_hook_active // false' <<<"$payload") == true ]] && exit 0

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0

# Nothing to compare against until WORK/ is committed at least once.
git ls-files --error-unmatch WORK/INDEX.md >/dev/null 2>&1 || exit 0
last=$(git log -1 --format=%H -- WORK/ 2>/dev/null)
[[ -n $last ]] || exit 0

head=$(git rev-parse HEAD)
[[ $last == "$head" ]] && exit 0

marker=".git/neurotrade-checkpoint-nag"
[[ -f $marker && $(cat "$marker" 2>/dev/null) == "$head" ]] && exit 0

commits=$(git rev-list --count "$last..$head" 2>/dev/null || echo 0)
lines=$(git diff --shortstat "$last..$head" 2>/dev/null \
        | grep -oE '[0-9]+ insertion' | grep -oE '[0-9]+' || echo 0)
lines=${lines:-0}

# 4 commits or ~600 inserted lines. Roughly one coherent subsystem — the unit a
# .doc.md should describe.
if (( commits >= 4 || lines >= 600 )); then
  echo "$head" > "$marker"
  cat >&2 <<MSG
WORK checkpoint due: $commits commit(s) and $lines inserted line(s) since WORK/ was last written.

Tell the user, in one short paragraph:
  - what coherent body of work has accumulated (name it, do not list commits)
  - the exact path you propose writing: WORK/phase <#> - <title>/<subtitle>.doc.md
  - that you recommend writing it and then clearing context before continuing
Then STOP and wait. Do not write the file until they agree.
MSG
  exit 2
fi
exit 0
