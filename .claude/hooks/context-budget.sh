#!/usr/bin/env bash
# Stop + PostToolUse: notice when this session's context has grown expensive.
#
# Cost per request is proportional to context size, and every tool call is a
# request, so a long-lived window multiplies the price of every step it takes.
# Measured on this project: one session reached a 460k median context over 1,473
# requests and was 86% of all spend (see WORK/cost-and-delegation.doc.md).
#
# Registered on PostToolUse as well as Stop because a Stop hook only fires at the
# end of a turn — a long autonomous turn can run hundreds of tool calls past any
# threshold without one firing. That is how the session above got there.
#
# The companion hook, work-checkpoint.sh, triggers on work accumulated. This one
# triggers on context spent, because the two come apart.
#
# Fires once per 50k band per session, shared across both events, and escalates:
# a soft notice below 250k, a hard stop above.

payload=$(cat)
[[ $(jq -r '.stop_hook_active // false' <<<"$payload") == true ]] && exit 0
# A subagent's tool calls must not consume the main loop's warning.
[[ -n $(jq -r '.agent_id // empty' <<<"$payload") ]] && exit 0

transcript=$(jq -r '.transcript_path // empty' <<<"$payload")
[[ -n $transcript && -f $transcript ]] || exit 0
session=$(jq -r '.session_id // "unknown"' <<<"$payload")

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0

# Context size is the last assistant turn's input: cache reads plus whatever was
# newly written. tail rather than a full scan — the file grows all session.
ctx=$(tail -n 200 "$transcript" 2>/dev/null | jq -rs '
  [ .[]
    | select(.message.usage != null)
    | (.message.usage.cache_read_input_tokens // 0)
      + (.message.usage.cache_creation_input_tokens // 0)
      + (.message.usage.input_tokens // 0)
  ] | last // 0' 2>/dev/null)
[[ $ctx =~ ^[0-9]+$ ]] || exit 0

# Below this a session is cheap enough that clearing costs more than it saves:
# the re-read of WORK/ and the working set is itself a real expense.
(( ctx < 150000 )) && exit 0

band=$(( ctx / 50000 ))
marker="$root/.git/neurotrade-context-band-$session"
[[ -f $marker && $(cat "$marker" 2>/dev/null) == "$band" ]] && exit 0
echo "$band" > "$marker"

if (( ctx < 250000 )); then
  cat >&2 <<MSG
Context budget: ~${ctx} tokens per request. Finish the current commit's worth of work, then
checkpoint (work-journal / commit-handoff) and recommend /clear. Do not start a new task in this
window.
MSG
else
  cat >&2 <<MSG
Context budget: ~${ctx} tokens per request — every further call costs that again.
STOP now. Tell the user in two sentences what is unfinished and propose checkpoint + /clear.
Continue in this window only if the user explicitly says so.
MSG
fi
exit 2
