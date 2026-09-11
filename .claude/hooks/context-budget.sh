#!/usr/bin/env bash
# Stop: notice when this session's context has grown expensive.
#
# Cost per request is proportional to context size, and every tool call is a
# request, so a long-lived window multiplies the price of every step it takes.
# Measured on this project: a window left open for a month reached 509k tokens
# per request and accounted for 80% of a day's spend, while the session that
# replaced it averaged 146k for comparable work.
#
# The companion hook, work-checkpoint.sh, triggers on work accumulated. This one
# triggers on context spent, because the two come apart: a long debugging stretch
# can burn a window without producing a single commit.
#
# Fires once per 100k band per session, so it escalates rather than nags.

payload=$(cat)
[[ $(jq -r '.stop_hook_active // false' <<<"$payload") == true ]] && exit 0

transcript=$(jq -r '.transcript_path // empty' <<<"$payload")
[[ -n $transcript && -f $transcript ]] || exit 0
session=$(jq -r '.session_id // "unknown"' <<<"$payload")

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0

# Context size is the last assistant turn's input: cache reads plus whatever was
# newly written. tail rather than a full scan — the file grows all session.
ctx=$(tail -n 400 "$transcript" 2>/dev/null | jq -rs '
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

band=$(( ctx / 100000 ))
marker="$root/.git/neurotrade-context-band-$session"
[[ -f $marker && $(cat "$marker" 2>/dev/null) == "$band" ]] && exit 0
echo "$band" > "$marker"

cat >&2 <<MSG
Context budget: this session is now carrying roughly ${ctx} tokens per request.

Every tool call from here costs that much again. Tell the user, in two sentences:
  - that the window has grown expensive and what is still unfinished in it
  - whether the right move is to checkpoint to WORK/ and clear, or to push on
    because the current task is nearly done

Then STOP and wait. Clearing mid-task is worse than finishing it; say which
applies rather than recommending a clear reflexively.
MSG
exit 2
