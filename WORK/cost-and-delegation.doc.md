# Why the working agreements say what they say

The rules in `CLAUDE.md` about delegation, batching, model tiers and clearing are not
preferences. This is the evidence behind them, kept here so `CLAUDE.md` can state the rule
and stop — it is injected into every request, and an argument only needs making once.

## How a session is billed

Cost per request is proportional to context size, and **every tool call is a request**. So
session cost is approximately `context size × number of requests`.

Two consequences, both counter-intuitive:

- A tool call that returns a lot is charged **twice** — once for the call, and again on
  every later request, because its output now sits in context permanently.
- A subagent's context is discarded; only its return value arrives. Delegation is therefore
  a **compression** mechanism, not merely a parallelism one.

## Measurement one — 2026-09-10

Cache reads outnumbered output tokens **571 : 1**, making re-read of context roughly **69%**
of spend and generated output about **6%**. One session ran **744 main-loop tool calls and 0
delegated ones**, and accounted for essentially a whole day's bill. That is what the hard
triggers in `CLAUDE.md` exist to prevent.

## Measurement two — 2026-09-11

Taken from the session transcripts in `~/.claude/projects/`, filtered to a 24-hour window by
message timestamp. Delegation was in force by this point.

| Session | Requests | Input tokens | Share | Avg context per request |
|---|---|---|---|---|
| Opened 2026-08-08, ran a month | 392 | 199.7M | 80% | 509k |
| Its replacement after a compaction | 339 | 49.5M | 20% | 146k |

Both windows did comparable work. The month-old one cost 3.5× more per step because it
carried a month of history; its largest single request read 968,738 tokens. It had been
compacted once in a month.

The usage dashboard reported "99% of your usage came from subagent-heavy sessions" while its
own per-agent breakdown summed to **5%**. That line labels the session, it does not attribute
the cost. The agents were not the expense; the window they ran in was.

**Conclusion: delegation is working and should not be cut back.** The remaining levers are
the length of a window and the model running it.

## Measurement three — 2026-09-12

All 13 project transcripts, whole history, weighted read 0.1× / create 1.25× / output 5×.
2,140 main-loop requests, every one `claude-opus-5`: **65% cache read, 25% cache create, 10%
output, ~0% fresh input.** The month-old session from measurement two was **86% of all measured
spend** — 1,473 requests, median context 460k, 33% of requests above 700k — and spawned only 6
subagents. 119 of its 778 tool calls were `TaskCreate`/`TaskUpdate`. A fresh session costs
29–57k before any work. Subagent transcripts are not stored under `~/.claude/projects/`, so the
dashboard's per-agent shares (test-author 15%, invariant-auditor 5%) could not be audited.

Changes made in response: sonnet as the user-scope default model; `context-budget.sh` also on
PostToolUse with 50k bands and a hard stop at 250k; task tools denied; `work-journal` and
`commit-handoff` moved into sonnet agents so they no longer run at peak context; `test-author`
capped at its own file and two fix rounds. Agents had `allowed-tools:` in frontmatter, which
subagents ignore — the key is `tools:` — so every agent was carrying every tool schema.
Downgrading agent tiers was rejected: three are already haiku, and the sonnet ones are
judgement work whose total share is small.

## What follows from this

- **Hard triggers, not judgement.** A rule that says "delegate when it seems worth it" loses
  to the pull of just reading the file. The table in `CLAUDE.md` names the triggers.
- **Batching.** N sequential independent calls are N full-context requests; the same N issued
  in one message is one.
- **Model tiers.** Cache read is most of the bill, so the model running the conversation sets
  the rate on nearly all of it. Sonnet for execution is roughly a fifth the cost at no useful
  loss. Opus is reserved for work whose failure mode is silent: §17 names backtest overfitting
  as the primary project risk, and a clean equity curve that is wrong is worth the expensive
  tier. Locating a file is not.
- **Clear between tasks.** A compaction resets the floor and the window regrows from there.
  Clearing after a checkpoint does not. One commit per window.

## The hooks that enforce it

| Hook | Event | Fires when |
|---|---|---|
| `work-context.sh` | SessionStart | always — injects `INDEX.md` and when to switch to opus |
| `work-checkpoint.sh` | Stop | 4 commits or ~600 inserted lines since `WORK/` was written |
| `context-budget.sh` | Stop, PostToolUse | context passes 150k, then once per 50k band; hard stop from 250k |

Both exit 2, which feeds their text back to Claude rather than to the user, and
both instruct it to propose and then wait. They trigger on different things on purpose: work
accumulated and context spent come apart, since a long debugging stretch can burn a window
without producing a single commit.

## Two smaller findings

**Unused plugins cost about 5,600 tokens of skill descriptions on every request**, roughly 4%
of a day's cache reads. `huggingface-skills` and `postman` are disabled in
`.claude/settings.json` for this repo, and since 2026-09-12 also at user scope along with
`github` (whose MCP server was failing auth on every session start). JSON
carries no comments, which is why the reason is recorded here.

**MCP tool schemas are already deferred** by the harness — only names are loaded until a tool
is actually fetched — so the Notion, Drive and context7 servers are not a meaningful per-request
cost and were checked rather than assumed.

## A mistake worth not repeating

During the calendar work, `library-researcher` was dispatched to establish the
`exchange_calendars` API *and* the installed package was probed inline four times. The probes
were right — they are where the pinned fixtures came from — and the agent was then redundant.
Decide which of the two answers a question before spending both.
