# NeuroTrade — one entry point for the whole monorepo.
#
# The repo will span Python, Go (api/) and TypeScript (ui/). Only Python exists
# today. The go-* and ts-* targets are here so the entry point does not change
# shape when those land in Phase 3: they report SKIP while the toolchain or the
# module is absent, and run for real the moment it appears.
#
# SKIP always means "nothing to run here", never "the command failed".

.DEFAULT_GOAL := help
SHELL := /bin/bash

PY := uv run

# Usage: $(call have,go) — true when the executable is on PATH.
have = command -v $(1) >/dev/null 2>&1

.PHONY: help doctor setup check fmt lint typecheck test clean show-config replay verify-replay ibkr-check paper-smoke backfill seed seed-fetch seed-ingest daily universe actions actions-check corpus-check docs-check \
        py-fmt py-lint py-typecheck py-test \
        go-fmt go-lint go-test \
        ts-fmt ts-lint ts-typecheck ts-test

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

doctor: ## Report which toolchains are installed
	@printf '%-8s' 'uv';   if $(call have,uv);   then uv --version;   else echo 'MISSING  → brew install uv'; fi
	@printf '%-8s' 'go';   if $(call have,go);   then go version;     else echo 'absent   (not needed until Phase 3)'; fi
	@printf '%-8s' 'pnpm'; if $(call have,pnpm); then pnpm --version; else echo 'absent   (not needed until Phase 3)'; fi

setup: ## Install dependencies for every toolchain present
	@if ! $(call have,uv); then echo 'uv is required: brew install uv'; exit 1; fi
	uv sync
	@if $(call have,go)   && [ -f go.work ];             then go work sync; else echo 'go: SKIP'; fi
	@if $(call have,pnpm) && [ -f pnpm-workspace.yaml ]; then pnpm install; else echo 'pnpm: SKIP'; fi
	$(PY) pre-commit install

# ── Aggregates ───────────────────────────────────────────────
check: lint typecheck test ## Full gate — run before declaring work done
	@echo '✓ check passed'

fmt: py-fmt go-fmt ts-fmt                ## Format every language
lint: py-lint go-lint ts-lint docs-check ## Lint every language and the docs
typecheck: py-typecheck ts-typecheck     ## Typecheck every language
test: py-test go-test ts-test            ## Test every language

# ── Documentation ────────────────────────────────────────────
# Docs drift because nothing fails when they go stale. This checks the parts
# that are mechanically verifiable: files a README names, `make` targets a doc
# claims, packages missing a README, and links between documents.
docs-check: ## Check the docs still describe the code
	$(PY) python scripts/check_docs.py

# ── Python ───────────────────────────────────────────────────
py-fmt:
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

py-lint:
	$(PY) ruff format --check src tests
	$(PY) ruff check src tests
	@if [ -f .importlinter ]; then $(PY) lint-imports; else echo 'import-linter: no contracts'; fi

py-typecheck:
	$(PY) mypy

py-test:
	$(PY) pytest

# ── Go — api/ gateway, Phase 3 ───────────────────────────────
go-fmt:
	@if $(call have,go) && [ -f api/go.mod ]; then cd api && gofmt -l -w .; else echo 'go-fmt: SKIP'; fi

go-lint:
	@if $(call have,go) && [ -f api/go.mod ]; then cd api && go vet ./...; else echo 'go-lint: SKIP'; fi

go-test:
	@if $(call have,go) && [ -f api/go.mod ]; then cd api && go test ./...; else echo 'go-test: SKIP'; fi

# ── TypeScript — ui/ dashboard, Phase 3 ──────────────────────
ts-fmt:
	@if $(call have,pnpm) && [ -f ui/package.json ]; then pnpm -C ui run format; else echo 'ts-fmt: SKIP'; fi

ts-lint:
	@if $(call have,pnpm) && [ -f ui/package.json ]; then pnpm -C ui run lint; else echo 'ts-lint: SKIP'; fi

ts-typecheck:
	@if $(call have,pnpm) && [ -f ui/package.json ]; then pnpm -C ui run typecheck; else echo 'ts-typecheck: SKIP'; fi

ts-test:
	@if $(call have,pnpm) && [ -f ui/package.json ]; then pnpm -C ui run test; else echo 'ts-test: SKIP'; fi

# ── Operations ───────────────────────────────────────────────
PROFILE ?= research

show-config: ## Print the resolved config and its hash. PROFILE=research|paper|live
	$(PY) neurotrade --profile $(PROFILE) config show

# LOG replays a file directly; SESSION looks under the configured data root.
# LOG defaults to the committed fixture, so `make replay` works on a fresh clone
# even though nothing writes session logs until the live engine lands.
LOG ?= tests/fixtures/session.jsonl

ibkr-check: ## Probe IB Gateway: reachable, and the account we expect
	$(PY) neurotrade --profile $(PROFILE) ibkr check

paper-smoke: ## Gate G2: submit a paper order, acknowledge, cancel
	$(PY) neurotrade --profile paper ibkr paper-smoke

# START has no default: a default would silently decide how much history the
# corpus holds. END defaults to yesterday inside the command; LIMIT and PASSES
# are optional.
backfill: ## Fill the corpus from IBKR. START=YYYY-MM-DD [END= LIMIT= PASSES=]
	@if [ -z "$(START)" ]; then echo 'START=YYYY-MM-DD is required'; exit 2; fi
	$(PY) neurotrade --profile $(PROFILE) ibkr backfill --start $(START) \
	  $(if $(END),--end $(END)) $(if $(LIMIT),--limit $(LIMIT)) $(if $(PASSES),--passes $(PASSES))

# Free vendor samples (§12.1 stage 2). `seed` does both halves; they are also
# separate targets because a fetch is a one-shot download that REFUSES to
# overwrite a snapshot already taken today, while an ingest is idempotent and
# worth re-running. SOURCE limits either to one vendor; SNAPSHOT ingests an
# older dated folder instead of the newest.
seed: seed-fetch seed-ingest ## Seed the corpus from free vendor samples. [SOURCE=]

seed-fetch: ## Download the vendor samples into a dated raw snapshot. [SOURCE=firstrate|kibot]
	$(PY) neurotrade --profile $(PROFILE) seed fetch $(if $(SOURCE),--source $(SOURCE))

seed-ingest: ## Normalise a fetched snapshot into derived/seed. [SOURCE= SNAPSHOT=YYYY-MM-DD]
	$(PY) neurotrade --profile $(PROFILE) seed ingest $(if $(SOURCE),--source $(SOURCE)) \
	  $(if $(SNAPSHOT),--snapshot $(SNAPSHOT))

# Daily bars from Yahoo Finance (§12.1 stage 3), including the `.TO` Canadian
# lines IBKR's crawler is slowest to reach. Unadjusted, and written to
# derived/daily/yfinance so they can never be mistaken for the minute corpus.
# START is required for the same reason `backfill` requires it: a default would
# silently decide how much history the corpus holds.
daily: ## Fill the daily-bar corpus from Yahoo. START=YYYY-MM-DD [END= LIMIT=]
	@if [ -z "$(START)" ]; then echo 'START=YYYY-MM-DD is required'; exit 2; fi
	$(PY) neurotrade --profile $(PROFILE) daily backfill --start $(START) \
	  $(if $(END),--end $(END)) $(if $(LIMIT),--limit $(LIMIT))

# Point-in-time membership over the daily corpus (§12.1 stage 3, second half).
# Reads derived/daily/yfinance and writes one file to derived/universe/yfinance.
# START is the first session to DECIDE, not the first session read: the screen
# reaches further back on its own for the trailing window.
universe: ## Build point-in-time universe membership. START=YYYY-MM-DD [END=]
	@if [ -z "$(START)" ]; then echo 'START=YYYY-MM-DD is required'; exit 2; fi
	$(PY) neurotrade --profile $(PROFILE) universe build --start $(START) \
	  $(if $(END),--end $(END))

# Corporate actions and the adjustment audit (§12.1 stage 5). `actions` fetches
# splits/dividends from Yahoo into derived/actions/yfinance; `actions-check`
# audits the daily corpus against them and exits non-zero on any gap no
# recorded action explains.
actions: ## Fetch splits and dividends from Yahoo. START=YYYY-MM-DD [END=]
	@if [ -z "$(START)" ]; then echo 'START=YYYY-MM-DD is required'; exit 2; fi
	$(PY) neurotrade --profile $(PROFILE) actions fetch --start $(START) \
	  $(if $(END),--end $(END))

actions-check: ## Audit the daily corpus for unexplained gaps. START=YYYY-MM-DD [END= THRESHOLD=]
	@if [ -z "$(START)" ]; then echo 'START=YYYY-MM-DD is required'; exit 2; fi
	$(PY) neurotrade --profile $(PROFILE) actions check --start $(START) \
	  $(if $(END),--end $(END)) $(if $(THRESHOLD),--threshold $(THRESHOLD))

# Corpus quality gate (§12.1 stage 5). Reports; never repairs. Exits non-zero
# on any finding, so it can gate a pipeline.
corpus-check: ## Audit the corpus for faults. START=YYYY-MM-DD [END= INTERVAL= SOURCE= LIMIT=]
	@if [ -z "$(START)" ]; then echo 'START=YYYY-MM-DD is required'; exit 2; fi
	$(PY) neurotrade --profile $(PROFILE) corpus check --start $(START) \
	  $(if $(END),--end $(END)) $(if $(INTERVAL),--interval $(INTERVAL)) \
	  $(if $(SOURCE),--source $(SOURCE)) $(if $(LIMIT),--limit $(LIMIT))

replay: ## Replay a session and print its digest. SESSION=YYYY-MM-DD or LOG=path
	@if [ -n "$(SESSION)" ]; then \
	   $(PY) neurotrade --profile $(PROFILE) replay --session $(SESSION); \
	 else \
	   $(PY) neurotrade --profile $(PROFILE) replay --log $(LOG); \
	 fi

verify-replay: ## Prove gate G1: replay twice, compare digests. SESSION= or LOG=
	@if [ -n "$(SESSION)" ]; then target="-s $(SESSION)"; else target="-l $(LOG)"; fi; \
	 first=$$($(PY) neurotrade --profile $(PROFILE) replay $$target 2>/dev/null); \
	 second=$$($(PY) neurotrade --profile $(PROFILE) replay $$target 2>/dev/null); \
	 echo "  run 1: $$first"; echo "  run 2: $$second"; \
	 if [ "$$first" = "$$second" ]; then echo '  ✓ deterministic'; \
	 else echo '  ✗ DIGESTS DIFFER — replay is not deterministic'; exit 1; fi

# ── Housekeeping ─────────────────────────────────────────────
clean: ## Remove build and tool caches. Never touches data/.
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
