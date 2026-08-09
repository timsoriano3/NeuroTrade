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

.PHONY: help doctor setup check fmt lint typecheck test clean \
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

# ── Aggregates ───────────────────────────────────────────────
check: lint typecheck test ## Full gate — run before declaring work done
	@echo '✓ check passed'

fmt: py-fmt go-fmt ts-fmt                ## Format every language
lint: py-lint go-lint ts-lint            ## Lint every language
typecheck: py-typecheck ts-typecheck     ## Typecheck every language
test: py-test go-test ts-test            ## Test every language

# ── Python ───────────────────────────────────────────────────
py-fmt:
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

py-lint:
	$(PY) ruff format --check src tests
	$(PY) ruff check src tests

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

# ── Housekeeping ─────────────────────────────────────────────
clean: ## Remove build and tool caches. Never touches data/.
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
