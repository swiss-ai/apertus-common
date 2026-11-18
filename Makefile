SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show available targets
	@grep -E '^[a-z-]+:.*##' Makefile | awk 'BEGIN {FS = ":.*## "}; {printf "%-12s %s\n", $$1, $$2}'

.PHONY: install
install:  ## Install locked dependencies including dev tools
	uv sync --locked

.PHONY: format
format:  ## Auto-format and autofix source files
	uv run --locked ruff check src tests scripts examples --fix
	uv run --locked ruff format src tests scripts examples

.PHONY: lint
lint:  ## Check style without modifying files
	uv run --locked ruff check src tests scripts examples
	uv run --locked ruff format --check src tests scripts examples

.PHONY: typecheck
typecheck:  ## Strict mypy + pyright + API completeness
	uv run --locked mypy
	uv run --locked pyright
	uv run --locked pyright --verifytypes apertus_common --ignoreexternal --outputjson | uv run --locked python -c "import json,sys; s=json.load(sys.stdin)['typeCompleteness']['completenessScore']; print(f'completeness: {s:.1%}'); sys.exit(0 if s == 1.0 else 1)"

.PHONY: examples
examples:  ## Type-check and execute example notebooks
	uv run --locked python scripts/check_notebooks.py

.PHONY: rules
rules:  ## Print the conformance rule table
	@uv run --locked python scripts/rules.py

.PHONY: test
test:  ## Run the test suite
	uv run --locked pytest

.PHONY: all
all: lint typecheck examples test check-dist  ## Everything CI runs

.PHONY: check-artifact
check-artifact:  ## Verify a required real artifact (set APERTUS_ENCODING)
	uv run --locked python scripts/check_artifact.py "$(APERTUS_ENCODING)"

.PHONY: harness
harness:  ## Run exact-token continuation example (set APERTUS_ENCODING)
	uv run --locked python examples/harness_roundtrip.py "$(APERTUS_ENCODING)"

.PHONY: bench
bench:  ## Measure parser throughput/memory; optionally set APERTUS_ENCODING
	uv run --locked python scripts/bench_parse.py $(if $(APERTUS_ENCODING),--artifact "$(APERTUS_ENCODING)")

.PHONY: build
build:  ## Build wheel and source distribution
	uv build

.PHONY: check-dist
check-dist: build  ## Verify isolated wheel installs, base and HF extra
	uv run --locked python scripts/check_distribution.py

.PHONY: prepare
prepare:  ## Run the weighted preparation example (set APERTUS_ENCODING)
	uv run --locked python examples/prepare_data.py "$(APERTUS_ENCODING)"

.PHONY: bench-prepare
bench-prepare:  ## Measure data preparation (set APERTUS_ENCODING)
	uv run --locked python scripts/bench_prepare.py --artifact "$(APERTUS_ENCODING)"
