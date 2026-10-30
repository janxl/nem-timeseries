# Thin wrapper over scripts/. Run `make` (or `make help`) for the list.
# Needs: bash, make, git, curl, python3 (3.11+). Run it from WSL2 (or Linux/macOS).
SHELL := /bin/bash

# The virtualenv lives in WSL's home, not in the repo: venvs on /mnt/c are very slow.
# Override with VENV=... ; every uv call below uses it.
VENV ?= $(HOME)/.venvs/nem-time-series
export UV_PROJECT_ENVIRONMENT := $(VENV)
# Cache and venv can be on different filesystems under WSL, where hardlinks fail.
export UV_LINK_MODE := copy

# How commands are run. Locally this is uv (the virtualenv above). Where the packages are already installed with
# pip, for example on Colab, use: make backtest RUN=   (empty, so the installed command is run directly)
RUN ?= uv run
export PATH := $(HOME)/.local/bin:$(PATH)

.DEFAULT_GOAL := help
.PHONY: help doctor setup backtest finetune report test

help:  ## show this list
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  make %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

doctor:  ## check this machine is ready (tools, disk, network) and say how to fix gaps
	@bash scripts/doctor.sh

# ---- first-time setup: make doctor; make setup

setup:  ## install uv if missing, then create the virtualenv (outside the repo) with all dependencies
	@command -v uv >/dev/null 2>&1 || { echo "installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv sync --extra models --extra dev --extra report
	@echo; echo "Ready. Virtualenv: $(VENV). Run commands with: uv run <cmd>"

# ---- backtest: forecast the next 48 half-hours from every midnight since 2025-01-01, score, save to results/
# Options: make backtest ARGS="--models naive_yesterday --start 2025-06-01 --force"

backtest:  ## run the rolling backtest for all registered models; writes results/<region>/ (metrics.md, .csv)
	$(RUN) nem-ts backtest $(ARGS)

# ---- fine-tune Chronos-2 (LoRA, price only) on 2023-24, saved to models/chronos2_ft; `make backtest` then scores it.
# Slow on CPU: use a GPU (the Colab notebook does). Quick local check: make finetune ARGS="--num-steps 20 --batch-size 8"

finetune:  ## fine-tune Chronos-2 on 2023-24 prices (LoRA) and save the checkpoint to models/
	$(RUN) nem-ts finetune $(ARGS)

# ---- Excel report of the daily forecasts: actual price vs P10/P50/P90, half-hour by half-hour (from the saved predictions)

report:  ## build results/<region>/daily_forecasts.xlsx: pick a day and see actual vs forecast (opens best in Excel)
	$(RUN) python scripts/build_daily_forecasts_xlsx.py $(ARGS)

test:  ## run the unit tests
	$(RUN) pytest -q
