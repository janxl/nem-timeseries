# nem-timeseries

Forecasting Victorian (VIC1) electricity prices a day ahead with Amazon's
[Chronos-2](https://huggingface.co/amazon/chronos-2) time-series foundation model, compared against baselines,
then fine-tuned. Everything here runs from one command per step and can be reproduced on a laptop or on Google Colab.

> **Results and discussion: [RESULTS.md](RESULTS.md).** This README is only about installing and running the project.

**Public data only.** Prices come from AEMO's public NEM data. No proprietary data is used or included
(see [datasets/README.md](datasets/README.md) for the source, attribution and how to rebuild it yourself).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janxl/nem-timeseries/blob/main/notebooks/colab.ipynb)

## The task in one paragraph

Every day at midnight (NEM time), predict the price for each of the next 48 half-hours (24 hours), with a P10, P50 (median)
and P90 band. The test period is every midnight from 1 Jan 2025 to 25 Oct 2026 (635 usable days); training data, where
a model needs it, is 2023 and 2024 only. Models: seasonal naive, LightGBM, and Chronos-2 zero-shot and fine-tuned.

## What you can do

| I want to... | Run | Takes |
|---|---|---|
| Reproduce the results table | `make backtest` | seconds (reuses the saved predictions) |
| Recompute every forecast from scratch | `make backtest ARGS="--force"` | about 1 to 2 hours on a laptop CPU |
| Re-run the fine-tuning | `make finetune`, then `make backtest` | about 5 minutes on a Colab GPU, hours on a CPU |
| Do all of it on a free GPU | open the Colab notebook (badge above) | roughly 15 to 30 minutes |

## Requirements

- **Linux, macOS, or Windows with WSL2.** On Windows run everything inside WSL2.
- **`make`, `git`, `curl` and Python 3.11 or newer.** `uv` (the installer) is fetched automatically by `make setup`.
- **No GPU needed.** A CUDA GPU is used automatically when present (the same code runs on both).
- **Internet** the first time: it downloads the Chronos-2 weights (a few hundred MB, Apache-2.0) from Hugging Face.
- About 10 GB of free disk space.

## Quick start

```bash
git clone https://github.com/janxl/nem-timeseries.git
cd nem-timeseries

make doctor     # checks your machine (tools, disk, network) and says how to fix anything missing
make setup      # creates the virtual environment and installs everything (CPU-only PyTorch)
make test       # 43 unit tests, no model downloads needed
make backtest   # reproduces the results table from the saved predictions
```

The table is written to `results/VIC1/metrics_vs_chronos2_zs.md` and printed to the screen. `make` on its own lists
every target.

**WSL2 note:** `make setup` puts the virtual environment in your Linux home (`~/.venvs/nem-time-series`), not in the repo,
because a virtual environment on the Windows drive (`/mnt/c`) is very slow. To use another location set `VENV`, for
example `export VENV=/some/path` before running `make`. Keeping the repo itself on the Linux side (`~/`) is faster still.

## Reproducing the results

The repository ships everything needed to reproduce the numbers without training or a GPU:

| What | Where | Size |
|---|---|---|
| Price and demand, 30-minute | `datasets/VIC1/actuals_30min.parquet` | 1.6 MB |
| Every model's saved forecasts | `results/VIC1/predictions_<model>.parquet` | 13 MB in total |
| The fine-tuned Chronos-2 adapter | `models/chronos2_ft/` | 4.7 MB |

### 1. Check the table (seconds)

```bash
make backtest
```

### 2. Recompute the forecasts yourself

```bash
make backtest ARGS="--force"                                   # all models
make backtest ARGS="--models chronos2_zs,lightgbm --force"     # just some
```

Rough times on an 8-core laptop CPU: naive and LightGBM take seconds (LightGBM trains in about 20 s); each Chronos-2
variant takes roughly 5 to 30 minutes depending on how much history it reads (the default 14 days is about 10 minutes).
Re-running should give the same numbers: the fine-tuned model computed on a CPU matched the Colab GPU run to within
$0.03/MWh on any single forecast.

Model names: `naive_yesterday`, `naive_last_week`, `lightgbm`, `chronos2_zs` (price only, 14-day context),
`chronos2_zs_cal` (adds calendar features), `chronos2_zs_ctx3d` / `ctx7d` / `ctx28d` / `ctx42d` (context-length
ablation), and `chronos2_ft` (the fine-tuned model).

Useful options (all go inside `ARGS="..."`):

| Option | Meaning |
|---|---|
| `--models a,b,c` | which models to run |
| `--start 2025-06-01` / `--end 2026-01-31` | restrict the forecast days |
| `--force` | recompute even if saved predictions exist |
| `--reference lightgbm` | which model the significance table compares everything against (default `chronos2_zs`) |
| `--region VIC1` | region (only VIC1 data is included, see below) |

Each reference gets its own files, so comparing against several models never overwrites anything.

> **Saved predictions are reused.** `make backtest` skips a model whose saved forecasts cover the same days. It cannot
> tell if the code or model changed since, so after changing a model or its settings, add `--force`.

### 3. Fine-tune Chronos-2 yourself

```bash
make finetune                                  # LoRA, 1000 steps, 14-day context, on 2023-24 prices only
make backtest ARGS="--models chronos2_ft --force"
```

This writes `models/chronos2_ft/` (overwriting the committed adapter) and scores it on the 2025+ days. The training data
stops at 31 Dec 2024, and the backtest refuses to score any forecast day a trained model could have seen.

- **On a GPU (for example Colab's free T4): about 5 minutes.**
- **On a laptop CPU: several hours.** Try a quick check first: `make finetune ARGS="--num-steps 20 --batch-size 8"`
  (about 2 minutes; the result is meaningless, it only proves the pipeline works).
- Other options: `--num-steps`, `--batch-size`, `--learning-rate`, `--context-days`, `--seed`, `--models-dir`.
  The defaults (1000 steps, batch 32, learning rate 1e-5) are untuned.

### 4. Rebuild the data

The dataset is a snapshot of AEMO's public data. [datasets/README.md](datasets/README.md) has a short, tested nemosis
script that rebuilds it for any region and period. Only VIC1 is included. For another region, build its file with that
script and pass `--region`.

## Run it on Google Colab (GPU)

1. Open the notebook with the badge at the top, or go to
   `https://colab.research.google.com/github/janxl/nem-timeseries/blob/main/notebooks/colab.ipynb`.
2. **Runtime > Change runtime type > GPU** (a T4 is enough).
3. Check the settings cell, then **Run all**. It clones the repo, installs it, confirms the GPU, runs the tests,
   fine-tunes, runs the backtest and shows the results and significance tables.

Notes:
- If you fork the repo, change `REPO_URL` in the settings cell. A private repo needs a token in the URL.
- The notebook calls `make ... RUN=`, which runs the pip-installed commands directly. Locally the Makefile uses `uv`.
  On Colab that would swap Colab's CUDA PyTorch for a CPU-only one.
- Colab ships an old `torchao` that the LoRA library refuses to work with, so the install cell removes it.
- Colab's disk is wiped when the runtime is recycled. Set `SAVE_TO_DRIVE = True` to copy the model and results to Drive,
  or download the zips offered at the end.

## Make targets

| Target | What it does |
|---|---|
| `make help` | list the targets |
| `make doctor` | check tools, disk and network, and say how to fix gaps |
| `make setup` | create the virtual environment and install (uses `uv`, installed if missing) |
| `make test` | run the unit tests |
| `make backtest` | rolling backtest for all models; writes `results/<region>/` |
| `make finetune` | LoRA fine-tune of Chronos-2 on 2023-24; writes `models/chronos2_ft/` |

Pass options with `ARGS="..."`. Where the packages are already installed (for example Colab), add `RUN=` to skip `uv`.

## What the output files are

| File | Contents |
|---|---|
| `results/VIC1/metrics.csv` | every model's scores |
| `results/VIC1/metrics_vs_<model>.md` | the results table plus a significance table against that reference model |
| `results/VIC1/significance_vs_<model>.csv` | the same significance numbers as a table |
| `results/VIC1/predictions_<model>.parquet` | every forecast: origin, step, actual, p10, p50, p90 |
| `models/chronos2_ft/` | the fine-tuned adapter and `meta.json` (settings, training range, seconds taken) |

## Project layout

```
datasets/VIC1/          AEMO price and demand (30-minute), committed snapshot, and how to rebuild it
src/nem_ts/             the package: data, backtest, metrics, significance, report, cli
src/nem_ts/models/      naive, lightgbm, chronos2 (zero-shot and fine-tune), registry
tests/                  unit tests (use toy data and a fake model, so no downloads)
results/VIC1/           saved forecasts, scores and significance tables
models/chronos2_ft/     the fine-tuned LoRA adapter
notebooks/colab.ipynb   the one-click GPU run
scripts/doctor.sh       the machine check behind `make doctor`
```

## Add your own model

A model is a small class with a `name`, a `context_length`, a `probabilistic` flag and one method:

```python
def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    # history: price (and calendar columns) up to 23:30 the day before, context_length rows
    # future:  calendar columns for the 48 half-hours to forecast
    # return a frame indexed like `future` with columns p10, p50, p90
```

See `src/nem_ts/models/base.py` for the interface and `naive.py` for the smallest example. Register it in
`src/nem_ts/models/registry.py`, then `make backtest ARGS="--models my_model"`. If it trains on data, set
`trained_through` so the backtest can refuse to score days it has seen. Adding Toto, TimesFM or similar follows the same
pattern.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `make doctor` reports a failed network check | the data sources are public and need no key; check your proxy or firewall |
| `chronos2_ft: skipped, no fine-tuned checkpoint` | pull `models/chronos2_ft/` (it is in the repo) or run `make finetune` |
| `ImportError: ... incompatible version of torchao` | `pip uninstall -y torchao` (the Colab notebook does this for you) |
| Colab says `No module named 'nem_ts...'` | re-run the clone and install cells; make sure you pulled the latest commit |
| Virtual environment is slow on Windows | keep it in the WSL home (the default) and, ideally, keep the repo there too |
| Results look stale after a code change | re-run with `--force` |

## Licences and credits

- **Code:** MIT (see [LICENSE](LICENSE)).
- **Data:** (c) Australian Energy Market Operator Limited, used under
  [AEMO's copyright permissions](https://www.aemo.com.au/Privacy-and-Legal-Notices/Copyright-Permissions-Notice).
- **Model:** the fine-tuned adapter is derived from [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2)
  (Apache-2.0; [paper](https://arxiv.org/abs/2510.15821)). Chronos-2 itself is downloaded from Hugging Face, not stored here.
