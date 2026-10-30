# nem-timeseries

Forecasting Victorian (VIC1) electricity prices a day ahead with Amazon's
[Chronos-2](https://huggingface.co/amazon/chronos-2), compared with baselines and then fine-tuned.
The prices come from Australia's National Electricity Market (NEM), published by the market operator AEMO.
Uses **public data only**.

**Headline:** Chronos-2 with no training of its own ties a LightGBM model trained on two years of this data, and a
5-minute fine-tune on a GPU makes it the best model (about 4% lower error than LightGBM, a gap unlikely to be luck).
The full results and caveats are in **[RESULTS.md](RESULTS.md)**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janxl/nem-timeseries/blob/main/notebooks/colab.ipynb)

Every midnight, predict the price for each of the next 48 half-hours. Trained on 2023-24, tested on every day from
1 Jan 2025.

## Before you start

You need Linux, macOS or Windows with WSL2 (run everything inside WSL2), plus `make`, `git`, `curl` and Python 3.11+.
No GPU is required. The first run downloads the Chronos-2 weights from Hugging Face (a few hundred MB).

## Step 1: Get the code

```bash
git clone https://github.com/janxl/nem-timeseries.git
cd nem-timeseries
```

## Step 2: Check your machine

```bash
make doctor
```

It checks your tools, disk space and network, and tells you how to fix anything that is missing.

## Step 3: Install

```bash
make setup
```

This installs `uv` if needed and creates a virtual environment with everything in it (a few minutes). On Windows/WSL2 the
environment lives in your Linux home, not in the repo, to keep it fast.

## Step 4: Check the install works (optional)

```bash
make test
```

43 quick tests; they need no downloads.

## Step 5: Reproduce the results table

```bash
make backtest
```

Takes seconds. The repo already contains the data, every model's saved forecasts and the fine-tuned model, so this just
scores them and prints the table. It is also saved to `results/VIC1/metrics_vs_chronos2_zs.md`.

## Step 6: Recompute the forecasts yourself (optional)

```bash
make backtest ARGS="--force"
```

Reruns every model from scratch: roughly 1 to 2 hours on a laptop CPU. To run only some models:

```bash
make backtest ARGS="--models lightgbm,chronos2_zs --force"
```

Model names: `naive_yesterday`, `naive_last_week`, `lightgbm`, `chronos2_zs`, `chronos2_zs_cal`, `chronos2_ft`, and
`chronos2_zs_ctx3d` / `ctx7d` / `ctx28d` / `ctx42d`. Other useful options: `--start 2025-06-01`, `--end 2026-01-31`,
and `--reference lightgbm` (compare everything against that model; it gets its own output files).

Saved forecasts are reused unless you pass `--force`, so add it whenever you have changed a model or its settings.

## Step 7: Fine-tune Chronos-2 yourself (optional)

```bash
make finetune
make backtest ARGS="--models chronos2_ft --force"
```

Takes about 5 minutes on a GPU but several hours on a laptop CPU, so a GPU (see Step 8) is the better place for it.
To just check that it runs, try `make finetune ARGS="--num-steps 20 --batch-size 8"` (about 2 minutes). This overwrites
the fine-tuned model in `models/chronos2_ft/`. It only ever trains on 2023-24; the test period is never used.

## Step 8: Or do it all on a free GPU in Google Colab

1. Click the Colab badge at the top.
2. Choose **Runtime > Change runtime type > GPU**.
3. Click **Run all**.

It clones the repo, installs it, fine-tunes, runs the backtest and shows the results (about 15 to 30 minutes). If you
fork the repo, change `REPO_URL` in the notebook's first cell.

## Where things are

| Path | What |
|---|---|
| `results/VIC1/` | results tables and every model's saved forecasts |
| `models/chronos2_ft/` | the fine-tuned model (a 4.7 MB adapter) |
| `datasets/VIC1/` | the price data, and [how to rebuild it from AEMO](datasets/README.md) |
| `src/nem_ts/` | the code; models are in `src/nem_ts/models/` and registered in `registry.py` |
| `notebooks/colab.ipynb` | the Colab notebook |

## If something goes wrong

- `chronos2_ft: skipped, no fine-tuned checkpoint`: run `make finetune`, or make sure `models/chronos2_ft/` is there.
- `ImportError ... torchao` (Colab): run `pip uninstall -y torchao`. The notebook already does this.
- Results look out of date after a code change: re-run with `--force`.

## Licences

Code: MIT. Data: (c) Australian Energy Market Operator Limited, used under
[AEMO's copyright permissions](https://aemo.com.au/privacy-and-legal-notices/copyright-permissions).
Model: derived from [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) (Apache-2.0).
