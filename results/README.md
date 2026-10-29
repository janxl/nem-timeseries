# results/

Final benchmark outputs for VIC1: 635 forecast days (NEM midnight, 2025-01-01 to 2026-10-25), 48 half-hours each.
Everything here is produced by `make backtest`; the fine-tuned model comes from `make finetune` (`models/chronos2_ft/`).

| file | what it is |
|---|---|
| `VIC1/metrics.csv` | scores for every model (MAE, RMSE, spike and negative-price MAE, coverage, pinball) |
| `VIC1/metrics_vs_<model>.md` | the results table, plus a paired significance table against that reference model |
| `VIC1/significance_vs_<model>.csv` | the same significance numbers as a table |
| `VIC1/predictions_<model>.parquet` | every forecast: origin, step, actual, p10, p50, p90 |

The fine-tuned predictions were produced on a Colab GPU; recomputing them on a laptop CPU from the saved weights gives
the same numbers (largest difference in any forecast: $0.03/MWh; MAE identical to four decimals).

**Saved predictions are reused.** `make backtest` skips any model whose saved predictions cover the same forecast days.
It cannot tell whether the code or the model changed since, so after changing a model or its settings, re-run it with
`make backtest ARGS="--force"` (or `--models <name> --force`).
