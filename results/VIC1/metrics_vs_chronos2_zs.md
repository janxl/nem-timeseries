# VIC1 backtest results

- 635 forecast origins (NEM midnight), 2025-01-01 to 2026-10-25; 28 more skipped because the actual prices had gaps.
- Each origin forecasts the next 48 half-hours (24 hours).
- Spike = actual price at or above $181.0/MWh (top 5% of the evaluated prices).

| model | MAE | RMSE | MAE spikes | MAE negative | P10-P90 coverage | Pinball |
|---|---:|---:|---:|---:|---:|---:|
| naive_yesterday | 57.89 | 326.30 | 322.36 | 36.01 | – | – |
| naive_last_week | 74.09 | 349.20 | 360.19 | 50.50 | – | – |
| lightgbm | 41.76 | 246.13 | 236.86 | 30.04 | 68.6% | 14.98 |
| chronos2_zs | 41.14 | 244.38 | 231.68 | 26.98 | 76.6% | 14.57 |
| chronos2_zs_cal | 43.13 | 281.20 | 269.79 | 25.32 | 79.6% | 14.98 |
| chronos2_ft | 40.02 | 244.06 | 228.28 | 25.96 | 77.8% | 14.19 |

Errors are in $/MWh. MAE, RMSE and pinball: lower is better; coverage should be near 80%.

## Is the difference real? Each model minus `chronos2_zs`

Paired by forecast day (635 origins); 95% confidence intervals from a 7-day block bootstrap. A positive difference means the model is **worse** than the reference. `*` marks an interval that excludes zero, i.e. a difference unlikely to be luck.

| model | MAE difference [95% CI] | better on % of days | pinball difference [95% CI] |
|---|---:|---:|---:|
| naive_yesterday | +16.75 [+11.71, +24.02] * | 23% | – |
| naive_last_week | +32.95 [+25.43, +43.21] * | 15% | – |
| lightgbm | +0.62 [-0.74, +1.92] | 46% | +0.41 [-0.15, +0.93] |
| chronos2_zs_cal | +1.99 [-0.37, +6.20] | 50% | +0.41 [-0.03, +1.14] |
| chronos2_ft | -1.11 [-1.52, -0.77] * | 60% | -0.39 [-0.55, -0.25] * |

Differences are in $/MWh. "Better on % of days" compares the daily MAE of the model with the reference.
