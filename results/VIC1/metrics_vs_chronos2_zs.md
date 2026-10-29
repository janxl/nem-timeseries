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
| chronos2_zs_ctx3d | 47.70 | 247.24 | 242.88 | 38.42 | 72.8% | 16.97 |
| chronos2_zs_ctx7d | 44.21 | 245.86 | 231.34 | 31.30 | 74.5% | 15.58 |
| chronos2_zs_ctx28d | 42.35 | 271.69 | 233.39 | 25.74 | 76.6% | 14.85 |
| chronos2_zs_ctx42d | 40.62 | 246.75 | 236.17 | 25.49 | 76.5% | 14.41 |

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
| chronos2_zs_ctx3d | +6.56 [+5.32, +7.69] * | 30% | +2.40 [+1.84, +3.13] * |
| chronos2_zs_ctx7d | +3.08 [+2.30, +3.88] * | 35% | +1.00 [+0.72, +1.34] * |
| chronos2_zs_ctx28d | +1.21 [-1.03, +5.29] | 57% | +0.28 [-0.32, +1.36] |
| chronos2_zs_ctx42d | -0.51 [-1.14, +0.13] | 58% | -0.17 [-0.35, +0.03] |

Differences are in $/MWh. "Better on % of days" compares the daily MAE of the model with the reference.
