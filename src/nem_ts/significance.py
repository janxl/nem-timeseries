"""Is the difference between two models real, or luck?

Each model gets one error per origin (forecast day): the mean over that day's 48 half-hours. Two models are then
compared day by day (a paired comparison), and a block bootstrap over the days gives a confidence interval for the
average difference. Blocks of consecutive days are resampled together because neighbouring days are correlated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nem_ts.metrics import QUANTILE_COLUMNS, QUANTILES

BLOCK_DAYS = 7
N_BOOT = 5000
LEVEL = 0.95


def per_origin_errors(df: pd.DataFrame, probabilistic: bool) -> pd.DataFrame:
    """One row per origin: mae (of the median) and pinball loss (NaN for point forecasts)."""
    out = pd.DataFrame({"mae": (df["actual"] - df["p50"]).abs().groupby(df["origin"]).mean()})
    if probabilistic:
        losses = []
        for col, q in zip(QUANTILE_COLUMNS, QUANTILES):
            diff = df["actual"] - df[col]
            losses.append(np.maximum(q * diff, (q - 1) * diff))
        out["pinball"] = (sum(losses) / len(losses)).groupby(df["origin"]).mean()
    else:
        out["pinball"] = np.nan
    return out.sort_index()


def block_bootstrap_ci(x: np.ndarray, block: int = BLOCK_DAYS, n_boot: int = N_BOOT, level: float = LEVEL,
                       seed: int = 0) -> tuple[float, float, float]:
    """Mean of x with a moving-block bootstrap confidence interval. x is ordered in time."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)  # ceil
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def compare_to_reference(results: dict[str, tuple[pd.DataFrame, bool]], reference: str, **kwargs) -> pd.DataFrame:
    """Each model minus the reference, per metric: mean difference, confidence interval, and share of days better.

    A positive difference means the model is worse (larger error) than the reference.
    """
    errors = {name: per_origin_errors(df, prob) for name, (df, prob) in results.items()}
    ref = errors[reference]
    rows = {}
    for name, err in errors.items():
        if name == reference:
            continue
        if not err.index.equals(ref.index):
            raise ValueError(f"{name} and {reference} were not scored on the same origins")
        row = {}
        for metric in ("mae", "pinball"):
            diff = (err[metric] - ref[metric]).to_numpy()
            if np.isnan(diff).any():
                mean = lo = hi = better = np.nan
            else:
                mean, lo, hi = block_bootstrap_ci(diff, **kwargs)
                better = float((diff < 0).mean())  # share of days where the model beats the reference
            row.update({f"{metric}_diff": mean, f"{metric}_lo": lo, f"{metric}_hi": hi, f"{metric}_better_days": better})
        rows[name] = row
    out = pd.DataFrame(rows).T
    out.attrs["reference"] = reference
    out.attrs["n_origins"] = len(ref)
    return out
