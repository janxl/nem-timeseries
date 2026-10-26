"""Forecast metrics.

Input is a long frame with columns actual, p10, p50, p90 (one row per origin and half-hour). The point
forecast is the median (p50). Coverage and pinball loss only make sense for models that produce real
quantiles; for the others they are NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_COLUMNS = ("p10", "p50", "p90")
SPIKE_QUANTILE = 0.95


def spike_threshold(actual: pd.Series, q: float = SPIKE_QUANTILE) -> float:
    """Price at or above which a half-hour counts as a spike (top 5% of the evaluated actuals)."""
    return float(actual.quantile(q))


def pinball_loss(actual: pd.Series, forecast: pd.Series, q: float) -> float:
    diff = actual - forecast
    return float(np.maximum(q * diff, (q - 1) * diff).mean())


def _mae(actual: pd.Series, forecast: pd.Series) -> float:
    return float((actual - forecast).abs().mean()) if len(actual) else float("nan")


def score(df: pd.DataFrame, *, spike_threshold: float, probabilistic: bool = True) -> dict[str, float]:
    """All metrics for one model. Pass the same spike_threshold when comparing models."""
    actual, p50 = df["actual"], df["p50"]
    spikes, negative = actual >= spike_threshold, actual < 0
    out = {
        "mae": _mae(actual, p50),
        "rmse": float(np.sqrt(((actual - p50) ** 2).mean())),
        "mae_spike": _mae(actual[spikes], p50[spikes]),
        "mae_negative": _mae(actual[negative], p50[negative]),
        "coverage_p10_p90": float(((actual >= df["p10"]) & (actual <= df["p90"])).mean()) if probabilistic else float("nan"),
        "pinball": float(np.mean([pinball_loss(actual, df[c], q) for c, q in zip(QUANTILE_COLUMNS, QUANTILES)]))
        if probabilistic
        else float("nan"),
        "n_forecasts": float(len(df)),
        "n_origins": float(df["origin"].nunique()) if "origin" in df else float("nan"),
        "n_spike": float(spikes.sum()),
        "n_negative": float(negative.sum()),
    }
    return out


def score_models(results: dict[str, tuple[pd.DataFrame, bool]]) -> pd.DataFrame:
    """Score several models on identical forecasts. results: name -> (predictions, probabilistic).

    Every model must cover the same origins and actuals, and one spike threshold is used for all of them.
    """
    names = list(results)
    first = results[names[0]][0]
    keys = ["origin", "step"]
    for name in names[1:]:
        other = results[name][0]
        same = len(other) == len(first) and np.array_equal(
            other[keys + ["actual"]].to_numpy(), first[keys + ["actual"]].to_numpy()
        )
        if not same:
            raise ValueError(f"{name} was not scored on the same origins and actuals as {names[0]}")
    threshold = spike_threshold(first["actual"])
    table = pd.DataFrame(
        {name: score(df, spike_threshold=threshold, probabilistic=prob) for name, (df, prob) in results.items()}
    ).T
    table.attrs["spike_threshold"] = threshold
    return table
