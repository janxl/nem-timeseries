"""Rolling-origin backtest: forecast the next 48 half-hours from every NEM midnight.

The same set of origins is used for every model, so results are directly comparable. A model only ever
receives data up to the half-hour before the origin, and calendar features for the half-hours it must forecast.
"""

from __future__ import annotations

import pandas as pd

from nem_ts.data import CALENDAR_COLUMNS, STEP, STEPS_PER_DAY
from nem_ts.metrics import QUANTILE_COLUMNS
from nem_ts.models.base import Forecaster

HORIZON = STEPS_PER_DAY
DEFAULT_TEST_START = "2025-01-01"


def valid_origins(frame: pd.DataFrame, start: str = DEFAULT_TEST_START, end: str | None = None,
                  horizon: int = HORIZON) -> tuple[pd.DatetimeIndex, int]:
    """Midnights whose next `horizon` actual prices are all present. Returns (origins, number skipped for gaps)."""
    last_start = frame.index.max() - (horizon - 1) * STEP  # latest origin whose targets fit in the data
    last = min(pd.Timestamp(end) if end else last_start, last_start).normalize()
    candidates = pd.date_range(pd.Timestamp(start).normalize(), last, freq="D")
    candidates = candidates[candidates + (horizon - 1) * STEP <= frame.index.max()]
    rrp = frame["rrp"]
    keep = [o for o in candidates if rrp.loc[o : o + (horizon - 1) * STEP].notna().all()]
    return pd.DatetimeIndex(keep), len(candidates) - len(keep)


def check_no_leakage(model: Forecaster, origins: pd.DatetimeIndex) -> None:
    """A model that was trained on data must not be scored on origins whose history or targets it trained on."""
    through = getattr(model, "trained_through", None)
    if through is not None and len(origins) and origins[0] <= through:
        raise ValueError(
            f"{model.name} was trained on data up to {through}, but the first test origin is {origins[0]}: "
            "the backtest would score data the model has seen"
        )


def run_backtest(model: Forecaster, frame: pd.DataFrame, origins: pd.DatetimeIndex,
                 horizon: int = HORIZON) -> pd.DataFrame:
    """One row per origin and half-hour: model, origin, step, timestamp, actual, p10, p50, p90."""
    check_no_leakage(model, origins)
    rows = []
    for origin in origins:
        history = frame.loc[: origin - STEP].iloc[-model.context_length :]
        if len(history) != model.context_length:
            raise ValueError(f"{origin}: only {len(history)} of {model.context_length} history rows available")
        target = frame.loc[origin : origin + (horizon - 1) * STEP]
        future = target[CALENDAR_COLUMNS]
        pred = model.predict(history, future)
        _check_prediction(pred, future, model.name, origin)
        rows.append(
            pd.DataFrame(
                {
                    "model": model.name,
                    "origin": origin,
                    "step": range(horizon),
                    "timestamp": target.index,
                    "actual": target["rrp"].to_numpy(),
                    **{c: pred[c].to_numpy(dtype=float) for c in QUANTILE_COLUMNS},
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _check_prediction(pred: pd.DataFrame, future: pd.DataFrame, name: str, origin: pd.Timestamp) -> None:
    if not isinstance(pred, pd.DataFrame) or not set(QUANTILE_COLUMNS) <= set(pred.columns):
        raise ValueError(f"{name}: predict() must return a DataFrame with columns {QUANTILE_COLUMNS}")
    if not pred.index.equals(future.index):
        raise ValueError(f"{name} at {origin}: prediction index does not match the {len(future)} future half-hours")
    if pred[list(QUANTILE_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} at {origin}: prediction contains NaN")
