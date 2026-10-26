"""LightGBM quantile baseline.

One model per quantile (P10, P50, P90), trained once on data up to `train_end` and then applied unchanged to
every test origin. Each training row is one half-hour of one midnight-origin forecast (the same 48-step setup as
the backtest), described by what is known at the origin: lagged prices, recent price statistics, and the calendar.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from nem_ts.data import STEP, STEPS_PER_DAY
from nem_ts.metrics import QUANTILE_COLUMNS, QUANTILES

CONTEXT_DAYS = 7
CONTEXT_LENGTH = CONTEXT_DAYS * STEPS_PER_DAY  # 336 half-hours of history
DEFAULT_TRAIN_END = pd.Timestamp("2024-12-31 23:30")  # the day before the test period starts
FEATURES = [
    "step", "dow", "is_holiday",
    "lag_1d", "lag_2d", "lag_7d", "same_hh_mean_7d", "same_hh_max_7d",
    "last_price", "yday_mean", "yday_max", "yday_min", "yday_std", "week_mean",
]


def make_features(prices: np.ndarray, dow: np.ndarray, is_holiday: np.ndarray) -> pd.DataFrame:
    """Features for the 48 half-hours after a midnight origin.

    prices: the 336 half-hour prices before the origin (oldest first, NaN allowed), i.e. 7 whole days ending at 23:30.
    dow, is_holiday: calendar values of the 48 half-hours being forecast.
    """
    if len(prices) != CONTEXT_LENGTH:
        raise ValueError(f"expected {CONTEXT_LENGTH} history prices, got {len(prices)}")
    days = np.asarray(prices, dtype=float).reshape(CONTEXT_DAYS, STEPS_PER_DAY)  # row = day, column = half-hour
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are fine: LightGBM handles NaN
        same_mean, same_max = np.nanmean(days, axis=0), np.nanmax(days, axis=0)
        yday = days[-1]
        stats = {
            "yday_mean": np.nanmean(yday), "yday_max": np.nanmax(yday), "yday_min": np.nanmin(yday),
            "yday_std": np.nanstd(yday), "week_mean": np.nanmean(days),
        }
    last = days[-1, -1]
    out = pd.DataFrame(
        {
            "step": np.arange(STEPS_PER_DAY), "dow": dow, "is_holiday": is_holiday,
            "lag_1d": days[-1], "lag_2d": days[-2], "lag_7d": days[0],
            "same_hh_mean_7d": same_mean, "same_hh_max_7d": same_max,
            "last_price": last,
        }
    )
    for name, value in stats.items():
        out[name] = value
    return out[FEATURES]


def build_training_set(frame: pd.DataFrame, train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """All midnight origins whose 48 targets end on or before train_end, with a full 7-day history."""
    first = (frame.index.min() + CONTEXT_LENGTH * STEP).ceil("D")
    last = (train_end - (STEPS_PER_DAY - 1) * STEP).floor("D")
    xs, ys = [], []
    for origin in pd.date_range(first, last, freq="D"):
        history = frame.loc[: origin - STEP, "rrp"].iloc[-CONTEXT_LENGTH:]
        target = frame.loc[origin : origin + (STEPS_PER_DAY - 1) * STEP]
        if len(history) != CONTEXT_LENGTH or len(target) != STEPS_PER_DAY:
            continue
        x = make_features(history.to_numpy(), target["dow"].to_numpy(), target["is_holiday"].to_numpy())
        keep = target["rrp"].notna().to_numpy()  # no target, no training row
        xs.append(x[keep])
        ys.append(target["rrp"][keep])
    if not xs:
        raise ValueError("no training origins: the data ends before train_end or is too short")
    return pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True)


class LightGBMQuantile:
    name = "lightgbm"
    probabilistic = True
    context_length = CONTEXT_LENGTH

    def __init__(self, params: dict | None = None):
        self.num_boost_round = 400
        self.params = {
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
            "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8,
            "seed": 0, "verbose": -1, **(params or {}),
        }
        self.num_boost_round = self.params.pop("num_boost_round", self.num_boost_round)
        self.models: dict[float, lgb.Booster] = {}
        self.trained_through: pd.Timestamp | None = None

    def fit(self, frame: pd.DataFrame, train_end: pd.Timestamp = DEFAULT_TRAIN_END) -> "LightGBMQuantile":
        x, y = build_training_set(frame, train_end)
        self.models = {
            q: lgb.train({"objective": "quantile", "alpha": q, **self.params}, lgb.Dataset(x, label=y),
                         num_boost_round=self.num_boost_round)
            for q in QUANTILES
        }
        self.trained_through = pd.Timestamp(train_end)
        self.n_training_rows = len(x)
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("call fit() first")
        x = make_features(history["rrp"].to_numpy(), future["dow"].to_numpy(), future["is_holiday"].to_numpy())
        preds = np.sort(np.column_stack([self.models[q].predict(x) for q in QUANTILES]), axis=1)  # no crossing
        return pd.DataFrame(dict(zip(QUANTILE_COLUMNS, preds.T)), index=future.index)
