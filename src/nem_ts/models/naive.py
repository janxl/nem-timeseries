"""Seasonal naive: repeat the price from the same time yesterday (lag 48) or last week (lag 336)."""

from __future__ import annotations

import pandas as pd

from nem_ts.data import STEPS_PER_DAY


class SeasonalNaive:
    probabilistic = False

    def __init__(self, lag_steps: int = STEPS_PER_DAY, name: str | None = None):
        if lag_steps < STEPS_PER_DAY:
            raise ValueError("lag must be at least one day, or the value would not be known at the origin")
        self.lag_steps = lag_steps
        self.context_length = lag_steps
        self.name = name or {48: "naive_yesterday", 336: "naive_last_week"}.get(lag_steps, f"naive_lag{lag_steps}")

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        # Fill gaps so one missing half-hour does not poison the forecast.
        prices = history["rrp"].ffill().bfill().to_numpy()
        n = len(future)
        values = prices[len(prices) - self.lag_steps : len(prices) - self.lag_steps + n]
        return pd.DataFrame({"p10": values, "p50": values, "p90": values}, index=future.index)
