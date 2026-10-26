"""The interface every model implements, so the backtest can treat them all the same way."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    """A model that forecasts the next 48 half-hours from history up to a midnight.

    name:           label used in result tables.
    context_length: how many past half-hours of history the backtest passes in.
    probabilistic:  False for point forecasts; coverage and pinball loss are then reported as NaN.
    """

    name: str
    context_length: int
    probabilistic: bool

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """history: rrp (plus demand and calendar columns) up to the half-hour before the origin, regular 30-minute
        index, exactly context_length rows, may contain NaN.
        future: calendar columns only, for the 48 half-hours to forecast.
        Returns a frame indexed like `future` with columns p10, p50, p90 (equal for point forecasts)."""
        ...
