import numpy as np
import pandas as pd
import pytest

from nem_ts.backtest import HORIZON, run_backtest, valid_origins
from nem_ts.data import CALENDAR_COLUMNS, STEP, calendar_features
from nem_ts.models.naive import SeasonalNaive


def toy_frame(days=30, start="2025-01-01"):
    """A price that repeats every day: the value is the half-hour of day (0-47)."""
    index = pd.date_range(start, periods=days * 48, freq=STEP, name="timestamp")
    df = pd.DataFrame({"rrp": (index.hour * 2 + index.minute // 30).astype(float)}, index=index)
    return df.join(calendar_features(index))


class Recorder:
    """Causal model that records what it was given and forecasts the mean of its history."""

    name, probabilistic = "recorder", True

    def __init__(self, context_length=96):
        self.context_length = context_length
        self.seen = []

    def predict(self, history, future):
        self.seen.append((history, future))
        m = float(history["rrp"].mean())
        return pd.DataFrame({"p10": m - 1, "p50": m, "p90": m + 1}, index=future.index)


def test_calendar_features():
    index = pd.date_range("2025-01-01", periods=3, freq=STEP)  # 1 Jan 2025 is a Wednesday and a public holiday
    cal = calendar_features(index)
    assert cal["half_hour"].tolist() == [0, 1, 2]
    assert cal["dow"].tolist() == [2, 2, 2]
    assert cal["is_holiday"].tolist() == [1, 1, 1]
    assert calendar_features(pd.date_range("2025-01-02", periods=1, freq=STEP))["is_holiday"].iloc[0] == 0


def test_origins_are_midnights_whose_targets_fit_in_the_data():
    frame = toy_frame(days=10)  # 1 Jan to 10 Jan
    origins, skipped = valid_origins(frame, start="2025-01-03")
    assert origins[0] == pd.Timestamp("2025-01-03") and origins[-1] == pd.Timestamp("2025-01-10")
    assert skipped == 0 and len(origins) == 8


def test_origins_with_a_gap_in_the_targets_are_skipped_and_counted():
    frame = toy_frame(days=10)
    frame.loc["2025-01-05 12:00", "rrp"] = np.nan
    origins, skipped = valid_origins(frame, start="2025-01-03")
    assert pd.Timestamp("2025-01-05") not in origins and skipped == 1


def test_seasonal_naive_is_exact_on_a_daily_repeating_series():
    frame = toy_frame()
    origins, _ = valid_origins(frame, start="2025-01-10")
    out = run_backtest(SeasonalNaive(48), frame, origins)
    assert (out["p50"] == out["actual"]).all()
    assert len(out) == len(origins) * HORIZON
    assert set(out.columns) == {"model", "origin", "step", "timestamp", "actual", "p10", "p50", "p90"}


def test_seasonal_naive_week_lag_copies_the_value_from_a_week_earlier():
    frame = toy_frame()
    frame["rrp"] = np.arange(len(frame), dtype=float)  # every value unique
    origins, _ = valid_origins(frame, start="2025-01-10", end="2025-01-10")
    out = run_backtest(SeasonalNaive(336), frame, origins)
    week_ago = frame["rrp"].shift(336).loc["2025-01-10":"2025-01-10 23:30"].to_numpy()
    assert np.array_equal(out["p50"].to_numpy(), week_ago)


def test_model_only_sees_data_before_the_origin_and_exactly_its_context_length():
    frame = toy_frame()
    model = Recorder(context_length=96)
    origins, _ = valid_origins(frame, start="2025-01-10")
    run_backtest(model, frame, origins)
    assert len(model.seen) == len(origins)
    for origin, (history, future) in zip(origins, model.seen):
        assert history.index.max() == origin - STEP  # last half-hour before the origin
        assert len(history) == 96
        assert future.index[0] == origin and len(future) == HORIZON
        assert list(future.columns) == CALENDAR_COLUMNS  # no prices from the future


def test_changing_future_prices_does_not_change_earlier_forecasts():
    frame = toy_frame()
    altered = frame.copy()
    altered.loc["2025-01-15":, "rrp"] = 9999.0
    origins, _ = valid_origins(frame, start="2025-01-10", end="2025-01-14")
    a = run_backtest(Recorder(), frame, origins)
    b = run_backtest(Recorder(), altered, origins)
    assert np.array_equal(a["p50"].to_numpy(), b["p50"].to_numpy())


def test_bad_predictions_are_rejected():
    frame = toy_frame()
    origins, _ = valid_origins(frame, start="2025-01-10", end="2025-01-10")

    class WrongIndex(Recorder):
        def predict(self, history, future):
            return super().predict(history, future).reset_index(drop=True)

    class HasNaN(Recorder):
        def predict(self, history, future):
            out = super().predict(history, future)
            out.iloc[0, 0] = np.nan
            return out

    with pytest.raises(ValueError, match="index"):
        run_backtest(WrongIndex(), frame, origins)
    with pytest.raises(ValueError, match="NaN"):
        run_backtest(HasNaN(), frame, origins)


def test_not_enough_history_is_an_error():
    frame = toy_frame(days=10)
    origins, _ = valid_origins(frame, start="2025-01-01", end="2025-01-01")
    with pytest.raises(ValueError, match="history rows"):
        run_backtest(Recorder(context_length=96), frame, origins)
