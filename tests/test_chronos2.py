import numpy as np
import pandas as pd

from nem_ts.backtest import run_backtest, valid_origins
from nem_ts.data import CALENDAR_COLUMNS, STEP, calendar_features
from nem_ts.models.chronos2 import Chronos2Forecaster


class FakePipeline:
    """Stands in for Chronos2Pipeline (no model download): records the calls and returns a fixed forecast."""

    def __init__(self):
        self.calls = []

    def predict_df(self, df, future_df=None, id_column="item_id", timestamp_column="timestamp", target="target",
                   prediction_length=None, quantile_levels=None, **kwargs):
        self.calls.append((df, future_df, prediction_length, quantile_levels))
        last = float(df[target].iloc[-1])
        ts = pd.date_range(df[timestamp_column].iloc[-1] + STEP, periods=prediction_length, freq=STEP)
        out = pd.DataFrame({id_column: df[id_column].iloc[0], timestamp_column: ts, "target_name": target})
        out["predictions"] = last
        for q, shift in zip(quantile_levels, (-1.0, 0.0, 1.0)):
            out[str(q)] = last + shift
        return out.iloc[::-1]  # deliberately unsorted: the model must sort by timestamp


def toy_frame(days=30):
    index = pd.date_range("2025-01-01", periods=days * 48, freq=STEP, name="timestamp")
    df = pd.DataFrame({"rrp": np.arange(len(index), dtype=float)}, index=index)
    return df.join(calendar_features(index))


def run(use_calendar):
    pipe = FakePipeline()
    model = Chronos2Forecaster(context_days=7, use_calendar=use_calendar, pipeline=pipe)
    frame = toy_frame()
    origins, _ = valid_origins(frame, start="2025-01-12", end="2025-01-13")
    return pipe, model, run_backtest(model, frame, origins), frame


def test_price_only_sends_no_covariates():
    pipe, model, out, _ = run(use_calendar=False)
    context, future_df, prediction_length, quantiles = pipe.calls[0]
    assert model.context_length == 7 * 48 and len(context) == 7 * 48
    assert list(context.columns) == ["id", "timestamp", "target"]
    assert future_df is None and prediction_length == 48 and quantiles == [0.1, 0.5, 0.9]
    assert len(out) == 2 * 48


def test_calendar_variant_sends_known_future_covariates():
    pipe, _, _, _ = run(use_calendar=True)
    context, future_df, _, _ = pipe.calls[0]
    assert list(context.columns) == ["id", "timestamp", "target"] + CALENDAR_COLUMNS
    assert list(future_df.columns) == ["id", "timestamp"] + CALENDAR_COLUMNS  # covariates, but no target
    assert len(future_df) == 48


def test_context_never_includes_the_origin_or_later():
    pipe, _, out, _ = run(use_calendar=False)
    for (context, *_), origin in zip(pipe.calls, out["origin"].unique()):
        assert context["timestamp"].max() == origin - STEP


def test_forecast_is_mapped_to_p10_p50_p90_in_time_order():
    _, _, out, frame = run(use_calendar=False)
    first = out[out["origin"] == out["origin"].min()]
    last_history = frame.loc[first["origin"].iloc[0] - STEP, "rrp"]
    assert (first["p50"] == last_history).all()
    assert (first["p10"] == last_history - 1).all() and (first["p90"] == last_history + 1).all()
    assert first["timestamp"].is_monotonic_increasing
