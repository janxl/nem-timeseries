import numpy as np
import pandas as pd
import pytest

from nem_ts.backtest import check_no_leakage, run_backtest, valid_origins
from nem_ts.data import STEP, calendar_features
from nem_ts.models.lgbm import CONTEXT_LENGTH, FEATURES, LightGBMQuantile, build_training_set, make_features


def toy_frame(days=60, start="2025-01-01"):
    """A price that repeats every day: the value is the half-hour of day (0-47)."""
    index = pd.date_range(start, periods=days * 48, freq=STEP, name="timestamp")
    df = pd.DataFrame({"rrp": (index.hour * 2 + index.minute // 30).astype(float)}, index=index)
    return df.join(calendar_features(index))


def test_features_pick_the_right_lags():
    prices = np.arange(CONTEXT_LENGTH, dtype=float)  # 7 days x 48; the value is its own position
    x = make_features(prices, dow=np.full(48, 2), is_holiday=np.zeros(48, dtype=int))
    assert list(x.columns) == FEATURES and len(x) == 48
    k = x["step"].to_numpy()
    assert (x["lag_1d"] == 288 + k).all()   # same half-hour yesterday (day 7 of 7)
    assert (x["lag_2d"] == 240 + k).all()
    assert (x["lag_7d"] == k).all()         # same half-hour a week ago (day 1 of 7)
    assert (x["same_hh_mean_7d"] == k + 144).all()  # mean of k, k+48, ..., k+288
    assert (x["same_hh_max_7d"] == k + 288).all()
    assert (x["last_price"] == CONTEXT_LENGTH - 1).all()
    assert (x["yday_max"] == CONTEXT_LENGTH - 1).all() and (x["yday_min"] == 288).all()


def test_features_cope_with_gaps_and_reject_wrong_length():
    prices = np.full(CONTEXT_LENGTH, np.nan)
    x = make_features(prices, np.zeros(48, dtype=int), np.zeros(48, dtype=int))
    assert x["lag_1d"].isna().all()
    with pytest.raises(ValueError, match="336"):
        make_features(np.zeros(10), np.zeros(48), np.zeros(48))


def test_training_set_only_uses_data_up_to_train_end():
    frame = toy_frame()
    train_end = pd.Timestamp("2025-02-09 23:30")
    # make everything after train_end absurd: if any of it leaked into training, y would show it
    frame.loc[train_end + STEP :, "rrp"] = 1e6
    x, y = build_training_set(frame, train_end)
    assert y.max() <= 47
    assert len(x) == len(y) > 0


def test_fit_predict_learns_a_repeating_daily_pattern():
    frame = toy_frame(days=60)
    model = LightGBMQuantile({"num_boost_round": 60, "min_data_in_leaf": 5}).fit(frame, pd.Timestamp("2025-02-14 23:30"))
    assert model.trained_through == pd.Timestamp("2025-02-14 23:30")
    origins, _ = valid_origins(frame, start="2025-02-20", end="2025-02-22")
    out = run_backtest(model, frame, origins)
    assert (out["p10"] <= out["p50"]).all() and (out["p50"] <= out["p90"]).all()
    assert (out["p50"] - out["actual"]).abs().mean() < 1.0


def test_leakage_guard_rejects_origins_the_model_trained_on():
    class Trained:
        name, context_length, probabilistic = "x", 48, True
        trained_through = pd.Timestamp("2025-02-14 23:30")

    check_no_leakage(Trained(), pd.DatetimeIndex(["2025-02-15"]))  # first midnight after training: fine
    with pytest.raises(ValueError, match="trained on data up to"):
        check_no_leakage(Trained(), pd.DatetimeIndex(["2025-02-14"]))
