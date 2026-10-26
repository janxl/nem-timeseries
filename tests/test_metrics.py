import math

import pandas as pd
import pytest

from nem_ts.metrics import pinball_loss, score, score_models, spike_threshold


def frame(actual, p10=None, p50=None, p90=None, origin=None):
    p50 = p50 if p50 is not None else actual
    return pd.DataFrame(
        {
            "origin": origin if origin is not None else [0] * len(actual),
            "step": range(len(actual)),
            "actual": actual,
            "p10": p10 if p10 is not None else p50,
            "p50": p50,
            "p90": p90 if p90 is not None else p50,
        }
    )


def test_mae_and_rmse_by_hand():
    # errors are 1, 0, 2, 4
    s = score(frame([1.0, 2.0, 3.0, 4.0], p50=[2.0, 2.0, 5.0, 0.0]), spike_threshold=100)
    assert s["mae"] == pytest.approx(7 / 4)
    assert s["rmse"] == pytest.approx(math.sqrt(21 / 4))


def test_spike_mae_uses_only_prices_at_or_above_the_threshold():
    s = score(frame([10.0, 100.0, 200.0], p50=[10.0, 90.0, 150.0]), spike_threshold=100.0)
    assert s["n_spike"] == 2
    assert s["mae_spike"] == pytest.approx((10 + 50) / 2)


def test_negative_mae_uses_only_negative_prices():
    s = score(frame([-10.0, 0.0, 5.0, -30.0], p50=[0.0, 0.0, 5.0, -10.0]), spike_threshold=100)
    assert s["n_negative"] == 2  # zero is not negative
    assert s["mae_negative"] == pytest.approx((10 + 20) / 2)


def test_coverage_counts_actuals_inside_the_band():
    actual = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = score(frame(actual, p10=[0.0] * 5, p50=actual, p90=[4.0] * 5), spike_threshold=100)
    assert s["coverage_p10_p90"] == pytest.approx(4 / 5)  # only 5.0 is above p90


def test_pinball_loss_by_hand():
    # y=10: q=0.1 -> 0.1*2=0.2 ; q=0.5 -> 0.5*2=1.0 ; q=0.9 -> 0.1*5=0.5
    df = frame([10.0], p10=[8.0], p50=[12.0], p90=[15.0])
    assert pinball_loss(df["actual"], df["p10"], 0.1) == pytest.approx(0.2)
    assert pinball_loss(df["actual"], df["p50"], 0.5) == pytest.approx(1.0)
    assert pinball_loss(df["actual"], df["p90"], 0.9) == pytest.approx(0.5)
    assert score(df, spike_threshold=100)["pinball"] == pytest.approx((0.2 + 1.0 + 0.5) / 3)


def test_point_forecasts_have_no_coverage_or_pinball():
    s = score(frame([1.0, 2.0]), spike_threshold=100, probabilistic=False)
    assert math.isnan(s["coverage_p10_p90"]) and math.isnan(s["pinball"])
    assert s["mae"] == 0.0


def test_spike_threshold_is_the_95th_percentile():
    assert spike_threshold(pd.Series(range(101))) == pytest.approx(95.0)


def test_score_models_uses_one_threshold_and_rejects_different_origins():
    a = frame([1.0, 2.0, 300.0, 4.0])
    b = frame([1.0, 2.0, 300.0, 4.0], p50=[0.0, 0.0, 0.0, 0.0])
    table = score_models({"a": (a, False), "b": (b, False)})
    assert table.attrs["spike_threshold"] == pytest.approx(spike_threshold(a["actual"]))
    assert table.loc["a", "mae"] == 0.0 and table.loc["b", "mae"] > 0
    other = frame([1.0, 2.0, 999.0, 4.0])
    with pytest.raises(ValueError, match="same origins"):
        score_models({"a": (a, False), "c": (other, False)})
