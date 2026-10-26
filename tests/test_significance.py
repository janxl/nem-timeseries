import numpy as np
import pandas as pd
import pytest

from nem_ts.significance import block_bootstrap_ci, compare_to_reference, per_origin_errors


def preds(errors_by_day, probabilistic=True, width=1.0):
    """48 half-hours per day; the p50 is off by that day's error, with a symmetric band of +-width."""
    rows = []
    for day, err in enumerate(errors_by_day):
        origin = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)
        for step in range(48):
            rows.append({"origin": origin, "step": step, "actual": 100.0, "p10": 100 + err - width,
                         "p50": 100.0 + err, "p90": 100 + err + width})
    return pd.DataFrame(rows)


def test_per_origin_mae_is_the_mean_over_the_days_48_half_hours():
    out = per_origin_errors(preds([3.0, -5.0]), probabilistic=True)
    assert out["mae"].tolist() == [3.0, 5.0]
    assert out.index.is_monotonic_increasing


def test_pinball_is_nan_for_point_forecasts():
    assert per_origin_errors(preds([1.0]), probabilistic=False)["pinball"].isna().all()


def test_bootstrap_of_a_constant_has_a_zero_width_interval():
    mean, lo, hi = block_bootstrap_ci(np.full(100, 5.0))
    assert mean == lo == hi == 5.0


def test_interval_excludes_zero_for_a_real_difference_and_includes_it_for_noise():
    rng = np.random.default_rng(1)
    _, lo, hi = block_bootstrap_ci(rng.normal(2.0, 1.0, 300))
    assert lo > 0
    _, lo, hi = block_bootstrap_ci(rng.normal(0.0, 1.0, 300))
    assert lo < 0 < hi


def test_bootstrap_is_reproducible_and_handles_short_series():
    x = np.arange(20, dtype=float)
    assert block_bootstrap_ci(x, seed=3) == block_bootstrap_ci(x, seed=3)
    assert block_bootstrap_ci(np.array([1.0, 2.0]), block=7)[0] == 1.5


def test_compare_to_reference_signs_and_win_rate():
    rng = np.random.default_rng(0)
    base = rng.uniform(5, 15, 120)
    results = {
        "ref": (preds(base), True),
        "worse": (preds(base + 3.0), True),   # always 3 worse
        "better": (preds(base - 2.0 * (base > 0)), True),  # always 2 better in MAE
    }
    table = compare_to_reference(results, "ref")
    assert "ref" not in table.index
    assert table.loc["worse", "mae_diff"] == pytest.approx(3.0)
    assert table.loc["worse", "mae_lo"] > 0                 # clearly worse
    assert table.loc["worse", "mae_better_days"] == 0.0
    assert table.loc["better", "mae_diff"] == pytest.approx(-2.0)
    assert table.loc["better", "mae_better_days"] == 1.0
    assert table.attrs["reference"] == "ref" and table.attrs["n_origins"] == 120


def test_point_forecasts_have_no_pinball_comparison():
    results = {"ref": (preds([1.0] * 20), True), "naive": (preds([2.0] * 20), False)}
    table = compare_to_reference(results, "ref")
    assert np.isnan(table.loc["naive", "pinball_diff"]) and table.loc["naive", "mae_diff"] == pytest.approx(1.0)


def test_different_origins_are_rejected():
    results = {"ref": (preds([1.0] * 20), True), "other": (preds([1.0] * 19), True)}
    with pytest.raises(ValueError, match="same origins"):
        compare_to_reference(results, "ref")
