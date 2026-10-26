import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nem_ts.backtest import check_no_leakage
from nem_ts.cli import app
from nem_ts.data import STEP, calendar_features
from nem_ts.models.chronos2 import CKPT_NAME, FT_NAME, FineTuneConfig, fine_tune, load_finetuned


class FakePipeline:
    """Stands in for Chronos2Pipeline: records fit() arguments, returns a 'tuned' pipeline."""

    def __init__(self):
        self.fit_args = None

    def fit(self, inputs, **kwargs):
        self.fit_args = (inputs, kwargs)
        return FakePipeline()


def toy_frame(days=60):
    index = pd.date_range("2025-01-01", periods=days * 48, freq=STEP, name="timestamp")
    df = pd.DataFrame({"rrp": np.arange(len(index), dtype=float)}, index=index)
    return df.join(calendar_features(index))


def test_fine_tune_uses_only_data_up_to_train_end_and_the_documented_settings(tmp_path):
    frame = toy_frame()
    train_end = pd.Timestamp("2025-01-31 23:30")
    frame.loc[train_end + STEP :, "rrp"] = 1e9  # anything after train_end must never reach the trainer
    pipe = FakePipeline()
    cfg = FineTuneConfig(num_steps=5, batch_size=4, train_end=str(train_end))
    model = fine_tune(frame, tmp_path / FT_NAME, cfg, pipeline=pipe)

    inputs, kw = pipe.fit_args
    assert len(inputs) == 1 and inputs[0]["target"].max() < 1e9
    assert len(inputs[0]["target"]) == 31 * 48
    assert inputs[0]["target"].dtype == np.float32
    assert kw["finetune_mode"] == "lora" and kw["prediction_length"] == 48
    assert kw["context_length"] == 672 and kw["min_past"] == 672  # every window has the full 14-day context
    assert kw["num_steps"] == 5 and kw["batch_size"] == 4 and kw["finetuned_ckpt_name"] == CKPT_NAME
    assert model.trained_through == train_end and model.name == FT_NAME


def test_fine_tune_writes_metadata_and_load_reads_it_back(tmp_path):
    out = tmp_path / FT_NAME
    fine_tune(toy_frame(), out, FineTuneConfig(num_steps=3, train_end="2025-01-31 23:30"), pipeline=FakePipeline())
    meta = json.loads((out / "meta.json").read_text())
    assert meta["train_end"] == "2025-01-31 23:30" and meta["finetune_mode"] == "lora" and meta["num_steps"] == 3

    seen = []
    model = load_finetuned(out, loader=lambda path: seen.append(path) or FakePipeline())
    assert seen == [str(out / CKPT_NAME)]
    assert model.trained_through == pd.Timestamp("2025-01-31 23:30") and model.context_length == 672


def test_a_fine_tuned_model_cannot_be_scored_on_data_it_trained_on(tmp_path):
    out = tmp_path / FT_NAME
    fine_tune(toy_frame(), out, FineTuneConfig(num_steps=1, train_end="2025-01-31 23:30"), pipeline=FakePipeline())
    model = load_finetuned(out, loader=lambda path: FakePipeline())
    check_no_leakage(model, pd.DatetimeIndex(["2025-02-01"]))
    with pytest.raises(ValueError, match="trained on data up to"):
        check_no_leakage(model, pd.DatetimeIndex(["2025-01-31"]))


def test_finetune_command_rejects_a_train_end_inside_the_test_period(tmp_path):
    out = CliRunner().invoke(app, ["finetune", "--data", str(tmp_path), "--models-dir", str(tmp_path / "m"),
                                   "--train-end", "2025-06-30 23:30"])
    assert out.exit_code != 0 and "before the test start" in out.output


def test_backtest_skips_a_model_whose_checkpoint_is_missing(tmp_path):
    root = tmp_path / "datasets" / "VIC1"
    root.mkdir(parents=True)
    index = pd.date_range("2025-01-01", periods=30 * 48, freq="30min", name="timestamp")
    pd.DataFrame({"timestamp": index, "rrp": (index.hour * 2 + index.minute // 30).astype(float), "demand": 1.0}
                 ).to_parquet(root / "actuals_30min.parquet", index=False)
    out = CliRunner().invoke(app, ["backtest", "--data", str(tmp_path / "datasets"), "--results", str(tmp_path / "r"),
                                   "--models-dir", str(tmp_path / "none"), "--start", "2025-01-10",
                                   "--models", "naive_yesterday,chronos2_ft"])
    assert out.exit_code == 0, out.output
    assert "chronos2_ft: skipped" in out.output and "make finetune" in out.output
