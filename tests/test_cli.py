import pandas as pd
from typer.testing import CliRunner

from nem_ts.cli import app

runner = CliRunner()


def write_toy_dataset(root, days=30):
    index = pd.date_range("2025-01-01", periods=days * 48, freq="30min", name="timestamp")
    df = pd.DataFrame({"timestamp": index, "rrp": (index.hour * 2 + index.minute // 30).astype(float), "demand": 1.0})
    (root / "VIC1").mkdir(parents=True)
    df.to_parquet(root / "VIC1" / "actuals_30min.parquet", index=False)


def test_backtest_command_saves_predictions_and_metrics(tmp_path):
    write_toy_dataset(tmp_path / "datasets")
    results = tmp_path / "results"
    args = ["backtest", "--data", str(tmp_path / "datasets"), "--results", str(results), "--start", "2025-01-10",
            "--models", "naive_yesterday,naive_last_week"]
    out = runner.invoke(app, args)
    assert out.exit_code == 0, out.output
    folder = results / "VIC1"
    assert (folder / "predictions_naive_yesterday.parquet").exists()
    assert (folder / "predictions_naive_last_week.parquet").exists()
    metrics = pd.read_csv(folder / "metrics.csv", index_col="model")
    assert metrics.loc["naive_yesterday", "mae"] == 0.0  # the toy price repeats every day
    assert "| naive_yesterday |" in (folder / "metrics_vs_naive_yesterday.md").read_text(encoding="utf-8")  # default reference not run: first model

    again = runner.invoke(app, args)  # second run reuses the saved predictions
    assert "using saved predictions" in again.output


def test_unknown_model_is_a_clear_error(tmp_path):
    write_toy_dataset(tmp_path / "datasets")
    out = runner.invoke(app, ["backtest", "--data", str(tmp_path / "datasets"), "--results", str(tmp_path / "r"),
                              "--start", "2025-01-10", "--models", "nope"])
    assert out.exit_code != 0


def test_each_reference_gets_its_own_files_and_none_overwrites_another(tmp_path):
    write_toy_dataset(tmp_path / "datasets")
    results = tmp_path / "results"
    base = ["backtest", "--data", str(tmp_path / "datasets"), "--results", str(results), "--start", "2025-01-10",
            "--models", "naive_yesterday,naive_last_week"]
    assert runner.invoke(app, base + ["--reference", "naive_yesterday"]).exit_code == 0
    folder = results / "VIC1"
    names = ("metrics_vs_naive_yesterday.md", "significance_vs_naive_yesterday.csv")
    before = {n: (folder / n).read_bytes() for n in names}

    out = runner.invoke(app, base + ["--reference", "naive_last_week"])
    assert out.exit_code == 0, out.output
    assert (folder / "significance_vs_naive_last_week.csv").exists()
    assert "`naive_last_week`" in (folder / "metrics_vs_naive_last_week.md").read_text(encoding="utf-8")
    assert {n: (folder / n).read_bytes() for n in names} == before  # the first reference's files are unchanged
    assert not (folder / "metrics.md").exists() and not (folder / "significance.csv").exists()


def test_a_reference_without_predictions_is_a_clear_error(tmp_path):
    write_toy_dataset(tmp_path / "datasets")
    out = runner.invoke(app, ["backtest", "--data", str(tmp_path / "datasets"), "--results", str(tmp_path / "r"),
                              "--start", "2025-01-10", "--models", "naive_yesterday,naive_last_week",
                              "--reference", "lightgbm"])
    assert out.exit_code != 0 and "no predictions" in out.output
