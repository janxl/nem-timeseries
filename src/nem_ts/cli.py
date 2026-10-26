"""Command line: `nem-ts backtest` (also: make backtest)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from nem_ts.backtest import DEFAULT_TEST_START, run_backtest, valid_origins
from nem_ts.data import load_actuals
from nem_ts.metrics import score_models
from nem_ts.models.chronos2 import DEFAULT_TRAIN_END, FineTuneConfig  # light: torch is only imported on use
from nem_ts.models.registry import REGISTRY, ModelNotReady, get_spec
from nem_ts.report import metrics_markdown, significance_markdown
from nem_ts.significance import BLOCK_DAYS, LEVEL, compare_to_reference

app = typer.Typer(help="Benchmark time-series models on NEM prices.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Benchmark time-series models on NEM prices."""


@app.command()
def backtest(
    models: str = typer.Option(",".join(REGISTRY), help="Comma-separated model names."),
    region: str = typer.Option("VIC1"),
    start: str = typer.Option(DEFAULT_TEST_START, help="First forecast origin (NEM midnight)."),
    end: Optional[str] = typer.Option(None, help="Last forecast origin (default: as late as the data allows)."),
    data: Optional[Path] = typer.Option(None, help="Folder holding <region>/actuals_30min.parquet (default: datasets/)."),
    results: Path = typer.Option(Path("results"), help="Where predictions and metrics are written."),
    force: bool = typer.Option(False, help="Recompute models even if saved predictions exist."),
    reference: str = typer.Option("chronos2_zs", help="Model the significance table compares the others against."),
    models_dir: Path = typer.Option(Path("models"), help="Where fine-tuned checkpoints are saved (make finetune)."),
) -> None:
    """Forecast the next 48 half-hours from every midnight, score the models, and save the results."""
    frame = load_actuals(region, data)
    origins, skipped = valid_origins(frame, start=start, end=end)
    out_dir = results / region
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"{region}: {len(origins)} origins ({origins[0].date()} to {origins[-1].date()}), "
               f"{skipped} skipped for gaps in the actual prices")

    for name in [m.strip() for m in models.split(",") if m.strip()]:
        spec = get_spec(name)
        path = out_dir / f"predictions_{name}.parquet"
        if path.exists() and not force and _same_origins(pd.read_parquet(path), origins):
            typer.echo(f"  {name}: using saved predictions ({path})")
            continue
        typer.echo(f"  {name}: running...")
        try:
            model = spec.factory(frame, models_dir)
        except ModelNotReady as err:
            typer.echo(f"  {name}: skipped, {err}")
            continue
        if getattr(model, "trained_through", None) is not None:
            typer.echo(f"  {name}: trained on data up to {model.trained_through}")
        try:
            run_backtest(model, frame, origins).to_parquet(path, index=False)
        except ValueError as err:
            raise typer.BadParameter(str(err)) from err

    scored = {}
    for name in REGISTRY:  # every saved model on these exact origins, not only the ones just run
        path = out_dir / f"predictions_{name}.parquet"
        if path.exists():
            preds = pd.read_parquet(path)
            if _same_origins(preds, origins):
                scored[name] = (preds, REGISTRY[name].probabilistic)
    if not scored:
        typer.echo("No model has predictions on these origins, nothing to score.")
        raise typer.Exit(1)
    table = score_models(scored)
    table.to_csv(out_dir / "metrics.csv", index_label="model")
    text = metrics_markdown(
        table,
        title=f"{region} backtest results",
        notes=[
            f"{len(origins)} forecast origins (NEM midnight), {origins[0].date()} to {origins[-1].date()}; "
            f"{skipped} more skipped because the actual prices had gaps.",
            "Each origin forecasts the next 48 half-hours (24 hours).",
            f"Spike = actual price at or above ${table.attrs['spike_threshold']:.1f}/MWh (top 5% of the evaluated prices).",
        ],
    )
    if len(scored) > 1:
        ref = reference if reference in scored else next(iter(scored))
        sig = compare_to_reference(scored, ref)
        sig.to_csv(out_dir / "significance.csv", index_label="model")
        text += "\n" + significance_markdown(sig, level=LEVEL, block_days=BLOCK_DAYS)
    (out_dir / "metrics.md").write_text(text, encoding="utf-8")
    typer.echo("\n" + text)
    typer.echo(f"Saved to {out_dir}/ (predictions_<model>.parquet, metrics.csv, metrics.md)")


@app.command()
def finetune(
    region: str = typer.Option("VIC1"),
    data: Optional[Path] = typer.Option(None, help="Folder holding <region>/actuals_30min.parquet (default: datasets/)."),
    models_dir: Path = typer.Option(Path("models"), help="Where the checkpoint is saved."),
    num_steps: int = typer.Option(FineTuneConfig.num_steps, help="Training steps (use a GPU for the full run)."),
    batch_size: int = typer.Option(FineTuneConfig.batch_size),
    learning_rate: float = typer.Option(FineTuneConfig.learning_rate),
    context_days: int = typer.Option(FineTuneConfig.context_days),
    train_end: str = typer.Option(DEFAULT_TRAIN_END, help="Last half-hour used for training; must be before the test start."),
    seed: int = typer.Option(FineTuneConfig.seed),
) -> None:
    """Fine-tune Chronos-2 (LoRA, price only) on 2023-24 and save the checkpoint for `backtest`."""
    from nem_ts.device import get_device
    from nem_ts.models.chronos2 import FT_NAME, fine_tune

    if pd.Timestamp(train_end) >= pd.Timestamp(DEFAULT_TEST_START):
        raise typer.BadParameter(f"train_end {train_end} must be before the test start {DEFAULT_TEST_START}")
    frame = load_actuals(region, data)
    out_dir = models_dir / FT_NAME
    typer.echo(f"Fine-tuning Chronos-2 on {region} prices up to {train_end} on {get_device()}: "
               f"{num_steps} steps, batch {batch_size}, lr {learning_rate:g}, context {context_days} days")
    cfg = FineTuneConfig(context_days=context_days, num_steps=num_steps, batch_size=batch_size,
                         learning_rate=learning_rate, train_end=train_end, seed=seed)
    fine_tune(frame, out_dir, cfg)
    typer.echo(f"Saved to {out_dir}/. Now run: make backtest")


def _same_origins(preds: pd.DataFrame, origins: pd.DatetimeIndex) -> bool:
    return pd.DatetimeIndex(preds["origin"].unique()).equals(origins)


if __name__ == "__main__":
    app()
