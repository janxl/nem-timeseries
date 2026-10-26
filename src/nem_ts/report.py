"""Turn a metrics table into markdown."""

from __future__ import annotations

import math

import pandas as pd

COLUMNS = [
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("mae_spike", "MAE spikes"),
    ("mae_negative", "MAE negative"),
    ("coverage_p10_p90", "P10-P90 coverage"),
    ("pinball", "Pinball"),
]


def _fmt(column: str, value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.1%}" if column == "coverage_p10_p90" else f"{value:.2f}"


def _diff_cell(row: pd.Series, metric: str) -> str:
    mean = row[f"{metric}_diff"]
    if mean is None or math.isnan(mean):
        return "–"
    lo, hi = row[f"{metric}_lo"], row[f"{metric}_hi"]
    mark = " *" if (lo > 0 or hi < 0) else ""  # the interval excludes zero
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]{mark}"


def significance_markdown(table: pd.DataFrame, *, level: float, block_days: int) -> str:
    """Each model against the reference model, with a bootstrap confidence interval on the difference."""
    ref, n = table.attrs["reference"], table.attrs["n_origins"]
    lines = [
        f"## Is the difference real? Each model minus `{ref}`",
        "",
        f"Paired by forecast day ({n} origins); {level:.0%} confidence intervals from a {block_days}-day block bootstrap. "
        "A positive difference means the model is **worse** than the reference. "
        "`*` marks an interval that excludes zero, i.e. a difference unlikely to be luck.",
        "",
        "| model | MAE difference [95% CI] | better on % of days | pinball difference [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for model, row in table.iterrows():
        better = "–" if math.isnan(row["mae_better_days"]) else f"{row['mae_better_days']:.0%}"
        lines.append(f"| {model} | {_diff_cell(row, 'mae')} | {better} | {_diff_cell(row, 'pinball')} |")
    lines += ["", "Differences are in $/MWh. \"Better on % of days\" compares the daily MAE of the model with the reference.", ""]
    return "\n".join(lines)


def metrics_markdown(table: pd.DataFrame, *, title: str, notes: list[str]) -> str:
    """A markdown table, one row per model. MAE, RMSE and pinball are in $/MWh."""
    lines = [f"# {title}", ""]
    lines += [f"- {n}" for n in notes] + [""]
    lines.append("| model | " + " | ".join(label for _, label in COLUMNS) + " |")
    lines.append("|---|" + "---:|" * len(COLUMNS))
    for model, row in table.iterrows():
        lines.append(f"| {model} | " + " | ".join(_fmt(col, row[col]) for col, _ in COLUMNS) + " |")
    lines += ["", "Errors are in $/MWh. MAE, RMSE and pinball: lower is better; coverage should be near 80%.", ""]
    return "\n".join(lines)
