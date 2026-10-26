"""Models the CLI can run by name. Add new models here (factories are called lazily, so heavy ones load on use).

A factory receives the full actuals frame and the folder of saved models. Models that need training fit themselves
(LightGBM) or load a checkpoint made earlier by `make finetune` (fine-tuned Chronos-2). They must record
`trained_through` so the backtest can refuse to score data they trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from nem_ts.models.base import Forecaster
from nem_ts.models.naive import SeasonalNaive


class ModelNotReady(Exception):
    """The model needs something that has not been made yet (e.g. a fine-tuned checkpoint)."""


def _lightgbm(frame: pd.DataFrame, models_dir: Path) -> Forecaster:
    from nem_ts.models.lgbm import LightGBMQuantile

    return LightGBMQuantile().fit(frame)


def _chronos(use_calendar: bool, context_days: int = 14, name: str | None = None):
    def factory(frame: pd.DataFrame, models_dir: Path) -> Forecaster:
        from nem_ts.models.chronos2 import Chronos2Forecaster  # needs the `models` extra (torch)

        return Chronos2Forecaster(context_days=context_days, use_calendar=use_calendar, name=name)

    return factory


def _chronos_finetuned(frame: pd.DataFrame, models_dir: Path) -> Forecaster:
    from nem_ts.models.chronos2 import FT_NAME, load_finetuned

    path = Path(models_dir) / FT_NAME
    if not (path / "meta.json").exists():
        raise ModelNotReady(f"no fine-tuned checkpoint in {path}: run `make finetune` first")
    return load_finetuned(path)


@dataclass(frozen=True)
class ModelSpec:
    factory: Callable[[pd.DataFrame, Path], Forecaster]
    probabilistic: bool


REGISTRY: dict[str, ModelSpec] = {
    "naive_yesterday": ModelSpec(lambda frame, models_dir: SeasonalNaive(48), probabilistic=False),
    "naive_last_week": ModelSpec(lambda frame, models_dir: SeasonalNaive(336), probabilistic=False),
    "lightgbm": ModelSpec(_lightgbm, probabilistic=True),
    "chronos2_zs": ModelSpec(_chronos(use_calendar=False), probabilistic=True),
    "chronos2_zs_cal": ModelSpec(_chronos(use_calendar=True), probabilistic=True),
    "chronos2_ft": ModelSpec(_chronos_finetuned, probabilistic=True),
}

# Context-length ablation (price only): how many days of history Chronos-2 sees. 14 days is chronos2_zs.
CONTEXT_ABLATION_DAYS = (3, 7, 28, 42)
for _days in CONTEXT_ABLATION_DAYS:
    _name = f"chronos2_zs_ctx{_days}d"
    REGISTRY[_name] = ModelSpec(_chronos(use_calendar=False, context_days=_days, name=_name), probabilistic=True)


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(REGISTRY)}")
    return REGISTRY[name]
