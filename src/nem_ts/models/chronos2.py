"""Amazon Chronos-2: zero-shot, or fine-tuned with LoRA on our own prices.

One forecaster class serves both. It is price only unless `use_calendar` is set, in which case the calendar features
(half-hour, weekday, holiday) are also passed as known-future covariates.

Fine-tuning (`fine_tune`) trains LoRA adapters on the price history up to `train_end` only, then saves a checkpoint
that `load_finetuned` reads back, so the backtest never retrains. Run it on a GPU where possible (e.g. Colab).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nem_ts.data import CALENDAR_COLUMNS, STEPS_PER_DAY
from nem_ts.metrics import QUANTILE_COLUMNS, QUANTILES

MODEL_ID = "amazon/chronos-2"
SERIES_ID = "series"
FT_NAME = "chronos2_ft"
CKPT_NAME = "finetuned-ckpt"
DEFAULT_TRAIN_END = "2024-12-31 23:30"  # the half-hour before the test period starts


def load_pipeline(model_id: str = MODEL_ID):
    """Chronos-2 (or a saved fine-tuned checkpoint) on the GPU when there is one, otherwise on the CPU."""
    from chronos import Chronos2Pipeline

    from nem_ts.device import get_device

    return Chronos2Pipeline.from_pretrained(model_id, device_map=get_device())


class Chronos2Forecaster:
    probabilistic = True

    def __init__(self, context_days: int = 14, use_calendar: bool = False, name: str | None = None, pipeline=None,
                 trained_through: pd.Timestamp | None = None):
        self.context_length = context_days * STEPS_PER_DAY
        self.use_calendar = use_calendar
        self.name = name or f"chronos2_zs{'_cal' if use_calendar else ''}"
        self.pipeline = pipeline if pipeline is not None else load_pipeline()
        self.trained_through = trained_through  # None for zero-shot: nothing is fitted to our data

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        context = pd.DataFrame({"id": SERIES_ID, "timestamp": history.index, "target": history["rrp"].to_numpy(float)})
        future_df = None
        if self.use_calendar:
            for col in CALENDAR_COLUMNS:
                context[col] = history[col].to_numpy()
            future_df = pd.DataFrame({"id": SERIES_ID, "timestamp": future.index})
            for col in CALENDAR_COLUMNS:
                future_df[col] = future[col].to_numpy()
        out = self.pipeline.predict_df(
            context, future_df=future_df, id_column="id", timestamp_column="timestamp", target="target",
            prediction_length=len(future), quantile_levels=list(QUANTILES),
        ).sort_values("timestamp")
        if not pd.DatetimeIndex(out["timestamp"]).equals(future.index):
            raise ValueError(f"{self.name}: forecast timestamps do not match the requested half-hours")
        values = np.sort(out[[str(q) for q in QUANTILES]].to_numpy(float), axis=1)  # no crossing quantiles
        return pd.DataFrame(dict(zip(QUANTILE_COLUMNS, values.T)), index=future.index)


@dataclass
class FineTuneConfig:
    context_days: int = 14
    num_steps: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-5  # the Chronos-2 docs suggest about 1e-5 for LoRA
    train_end: str = DEFAULT_TRAIN_END
    seed: int = 0
    finetune_mode: str = "lora"


def fine_tune(frame: pd.DataFrame, out_dir: Path | str, config: FineTuneConfig | None = None,
              pipeline=None) -> Chronos2Forecaster:
    """Fine-tune on the price history up to config.train_end (never later) and save the checkpoint in out_dir."""
    cfg = config or FineTuneConfig()
    out_dir = Path(out_dir)
    train_end = pd.Timestamp(cfg.train_end)
    prices = frame.loc[:train_end, "rrp"].to_numpy(np.float32)
    n_missing = int(np.isnan(prices).sum())
    context_length = cfg.context_days * STEPS_PER_DAY

    np.random.seed(cfg.seed)  # Chronos-2 draws its training windows from numpy's global generator
    pipeline = pipeline if pipeline is not None else load_pipeline()
    try:
        import torch

        torch.manual_seed(cfg.seed)
    except ImportError:
        pass

    started = time.time()
    tuned = pipeline.fit(
        [{"target": prices}],
        prediction_length=STEPS_PER_DAY,
        finetune_mode=cfg.finetune_mode,
        context_length=context_length,
        min_past=context_length,  # every training window has the full context, as in the backtest
        learning_rate=cfg.learning_rate,
        num_steps=cfg.num_steps,
        batch_size=cfg.batch_size,
        output_dir=str(out_dir),
        finetuned_ckpt_name=CKPT_NAME,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        **asdict(cfg), "n_training_prices": len(prices), "n_missing": n_missing,
        "trained_on": [str(frame.index[0]), str(min(train_end, frame.index[-1]))],
        "seconds": round(time.time() - started), "base_model": MODEL_ID,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return Chronos2Forecaster(cfg.context_days, name=FT_NAME, pipeline=tuned, trained_through=train_end)


def load_finetuned(out_dir: Path | str, loader=load_pipeline) -> Chronos2Forecaster:
    """Load a checkpoint written by fine_tune (the LoRA adapter is merged into the base model on load)."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    return Chronos2Forecaster(
        meta["context_days"], name=FT_NAME, pipeline=loader(str(out_dir / CKPT_NAME)),
        trained_through=pd.Timestamp(meta["train_end"]),
    )
