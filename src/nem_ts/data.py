"""Load the exported actuals and build calendar features.

All timestamps are NEM time (AEST, UTC+10, no daylight saving), timezone-naive, labelling the start of a
30-minute period. See datasets/README.md.
"""

from __future__ import annotations

from pathlib import Path

import holidays
import pandas as pd

STEP = pd.Timedelta(minutes=30)
STEPS_PER_DAY = 48
CALENDAR_COLUMNS = ["half_hour", "dow", "is_holiday"]

DEFAULT_DATASETS = Path(__file__).resolve().parents[2] / "datasets"


def calendar_features(index: pd.DatetimeIndex, state: str = "VIC") -> pd.DataFrame:
    """Features known in advance for every timestamp: half-hour of day (0-47), weekday (0=Mon), public holiday."""
    years = range(int(index.year.min()), int(index.year.max()) + 1)
    holiday_dates = pd.to_datetime(list(holidays.Australia(subdiv=state, years=years)))
    return pd.DataFrame(
        {
            "half_hour": index.hour * 2 + index.minute // 30,
            "dow": index.dayofweek,
            "is_holiday": index.normalize().isin(holiday_dates).astype(int),
        },
        index=index,
    )


def load_actuals(region: str = "VIC1", root: Path | str | None = None) -> pd.DataFrame:
    """30-minute actuals on a complete regular index (missing periods are NaN), plus calendar features.

    Columns: rrp, demand (past-only, ends 9 Jul 2026), half_hour, dow, is_holiday.
    """
    path = Path(root or DEFAULT_DATASETS) / region / "actuals_30min.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: see datasets/README.md for how to get the data")
    df = pd.read_parquet(path).set_index("timestamp").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=STEP, name="timestamp"))
    return df.join(calendar_features(df.index))
