# datasets/

A small snapshot of public AEMO data, committed so the benchmark (and the Colab notebook) runs on identical inputs.

## Source

- **AEMO NEM data:** dispatch prices (`DISPATCHPRICE`, column `RRP`) and regional demand (`DISPATCHREGIONSUM`,
  column `TOTALDEMAND`), 5-minute, averaged here to 30 minutes.
- **Attribution:** Data (c) Australian Energy Market Operator Limited. Used under AEMO's
  [copyright permissions](https://www.aemo.com.au/Privacy-and-Legal-Notices/Copyright-Permissions-Notice).
- Only public data is used.

## Files

`VIC1/actuals_30min.parquet`, with `VIC1/manifest.json` describing it.

| column | meaning |
|---|---|
| `timestamp` | start of the 30-minute period, NEM time (AEST, UTC+10, no daylight saving), timezone-naive |
| `rrp` | regional reference price, $/MWh: mean of the six 5-minute dispatch prices in the period |
| `demand` | `TOTALDEMAND`, MW: mean of the six 5-minute values in the period |

A value is missing (NaN) if any of its six 5-minute intervals is missing.

## Coverage (VIC1)

| | first | last | missing half-hours |
|---|---|---|---|
| `rrp` | 2023-01-01 00:00 | 2026-10-25 23:30 | 130 (on 28 days) |
| `demand` | 2023-01-01 00:00 | 2026-07-09 13:30 | no values after 9 Jul 2026 in this snapshot |

Use `rrp` for anything that must cover the whole period. The benchmark currently uses `rrp` only.

## Get your own data

Everything here can be rebuilt from AEMO with [`nemosis`](https://github.com/UNSW-CEEM/NEMOSIS) (installed by
`make setup`). This reproduces the file above for any region and period. AEMO stamps each 5-minute value with the
**end** of its interval, which is why the bins are right-closed and labelled by their start.

```python
from pathlib import Path

import pandas as pd
from nemosis import dynamic_data_compiler

REGION, START, END = "VIC1", "2023/01/01 00:00:00", "2026/10/26 00:00:00"   # nemosis wants YYYY/MM/DD


def pull(table, column):
    df = dynamic_data_compiler(
        START, END, table, "nemosis_cache", fformat="parquet",
        select_columns=["SETTLEMENTDATE", "REGIONID", "INTERVENTION", column],
        filter_cols=["REGIONID", "INTERVENTION"], filter_values=([REGION], [0]),   # 0 = no intervention pricing
    )
    return df.set_index("SETTLEMENTDATE")[column]


def half_hourly(s):
    g = s.resample("30min", closed="right", label="left").agg(["mean", "count"])
    return g["mean"].where(g["count"] == 6)   # only complete half-hours


out = pd.DataFrame({
    "rrp": half_hourly(pull("DISPATCHPRICE", "RRP")),
    "demand": half_hourly(pull("DISPATCHREGIONSUM", "TOTALDEMAND")),
})
out.index.name = "timestamp"
out = out.dropna(how="all").reset_index()
Path(f"datasets/{REGION}").mkdir(parents=True, exist_ok=True)
out.to_parquet(f"datasets/{REGION}/actuals_30min.parquet", index=False)
```

The first run downloads AEMO's monthly archives (a few MB per month and table) into `nemosis_cache/`, which can take a
while for several years.
