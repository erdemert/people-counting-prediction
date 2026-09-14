"""Load a brand's raw hourly CSV -> clean daily per-store panel.

Anomaly handling: single-day spikes (>8x a store's own 15-day rolling
median, reverting the next day) are winsorized to the rolling median rather
than dropped, so every store keeps a continuous daily series. Sustained
step-changes (new stores ramping up, seasonal closures) are left untouched --
they are real signal, not noise.
"""
import os
import numpy as np
import pandas as pd

from config import DATA_ROOT, RAW_COLUMNS, BRANDS

ANOMALY_RATIO = 8.0
ANOMALY_ABS_FLOOR = 50


def load_raw(brand_key: str) -> pd.DataFrame:
    cfg = BRANDS[brand_key]
    path = os.path.join(DATA_ROOT, cfg["csv"])
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS,
                      usecols=["store_id", "store_name", "count_in", "ts"],
                      parse_dates=["ts"])
    df["date"] = df["ts"].dt.normalize()
    daily = df.groupby(["store_id", "store_name", "date"], as_index=False)["count_in"].sum()
    return daily


def _winsorize_store(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    med = g["count_in"].rolling(15, center=True, min_periods=5).median()
    ratio = g["count_in"] / med.replace(0, np.nan)
    next_val = g["count_in"].shift(-1)
    next_med = med.shift(-1)
    reverted = next_val <= 2 * next_med.replace(0, np.nan)
    is_anomaly = (ratio > ANOMALY_RATIO) & (g["count_in"] > ANOMALY_ABS_FLOOR) & reverted.fillna(True)
    g.loc[is_anomaly, "count_in"] = med[is_anomaly].round()
    g["was_anomaly"] = is_anomaly
    return g


def clean_and_reindex(daily: pd.DataFrame) -> pd.DataFrame:
    cleaned = daily.groupby("store_id", group_keys=False)[daily.columns].apply(_winsorize_store)

    out = []
    for store_id, g in cleaned.groupby("store_id"):
        store_name = g["store_name"].iloc[0]
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        s = g.set_index("date")["count_in"].reindex(full_idx)
        s = s.interpolate(limit=2)  # bridge short gaps only
        frame = pd.DataFrame({"date": full_idx, "store_id": store_id,
                               "store_name": store_name, "count_in": s.values})
        out.append(frame)
    panel = pd.concat(out, ignore_index=True)
    panel = panel.dropna(subset=["count_in"])
    panel["count_in"] = panel["count_in"].clip(lower=0)
    return panel


def build_panel(brand_key: str) -> pd.DataFrame:
    region_fn = BRANDS[brand_key]["region_fn"]
    raw = load_raw(brand_key)
    panel = clean_and_reindex(raw)
    panel["region"] = panel["store_name"].map(region_fn)
    panel["brand"] = brand_key
    return panel.sort_values(["store_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "bluemint"
    p = build_panel(key)
    print(p.shape, p["store_id"].nunique(), p["date"].min(), p["date"].max())
    print(p.groupby("region")["store_id"].nunique())

