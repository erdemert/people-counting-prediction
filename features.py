"""Calendar/holiday feature engineering shared by all four models."""
import numpy as np
import pandas as pd

from holidays import get_region_events

MAX_HORIZON_DAYS = 60


def _region_lookup_frame(region: str) -> pd.DataFrame:
    events = get_region_events(region)
    if not events:
        return pd.DataFrame(columns=["event_date", "label"])
    ev = pd.DataFrame(events, columns=["event_date", "label", "kind"])
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    return ev.sort_values("event_date")


def add_calendar_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have 'date' (datetime64) and 'region' columns. Adds dow, month,
    is_weekend, days_to_next_holiday, days_since_last_holiday, holiday_today."""
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek.astype("category")
    df["month"] = df["date"].dt.month.astype("category")
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

    out_next = np.full(len(df), MAX_HORIZON_DAYS, dtype=float)
    out_prev = np.full(len(df), MAX_HORIZON_DAYS, dtype=float)
    out_today = np.array(["none"] * len(df), dtype=object)

    for region, idx in df.groupby("region").groups.items():
        ev = _region_lookup_frame(region)
        if ev.empty:
            continue
        dates = df.loc[idx, "date"].values.astype("datetime64[D]")
        ev_dates = ev["event_date"].values.astype("datetime64[D]")
        ev_labels = ev["label"].values
        pos = np.searchsorted(ev_dates, dates, side="left")

        next_days = np.full(len(dates), MAX_HORIZON_DAYS, dtype=float)
        valid_next = pos < len(ev_dates)
        next_days[valid_next] = (ev_dates[pos[valid_next]] - dates[valid_next]).astype("timedelta64[D]").astype(float)

        # An exact hit (next_days == 0) means dates[i] IS an event day itself --
        # searchsorted(side="left") puts `pos` AT the match in that case, not
        # after it, so the "today" check must look at `pos`, not `pos - 1`.
        # (Previously this checked `prev_days == 0` using `pos - 1`, which
        # only ever matched a date exactly one day after some *earlier*
        # event -- so holiday_today was silently "none" on every actual
        # holiday day, for every region, the whole time.)
        exact_today = valid_next & (next_days == 0)

        prev_pos = pos - 1
        prev_days = np.full(len(dates), MAX_HORIZON_DAYS, dtype=float)
        valid_prev = prev_pos >= 0
        prev_days[valid_prev] = (dates[valid_prev] - ev_dates[prev_pos[valid_prev]]).astype("timedelta64[D]").astype(float)
        prev_days[exact_today] = 0  # the "last" holiday on a holiday day is today itself

        today_label = np.array(["none"] * len(dates), dtype=object)
        today_label[exact_today] = ev_labels[pos[exact_today]]

        out_next[idx] = np.minimum(next_days, MAX_HORIZON_DAYS)
        out_prev[idx] = np.minimum(prev_days, MAX_HORIZON_DAYS)
        out_today[idx] = today_label

    df["days_to_next_holiday"] = out_next
    df["days_since_last_holiday"] = out_prev
    df["holiday_today"] = pd.Categorical(out_today)
    return df


def add_lag_features(df: pd.DataFrame, target_col="count_in",
                      lags=(1, 7, 14, 28), roll_windows=(7, 28)) -> pd.DataFrame:
    """df: one row per (store_id, date), sorted by store_id, date."""
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
    g = df.groupby("store_id")[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in roll_windows:
        shifted = g.shift(1)
        df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=max(2, w // 3)).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=max(2, w // 3)).std().reset_index(level=0, drop=True)
    return df


TABULAR_CAT_FEATURES = ["store_id", "region", "dow", "month", "holiday_today"]
TABULAR_NUM_FEATURES = ["is_weekend", "days_to_next_holiday", "days_since_last_holiday",
                         "lag_1", "lag_7", "lag_14", "lag_28",
                         "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28"]
