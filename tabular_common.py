"""Shared feature prep, categorical encoding, and recursive multi-step
forecasting loop used by both the LightGBM and CatBoost benchmarks."""
import numpy as np
import pandas as pd

from features import add_calendar_holiday_features, add_lag_features, \
    TABULAR_CAT_FEATURES, TABULAR_NUM_FEATURES

LAGS = (1, 7, 14, 28)
CAT_COLS = TABULAR_CAT_FEATURES
NUM_COLS = TABULAR_NUM_FEATURES
FEATURE_COLS = CAT_COLS + NUM_COLS


class CategoryEncoder:
    """Plain string -> int encoder, unseen values fall back to a shared
    'unknown' bucket instead of raising."""

    def __init__(self):
        self.maps = {}

    def fit(self, df: pd.DataFrame, cols):
        for c in cols:
            vocab = sorted(df[c].astype(str).unique().tolist())
            self.maps[c] = {v: i for i, v in enumerate(vocab)}
        return self

    def transform(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            m = self.maps[c]
            unknown = len(m)
            df[c] = df[c].astype(str).map(lambda v: m.get(v, unknown)).astype(int)
        return df


def build_training_frame(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["store_id", "date"]).reset_index(drop=True)
    df = add_calendar_holiday_features(df)
    df = add_lag_features(df)
    return df


def recursive_forecast(model, encoder: CategoryEncoder, train_panel: pd.DataFrame,
                        store_meta: pd.DataFrame, forecast_dates_by_store: dict) -> pd.DataFrame:
    """model: any object with .predict(X_dataframe) -> array.
    store_meta: one row per store_id with static 'region' column.
    forecast_dates_by_store: store_id -> sorted list of dates to predict (subset of the horizon
    this store actually has ground truth for)."""
    history = {sid: g.set_index("date")["count_in"].copy()
               for sid, g in train_panel.groupby("store_id")}
    all_dates = sorted({d for ds in forecast_dates_by_store.values() for d in ds})
    preds = []

    for d in all_dates:
        active = [sid for sid, ds in forecast_dates_by_store.items() if d in ds]
        if not active:
            continue
        rows = []
        for sid in active:
            s = history[sid]
            row = {"store_id": sid, "date": d}
            for lag in LAGS:
                ld = d - pd.Timedelta(days=lag)
                row[f"lag_{lag}"] = s.get(ld, np.nan)
            recent = s[s.index < d]
            r7, r28 = recent.tail(7), recent.tail(28)
            row["roll_mean_7"] = r7.mean() if len(r7) else np.nan
            row["roll_std_7"] = r7.std() if len(r7) > 1 else np.nan
            row["roll_mean_28"] = r28.mean() if len(r28) else np.nan
            row["roll_std_28"] = r28.std() if len(r28) > 1 else np.nan
            rows.append(row)
        batch = pd.DataFrame(rows).merge(store_meta, on="store_id", how="left")
        batch = add_calendar_holiday_features(batch)
        X = encoder.transform(batch, CAT_COLS)[FEATURE_COLS]
        yhat = np.clip(model.predict(X), 0, None)
        batch["y_pred"] = yhat
        preds.append(batch[["store_id", "date", "y_pred"]])
        for sid, val in zip(batch["store_id"], yhat):
            history[sid].loc[d] = val

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["store_id", "date", "y_pred"])


def time_split(panel: pd.DataFrame, horizon: int):
    cutoff = panel["date"].max() - pd.Timedelta(days=horizon)
    train = panel[panel["date"] <= cutoff].copy()
    test = panel[panel["date"] > cutoff].copy()
    return train, test, cutoff
