import numpy as np
import pandas as pd


def _safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return y_true, y_pred


def mae(y_true, y_pred):
    y_true, y_pred = _safe(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = _safe(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred):
    y_true, y_pred = _safe(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def wape(y_true, y_pred):
    y_true, y_pred = _safe(y_true, y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def evaluate(df: pd.DataFrame, y_true_col="y_true", y_pred_col="y_pred", group_col=None):
    """Returns a dict of overall metrics, and optionally a per-group breakdown."""
    overall = dict(
        MAE=mae(df[y_true_col], df[y_pred_col]),
        RMSE=rmse(df[y_true_col], df[y_pred_col]),
        sMAPE=smape(df[y_true_col], df[y_pred_col]),
        WAPE=wape(df[y_true_col], df[y_pred_col]),
        n=len(df),
    )
    if group_col is None:
        return overall, None
    rows = []
    for key, g in df.groupby(group_col):
        rows.append(dict(**{group_col: key}, MAE=mae(g[y_true_col], g[y_pred_col]),
                          RMSE=rmse(g[y_true_col], g[y_pred_col]),
                          sMAPE=smape(g[y_true_col], g[y_pred_col]),
                          WAPE=wape(g[y_true_col], g[y_pred_col]), n=len(g)))
    return overall, pd.DataFrame(rows).sort_values("WAPE")
