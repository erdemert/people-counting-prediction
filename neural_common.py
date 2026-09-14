"""Shared TimeSeriesDataSet construction for the TFT and DeepAR benchmarks."""
import os
import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

from features import add_calendar_holiday_features

KNOWN_CATS = ["dow", "month", "holiday_today"]
KNOWN_REALS = ["days_to_next_holiday", "days_since_last_holiday", "is_weekend"]
STATIC_CATS = ["region"]


def build_full_frame(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["store_id", "date"]).reset_index(drop=True)
    df = add_calendar_holiday_features(df)
    df["store_id"] = df["store_id"].astype(str)
    df["dow"] = df["dow"].astype(str)
    df["month"] = df["month"].astype(str)
    df["holiday_today"] = df["holiday_today"].astype(str)
    global_min = df["date"].min()
    df["time_idx"] = (df["date"] - global_min).dt.days.astype(int)
    df["count_in"] = df["count_in"].astype(float)
    return df


def make_datasets(full_df: pd.DataFrame, cutoff_idx: int, horizon: int, max_encoder_length: int = 60,
                   normalizer_transform: str = "softplus"):
    train_df = full_df[full_df["time_idx"] <= cutoff_idx]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="count_in",
        group_ids=["store_id"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length // 2,
        max_prediction_length=horizon,
        min_prediction_length=horizon,
        static_categoricals=STATIC_CATS,
        time_varying_known_categoricals=KNOWN_CATS,
        time_varying_known_reals=KNOWN_REALS + ["time_idx"],
        time_varying_unknown_reals=["count_in"],
        target_normalizer=GroupNormalizer(groups=["store_id"], transformation=normalizer_transform),
        categorical_encoders={"holiday_today": NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, full_df, predict=True, stop_randomization=True)
    return training, validation


def make_dataloaders(training, validation, batch_size=512, num_workers=None):
    # batch_size=64 with num_workers=0 (the old defaults) left the GPU
    # ~35% utilized and starved on CPU-side batch construction -- for a
    # panel this size (100Ks of overlapping windows/epoch) that made a
    # single epoch take 15+ minutes for a ~24K-param model. Bigger batches
    # + parallel workers fixes the actual bottleneck (data loading, not
    # compute).
    if num_workers is None:
        num_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    persistent = num_workers > 0
    train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers,
                                       persistent_workers=persistent)
    val_dl = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=num_workers,
                                       persistent_workers=persistent)
    return train_dl, val_dl


class EpochLogger(pl.Callback):
    """One log line per epoch (train/val loss + metrics) through our own
    logger, instead of Lightning's progress bar -- a tqdm-style bar renders
    badly once stdout is piped/redirected to a log file, which is exactly
    how this benchmark is normally run."""

    def __init__(self, logger, tag: str):
        # NOTE: don't call this attribute `self.log` -- Lightning's Callback
        # base class binds its own `log` method onto instances (so callbacks
        # can log metrics the same way a LightningModule does), which would
        # silently clobber a plain logger stored under that name.
        self._logger = logger
        self.tag = tag
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        import time
        self._t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        import time
        elapsed = time.time() - self._t0 if self._t0 else 0
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        parts = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        self._logger.info(f"[{self.tag}] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
                           f"({elapsed:.0f}s){': ' + parts if parts else ''}")


def make_trainer(max_epochs: int, accelerator: str = "gpu", devices=1, callbacks=None):
    # devices=1 on purpose: run_all.py/run_benchmark.py loop sequentially over
    # many brands/models in one script. devices="auto" on a multi-GPU box makes
    # Lightning launch DDP, which re-execs this whole script per extra GPU rank
    # -- wrong fit for a sequential loop like this one. Single GPU per model is
    # already fast enough at this data scale; parallelize across brands
    # instead (e.g. run separate processes pinned to CUDA_VISIBLE_DEVICES=0/1)
    # if you want to use both GPUs at once.
    return pl.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, devices=devices,
        enable_progress_bar=False, logger=False, enable_checkpointing=False,
        gradient_clip_val=0.1, callbacks=callbacks or [],
    )


def predictions_to_frame(model, val_dl, validation) -> pd.DataFrame:
    """Returns a long dataframe: store_id, time_idx, y_pred (point / median forecast)."""
    raw = model.predict(val_dl, mode="prediction", return_index=True)
    preds = raw.output
    index = raw.index
    rows = []
    for i in range(preds.shape[0]):
        sid = index.iloc[i]["store_id"]
        start_idx = int(index.iloc[i]["time_idx"])
        for h in range(preds.shape[1]):
            rows.append({"store_id": sid, "time_idx": start_idx + h, "y_pred": float(preds[i, h])})
    return pd.DataFrame(rows)
