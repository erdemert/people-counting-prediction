import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from neural_common import build_full_frame, make_datasets, make_dataloaders, make_trainer, \
    predictions_to_frame, EpochLogger


def run(panel, horizon: int, max_epochs: int = 10, max_encoder_length: int = 60, accelerator: str = "gpu",
        logger=None, tag: str = "tft", batch_size: int = 512,
        limit_train_batches=300, limit_val_batches=60):
    torch.manual_seed(42)
    full_df = build_full_frame(panel)
    cutoff_idx = full_df["time_idx"].max() - horizon

    training, validation = make_datasets(full_df, cutoff_idx, horizon, max_encoder_length)
    train_dl, val_dl = make_dataloaders(training, validation, batch_size=batch_size)

    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=0.03, hidden_size=16, attention_head_size=1,
        dropout=0.1, hidden_continuous_size=8, loss=QuantileLoss(),
        log_interval=0, reduce_on_plateau_patience=4,
    )
    callbacks = [EpochLogger(logger, tag)] if logger else None
    trainer = make_trainer(max_epochs, accelerator=accelerator, callbacks=callbacks,
                            limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    pred_frame = predictions_to_frame(model, val_dl, validation)

    id_map = full_df[["store_id", "store_name", "region", "date", "time_idx", "count_in"]]
    merged = pred_frame.merge(id_map, on=["store_id", "time_idx"], how="left")
    merged = merged.rename(columns={"count_in": "y_true"})
    merged["y_pred"] = merged["y_pred"].clip(lower=0)
    return merged[["store_id", "store_name", "region", "date", "y_true", "y_pred"]], model
