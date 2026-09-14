import torch
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss

from neural_common import build_full_frame, make_datasets, make_dataloaders, make_trainer, \
    predictions_to_frame, EpochLogger


def run(panel, horizon: int, max_epochs: int = 10, max_encoder_length: int = 60, accelerator: str = "gpu",
        logger=None, tag: str = "deepar"):
    torch.manual_seed(42)
    full_df = build_full_frame(panel)
    cutoff_idx = full_df["time_idx"].max() - horizon

    training, validation = make_datasets(full_df, cutoff_idx, horizon, max_encoder_length,
                                          normalizer_transform=None)
    train_dl, val_dl = make_dataloaders(training, validation)

    model = DeepAR.from_dataset(
        training, learning_rate=0.03, hidden_size=16, rnn_layers=2, dropout=0.1,
        loss=NormalDistributionLoss(), log_interval=0,
    )
    callbacks = [EpochLogger(logger, tag)] if logger else None
    trainer = make_trainer(max_epochs, accelerator=accelerator, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    pred_frame = predictions_to_frame(model, val_dl, validation)

    id_map = full_df[["store_id", "store_name", "region", "date", "time_idx", "count_in"]]
    merged = pred_frame.merge(id_map, on=["store_id", "time_idx"], how="left")
    merged = merged.rename(columns={"count_in": "y_true"})
    merged["y_pred"] = merged["y_pred"].clip(lower=0)
    return merged[["store_id", "store_name", "region", "date", "y_true", "y_pred"]], model
