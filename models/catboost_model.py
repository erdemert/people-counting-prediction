from catboost import CatBoostRegressor

from tabular_common import (build_training_frame, CategoryEncoder, recursive_forecast,
                             time_split, CAT_COLS, NUM_COLS, FEATURE_COLS)


def run(panel, horizon: int):
    train_panel, test_panel, cutoff = time_split(panel, horizon)

    train_frame = build_training_frame(train_panel)
    encoder = CategoryEncoder().fit(train_frame, CAT_COLS)
    train_enc = encoder.transform(train_frame, CAT_COLS)

    X = train_enc[FEATURE_COLS]
    y = train_enc["count_in"]
    cat_idx = [FEATURE_COLS.index(c) for c in CAT_COLS]

    model = CatBoostRegressor(
        iterations=600, depth=8, learning_rate=0.04, loss_function="RMSE",
        random_seed=42, verbose=False, allow_writing_files=False,
    )
    model.fit(X, y, cat_features=cat_idx)

    store_meta = train_panel.drop_duplicates("store_id")[["store_id", "region"]]
    forecast_dates_by_store = {
        sid: sorted(g["date"].tolist()) for sid, g in test_panel.groupby("store_id")
    }
    preds = recursive_forecast(model, encoder, train_panel, store_meta, forecast_dates_by_store)

    merged = test_panel.merge(preds, on=["store_id", "date"], how="inner")
    merged = merged.rename(columns={"count_in": "y_true"})
    return merged[["store_id", "store_name", "region", "date", "y_true", "y_pred"]], model
