"""Run the LightGBM / CatBoost / TFT / DeepAR benchmark for one brand.

Standalone usage:
    python run_benchmark.py --brand koton --horizon 28 --epochs 30
    python run_benchmark.py --brand bluemint --models lightgbm,catboost --accelerator cpu

Also importable: `run_brand(...)` is what run_all.py calls per brand.
"""
import argparse
import os
import time

import pandas as pd

from config import BRANDS
from data_prep import build_panel
from metrics import evaluate
from logging_setup import get_logger

MODEL_RUNNERS = {}


def _lazy_import_runners():
    if MODEL_RUNNERS:
        return
    from models import lightgbm_model, catboost_model, tft_model, deepar_model
    MODEL_RUNNERS["lightgbm"] = lambda panel, horizon, epochs, enc_len, accel, log, tag, bs: lightgbm_model.run(panel, horizon)
    MODEL_RUNNERS["catboost"] = lambda panel, horizon, epochs, enc_len, accel, log, tag, bs: catboost_model.run(panel, horizon)
    MODEL_RUNNERS["tft"] = lambda panel, horizon, epochs, enc_len, accel, log, tag, bs: tft_model.run(panel, horizon, epochs, enc_len, accel, log, tag, bs)
    MODEL_RUNNERS["deepar"] = lambda panel, horizon, epochs, enc_len, accel, log, tag, bs: deepar_model.run(panel, horizon, epochs, enc_len, accel, log, tag, bs)


def run_brand(brand: str, horizon: int = 28, epochs: int = 30, encoder_length: int = 90,
              models=("lightgbm", "catboost", "tft", "deepar"), accelerator: str = "gpu",
              max_stores: int = None, output_root: str = "outputs", logger=None,
              batch_size: int = 512) -> pd.DataFrame:
    """Trains+evaluates each requested model for one brand. Returns the summary DataFrame.
    Never raises for a single model's failure -- it's recorded in the summary with status=error
    so a full batch run keeps going."""
    log = logger or get_logger(f"benchmark.{brand}")
    _lazy_import_runners()
    for m in models:
        if m not in MODEL_RUNNERS:
            raise SystemExit(f"Unknown model '{m}'. Choices: {list(MODEL_RUNNERS)}")

    out_dir = os.path.join(output_root, brand)
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"building panel for brand={brand} ...")
    panel = build_panel(brand)
    if max_stores:
        keep = panel["store_id"].drop_duplicates().head(max_stores)
        panel = panel[panel["store_id"].isin(keep)]

    # Drop stores with zero rows before the train/test cutoff -- a store that
    # only started appearing inside the last `horizon` days has no history to
    # learn from, and forecasting it crashes every model differently (KeyError
    # in the tabular recursive loop, "unknown category" in the neural
    # TimeSeriesDataSet). Excluding it here, once, keeps all 4 models
    # consistent instead of patching each one separately.
    cutoff = panel["date"].max() - pd.Timedelta(days=horizon)
    has_history = panel.loc[panel["date"] <= cutoff, "store_id"].unique()
    too_new = set(panel["store_id"].unique()) - set(has_history)
    if too_new:
        names = panel.loc[panel["store_id"].isin(too_new), "store_name"].unique().tolist()
        log.warning(f"[{brand}] dropping {len(too_new)} store(s) with no history before the "
                    f"{horizon}-day cutoff (too new to forecast): {names}")
        panel = panel[panel["store_id"].isin(has_history)]

    n_stores = panel["store_id"].nunique()
    log.info(f"panel ready: {len(panel):,} rows, {n_stores} stores, "
             f"{panel['date'].min().date()} .. {panel['date'].max().date()}, "
             f"regions={sorted(panel['region'].unique())}")

    summary_rows = []
    for name in models:
        log.info(f"[{brand}] {name}: starting (horizon={horizon}, epochs={epochs}, "
                 f"encoder_length={encoder_length}, accelerator={accelerator})")
        t0 = time.time()
        try:
            preds, _ = MODEL_RUNNERS[name](panel, horizon, epochs, encoder_length, accelerator,
                                            log, f"{brand}:{name}", batch_size)
        except Exception as e:
            elapsed = time.time() - t0
            log.error(f"[{brand}] {name}: FAILED after {elapsed:.0f}s -- {e}", exc_info=True)
            summary_rows.append(dict(brand=brand, model=name, MAE=None, RMSE=None, sMAPE=None,
                                      WAPE=None, n=0, seconds=round(elapsed, 1), status=f"error: {e}"[:300]))
            continue
        elapsed = time.time() - t0
        preds.to_csv(os.path.join(out_dir, f"preds_{name}.csv"), index=False)
        overall, by_region = evaluate(preds, "y_true", "y_pred", "region")
        if by_region is not None:
            by_region.to_csv(os.path.join(out_dir, f"by_region_{name}.csv"), index=False)
        summary_rows.append(dict(brand=brand, model=name, **overall, seconds=round(elapsed, 1), status="ok"))
        log.info(f"[{brand}] {name}: done in {elapsed:.0f}s -- "
                 f"WAPE={overall['WAPE']:.2f} sMAPE={overall['sMAPE']:.2f} "
                 f"MAE={overall['MAE']:.2f} RMSE={overall['RMSE']:.2f}")

    new_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(summary_path):
        # Merge, don't clobber -- e.g. re-running with --models tft,deepar
        # after LightGBM/CatBoost already succeeded shouldn't erase those rows.
        prior = pd.read_csv(summary_path)
        prior = prior[~prior["model"].isin(models)]
        new_summary = pd.concat([prior, new_summary], ignore_index=True)
    summary = new_summary.sort_values("WAPE")
    summary.to_csv(summary_path, index=False)
    log.info(f"[{brand}] summary:\n{summary.to_string(index=False)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True, choices=list(BRANDS.keys()))
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--epochs", type=int, default=30, help="neural models only")
    ap.add_argument("--encoder-length", type=int, default=90, help="neural models only")
    ap.add_argument("--models", default="lightgbm,catboost,tft,deepar")
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "auto"])
    ap.add_argument("--max-stores", type=int, default=None,
                     help="optional cap on number of stores, for a quick trial run")
    ap.add_argument("--batch-size", type=int, default=512, help="neural models only")
    ap.add_argument("--output-root", default="outputs")
    args = ap.parse_args()

    logger = get_logger(f"benchmark.{args.brand}", log_file=os.path.join("logs", f"{args.brand}.log"))
    run_brand(args.brand, args.horizon, args.epochs, args.encoder_length,
              tuple(m.strip() for m in args.models.split(",") if m.strip()),
              args.accelerator, args.max_stores, args.output_root, logger, args.batch_size)


if __name__ == "__main__":
    main()
