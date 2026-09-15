"""Single entry point: runs LightGBM, CatBoost, TFT and DeepAR for every brand,
one after another, with full logging, and writes a cross-brand leaderboard.

    python run_all.py
    python run_all.py --brands koton,lcw --epochs 40
    python run_all.py --models lightgbm,catboost   # skip the GPU models entirely

Defaults to GPU for TFT/DeepAR (--accelerator gpu). Logs:
  logs/run_all_<timestamp>.log   -- one line per brand/model as it starts/finishes
  logs/<brand>.log               -- full detail for that brand's run
  outputs/<brand>/summary.csv    -- per-brand model comparison
  outputs/all_brands_summary.csv -- everything combined, plus a leaderboard printed at the end
"""
import argparse
import os
import time
from datetime import datetime

import pandas as pd

from config import BRANDS
from run_benchmark import run_brand
from logging_setup import get_logger

ALL_MODELS = ("lightgbm", "catboost", "tft", "deepar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brands", default="all", help="comma-separated brand keys, or 'all'")
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--epochs", type=int, default=30, help="neural models only")
    ap.add_argument("--encoder-length", type=int, default=90, help="neural models only")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "auto"])
    ap.add_argument("--max-stores", type=int, default=None,
                     help="optional cap on stores per brand, for a fast end-to-end trial")
    ap.add_argument("--batch-size", type=int, default=512, help="neural models only")
    ap.add_argument("--limit-train-batches", type=int, default=300,
                     help="neural models only -- cap batches/epoch, the actual lever for TFT/DeepAR "
                          "epoch time on a large panel (see run_benchmark.py --help)")
    ap.add_argument("--limit-val-batches", type=int, default=60, help="neural models only")
    ap.add_argument("--num-workers", type=int, default=None,
                     help="neural models only -- default is (cpu_count - 1)")
    ap.add_argument("--precision", default="32-true", choices=["32-true", "16-mixed", "bf16-mixed"],
                     help="neural models only")
    ap.add_argument("--output-root", default="outputs")
    args = ap.parse_args()

    brands = list(BRANDS.keys()) if args.brands == "all" else [b.strip() for b in args.brands.split(",")]
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    for b in brands:
        if b not in BRANDS:
            raise SystemExit(f"Unknown brand '{b}'. Choices: {list(BRANDS.keys())}")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = get_logger("run_all", log_file=os.path.join("logs", f"run_all_{run_tag}.log"))

    master_log.info("=" * 78)
    master_log.info(f"BATCH RUN START -- brands={brands} models={models} horizon={args.horizon} "
                     f"epochs={args.epochs} encoder_length={args.encoder_length} "
                     f"accelerator={args.accelerator} max_stores={args.max_stores}")
    master_log.info("=" * 78)

    batch_t0 = time.time()
    all_summaries = []
    brand_status = []

    for i, brand in enumerate(brands, 1):
        master_log.info(f"[{i}/{len(brands)}] {brand}: STARTING")
        brand_logger = get_logger(f"benchmark.{brand}", log_file=os.path.join("logs", f"{brand}.log"))
        t0 = time.time()
        try:
            summary = run_brand(
                brand, horizon=args.horizon, epochs=args.epochs,
                encoder_length=args.encoder_length, models=models,
                accelerator=args.accelerator, max_stores=args.max_stores,
                output_root=args.output_root, logger=brand_logger,
                batch_size=args.batch_size, limit_train_batches=args.limit_train_batches,
                limit_val_batches=args.limit_val_batches, num_workers=args.num_workers,
                precision=args.precision,
            )
            all_summaries.append(summary)
            elapsed = time.time() - t0
            best = summary.iloc[0]
            master_log.info(f"[{i}/{len(brands)}] {brand}: DONE in {elapsed:.0f}s -- "
                             f"best model={best['model']} WAPE={best['WAPE']:.2f}")
            brand_status.append((brand, "ok", elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            master_log.error(f"[{i}/{len(brands)}] {brand}: BRAND-LEVEL FAILURE after {elapsed:.0f}s -- {e}",
                              exc_info=True)
            brand_status.append((brand, f"error: {e}"[:300], elapsed))

    total_elapsed = time.time() - batch_t0
    master_log.info("=" * 78)
    master_log.info(f"BATCH RUN FINISHED in {total_elapsed / 60:.1f} min")
    master_log.info("Per-brand status:")
    for brand, status, elapsed in brand_status:
        master_log.info(f"  {brand:20s} {status:10s} {elapsed:8.0f}s")

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = os.path.join(args.output_root, "all_brands_summary.csv")
        combined.to_csv(combined_path, index=False)

        ok = combined[combined["status"] == "ok"].copy()
        master_log.info("=" * 78)
        master_log.info("LEADERBOARD -- best model per brand (by WAPE):")
        for brand, g in ok.groupby("brand"):
            best = g.sort_values("WAPE").iloc[0]
            master_log.info(f"  {brand:20s} -> {best['model']:10s} WAPE={best['WAPE']:.2f} sMAPE={best['sMAPE']:.2f}")

        if not ok.empty:
            ok["rank"] = ok.groupby("brand")["WAPE"].rank()
            avg_rank = ok.groupby("model")["rank"].mean().sort_values()
            master_log.info("-" * 78)
            master_log.info("Average rank across brands (lower = better):")
            for model, r in avg_rank.items():
                master_log.info(f"  {model:10s} avg_rank={r:.2f}")

        master_log.info(f"\nCombined results: {combined_path}")
    else:
        master_log.warning("No brand completed successfully -- nothing to summarize.")


if __name__ == "__main__":
    main()
