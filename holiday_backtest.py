"""Holiday / pre-holiday focused backtest -- a SEPARATE evaluation from
run_benchmark.py's summary.csv, answering a different question.

run_benchmark.py always tests on "the last `horizon` days of whatever data
exists" -- which usually lands in a random, holiday-free stretch of the
calendar (e.g. mid-August), so it never actually tests whether a model can
predict a holiday spike at all.

This script instead lets you pick an explicit training cutoff positioned
right before a real, known holiday (any region -- Ramazan Bayrami for
Turkey stores, Chinese New Year for China stores, Thanksgiving for US
stores, etc: see holidays.py), then breaks the evaluation window into three
buckets using the SAME per-region holiday calendars already used as model
features, rather than one blended average:
  - "holiday"     -- the holiday day(s)/eve/named event itself
  - "pre_holiday" -- the run-up window before one (default: 5 days)
  - "baseline"    -- everything else in the test window

Because every row already carries its own region, this works correctly on
multi-region brands (Koton, Alexander Wang) without needing one global
"the holiday" -- a Turkish store's rows get judged against Bayram, a
Chinese store's against CNY, a US store's against Thanksgiving, all in the
same run.

Usage:
    python holiday_backtest.py --brand koton --test-end 2026-03-15 --horizon 14
    python holiday_backtest.py --brand alexander_wang --test-end 2026-02-10 --models tft,deepar

Reuses the exact same model code as run_benchmark.py unchanged -- the trick
is simply truncating the panel to `test_end + horizon` before handing it to
the same lightgbm_model/catboost_model/tft_model/deepar_model.run()
functions, so their existing "train on everything before the last
`horizon` days" logic naturally lands on the chosen cutoff.
"""
import argparse
import os

import pandas as pd

from config import BRANDS
from data_prep import build_panel
from features import add_calendar_holiday_features
from metrics import evaluate
from logging_setup import get_logger

PRE_HOLIDAY_WINDOW = 5  # days-to-next-holiday counted as "pre-holiday build-up"

MODEL_RUNNERS = {}


def _lazy_import_runners():
    if MODEL_RUNNERS:
        return
    from models import lightgbm_model, catboost_model, tft_model, deepar_model
    MODEL_RUNNERS["lightgbm"] = lambda panel, horizon, kw: lightgbm_model.run(panel, horizon)
    MODEL_RUNNERS["catboost"] = lambda panel, horizon, kw: catboost_model.run(panel, horizon)
    MODEL_RUNNERS["tft"] = lambda panel, horizon, kw: tft_model.run(panel, horizon, **kw)
    MODEL_RUNNERS["deepar"] = lambda panel, horizon, kw: deepar_model.run(panel, horizon, **kw)


def tag_holiday_period(preds: pd.DataFrame) -> pd.DataFrame:
    """preds must have 'date' and 'region' columns. Adds 'period'
    ('holiday' / 'pre_holiday' / 'baseline') and 'holiday_label' (the raw
    event name from holidays.py, or 'none')."""
    tagged = add_calendar_holiday_features(preds[["date", "region"]].copy())
    period = pd.Series("baseline", index=tagged.index)
    is_pre = (tagged["days_to_next_holiday"] > 0) & (tagged["days_to_next_holiday"] <= PRE_HOLIDAY_WINDOW)
    period[is_pre] = "pre_holiday"
    period[tagged["holiday_today"].astype(str) != "none"] = "holiday"
    out = preds.copy()
    out["period"] = period.values
    out["holiday_label"] = tagged["holiday_today"].astype(str).values
    return out


def run_holiday_backtest(brand: str, test_end: str, horizon: int = 28,
                          models=("lightgbm", "catboost", "tft", "deepar"),
                          accelerator: str = "gpu", output_root: str = "outputs",
                          logger=None, **neural_overrides):
    """Returns (period_summary_df, region_period_df, day_summary_df). All also written to disk."""
    log = logger or get_logger(f"holiday.{brand}")
    _lazy_import_runners()

    out_dir = os.path.join(output_root, brand)
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"[{brand}] building panel ...")
    panel = build_panel(brand)

    cutoff = pd.Timestamp(test_end)
    window_end = cutoff + pd.Timedelta(days=horizon)
    if window_end > panel["date"].max():
        raise SystemExit(f"test_end={test_end} + horizon={horizon}d ({window_end.date()}) is beyond "
                          f"the data's last date ({panel['date'].max().date()}) -- pick an earlier "
                          f"test_end or shorter horizon.")
    if cutoff <= panel["date"].min():
        raise SystemExit(f"test_end={test_end} is at/before the data's first date "
                          f"({panel['date'].min().date()}) -- no training history would be left.")

    truncated = panel[panel["date"] <= window_end].copy()

    # Same "drop stores with no training history before cutoff" guard as
    # run_benchmark.py's run_brand() -- a store that only opened after our
    # chosen historical cutoff has nothing to train on and would crash every
    # model differently.
    has_history = truncated.loc[truncated["date"] <= cutoff, "store_id"].unique()
    too_new = set(truncated["store_id"].unique()) - set(has_history)
    if too_new:
        names = truncated.loc[truncated["store_id"].isin(too_new), "store_name"].unique().tolist()
        log.warning(f"[{brand}] dropping {len(too_new)} store(s) with no history before "
                    f"{cutoff.date()}: {names}")
        truncated = truncated[truncated["store_id"].isin(has_history)]

    # Symmetric guard: a store that stopped reporting *before* window_end
    # (closed, a pop-up that ended, etc.) has no data to evaluate against in
    # our chosen holiday window at all. Worse, for TFT/DeepAR this silently
    # corrupts results rather than erroring: pytorch-forecasting's
    # predict=True picks each *group's own* last available window, not one
    # aligned to a single global cutoff -- so a closed store gets evaluated
    # on its own old last-active window (sometimes a year+ earlier) instead
    # of being skipped, showing up as bogus negative day_offset rows utterly
    # unrelated to the holiday being tested. LightGBM/CatBoost don't have
    # this failure mode (their recursive forecast only ever produces rows
    # for dates that exist in the real test slice), so this guard mainly
    # protects the neural models, but is applied to all four for consistency.
    STALE_TOLERANCE_DAYS = 3
    last_date = truncated.groupby("store_id")["date"].max()
    still_open = last_date[last_date >= window_end - pd.Timedelta(days=STALE_TOLERANCE_DAYS)].index
    closed_early = set(truncated["store_id"].unique()) - set(still_open)
    if closed_early:
        names = truncated.loc[truncated["store_id"].isin(closed_early), "store_name"].unique().tolist()
        log.warning(f"[{brand}] dropping {len(closed_early)} store(s) whose data ends before the "
                    f"{window_end.date()} evaluation window (closed/discontinued): {names}")
        truncated = truncated[truncated["store_id"].isin(still_open)]

    log.info(f"[{brand}] backtest window: train up to {cutoff.date()}, evaluate "
             f"{(cutoff + pd.Timedelta(days=1)).date()} .. {window_end.date()} ({horizon}d), "
             f"{truncated['store_id'].nunique()} stores, regions={sorted(truncated['region'].unique())}")

    neural_kwargs = dict(
        max_epochs=neural_overrides.get("epochs", 30),
        max_encoder_length=neural_overrides.get("encoder_length", 90),
        accelerator=accelerator,
        batch_size=neural_overrides.get("batch_size", 512),
        limit_train_batches=neural_overrides.get("limit_train_batches", 300),
        limit_val_batches=neural_overrides.get("limit_val_batches", 60),
        num_workers=neural_overrides.get("num_workers"),
        precision=neural_overrides.get("precision", "32-true"),
    )

    all_tagged = []
    for name in models:
        kw = dict(neural_kwargs, logger=log, tag=f"{brand}:holiday:{name}") if name in ("tft", "deepar") else {}
        log.info(f"[{brand}] {name}: starting holiday backtest")
        try:
            preds, _ = MODEL_RUNNERS[name](truncated, horizon, kw)
        except Exception as e:
            log.error(f"[{brand}] {name}: FAILED -- {e}", exc_info=True)
            continue
        tagged = tag_holiday_period(preds)
        tagged["model"] = name
        all_tagged.append(tagged)
        log.info(f"[{brand}] {name}: done ({len(tagged)} test rows)")

    if not all_tagged:
        log.warning(f"[{brand}] no model produced results -- nothing to report")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(all_tagged, ignore_index=True)
    tag = cutoff.date().isoformat()
    # day_offset: 1 = first day forecast, 2 = second, etc. Needed because
    # recursive tabular models (LightGBM/CatBoost) compound error the further
    # into the horizon they go, while TFT/DeepAR forecast the whole horizon
    # directly and don't -- so comparing period buckets alone can unfairly
    # penalize the recursive models if "baseline" happens to sit later in the
    # horizon than "holiday" does. Keeping day_offset (and region, since a
    # holiday for one region can be a baseline day for another on the exact
    # same date) lets you control for that: compare models at matched
    # forecast depth, or trace the full day-by-day trend into and through
    # the holiday, instead of only three collapsed buckets.
    combined["day_offset"] = (combined["date"] - cutoff).dt.days
    combined.to_csv(os.path.join(out_dir, f"holiday_preds_{tag}.csv"), index=False)

    period_rows = []
    region_rows = []
    day_rows = []
    for (model, period), g in combined.groupby(["model", "period"]):
        overall, _ = evaluate(g, "y_true", "y_pred")
        period_rows.append(dict(brand=brand, model=model, period=period, **overall))
    for (model, period, region), g in combined.groupby(["model", "period", "region"]):
        overall, _ = evaluate(g, "y_true", "y_pred")
        region_rows.append(dict(brand=brand, model=model, period=period, region=region, **overall))
    for (model, region, day_offset), g in combined.groupby(["model", "region", "day_offset"]):
        overall, _ = evaluate(g, "y_true", "y_pred")
        # period/date are deterministic for a given (region, day_offset) since
        # cutoff is fixed -- every row in g shares the same one, just read it
        # off the first row rather than re-deriving it.
        day_rows.append(dict(brand=brand, model=model, region=region, day_offset=day_offset,
                              date=g["date"].iloc[0].date(), period=g["period"].iloc[0],
                              holiday_label=g["holiday_label"].iloc[0], **overall))

    period_summary = pd.DataFrame(period_rows).sort_values(["period", "model"])
    region_summary = pd.DataFrame(region_rows).sort_values(["period", "region", "model"])
    day_summary = pd.DataFrame(day_rows).sort_values(["region", "day_offset", "model"])

    period_path = os.path.join(out_dir, f"holiday_eval_{tag}.csv")
    region_path = os.path.join(out_dir, f"holiday_eval_by_region_{tag}.csv")
    day_path = os.path.join(out_dir, f"holiday_eval_by_day_{tag}.csv")
    period_summary.to_csv(period_path, index=False)
    region_summary.to_csv(region_path, index=False)
    day_summary.to_csv(day_path, index=False)

    log.info(f"[{brand}] holiday/pre-holiday/baseline results:\n{period_summary.to_string(index=False)}")
    log.info(f"[{brand}] saved: {period_path}\n[{brand}] saved (by region): {region_path}\n"
             f"[{brand}] saved (by day, for a horizon-depth-controlled/fair comparison and the "
             f"full day-by-day trend): {day_path}")
    return period_summary, region_summary, day_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True, choices=list(BRANDS.keys()))
    ap.add_argument("--test-end", required=True,
                     help="Train up through this date (YYYY-MM-DD); the following --horizon days "
                          "are the evaluation window. Pick a date shortly before a real holiday for "
                          "the region you want to test -- e.g. a Ramazan Bayrami eve for Turkey "
                          "stores, Chinese New Year for China stores, Thanksgiving for US stores "
                          "(see holidays.py for exact dates per region/year).")
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--models", default="lightgbm,catboost,tft,deepar")
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "auto"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--encoder-length", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit-train-batches", type=int, default=300)
    ap.add_argument("--limit-val-batches", type=int, default=60)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--precision", default="32-true", choices=["32-true", "16-mixed", "bf16-mixed"])
    ap.add_argument("--output-root", default="outputs")
    args = ap.parse_args()

    logger = get_logger(f"holiday.{args.brand}", log_file=os.path.join("logs", f"{args.brand}_holiday.log"))
    run_holiday_backtest(
        args.brand, args.test_end, args.horizon,
        tuple(m.strip() for m in args.models.split(",") if m.strip()),
        args.accelerator, args.output_root, logger,
        epochs=args.epochs, encoder_length=args.encoder_length, batch_size=args.batch_size,
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
        num_workers=args.num_workers, precision=args.precision,
    )


if __name__ == "__main__":
    main()
