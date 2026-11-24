"""Train the Week 6 anomaly model on genuine train PDFs; set its threshold on genuine dev.

    uv run python scripts/train_anomaly.py

1. Features (tamper/detect/anomaly.py) for every genuine train and dev statement, read
   fresh from its PDF one at a time (parse, then the profile's own steps). Cached in
   data/tamper/anomaly_features/<split>.parquet. The cache is keyed by split name only:
   after changing the feature code or the batch, pass --refresh to recompute it.
2. IsolationForest (300 trees, random_state 42) fitted on genuine TRAIN only.
3. Threshold = the 98th percentile of the genuine DEV scores, so 2% of genuine dev
   statements score above it BY CONSTRUCTION (the target, not a measured rate). The rate
   on statements the threshold never saw is the genuine-train share in the report.
4. Writes models/week6_iforest.pkl (model, threshold, feature names) and
   reports/week6_anomaly_fit.json (counts, threshold, shares above it, timings).

Genuine train and dev only: no tampered copies, no test statements.
"""

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import IsolationForest

from statementscreen.parse import parse_one
from statementscreen.splits import load_split
from statementscreen.tamper.detect.anomaly import FEATURES, MODEL_PATH, features, profile_of

SEED = 42
TREES = 300
FALSE_ALARM_TARGET = 0.02


def feature_table(batch: Path, split: str, out: Path, refresh: bool) -> tuple[pl.DataFrame, dict]:
    if out.is_file() and not refresh:
        print(f"{split}: using cached {out}")
        return pl.read_parquet(out), {"cached": True}
    ids = load_split(split, batch / "manifest.csv")
    rows, skipped = [], []
    start = time.perf_counter()
    for n, sid in enumerate(ids, start=1):
        parsed, _ = parse_one(batch / "pdfs" / f"{sid}.pdf")
        feats = features(parsed, profile_of(parsed))
        if feats is None:
            skipped.append(sid)
        else:
            rows.append({"statement_id": sid, "layout": parsed.layout, **feats})
        if n % 100 == 0:
            print(f"  {split}: {n}/{len(ids)}, {time.perf_counter() - start:.0f} s", flush=True)
    seconds = time.perf_counter() - start
    table = pl.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    return table, {"cached": False, "statements": len(ids), "skipped_no_rows": skipped,
                   "seconds": round(seconds, 1),
                   "seconds_per_pdf": round(seconds / max(len(ids), 1), 4)}  # fmt: skip


def matrix(table: pl.DataFrame) -> np.ndarray:
    """log1p of each feature, as anomaly.vector does for one statement."""
    return np.log1p(np.clip(table.select(FEATURES).to_numpy(), 0.0, None))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=Path, default=Path("data/week1_n2000"))
    p.add_argument("--cache", type=Path, default=Path("data/tamper/anomaly_features"))
    p.add_argument("--refresh", action="store_true", help="recompute cached features")
    args = p.parse_args(argv)

    train, train_run = feature_table(args.batch, "train", args.cache / "train.parquet",
                                     args.refresh)  # fmt: skip
    dev, dev_run = feature_table(args.batch, "dev", args.cache / "dev.parquet", args.refresh)

    model = IsolationForest(n_estimators=TREES, random_state=SEED).fit(matrix(train))
    train_scores = -model.score_samples(matrix(train))
    dev_scores = -model.score_samples(matrix(dev))
    threshold = float(np.quantile(dev_scores, 1 - FALSE_ALARM_TARGET))
    dev_flagged = dev.with_columns(pl.Series("score", dev_scores)).filter(
        pl.col("score") > threshold)  # fmt: skip

    saved = {"model": model, "threshold": threshold, "features": list(FEATURES),
             "trained_on": "genuine train", "threshold_from": "genuine dev",
             "false_alarm_target": FALSE_ALARM_TARGET, "seed": SEED, "trees": TREES,
             "sklearn_version": sklearn.__version__}  # fmt: skip
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(saved, f)

    report = {
        "what": "Week 6 anomaly model: IsolationForest on genuine train, threshold on dev",
        "features": list(FEATURES),
        "model_file": f"models/{MODEL_PATH.name}",
        "seed": SEED,
        "trees": TREES,
        "sklearn_version": sklearn.__version__,
        "train": {"statements": train.height, **train_run,
                  "above_threshold": int((train_scores > threshold).sum())},
        "dev": {"statements": dev.height, **dev_run,
                "above_threshold": dev_flagged.height,
                "above_threshold_share": round(dev_flagged.height / max(dev.height, 1), 4),
                "above_threshold_by_layout": dict(sorted(
                    dev_flagged.group_by("layout").len().iter_rows()))},
        "threshold": round(threshold, 6),
        "false_alarm_target": FALSE_ALARM_TARGET,
        "score_quantiles": {
            split: {q: round(float(np.quantile(s, float(q))), 6)
                    for q in ("0.5", "0.9", "0.98", "1.0")}
            for split, s in (("train", train_scores), ("dev", dev_scores))
        },
        "feature_medians": {
            split: {k: round(float(t[k].median()), 6) for k in FEATURES}
            for split, t in (("train", train), ("dev", dev))
        },
        "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
    }  # fmt: skip
    out = Path("reports/week6_anomaly_fit.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MODEL_PATH} and {out}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
