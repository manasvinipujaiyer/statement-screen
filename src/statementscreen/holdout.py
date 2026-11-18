"""Layout hold-out for the Week 4 models: a measurement only, on train and dev.

    uv run python -m statementscreen.holdout data/features/ data/week1_n2000/

For each direction (columns -> drcr, drcr -> columns), every label's model is retrained
on the train statements of one layout, with that label's FROZEN setting (num_leaves,
learning_rate, min_child_samples from data/models/models.json) and the same sampling,
weights, early stopping and seed as Week 4 (statementscreen.train). It is scored at the
FROZEN threshold on the dev statements of the other layout, next to the frozen
both-layouts model on the same dev statements. Nothing is retuned, and the frozen models
in data/models/ are never touched.

A statement's layout is the one the parser detected (patterns statement_results; 100%
correct in Week 2). Adds a "layout_holdout" section to reports/week5_stress.json.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp

from statementscreen.compare import scores
from statementscreen.features import load_set
from statementscreen.splits import SPLITS_DIR, SplitError
from statementscreen.summary import peak_memory_bytes
from statementscreen.train import (
    LABELS,
    _with_labels,
    feature_names,
    fit,
    label_rows,
    load_labels,
    load_train_sample,
    to_matrix,
)

LAYOUTS = ("columns", "drcr")
SETTING_KEYS = ("num_leaves", "learning_rate", "min_child_samples")


def layouts(patterns_root: Path, name: str) -> dict[str, str]:
    results = pl.read_parquet(patterns_root / name / "statement_results.parquet",
                              columns=["statement_id", "layout"])  # fmt: skip
    return dict(zip(results["statement_id"], results["layout"], strict=True))


def _prf(y: np.ndarray, flagged: np.ndarray) -> dict:
    return scores(int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum()),
                  int((~flagged & (y == 1)).sum()))  # fmt: skip


def run(features: Path, batch: Path, patterns_root: Path, models: Path,
        splits_dir: Path = SPLITS_DIR) -> dict:  # fmt: skip
    started = time.perf_counter()
    meta = json.loads((models / "models.json").read_text(encoding="utf-8"))
    names = feature_names(len(meta["tfidf_ngrams"]))
    train_layout = layouts(patterns_root, "train")
    dev_layout = layouts(patterns_root, "dev")

    sample = load_train_sample(features, load_labels(batch, "train", splits_dir))
    X, rows = sample["X"], sample["rows"]
    row_layout = np.array([train_layout[s] for s in rows["statement_id"]])

    dev_labels = load_labels(batch, "dev", splits_dir)
    xs, ys = [], []
    for dense, tfidf in load_set(features, "dev"):
        xs.append(to_matrix(dense, tfidf))
        ys.append(_with_labels(dense, dev_labels))
    X_dev, y_dev = sp.vstack(xs, format="csr"), pl.concat(ys)
    dev_row_layout = np.array([dev_layout[s] for s in y_dev["statement_id"]])
    frozen = {label: lgb.Booster(model_file=str(models / f"{label}.txt")) for label in LABELS}

    out = {}
    for train_on in LAYOUTS:
        test_on = next(x for x in LAYOUTS if x != train_on)
        on_dev = dev_row_layout == test_on
        X_test = X_dev[np.flatnonzero(on_dev)]
        direction = {"train_layout": train_on, "dev_layout": test_on,
                     "dev_rows": int(on_dev.sum()),
                     "dev_statements": len({s for s, k in dev_layout.items() if k == test_on}),
                     "labels": {}}  # fmt: skip
        for label in LABELS:
            chosen = meta["labels"][label]
            setting = {k: chosen[k] for k in SETTING_KEYS}
            fit_mask, y, w = label_rows(rows, label, holdout=False)
            val_mask, yv, wv = label_rows(rows, label, holdout=True)
            keep_fit = row_layout[fit_mask] == train_on
            keep_val = row_layout[val_mask] == train_on
            booster = fit(X[np.flatnonzero(fit_mask)[keep_fit]], y[keep_fit], w[keep_fit],
                          X[np.flatnonzero(val_mask)[keep_val]], yv[keep_val], wv[keep_val],
                          setting, names)  # fmt: skip
            truth = y_dev[label].to_numpy()[on_dev].astype(np.int8)
            threshold = chosen["threshold"]
            held = booster.predict(X_test, num_iteration=booster.best_iteration)
            both = frozen[label].predict(X_test)
            direction["labels"][label] = {
                "holdout": {**_prf(truth, held >= threshold),
                            "train_positives": int(y[keep_fit].sum()),
                            "train_sampled_negatives": int((y[keep_fit] == 0).sum()),
                            "best_iteration": booster.best_iteration},
                "both_layouts": _prf(truth, both >= threshold),
                "setting": setting,
                "threshold": threshold,
            }  # fmt: skip
            print(f"  {train_on} -> {test_on} {label}: hold-out F1 "
                  f"{direction['labels'][label]['holdout']['f1']}, both-layouts F1 "
                  f"{direction['labels'][label]['both_layouts']['f1']}", flush=True)  # fmt: skip
        out[f"{train_on}_to_{test_on}"] = direction
    return {
        "directions": out,
        "seconds_total": round(time.perf_counter() - started, 1),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
        "note": "measurement only: frozen settings and thresholds, nothing retuned",
    }


def _n(x: float | None) -> str:
    return "  n/a " if x is None else f"{x:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.holdout",
        description="Layout hold-out of the Week 4 models (measurement only).",
    )
    p.add_argument("features", type=Path, help="root folder written by statementscreen.features")
    p.add_argument("batch", type=Path, help="the reference batch (truth for train and dev)")
    p.add_argument("--patterns", type=Path, default=Path("data/patterns"))
    p.add_argument("--models", type=Path, default=Path("data/models"))
    p.add_argument("--report", type=Path, default=Path("reports/week5_stress.json"))
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    args = p.parse_args(argv)
    try:
        section = run(args.features, args.batch, args.patterns, args.models, args.splits_dir)
    except (SplitError, ValueError, FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    report = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    report["layout_holdout"] = section
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Added layout_holdout to {args.report} ({section['seconds_total']} s, peak memory "
          f"{section['peak_memory_mb']} MB)")  # fmt: skip
    for d in section["directions"].values():
        print(f"  train {d['train_layout']} -> dev {d['dev_layout']} "
              f"({d['dev_statements']} dev statements, {d['dev_rows']} rows)")  # fmt: skip
        for label, r in d["labels"].items():
            h, b = r["holdout"], r["both_layouts"]
            print(f"    {label:9} hold-out P {_n(h['precision'])} R {_n(h['recall'])} "
                  f"F1 {_n(h['f1'])} | both layouts P {_n(b['precision'])} "
                  f"R {_n(b['recall'])} F1 {_n(b['f1'])}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
