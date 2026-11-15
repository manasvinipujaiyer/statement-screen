"""Train one LightGBM model per label on train, tune on dev, then freeze.

    uv run python -m statementscreen.train data/features/ data/week1_n2000/ --out data/models/

Labels (ground truth, for the train and dev splits only, loaded with load_split, which
checks the manifest's hash; the test split stays closed and no evaluation batch is read):
is_salary, is_emi, is_bounce, and is_rent (category == "rent").

Features: data/features/<train|dev>/ (statementscreen.features): 30 dense features plus
the TF-IDF columns, named tfidf_0000... because LightGBM refuses some characters in
feature names; models.json maps each name back to its character n-gram.

Memory: the parts are streamed, and only a sample of train rows is kept:
  - one fixed-seed sample of NEG_RATE (10%) of all train rows, plus every row that is
    positive for any label;
  - for a label, its training rows are all of its positives plus the sampled rows that
    are negative for it. Sampled negatives get weight 1 / NEG_RATE, so predicted
    probabilities stay calibrated to the full data. Sample sizes are recorded.
HOLDOUT (10%) of train statements, chosen with a fixed seed, are kept out of fitting
and used only for early stopping.

Tuning: every setting in GRID (8: num_leaves x learning_rate x min_child_samples) is
trained for every label with deterministic settings (deterministic=True, row-wise, fixed
seed, NUM_THREADS = 2). Each is scored on all dev rows; its decision threshold gives the
best dev F1, placed in the middle (log-odds) of the gap of thresholds that all give it,
not on the edge of the dev positives (best_threshold). Per label, the setting with the
best dev F1 is kept; ties go to the best dev PR-AUC, then the earlier setting (better).
Every setting tried is logged in the report. The kept models are then frozen in --out:

    <label>.txt     the LightGBM model, at its best iteration
    models.json     per label: setting, threshold, best iteration, dev F1 and PR-AUC,
                    sample sizes; the feature names and the TF-IDF n-gram of each

and --out is never overwritten without --overwrite. Writes reports/week4_training.json
with every setting, timings, peak memory (ru_maxrss, converted by
summary.peak_memory_bytes: bytes on macOS, KB on Linux), training rows and model sizes.
"""

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, precision_recall_curve

from statementscreen.features import DENSE_FEATURES, KEYS, load_set
from statementscreen.generate import OutputFolderError, prepare_output_folder
from statementscreen.splits import SPLITS_DIR, SplitError, load_split
from statementscreen.summary import peak_memory_bytes

LABELS = ("is_salary", "is_emi", "is_bounce", "is_rent")
SEED = 42
NEG_RATE = 0.10
HOLDOUT = 0.10
NUM_THREADS = 2
MAX_ROUNDS = 1000
EARLY_STOPPING = 50
GRID = [{"num_leaves": leaves, "learning_rate": rate, "min_child_samples": child}
        for leaves in (15, 63) for rate in (0.05, 0.1) for child in (20, 100)]  # fmt: skip
BASE_PARAMS = {"objective": "binary", "num_threads": NUM_THREADS, "deterministic": True,
               "force_row_wise": True, "seed": SEED, "verbose": -1}  # fmt: skip
MODEL_FILES = (*(f"{label}.txt" for label in LABELS), "models.json")


# ---------------------------------------------------------------- data


def load_labels(batch: Path, split: str, splits_dir: Path = SPLITS_DIR) -> pl.DataFrame:
    """statement_id, row and the four labels for one split (train or dev), from the truth."""
    ids = load_split(split, batch / "manifest.csv", splits_dir)
    truth = (pl.scan_parquet(batch / "transactions.parquet")
             .filter(pl.col("statement_id").is_in(ids))
             .select("statement_id", pl.col("row").cast(pl.Int64), "is_salary", "is_emi",
                     "is_bounce", (pl.col("category") == "rent").alias("is_rent"))
             .collect())  # fmt: skip
    return truth


def feature_names(n_tfidf: int) -> list[str]:
    return [*DENSE_FEATURES, *(f"tfidf_{i:04d}" for i in range(n_tfidf))]


def to_matrix(dense: pl.DataFrame, tfidf: sp.csr_matrix) -> sp.csr_matrix:
    """Dense features (nulls -> NaN, LightGBM's missing) next to the TF-IDF columns."""
    values = dense.select(pl.col(DENSE_FEATURES).cast(pl.Float32)).to_numpy()
    return sp.hstack([sp.csr_matrix(values), tfidf], format="csr", dtype=np.float32)


def _with_labels(dense: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    joined = dense.select(KEYS).with_row_index("i").join(labels, on=KEYS, how="left")
    joined = joined.sort("i")
    missing = joined.filter(pl.col("is_salary").is_null()).height
    if missing:
        raise ValueError(f"{missing} feature rows have no truth row")
    return joined


def holdout_statements(ids: Sequence[str], share: float = HOLDOUT, seed: int = SEED) -> set[str]:
    ordered = sorted(set(ids))
    rng = np.random.default_rng([seed, 1])
    k = max(1, round(share * len(ordered)))
    return {ordered[i] for i in rng.permutation(len(ordered))[:k]}


def load_train_sample(features: Path, labels: pl.DataFrame, neg_rate: float = NEG_RATE,
                      seed: int = SEED) -> dict:  # fmt: skip
    """Stream the train parts, keeping the NEG_RATE sample plus every positive row."""
    holdout = holdout_statements(labels["statement_id"].unique().to_list(), seed=seed)
    rng = np.random.default_rng([seed, 2])
    xs, kept = [], []
    total = 0
    for dense, tfidf in load_set(features, "train"):
        joined = _with_labels(dense, labels)
        sampled = rng.random(joined.height) < neg_rate
        positive = np.zeros(joined.height, dtype=bool)
        for label in LABELS:
            positive |= joined[label].to_numpy()
        keep = sampled | positive
        xs.append(to_matrix(dense, tfidf)[keep])
        part = joined.filter(pl.Series(keep)).drop("i")
        in_holdout = pl.col("statement_id").is_in(holdout).alias("holdout")
        kept.append(part.with_columns(pl.Series("sampled", sampled[keep]), in_holdout))
        total += joined.height
    return {"X": sp.vstack(xs, format="csr"), "rows": pl.concat(kept), "train_rows": total,
            "holdout_statements": len(holdout)}  # fmt: skip


def label_rows(rows: pl.DataFrame, label: str, holdout: bool, neg_rate: float = NEG_RATE):
    """(row mask, 0/1 targets, weights) of one label's fitting or early-stopping rows."""
    pos = rows[label].to_numpy()
    use = (pos | rows["sampled"].to_numpy()) & (rows["holdout"].to_numpy() == holdout)
    weight = np.where(pos, 1.0, 1.0 / neg_rate)
    return use, pos[use].astype(np.int8), weight[use]


# ---------------------------------------------------------------- fitting and choosing


def fit(X, y, w, X_valid, y_valid, w_valid, setting: dict, names: list[str]) -> lgb.Booster:
    params = {**BASE_PARAMS, **setting}
    train = lgb.Dataset(X, y, weight=w, feature_name=names, free_raw_data=True)
    valid = lgb.Dataset(X_valid, y_valid, weight=w_valid, reference=train)
    return lgb.train(params, train, num_boost_round=MAX_ROUNDS, valid_sets=[valid],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])  # fmt: skip


def _logit(p: float) -> float:
    p = min(max(p, 1e-15), 1 - 1e-15)
    return float(np.log(p / (1 - p)))


def _prf(y: np.ndarray, flagged: np.ndarray) -> tuple[float, float, float]:
    tp = int((flagged & (y == 1)).sum())
    p = tp / max(int(flagged.sum()), 1)
    r = tp / max(int(y.sum()), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_threshold(y: np.ndarray, scores: np.ndarray) -> dict:
    """The decision threshold (flag when score >= threshold) with the best dev F1.

    Every threshold in the gap between the best-F1 cut (the lowest positive-side score
    that achieves it, "threshold_edge") and the next-lower dev score ("gap_below") gives
    exactly the same predictions on dev. The threshold is put in the middle of that gap
    (in log-odds), so it doesn't sit on the edge of the dev positives. Also returns
    precision, recall and F1 at the threshold, and the PR-AUC (average precision)."""
    if y.sum() == 0:
        return {"threshold": None, "threshold_edge": None, "gap_below": None,
                "precision": None, "recall": None, "f1": None, "pr_auc": None}  # fmt: skip
    precision, recall, thresholds = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]  # the last point has no threshold
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    edge = float(thresholds[int(np.argmax(f1))])
    below = scores[scores < edge]
    gap_below = float(below.max()) if below.size else None
    threshold = edge
    if gap_below is not None:
        middle = float(1 / (1 + np.exp(-(_logit(gap_below) + _logit(edge)) / 2)))
        same = np.array_equal(scores >= middle, scores >= edge)
        threshold = middle if gap_below < middle < edge and same else edge
    pr, rc, f = _prf(y, scores >= threshold)
    return {"threshold": threshold, "threshold_edge": edge, "gap_below": gap_below,
            "precision": pr, "recall": rc, "f1": f,
            "pr_auc": float(average_precision_score(y, scores))}  # fmt: skip


def better(result: dict, best: dict | None) -> bool:
    """Best dev F1; on a tie, best dev PR-AUC; on a tie again, the earlier setting."""
    return best is None or (result["f1"], result["pr_auc"]) > (best["f1"], best["pr_auc"])


def run(features: Path, batch: Path, out: Path, splits_dir: Path = SPLITS_DIR,
        overwrite: bool = False, grid: Sequence[dict] = GRID) -> dict:  # fmt: skip
    prepare_output_folder(out, overwrite, allowed=MODEL_FILES)
    started = time.perf_counter()
    vec_meta = json.loads((features / "tfidf_run.json").read_text(encoding="utf-8"))
    vec = joblib.load(features / "tfidf.joblib")
    ngrams = [str(g) for g in vec.get_feature_names_out()]
    names = feature_names(len(ngrams))

    sample = load_train_sample(features, load_labels(batch, "train", splits_dir))
    X, rows = sample["X"], sample["rows"]
    dev_labels = load_labels(batch, "dev", splits_dir)
    dev_x, dev_y = [], []
    for dense, tfidf in load_set(features, "dev"):
        dev_x.append(to_matrix(dense, tfidf))
        dev_y.append(_with_labels(dense, dev_labels))
    X_dev, y_dev = sp.vstack(dev_x, format="csr"), pl.concat(dev_y)
    del dev_x, dev_y

    tried, chosen, models, sizes = [], {}, {}, {}
    for label in LABELS:
        fit_mask, y, w = label_rows(rows, label, holdout=False)
        val_mask, yv, wv = label_rows(rows, label, holdout=True)
        sizes[label] = {"positives": int(y.sum()), "sampled_negatives": int((y == 0).sum()),
                        "holdout_positives": int(yv.sum()),
                        "holdout_sampled_negatives": int((yv == 0).sum()),
                        "negative_weight": 1 / NEG_RATE}  # fmt: skip
        truth = y_dev[label].to_numpy().astype(np.int8)
        best = None
        for k, setting in enumerate(grid):
            t0 = time.perf_counter()
            booster = fit(X[fit_mask], y, w, X[val_mask], yv, wv, setting, names)
            scores = booster.predict(X_dev, num_iteration=booster.best_iteration)
            result = {"label": label, "setting": k, **setting,
                      "best_iteration": booster.best_iteration, **best_threshold(truth, scores),
                      "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
            tried.append(result)
            what = (f"leaves {setting['num_leaves']:>2}, lr {setting['learning_rate']}, "
                    f"min_child {setting['min_child_samples']:>3}")  # fmt: skip
            score = f"dev F1 {result['f1']:.4f}, PR-AUC {result['pr_auc']:.4f}"
            cost = f"{result['best_iteration']} trees, {result['seconds']} s"
            print(f"  {label:9} setting {k}: {what} -> {score} ({cost})", flush=True)
            if better(result, best[0] if best else None):
                best = (result, booster)
        chosen[label], models[label] = best

    model_meta = {"labels": {}, "feature_names": names, "dense_features": DENSE_FEATURES,
                  "tfidf_ngrams": ngrams,
                  "tfidf_vocabulary_sha256": vec_meta["vocabulary_sha256"],
                  "params": BASE_PARAMS, "seed": SEED, "neg_rate": NEG_RATE,
                  "holdout": HOLDOUT, "frozen": True}  # fmt: skip
    model_sizes = {}
    for label in LABELS:
        result, booster = chosen[label], models[label]
        path = out / f"{label}.txt"
        booster.save_model(str(path), num_iteration=booster.best_iteration)
        model_sizes[label] = path.stat().st_size
        model_meta["labels"][label] = {**result, "sample": sizes[label]}
    (out / "models.json").write_text(json.dumps(model_meta, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")  # fmt: skip
    return {
        "chosen": {label: chosen[label] for label in LABELS},
        "tried": tried,
        "grid": list(grid),
        "sample_sizes": sizes,
        "train_rows_total": sample["train_rows"],
        "train_rows_kept": X.shape[0],
        "holdout_statements": sample["holdout_statements"],
        "dev_rows": X_dev.shape[0],
        "features": len(names),
        "model_bytes": model_sizes,
        "params": BASE_PARAMS,
        "seconds_total": round(time.perf_counter() - started, 1),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
        "lightgbm": lgb.__version__,
        "python": platform.python_version(),
        "platform": sys.platform,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.train",
        description="Train, tune on dev and freeze one LightGBM model per label.",
    )
    p.add_argument("features", type=Path, help="root folder written by statementscreen.features")
    p.add_argument("batch", type=Path, help="the reference batch (truth for train and dev)")
    p.add_argument("--out", type=Path, default=Path("data/models"))
    p.add_argument("--report", type=Path, default=Path("reports/week4_training.json"))
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--overwrite", action="store_true", help="replace frozen models")
    args = p.parse_args(argv)
    try:
        report = run(args.features, args.batch, args.out, args.splits_dir, args.overwrite)
    except (SplitError, OutputFolderError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Froze {len(LABELS)} models in {args.out}; wrote {args.report}")
    print(f"  train rows kept {report['train_rows_kept']} of {report['train_rows_total']}, "
          f"{report['seconds_total']} s, peak memory {report['peak_memory_mb']} MB")  # fmt: skip
    for label, c in report["chosen"].items():
        s = report["sample_sizes"][label]
        print(f"  {label:9} setting {c['setting']} threshold {c['threshold']:.3f}: dev "
              f"P {c['precision']:.4f} R {c['recall']:.4f} F1 {c['f1']:.4f} "
              f"PR-AUC {c['pr_auc']:.4f}; fitted on {s['positives']} positives + "
              f"{s['sampled_negatives']} sampled negatives; "
              f"{report['model_bytes'][label]} bytes")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
