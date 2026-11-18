"""Transaction features for the Week 4 classifier, built one statement at a time.

    uv run python -m statementscreen.features data/patterns/ --sets train dev shift_karan_v1 \\
        --out data/features/

Reads ONLY pattern outputs, data/patterns/<set>/rows.parquet and group_features.parquet
(their narration and layout came from the parser). Never transactions.parquet, labels,
the manifest, group_rules.parquet or generator code. Columns are read by name from fixed
allowlists (ROW_INPUT, GROUP_INPUT), so the rule flags (is_salary, is_emi, is_bounce,
is_rent), flag_reason and the cleaning "style" are never even loaded: the model must
learn on its own, not copy the rules or memorise our narration formats.

Per row:
  row      is_credit, channel (one-hot), log_amount, day_of_month, position (0 = first
           row of the statement, 1 = last), cut
  group    joined by group_id: months present, month coverage, transactions per month,
           gap mean and SD, circular day-of-month measures, amount CV, share near the
           median, group size (all prefixed grp_)
  row vs   amount_to_group_median, day_from_group_medoid (circular)
  group
  text     character 2-4-gram TF-IDF of the narration, lowercased, every digit replaced
           by "#", max 2000 features, fitted on a fixed-seed sample of TRAIN rows only

The TF-IDF vectoriser is always fitted from <patterns>/train/rows.parquet (fit_tfidf can
read nothing else) and saved to <out>/tfidf.joblib with tfidf_run.json. Outputs per set,
in <out>/<set>/, one pair of parts per chunk of CHUNK statements, rows in the same order:

    parts/NNNNN.parquet   statement_id, row and the dense features
    parts/NNNNN.npz       the TF-IDF rows (scipy sparse CSR, float32)
    features_run.json     counts, columns, seconds per statement, peak memory (ru_maxrss
                          via summary.peak_memory_bytes: bytes on macOS, KB on Linux)

"test" is refused unless STATEMENTSCREEN_OPEN_TEST=1, like the split loaders.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from statementscreen.generate import OutputFolderError, prepare_output_folder
from statementscreen.patterns.clean import CHANNELS
from statementscreen.splits import OPEN_TEST_ENV
from statementscreen.summary import peak_memory_bytes

CHUNK = 100
DAYS = 31
ROW_INPUT = ["statement_id", "row", "posted", "amount_paise", "direction", "channel", "cut",
             "narration", "group_id"]  # fmt: skip
GROUP_INPUT = ["statement_id", "group_id", "n_txn", "median_amount_paise", "dom_medoid",
               "months_present", "month_coverage", "txn_per_month", "gap_mean_days",
               "gap_std_days", "dom_mean_deviation", "dom_share_on_time", "dom_arc_span",
               "amount_cv", "amount_share_near_median"]  # fmt: skip
# Never read, never features: what the rules decided, and the format they recognised.
EXCLUDED = ("is_salary", "is_emi", "is_bounce", "is_rent", "flag_reason", "style")
GROUP_FEATURES = {"n_txn": "grp_size", "months_present": "grp_months_present",
                  "month_coverage": "grp_month_coverage", "txn_per_month": "grp_txn_per_month",
                  "gap_mean_days": "grp_gap_mean_days", "gap_std_days": "grp_gap_std_days",
                  "dom_mean_deviation": "grp_dom_mean_deviation",
                  "dom_share_on_time": "grp_dom_share_on_time",
                  "dom_arc_span": "grp_dom_arc_span", "amount_cv": "grp_amount_cv",
                  "amount_share_near_median": "grp_share_near_median"}  # fmt: skip
CHANNEL_COLUMNS = [f"channel_{c.lower()}" for c in CHANNELS]
DENSE_FEATURES = ["is_credit", *CHANNEL_COLUMNS, "log_amount", "day_of_month", "position", "cut",
                  *GROUP_FEATURES.values(), "amount_to_group_median",
                  "day_from_group_medoid"]  # fmt: skip
KEYS = ["statement_id", "row"]


@dataclass(frozen=True)
class TfidfConfig:
    sample_rows: int = 100_000
    seed: int = 42
    max_features: int = 2000
    ngram_min: int = 2
    ngram_max: int = 4
    min_df: int = 5


def normalise(narration: str | None) -> str:
    """Lowercase, every digit replaced by '#': references differ, their shape doesn't."""
    return re.sub(r"\d", "#", (narration or "").lower())


def _check_set(name: str) -> None:
    if name == "test" and os.environ.get(OPEN_TEST_ENV) != "1":
        raise ValueError(f"the test split is closed; set {OPEN_TEST_ENV}=1 to open it")


# ---------------------------------------------------------------- text


def fit_tfidf(patterns_root: Path, cfg: TfidfConfig | None = None) -> tuple[TfidfVectorizer, dict]:
    """Fit the vectoriser on a fixed-seed sample of TRAIN rows. Reads nothing else."""
    cfg = cfg or TfidfConfig()
    path = patterns_root / "train" / "rows.parquet"
    narrations = pl.read_parquet(path, columns=["narration"])["narration"]
    sample = narrations.sample(n=min(cfg.sample_rows, len(narrations)), seed=cfg.seed)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(cfg.ngram_min, cfg.ngram_max),
                          max_features=cfg.max_features, min_df=cfg.min_df,
                          sublinear_tf=True, dtype=np.float32)  # fmt: skip
    vec.fit([normalise(n) for n in sample])
    vocab = sorted(vec.vocabulary_)
    vocab_sha = hashlib.sha256("\n".join(vocab).encode()).hexdigest()
    meta = {"fitted_on": "train", "train_rows": len(narrations), "sample_rows": len(sample),
            "config": asdict(cfg), "features": len(vocab),
            "vocabulary_sha256": vocab_sha}  # fmt: skip
    return vec, meta


# ---------------------------------------------------------------- one statement


def statement_features(rows: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """One statement's pattern rows and group features -> keys + dense features, sorted
    by row. Output columns: KEYS + DENSE_FEATURES + narration (for the TF-IDF)."""
    rows = rows.select(ROW_INPUT).sort("row")
    joined = rows.join(groups.select(GROUP_INPUT), on=["statement_id", "group_id"],
                       how="left", validate="m:1").sort("row")  # fmt: skip
    n = joined.height
    day = pl.col("posted").dt.day().cast(pl.Int32)
    away = (day - pl.col("dom_medoid")).abs() % DAYS
    f32 = pl.Float32
    return joined.select(
        *KEYS,
        (pl.col("direction") == "credit").cast(pl.Int8).alias("is_credit"),
        *[(pl.col("channel") == c).cast(pl.Int8).alias(col)
          for c, col in zip(CHANNELS, CHANNEL_COLUMNS, strict=True)],
        (pl.col("amount_paise").abs() / 100).log1p().cast(f32).alias("log_amount"),
        day.cast(pl.Int8).alias("day_of_month"),
        (pl.int_range(pl.len()) / max(n - 1, 1)).cast(f32).alias("position"),
        pl.col("cut").cast(pl.Int8),
        *[pl.col(src).cast(f32).alias(dst) for src, dst in GROUP_FEATURES.items()],
        (pl.col("amount_paise").abs() / pl.col("median_amount_paise")).cast(f32)
        .alias("amount_to_group_median"),
        pl.min_horizontal(away, DAYS - away).cast(f32).alias("day_from_group_medoid"),
        "narration",
    )  # fmt: skip


# ---------------------------------------------------------------- one set


def build_set(patterns_root: Path, name: str, out_root: Path, vec: TfidfVectorizer,
              tfidf_meta: dict, overwrite: bool = False) -> dict:  # fmt: skip
    _check_set(name)
    src = patterns_root / name
    rows_path, groups_path = src / "rows.parquet", src / "group_features.parquet"
    ids = sorted(pl.read_parquet(rows_path, columns=["statement_id"])["statement_id"].unique())
    out = out_root / name
    prepare_output_folder(out, overwrite, allowed=("parts", "features_run.json"))
    (out / "parts").mkdir()

    seconds: list[float] = []
    n_rows = 0
    started = time.perf_counter()
    for k in range(0, len(ids), CHUNK):
        chunk = ids[k : k + CHUNK]
        rows = (pl.scan_parquet(rows_path).filter(pl.col("statement_id").is_in(chunk))
                .select(ROW_INPUT).collect())  # fmt: skip
        groups = (pl.scan_parquet(groups_path).filter(pl.col("statement_id").is_in(chunk))
                  .select(GROUP_INPUT).collect())  # fmt: skip
        rows_by = rows.partition_by("statement_id", as_dict=True)
        groups_by = groups.partition_by("statement_id", as_dict=True)
        frames = []
        for sid in chunk:
            t0 = time.perf_counter()
            frames.append(statement_features(rows_by[(sid,)], groups_by[(sid,)]))
            seconds.append(time.perf_counter() - t0)
        dense = pl.concat(frames)
        text = vec.transform([normalise(t) for t in dense["narration"]]).tocsr()
        dense = dense.drop("narration")
        if text.shape[0] != dense.height:
            raise RuntimeError(f"TF-IDF rows {text.shape[0]} != dense rows {dense.height}")
        dense.write_parquet(out / "parts" / f"{k:05d}.parquet")
        sp.save_npz(out / "parts" / f"{k:05d}.npz", text)
        n_rows += dense.height
        print(f"  {name}: {min(k + CHUNK, len(ids))}/{len(ids)} statements", flush=True)
    wall = time.perf_counter() - started
    summary = {
        "set": name,
        "statements": len(ids),
        "rows": n_rows,
        "dense_features": DENSE_FEATURES,
        "tfidf_features": len(vec.vocabulary_),
        "tfidf_vocabulary_sha256": tfidf_meta["vocabulary_sha256"],
        "inputs": {"rows.parquet": ROW_INPUT, "group_features.parquet": GROUP_INPUT},
        "never_read": list(EXCLUDED) + ["group_rules.parquet"],
        "seconds_total": round(wall, 1),
        "seconds_per_statement": round(sum(seconds) / max(len(seconds), 1), 4),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (out / "features_run.json").write_text(text, encoding="utf-8")
    return summary


def load_set(features_root: Path, name: str):
    """Yield (dense frame, TF-IDF CSR) per part, in order. For training and scoring."""
    _check_set(name)
    for part in sorted((features_root / name / "parts").glob("*.parquet")):
        yield pl.read_parquet(part), sp.load_npz(part.with_suffix(".npz")).tocsr()


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.features",
        description="Build classifier features from pattern outputs, one statement at a time.",
    )
    p.add_argument("patterns", type=Path, help="root folder written by statementscreen.patterns")
    p.add_argument("--sets", nargs="+", required=True,
                   help="pattern folders to build, e.g. train dev shift_karan_v1")  # fmt: skip
    p.add_argument("--out", type=Path, default=Path("data/features"))
    p.add_argument("--overwrite", action="store_true", help="replace previous features")
    saved = "load the frozen <out>/tfidf.joblib instead of refitting (never rewrites it)"
    p.add_argument("--use-saved-tfidf", action="store_true", help=saved)
    args = p.parse_args(argv)
    try:
        for name in args.sets:
            _check_set(name)
        if args.use_saved_tfidf:
            vec = joblib.load(args.out / "tfidf.joblib")
            meta = json.loads((args.out / "tfidf_run.json").read_text(encoding="utf-8"))
            print(f"TF-IDF loaded from {args.out / 'tfidf.joblib'} (not refitted)")
        else:
            vec, meta = fit_tfidf(args.patterns)
            args.out.mkdir(parents=True, exist_ok=True)
            joblib.dump(vec, args.out / "tfidf.joblib")
            text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
            (args.out / "tfidf_run.json").write_text(text, encoding="utf-8")
            print(f"TF-IDF fitted on {meta['sample_rows']} of {meta['train_rows']} train rows: "
                  f"{meta['features']} features")  # fmt: skip
        for name in args.sets:
            s = build_set(args.patterns, name, args.out, vec, meta, args.overwrite)
            print(f"Wrote {args.out / name}: {s['statements']} statements, {s['rows']} rows, "
                  f"{len(s['dense_features'])} dense + {s['tfidf_features']} TF-IDF features "
                  f"in {s['seconds_total']} s ({s['seconds_per_statement']} s per statement, "
                  f"peak memory {s['peak_memory_mb']} MB)")  # fmt: skip
    except (ValueError, OutputFolderError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
