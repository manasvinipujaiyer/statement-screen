"""Score the test split and the shift batch ONCE, as pre-registered.

    STATEMENTSCREEN_OPEN_TEST=1 uv run python -m statementscreen.score_once

Before anything from the test split or the shift batch is read, it checks, in order:
  1. reports/week5_preregistration.json exists, its frozen commit is an ancestor of HEAD,
     and every hash in it (source files, configs, models, vectoriser, splits) still
     matches (preregister.frozen_hashes);
  2. no first result exists yet (a re-run needs --post-hoc and is appended, labelled
     post-hoc, next to the first result, which is never changed);
  3. the frozen models' dev F1 equals the one recorded at training, for every label.

Then, for dev, test and the shift batch, it computes exactly the pre-registered metrics:
the label metrics of the rules and the frozen models (compare.score_set: precision,
recall, F1 with paired bootstrap intervals, PR-AUC, has_salary, false salary) and every
profile metric of step 4 (score_profile), with 95% bootstrap intervals over statements
(1,000 resamples, seed 42; income intervals count a missing estimate as a 100% error).
Test is set next to dev, and every metric where dev is better than the far end of the
test interval is listed as overstated by dev.

Writes reports/week5_test_and_shift.json and logs a local MLflow run (experiment
week5_profile). Records seconds and peak memory (ru_maxrss via summary.peak_memory_bytes:
bytes on macOS, KB on Linux).
"""

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from statementscreen.compare import EvalSet, assemble, load_models, score_set
from statementscreen.eval_batches import load_eval
from statementscreen.preregister import DEFAULT_OUT as PREREG
from statementscreen.preregister import frozen_hashes, git, sha256
from statementscreen.score_profile import build_report
from statementscreen.splits import SPLITS_DIR, SplitError, load_split
from statementscreen.summary import peak_memory_bytes
from statementscreen.tracking import DEFAULT_DIR, TrackingRefused, log_flat_run
from statementscreen.train import LABELS

B = 1000
SEED = 42
EXPERIMENT = "week5_profile"
DEFAULT_OUT = Path("reports/week5_test_and_shift.json")
# Profile headline metrics, and whether higher is better.
HEADLINE = {
    "income_median_ape": False, "income_within_5pct": True, "income_within_10pct": True,
    "share_without_estimate": False, "has_salary_precision": True,
    "has_salary_recall": True, "employer_exact": True, "employer_fuzzy": True,
    "customer_type_accuracy": True, "emi_count_exact": True, "emi_total_exact": True,
    "bounce_count_exact": True,
}  # fmt: skip


# ---------------------------------------------------------------- guards


def verify(prereg: dict, root: Path = Path(".")) -> list[str]:
    """Every hash that changed since the pre-registration (empty when nothing did)."""
    now = frozen_hashes(root)
    problems = []
    for section in ("code", "configs", "artefacts"):
        old, new = prereg["hashes"][section], now[section]
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                problems.append(f"{section}: {key} differs from the pre-registration")
    return problems


def is_ancestor(commit: str) -> bool:
    done = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          capture_output=True)  # fmt: skip
    return done.returncode == 0


def first_result_exists(path: Path) -> bool:
    if not path.exists():
        return False
    return bool(json.loads(path.read_text(encoding="utf-8")).get("first_result"))


def write(path: Path, report: dict, post_hoc: bool) -> None:
    """The first result is written once; a post-hoc re-run is appended, never replacing it."""
    if first_result_exists(path):
        if not post_hoc:
            raise ValueError(f"{path} already holds the one-time result; pass --post-hoc")
        old = json.loads(path.read_text(encoding="utf-8"))
        new = {k: v for k, v in report.items() if k != "first_result"}
        old.setdefault("post_hoc_runs", []).append({**new, "label": "post-hoc"})
        report = old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")  # fmt: skip


# ---------------------------------------------------------------- profile intervals


def _arrays(table: pl.DataFrame) -> dict[str, np.ndarray]:
    t_income = table["t_income"].to_numpy()
    ape = table["ape"].cast(pl.Float64).to_numpy()
    valid = t_income > 0
    return {
        "ape": np.where(valid, np.where(np.isnan(ape), 1.0, ape), np.nan),
        "valid": valid,
        "missing": valid & np.isnan(ape),
        "p_has": table["p_has_salary"].to_numpy().astype(bool),
        "t_has": table["t_has_salary"].to_numpy().astype(bool),
        "exact": table["employer_exact"].to_numpy().astype(bool),
        "fuzzy": table["employer_fuzzy"].to_numpy().astype(bool),
        "type_ok": (table["p_type"] == table["t_type"]).to_numpy(),
        "emi_ok": (table["p_emi_count"] == table["t_emi_count"]).to_numpy(),
        "emi_total_ok": (table["p_emi_total"] == table["t_emi_total"]).to_numpy(),
        "bounce_ok": (table["p_bounces"] == table["t_bounces"]).to_numpy(),
    }


def _median(values: np.ndarray) -> float:
    kept = values[~np.isnan(values)]
    return float(np.median(kept)) if kept.size else np.nan


def _stats(a: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Every headline metric for each row of statement indices idx (resamples x n)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ape, valid = a["ape"][idx], a["valid"][idx]
        p, t = a["p_has"][idx], a["t_has"][idx]
        tp = (p & t).sum(axis=1)
        return {
            "income_median_ape": np.array([_median(r) for r in ape]),
            "income_within_5pct": ((ape <= 0.05) & valid).sum(axis=1) / valid.sum(axis=1),
            "income_within_10pct": ((ape <= 0.10) & valid).sum(axis=1) / valid.sum(axis=1),
            "share_without_estimate": a["missing"][idx].sum(axis=1) / valid.sum(axis=1),
            "has_salary_precision": tp / p.sum(axis=1),
            "has_salary_recall": tp / t.sum(axis=1),
            "employer_exact": (a["exact"][idx] & t).sum(axis=1) / t.sum(axis=1),
            "employer_fuzzy": (a["fuzzy"][idx] & t).sum(axis=1) / t.sum(axis=1),
            "customer_type_accuracy": a["type_ok"][idx].mean(axis=1),
            "emi_count_exact": a["emi_ok"][idx].mean(axis=1),
            "emi_total_exact": a["emi_total_ok"][idx].mean(axis=1),
            "bounce_count_exact": a["bounce_ok"][idx].mean(axis=1),
        }  # fmt: skip


def _finite(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


def profile_intervals(table: pl.DataFrame, n: int = B, seed: int = SEED) -> dict:
    """Point value and 95% bootstrap interval (over statements) of each headline metric."""
    a = _arrays(table)
    point = _stats(a, np.arange(table.height)[None, :])
    idx = np.random.default_rng(seed).integers(0, table.height, size=(n, table.height))
    boot = _stats(a, idx)
    out = {}
    for name in HEADLINE:
        values = boot[name][~np.isnan(boot[name])]
        low, high = np.percentile(values, [2.5, 97.5]) if values.size else (np.nan, np.nan)
        out[name] = {"value": _finite(point[name][0]), "low": _finite(low),
                     "high": _finite(high), "resamples_used": int(values.size)}  # fmt: skip
    return out


def dev_overstated(dev: dict, test: dict) -> list[str]:
    """Headline metrics where dev is better than the far end of the test interval."""
    out = []
    for name, higher_is_better in HEADLINE.items():
        d, t = dev[name]["value"], test[name]
        if d is None or t["low"] is None:
            continue
        if (higher_is_better and d > t["high"]) or (not higher_is_better and d < t["low"]):
            out.append(name)
    return out


def labels_overstated(dev: dict, test: dict) -> list[str]:
    out = []
    for label in LABELS:
        for method in ("rules", "model"):
            d = dev["labels"][label][method]["f1"]
            high = test["labels"][label][method]["f1_ci"]["high"]
            if d is not None and high is not None and d > high:
                out.append(f"{label} {method} F1")
    return out


# ---------------------------------------------------------------- one set


def score(es: EvalSet, args, meta: dict, boosters: dict) -> dict:
    frame, statements = assemble(es, args.patterns, args.features, meta, boosters,
                                 args.splits_dir)  # fmt: skip
    labels = score_set(frame, statements, by_format_group=es.split is None)
    manifest = es.batch / "manifest.csv"
    if es.split:
        ids = load_split(es.split, manifest, args.splits_dir)
    else:
        ids = load_eval(es.name, manifest, args.splits_dir)
    profile, table = build_report(args.profiles / es.name, es.batch, ids)
    profile["intervals"] = profile_intervals(table)
    return {"labels": labels, "profile": profile}


def run(args) -> dict:
    started = time.perf_counter()
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if not is_ancestor(prereg["frozen_commit"]):
        raise ValueError(f"pre-registered commit {prereg['frozen_commit'][:12]} is not in "
                         "this branch's history")  # fmt: skip
    problems = verify(prereg)
    if problems:
        raise ValueError("changed since the pre-registration:\n  " + "\n  ".join(problems))
    if first_result_exists(args.out) and not args.post_hoc:
        raise ValueError(f"{args.out} already holds the one-time result; pass --post-hoc")
    meta, boosters = load_models(args.models)

    dev = score(EvalSet("dev", args.reference, "dev"), args, meta, boosters)
    for label in LABELS:
        if abs(dev["labels"]["labels"][label]["model"]["f1"] - meta["labels"][label]["f1"]) > 1e-9:
            raise ValueError(f"model dev F1 for {label} differs from the frozen model; "
                             "stopping before test and shift are read")  # fmt: skip
    test = score(EvalSet("test", args.reference, "test"), args, meta, boosters)
    shift = score(EvalSet(args.shift_name, args.shift_batch, None), args, meta, boosters)

    dev_p, test_p = dev["profile"]["intervals"], test["profile"]["intervals"]
    over = dev_overstated(dev_p, test_p) + labels_overstated(dev["labels"], test["labels"])
    summary = (f"dev overstated {len(over)} metric(s): dev was better than the far end of "
               f"the test 95% interval for {', '.join(over)}" if over else
               "dev did not overstate any pre-registered metric: every dev value is inside "
               "the test 95% interval or worse")  # fmt: skip
    return {
        "first_result": True,
        "preregistration": {"file": str(args.prereg), "sha256": sha256(args.prereg),
                            "frozen_commit": prereg["frozen_commit"]},
        "scored_at_commit": git("rev-parse", "HEAD"),
        "sets": {"dev": dev, "test": test, args.shift_name: shift},
        "test_vs_dev": {"overstated_by_dev": over, "summary": summary},
        "bootstrap": {"resamples": B, "seed": SEED, "unit": "statement", "level": 0.95},
        "notes": ["dev was used for development and tuning", "test and the shift batch "
                  "were each scored once, after the pre-registration was committed",
                  "the shift batch's label metrics repeat Week 4's frozen models",
                  "all data is synthetic; real-statement checks are still to come"],
        "run": {"seconds_total": round(time.perf_counter() - started, 1),
                "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1)},
    }  # fmt: skip


# ---------------------------------------------------------------- CLI


def _p(x: float | None) -> str:
    return "  n/a  " if x is None else f"{x:7.2%}"


def _ci(m: dict) -> str:
    return "n/a" if m["low"] is None else f"[{m['low']:.2%}, {m['high']:.2%}]"


def flat_metrics(report: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, s in report["sets"].items():
        for metric, m in s["profile"]["intervals"].items():
            if m["value"] is not None:
                out[f"{name}.profile.{metric}"] = m["value"]
        for label, t in s["labels"]["labels"].items():
            for method in ("rules", "model"):
                if t[method]["f1"] is not None:
                    out[f"{name}.{label}.{method}.f1"] = t[method]["f1"]
    out["seconds_total"] = report["run"]["seconds_total"]
    out["peak_memory_mb"] = report["run"]["peak_memory_mb"]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.score_once",
        description="Score the test split and the shift batch once, as pre-registered.",
    )
    p.add_argument("--prereg", type=Path, default=PREREG)
    p.add_argument("--reference", type=Path, default=Path("data/week1_n2000"))
    p.add_argument("--shift-batch", type=Path, default=Path("data/week4_shift_n300"))
    p.add_argument("--shift-name", default="shift_karan_v1")
    p.add_argument("--patterns", type=Path, default=Path("data/patterns"))
    p.add_argument("--features", type=Path, default=Path("data/features"))
    p.add_argument("--profiles", type=Path, default=Path("data/profiles"))
    p.add_argument("--models", type=Path, default=Path("data/models"))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--post-hoc", action="store_true", help="a re-run, labelled post-hoc")
    p.add_argument("--tracking-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    try:
        report = run(args)
        write(args.out, report, args.post_hoc)
    except (SplitError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"Wrote {args.out}{' (post-hoc re-run appended)' if args.post_hoc else ''} in "
          f"{report['run']['seconds_total']} s, peak memory {report['run']['peak_memory_mb']}"
          " MB")  # fmt: skip
    names = list(report["sets"])
    print(f"  profile {'':24}" + "".join(f"{n:>34}" for n in names))
    for metric in HEADLINE:
        cells = "".join(f"{_p(report['sets'][n]['profile']['intervals'][metric]['value'])} "
                        f"{_ci(report['sets'][n]['profile']['intervals'][metric]):>26}"
                        for n in names)  # fmt: skip
        print(f"  {metric:32}{cells}")
    print(f"  label F1 (rules / model) {'':7}" + "".join(f"{n:>34}" for n in names))
    for label in LABELS:
        cells = ""
        for n in names:
            t = report["sets"][n]["labels"]["labels"][label]
            r, m = t["rules"]["f1"], t["model"]["f1"]
            cells += f"{_p(r):>16} / {_p(m):<15}"
        print(f"  {label:32}{cells}")
    print(f"  {report['test_vs_dev']['summary']}")
    if args.no_mlflow:
        return 0
    params = {
        "frozen_commit": report["preregistration"]["frozen_commit"],
        "post_hoc": str(args.post_hoc),
        "bootstrap_resamples": str(B),
        "seed": str(SEED),
    }
    name = "score_once_post_hoc" if args.post_hoc else "score_once"
    try:
        run_id, seen = log_flat_run(name, params, flat_metrics(report), [args.out, args.prereg],
                                    EXPERIMENT, args.tracking_dir)  # fmt: skip
    except TrackingRefused as e:
        print(f"MLflow: {e}", file=sys.stderr)
        return 3  # the report is written; only tracking was refused
    print(f"  MLflow run {run_id} logged to {args.tracking_dir}/ (experiment '{EXPERIMENT}')")
    for w in seen:
        print(f"  MLflow setup warning (not about the store, logged as a tag): {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
