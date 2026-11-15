"""Rules vs the frozen LightGBM models, on dev and (once) on the shift batch.

    # first, a dry run on dev only: prints, writes nothing, never reads the shift batch
    uv run python -m statementscreen.compare --dev-only
    # then the one-time comparison
    uv run python -m statementscreen.compare

This is the scoring side: it reads the ground truth (transactions.parquet, labels,
manifest; for the shift batch also narration_formats.parquet), the Week 3 pattern outputs
(the rules' flags and has_salary), the Week 4 features and the frozen models. Dev is
loaded with load_split (manifest hash checked, test split closed), the shift batch with
load_eval (checked against its registration).

For each label: precision, recall and F1 for the rules and for the model, plus the
model's PR-AUC. Per statement: has_salary (for the model: any row predicted as salary;
the truth: the statement has a true employer). False salary flags on gig and
self-employed statements. Breakdowns by layout, hard-case tag and, on the shift batch,
Karan format group (dialect_report.format_group of the line that wrote each narration;
"default_narration" for the tax rows that kept today's text). 95% intervals for every F1
and for model minus rules, by a paired bootstrap over statements (B resamples, fixed
seed; the same resampled statements for both methods). Global feature importance (gain)
and, for up to MAX_EXPLAINED model misses and false flags per set, the top 3
pred_contrib features in plain words. The synthetic-to-shift drop: dev F1 minus shift F1.

Scored once: before the shift batch is read, the model's dev F1 must equal the one
frozen in models.json for every label, or nothing more is done. The report records the
frozen models' SHA-256. A report that already holds the shift result is never replaced:
--post-hoc keeps the first result and appends the new one, labelled post-hoc.

Writes reports/week4_classifier.json and logs a local MLflow run (experiment
week4_classifier; --tracking-dir, --no-mlflow). Records seconds and peak memory
(ru_maxrss via summary.peak_memory_bytes: bytes on macOS, kilobytes on Linux).
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from statementscreen.dialect_report import format_group
from statementscreen.eval_batches import load_eval
from statementscreen.features import KEYS, load_set
from statementscreen.pattern_scoring import load_truth, prf
from statementscreen.splits import SPLITS_DIR, SplitError, load_split
from statementscreen.summary import peak_memory_bytes
from statementscreen.synth.dialect import load_dialect
from statementscreen.tracking import DEFAULT_DIR, TrackingRefused, log_flat_run
from statementscreen.train import LABELS, to_matrix

B = 1000
SEED = 42
MAX_EXPLAINED = 30
EXPERIMENT = "week4_classifier"
METHODS = ("rules", "model")
DEFAULT_REPORT = Path("reports/week4_classifier.json")
DEFAULT_NARRATION = "default_narration"


@dataclass(frozen=True)
class EvalSet:
    name: str  # "dev" or a registered evaluation batch
    batch: Path  # the generated batch (truth)
    split: str | None  # "dev" for a split of the reference batch, None for an eval batch


# ---------------------------------------------------------------- metrics (pure)


def scores(tp: int, fp: int, fn: int) -> dict:
    """pattern_scoring.prf, except that F1 is always 2tp / (2tp + fp + fn) when that is
    defined: a method that flags nothing while positives exist has F1 0, not "undefined"
    (its precision stays undefined). The bootstrap uses the same definition. Added after
    the first shift scoring (Week 4 decision), so the report's re-run is post-hoc."""
    out = prf(tp, fp, fn)
    if out["f1"] is None and 2 * tp + fp + fn > 0:
        out["f1"] = 2 * tp / (2 * tp + fp + fn)
    return out


def counts(frame: pl.DataFrame, label: str, method: str) -> tuple[int, int, int]:
    """(tp, fp, fn) of one method's predictions (r_ or m_<label>) against t_<label>."""
    t, p = frame[f"t_{label}"], frame[f"{method[0]}_{label}"]
    return int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())


def label_table(frame: pl.DataFrame) -> dict:
    """Per label, rules and model P/R/F1 (and counts) over the rows of frame."""
    return {label: {m: scores(*counts(frame, label, m)) for m in METHODS} for label in LABELS}


def per_statement(frame: pl.DataFrame, label: str, method: str,
                  ids: Sequence[str]) -> np.ndarray:  # fmt: skip
    """(statements, 3) array of tp, fp, fn per statement, in the order of ids."""
    t, p = pl.col(f"t_{label}"), pl.col(f"{method[0]}_{label}")
    agg = frame.group_by("statement_id").agg(
        (p & t).sum().alias("tp"), (p & ~t).sum().alias("fp"), (~p & t).sum().alias("fn")
    )
    order = pl.DataFrame({"statement_id": list(ids)})
    joined = order.join(agg, on="statement_id", how="left").fill_null(0)
    return joined.select("tp", "fp", "fn").to_numpy().astype(np.int64)


def _f1(c: np.ndarray) -> np.ndarray:
    """F1 = 2tp / (2tp + fp + fn) along the last axis; NaN where it is undefined."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def _interval(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"low": None, "high": None, "includes_zero": None, "resamples_used": 0}
    low, high = (float(v) for v in np.percentile(values, [2.5, 97.5]))
    return {"low": low, "high": high, "includes_zero": low <= 0 <= high,
            "resamples_used": int(values.size)}  # fmt: skip


def bootstrap(rules: np.ndarray, model: np.ndarray, n: int = B, seed: int = SEED) -> dict:
    """Paired bootstrap over statements: the same resampled statements for both methods.
    rules, model: (statements, 3) tp/fp/fn. 95% percentile intervals for each F1 and for
    model minus rules; resamples where an F1 is undefined are left out (and counted)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rules), size=(n, len(rules)))
    f1_rules, f1_model = _f1(rules[idx].sum(axis=1)), _f1(model[idx].sum(axis=1))
    return {"rules": _interval(f1_rules), "model": _interval(f1_model),
            "model_minus_rules": _interval(f1_model - f1_rules)}  # fmt: skip


def _with_intervals(table: dict, frame: pl.DataFrame, label: str, ids: Sequence[str],
                    n: int, seed: int) -> dict:  # fmt: skip
    ci = bootstrap(per_statement(frame, label, "rules", ids),
                   per_statement(frame, label, "model", ids), n, seed)  # fmt: skip
    for m in METHODS:
        table[m]["f1_ci"] = ci[m]
    f1s = [table[m]["f1"] for m in METHODS]
    diff = f1s[1] - f1s[0] if None not in f1s else None
    table["model_minus_rules"] = {"f1": diff, "ci": ci["model_minus_rules"]}
    return table


def false_salary(statements: pl.DataFrame) -> dict:
    out = {}
    for kind in ("gig", "self_employed"):
        group = statements.filter(pl.col("customer_type") == kind)
        out[kind] = {"statements": group.height,
                     **{m: int(group[f"{m[0]}_has_salary"].sum()) for m in METHODS}}  # fmt: skip
    return out


def breakdowns(frame: pl.DataFrame, by_format_group: bool) -> dict:
    layouts = sorted(frame["layout"].unique())
    out = {"by_layout": {k: label_table(frame.filter(pl.col("layout") == k)) for k in layouts}}
    tags = sorted({t for ts in frame["tags"].unique() for t in ts.split("|") if t})
    by_tag = {"none": label_table(frame.filter(pl.col("tags") == ""))}
    for tag in tags:
        has_tag = pl.col("tags").str.split("|").list.contains(tag)
        by_tag[tag] = label_table(frame.filter(has_tag))
    out["by_hard_case"] = by_tag
    if by_format_group:
        groups = {}
        for k in sorted(frame["format_group"].unique()):
            part = frame.filter(pl.col("format_group") == k)
            groups[k] = {"rows": part.height, **label_table(part)}
        out["by_format_group"] = groups
    return out


def score_set(frame: pl.DataFrame, statements: pl.DataFrame, by_format_group: bool,
              n: int = B, seed: int = SEED) -> dict:  # fmt: skip
    """Everything for one set. frame: one row per transaction with t_/r_/m_/s_<label>,
    statement_id, layout, tags (and format_group); statements: one row per statement with
    customer_type and t_/r_/m_has_salary."""
    ids = sorted(statements["statement_id"])
    labels = {}
    for label in LABELS:
        table = {m: scores(*counts(frame, label, m)) for m in METHODS}
        truth = frame[f"t_{label}"].to_numpy()
        pr_auc = average_precision_score(truth, frame[f"s_{label}"]) if truth.any() else None
        table["model"]["pr_auc"] = None if pr_auc is None else float(pr_auc)
        labels[label] = _with_intervals(table, frame, label, ids, n, seed)
    hs = {m: scores(*counts(statements, "has_salary", m)) for m in METHODS}
    hs = _with_intervals(hs, statements, "has_salary", ids, n, seed)
    return {"statements": statements.height, "rows": frame.height, "labels": labels,
            "has_salary": hs, "false_salary": false_salary(statements),
            "breakdown": breakdowns(frame, by_format_group)}  # fmt: skip


def drops(dev: dict, shift: dict) -> dict:
    """Synthetic-to-shift drop per label and method: dev F1 minus shift F1."""
    out = {}
    for label in LABELS:
        out[label] = {}
        for m in METHODS:
            a, b = dev["labels"][label][m]["f1"], shift["labels"][label][m]["f1"]
            out[label][m] = a - b if a is not None and b is not None else None
    return out


# ---------------------------------------------------------------- plain words

GROUP_WORDS = {
    "grp_size": "its group has {v:.0f} transactions",
    "grp_months_present": "its group appears in {v:.0f} months",
    "grp_month_coverage": "its group covers {v:.0%} of the statement's months",
    "grp_txn_per_month": "its group has {v:.1f} transactions per month",
    "grp_gap_mean_days": "its group's average gap is {v:.1f} days",
    "grp_gap_std_days": "its group's gaps vary by {v:.1f} days (SD)",
    "grp_dom_mean_deviation": "its group is {v:.1f} days from its usual day on average",
    "grp_dom_share_on_time": "{v:.0%} of its group lands within 3 days of the usual day",
    "grp_dom_arc_span": "its group's days of the month span {v:.0f} days",
    "grp_amount_cv": "its group's amounts vary by {v:.2f} (coefficient of variation)",
    "grp_share_near_median": "{v:.0%} of its group is within 5% of the group's median amount",
}


def describe(name: str, value: float, ngrams: Sequence[str]) -> str:
    """One feature's value in plain words."""
    if name.startswith("tfidf_"):
        gram = ngrams[int(name.removeprefix("tfidf_"))]
        return f"narration {'contains' if value > 0 else 'does not contain'} '{gram}'"
    if value is None or math.isnan(value):
        return f"{name} is missing"
    if name == "is_credit":
        return "money in (a credit)" if value else "money out (a debit)"
    if name.startswith("channel_"):
        channel = name.removeprefix("channel_").upper()
        return f"channel is {channel}" if value else f"channel is not {channel}"
    if name == "log_amount":
        return f"amount is INR {math.expm1(value):,.0f}"
    if name == "day_of_month":
        return f"posted on day {value:.0f} of the month"
    if name == "position":
        return f"{value:.0%} of the way down the statement"
    if name == "cut":
        return "narration is cut short" if value else "narration is not cut short"
    if name == "amount_to_group_median":
        return f"amount is {value:.2f} times its group's median"
    if name == "day_from_group_medoid":
        return f"{value:.0f} days from its group's usual day"
    return GROUP_WORDS[name].format(v=value) if name in GROUP_WORDS else f"{name} = {value:g}"


def top_contributions(contrib: np.ndarray, values: np.ndarray, names: Sequence[str],
                      ngrams: Sequence[str], toward: int, k: int = 3) -> list[dict]:  # fmt: skip
    """The k features pushing hardest toward the wrong answer: toward=+1 for a false flag
    (pushed up), -1 for a miss (pushed down). contrib excludes the bias term."""
    out = []
    for i in np.argsort(-toward * contrib)[:k]:
        if toward * contrib[i] <= 0:
            break
        value = float(values[i])
        out.append({"feature": names[i], "value": None if math.isnan(value) else value,
                    "contribution_log_odds": round(float(contrib[i]), 4),
                    "in_words": describe(names[i], value, ngrams)})  # fmt: skip
    return out


def pick_errors(frame: pl.DataFrame, limit: int = MAX_EXPLAINED) -> list[dict]:
    """Up to `limit` model errors (misses and false flags), taken in turn from each label
    and kind, so no label crowds out the rest. Deterministic order."""
    pools = []
    for label in LABELS:
        t, m = pl.col(f"t_{label}"), pl.col(f"m_{label}")
        for kind, cond in (("miss", t & ~m), ("false flag", m & ~t)):
            rows = frame.filter(cond).sort(KEYS).select(*KEYS, f"s_{label}")
            pools.append([{"label": label, "kind": kind, "statement_id": r["statement_id"],
                           "row": r["row"], "score": r[f"s_{label}"]}
                          for r in rows.iter_rows(named=True)])  # fmt: skip
    out: list[dict] = []
    while len(out) < limit and any(pools):
        for pool in pools:
            if pool and len(out) < limit:
                out.append(pool.pop(0))
    return out


# ---------------------------------------------------------------- loading


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(models_dir: Path) -> tuple[dict, dict[str, lgb.Booster]]:
    meta = json.loads((models_dir / "models.json").read_text(encoding="utf-8"))
    if not meta.get("frozen"):
        raise ValueError(f"{models_dir} models are not frozen")
    boosters = {label: lgb.Booster(model_file=str(models_dir / f"{label}.txt"))
                for label in LABELS}  # fmt: skip
    return meta, boosters


def predict(es: EvalSet, features_root: Path, meta: dict, boosters: dict) -> pl.DataFrame:
    """KEYS, s_<label> (model probability) and m_<label> (at the frozen threshold)."""
    parts = []
    for dense, tfidf in load_set(features_root, es.name):
        X = to_matrix(dense, tfidf)
        scores = [pl.Series(f"s_{label}", boosters[label].predict(X)) for label in LABELS]
        parts.append(dense.select(KEYS).with_columns(scores))
    model = pl.concat(parts)
    flags = [(pl.col(f"s_{label}") >= meta["labels"][label]["threshold"]).alias(f"m_{label}")
             for label in LABELS]  # fmt: skip
    return model.with_columns(flags)


def _format_groups(batch: Path) -> pl.DataFrame | None:
    """KEYS and format_group, from the batch's narration_formats.parquet (if any)."""
    path = batch / "narration_formats.parquet"
    if not path.exists():
        return None
    fmts = pl.read_parquet(path).with_columns(pl.col("row").cast(pl.Int64))
    group_of = {f.line: format_group(f) for f in load_dialect(fmts["dialect"][0]).formats}
    groups = [group_of.get(line, DEFAULT_NARRATION) if line is not None else DEFAULT_NARRATION
              for line in fmts["format_line"]]  # fmt: skip
    return fmts.select(KEYS).with_columns(pl.Series("format_group", groups))


def assemble(es: EvalSet, patterns_root: Path, features_root: Path, meta: dict,
             boosters: dict, splits_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:  # fmt: skip
    """One set's per-row frame (truth, rules, model) and per-statement frame."""
    manifest_path = es.batch / "manifest.csv"
    if es.split:
        ids = load_split(es.split, manifest_path, splits_dir)
    else:
        ids = load_eval(es.name, manifest_path, splits_dir)
    truth = load_truth(es.batch, ids)
    patterns = patterns_root / es.name
    rules = pl.read_parquet(patterns / "rows.parquet",
                            columns=["statement_id", "row", *LABELS, "narration"])  # fmt: skip
    rules = rules.rename({label: f"r_{label}" for label in LABELS})
    model = predict(es, features_root, meta, boosters)

    t = truth.rows.select("statement_id", "row", "category", *[f"t_{x}" for x in LABELS])
    frame = t.join(rules, on=KEYS, how="full", coalesce=True)
    frame = frame.join(model, on=KEYS, how="full", coalesce=True)
    broken = frame.filter(pl.col("category").is_null() | pl.col("r_is_salary").is_null()
                          | pl.col("s_is_salary").is_null()).height  # fmt: skip
    if broken:
        raise ValueError(f"{es.name}: truth, rules and model rows don't line up "
                         f"({broken} rows missing from one of them)")  # fmt: skip
    man = pl.DataFrame([{"statement_id": sid, "layout": m["layout"],
                         "customer_type": m["customer_type"], "tags": m["hard_case_tags"] or ""}
                        for sid, m in truth.manifest.items()])  # fmt: skip
    frame = frame.join(man, on="statement_id", how="left")
    groups = _format_groups(es.batch)
    if groups is not None:
        frame = frame.join(groups, on=KEYS, how="left")
    frame = frame.sort(KEYS)

    results = pl.read_parquet(patterns / "statement_results.parquet",
                              columns=["statement_id", "has_salary"])  # fmt: skip
    model_has = frame.group_by("statement_id").agg(pl.col("m_is_salary").any())
    true_has = pl.DataFrame({"statement_id": list(truth.labels),
                             "t_has_salary": [lab.get("true_employer") is not None
                                              for lab in truth.labels.values()]})  # fmt: skip
    statements = (man.join(results.rename({"has_salary": "r_has_salary"}), on="statement_id",
                           how="left")
                  .join(model_has.rename({"m_is_salary": "m_has_salary"}), on="statement_id",
                        how="left")
                  .join(true_has, on="statement_id", how="left")
                  .sort("statement_id"))  # fmt: skip
    return frame, statements


def explain(es: EvalSet, frame: pl.DataFrame, features_root: Path, meta: dict,
            boosters: dict, limit: int = MAX_EXPLAINED) -> list[dict]:  # fmt: skip
    """Top 3 pred_contrib features, in plain words, for up to `limit` model errors."""
    picked = pick_errors(frame, limit)
    wanted = {(e["statement_id"], e["row"]) for e in picked}
    names, ngrams = meta["feature_names"], meta["tfidf_ngrams"]
    rows: dict[tuple[str, int], tuple] = {}  # key -> (one-row matrix, its values)
    for dense, tfidf in load_set(features_root, es.name) if wanted else []:
        keys = list(zip(dense["statement_id"], dense["row"], strict=True))
        hit = [i for i, key in enumerate(keys) if key in wanted]
        if hit:
            X = to_matrix(dense, tfidf)[hit]
            values = X.toarray()
            for j, i in enumerate(hit):
                rows[keys[i]] = (X[j], values[j])
    info = frame.select(*KEYS, "narration", "category").rows_by_key(KEYS, named=True,
                                                                    unique=True)  # fmt: skip
    out = []
    for e in picked:
        key = (e["statement_id"], e["row"])
        x, values = rows[key]
        contrib = boosters[e["label"]].predict(x, pred_contrib=True)
        contrib = contrib.toarray() if hasattr(contrib, "toarray") else np.asarray(contrib)
        toward = 1 if e["kind"] == "false flag" else -1
        top = top_contributions(contrib[0, :-1], values, names, ngrams, toward)
        out.append({**e, "narration": info[key]["narration"],
                    "true_category": info[key]["category"],
                    "threshold": meta["labels"][e["label"]]["threshold"],
                    "top_features": top})  # fmt: skip
    return out


def importance(meta: dict, boosters: dict, k: int = 15) -> dict:
    """Per label, the k features with the most total gain, in readable form."""
    names, ngrams = meta["feature_names"], meta["tfidf_ngrams"]
    out = {}
    for label in LABELS:
        gain = boosters[label].feature_importance(importance_type="gain")
        total = float(gain.sum()) or 1.0
        top = []
        for i in np.argsort(-gain)[:k]:
            if gain[i] <= 0:
                break
            name = names[i]
            reads = (f"narration n-gram '{ngrams[int(name.removeprefix('tfidf_'))]}'"
                     if name.startswith("tfidf_") else name)  # fmt: skip
            top.append({"feature": name, "reads_as": reads,
                        "gain_share": round(float(gain[i]) / total, 4)})  # fmt: skip
        out[label] = top
    return out


# ---------------------------------------------------------------- the one-time report


def already_scored(path: Path) -> bool:
    """Does the report at path hold the one-time shift result?"""
    if not path.exists():
        return False
    return bool(json.loads(path.read_text(encoding="utf-8")).get("shift_first_result"))


def write_report(path: Path, report: dict, post_hoc: bool) -> dict:
    """Write the report. One that already holds the shift result is never replaced: with
    post_hoc=True the new result is appended, labelled post-hoc; otherwise refused."""
    if already_scored(path):
        if not post_hoc:
            raise ValueError(f"{path} already holds the one-time shift result; pass "
                             "--post-hoc to add a re-run, labelled post-hoc")  # fmt: skip
        old = json.loads(path.read_text(encoding="utf-8"))
        new = {k: v for k, v in report.items() if k != "shift_first_result"}
        old.setdefault("post_hoc_runs", []).append({**new, "label": "post-hoc"})
        report = old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")  # fmt: skip
    return report


def flat_metrics(report: dict) -> dict[str, float]:
    m: dict[str, float] = {}
    for set_name, s in report["sets"].items():
        for label, t in s["labels"].items():
            for method in METHODS:
                for k in ("precision", "recall", "f1"):
                    if t[method][k] is not None:
                        m[f"{set_name}.{label}.{method}.{k}"] = t[method][k]
            if t["model"].get("pr_auc") is not None:
                m[f"{set_name}.{label}.model.pr_auc"] = t["model"]["pr_auc"]
            ci = t["model_minus_rules"]["ci"]
            if ci["low"] is not None:
                m[f"{set_name}.{label}.diff_ci_low"] = ci["low"]
                m[f"{set_name}.{label}.diff_ci_high"] = ci["high"]
    for label, d in report.get("synthetic_to_shift_drop", {}).items():
        for method, v in d.items():
            if v is not None:
                m[f"drop.{label}.{method}"] = v
    m["seconds_total"] = report["run"]["seconds_total"]
    m["peak_memory_mb"] = report["run"]["peak_memory_mb"]
    return m


def _n(x: float | None) -> str:
    return "  n/a " if x is None else f"{x:.4f}"


def _span(ci: dict) -> str:
    if ci["low"] is None:
        return "n/a"
    zero = " (includes 0)" if ci["includes_zero"] else ""
    return f"[{ci['low']:+.4f}, {ci['high']:+.4f}]{zero}"


def print_set(name: str, s: dict) -> None:
    print(f"  {name}: {s['statements']} statements, {s['rows']} rows")
    for label, t in [*s["labels"].items(), ("has_salary", s["has_salary"])]:
        r, m = t["rules"], t["model"]
        auc = f", PR-AUC {_n(m['pr_auc'])}" if "pr_auc" in m else ""
        print(f"    {label:10} rules P {_n(r['precision'])} R {_n(r['recall'])} "
              f"F1 {_n(r['f1'])} | model P {_n(m['precision'])} R {_n(m['recall'])} "
              f"F1 {_n(m['f1'])}{auc}")  # fmt: skip
        print(f"    {'':10} model minus rules F1, 95% CI {_span(t['model_minus_rules']['ci'])}")
    for kind, f in s["false_salary"].items():
        print(f"    false salary on {kind}: rules {f['rules']}, model {f['model']} "
              f"of {f['statements']} statements")  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m statementscreen.compare",
                                description="Compare the rules and the frozen models.")  # fmt: skip
    p.add_argument("--dev-batch", type=Path, default=Path("data/week1_n2000"))
    p.add_argument("--shift-batch", type=Path, default=Path("data/week4_shift_n300"))
    p.add_argument("--shift-name", default="shift_karan_v1")
    p.add_argument("--patterns", type=Path, default=Path("data/patterns"))
    p.add_argument("--features", type=Path, default=Path("data/features"))
    p.add_argument("--models", type=Path, default=Path("data/models"))
    p.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    dry_run = "dry run: print dev, write nothing, never read the shift batch"
    p.add_argument("--dev-only", action="store_true", help=dry_run)
    p.add_argument("--post-hoc", action="store_true",
                   help="re-score the shift batch; the first result is kept")  # fmt: skip
    p.add_argument("--tracking-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)

    started = time.perf_counter()
    try:
        if not args.dev_only and already_scored(args.out) and not args.post_hoc:
            raise ValueError(f"{args.out} already holds the one-time shift result; pass "
                             "--post-hoc to add a re-run, labelled post-hoc")  # fmt: skip
        meta, boosters = load_models(args.models)
        dev = EvalSet("dev", args.dev_batch, "dev")
        frame, statements = assemble(dev, args.patterns, args.features, meta, boosters,
                                     args.splits_dir)  # fmt: skip
        sets = {"dev": score_set(frame, statements, by_format_group=False)}
        mismatched = [label for label in LABELS
                      if abs(sets["dev"]["labels"][label]["model"]["f1"]
                             - meta["labels"][label]["f1"]) > 1e-9]  # fmt: skip
        if mismatched:
            raise ValueError(f"model dev F1 differs from the frozen models for {mismatched}; "
                             "stopping before the shift batch is read")  # fmt: skip
        if args.dev_only:
            print("Dry run on dev (nothing written; the shift batch was not read):")
            print_set("dev", sets["dev"])
            print("  model dev F1 equals the frozen models' for every label")
            return 0
        explained = {"dev": explain(dev, frame, args.features, meta, boosters)}
        shift = EvalSet(args.shift_name, args.shift_batch, None)
        frame, statements = assemble(shift, args.patterns, args.features, meta, boosters,
                                     args.splits_dir)  # fmt: skip
        sets[shift.name] = score_set(frame, statements, by_format_group=True)
        explained[shift.name] = explain(shift, frame, args.features, meta, boosters)
    except (SplitError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    chosen_keys = ("setting", "num_leaves", "learning_rate", "min_child_samples", "threshold",
                   "best_iteration")  # fmt: skip
    report = {
        "sets": sets,
        "shift_first_result": True,
        "shift_set": args.shift_name,
        "synthetic_to_shift_drop": drops(sets["dev"], sets[args.shift_name]),
        "importance_gain": importance(meta, boosters),
        "explained_errors": explained,
        "models": {
            "sha256": {label: sha256(args.models / f"{label}.txt") for label in LABELS},
            "models_json_sha256": sha256(args.models / "models.json"),
            "chosen": {label: {k: meta["labels"][label][k] for k in chosen_keys}
                       for label in LABELS},
        },
        "bootstrap": {"resamples": B, "seed": SEED, "unit": "statement", "paired": True},
        "notes": [
            "dev was used for tuning (settings and thresholds)",
            "the shift batch was scored once, after every setting was frozen",
            "the shift formats were written by an independent collaborator who had not "
            "seen the code",
            "all data is synthetic",
            "the test split is still closed",
        ],
        "run": {"seconds_total": round(time.perf_counter() - started, 1),
                "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1)},
    }  # fmt: skip
    try:
        write_report(args.out, report, args.post_hoc)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    run = report["run"]
    print(f"Wrote {args.out}{' (post-hoc re-run appended)' if args.post_hoc else ''} in "
          f"{run['seconds_total']} s, peak memory {run['peak_memory_mb']} MB")  # fmt: skip
    for name, s in sets.items():
        print_set(name, s)
    for label, d in report["synthetic_to_shift_drop"].items():
        print(f"  dev-to-shift F1 drop {label:9} rules {_n(d['rules'])}  model {_n(d['model'])}")
    if args.no_mlflow:
        return 0
    run_params = {f"{label}.{k}": str(v) for label, c in report["models"]["chosen"].items()
                  for k, v in c.items()}  # fmt: skip
    run_params |= {"bootstrap_resamples": str(B), "seed": str(SEED),
                   "post_hoc": str(args.post_hoc)}  # fmt: skip
    name = "compare_post_hoc" if args.post_hoc else "compare"
    try:
        run_id, seen = log_flat_run(name, run_params, flat_metrics(report), [args.out],
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
