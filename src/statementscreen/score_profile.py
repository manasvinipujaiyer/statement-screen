"""Score income profiles against the ground truth. The profile module never imports this.

    uv run python -m statementscreen.score_profile data/profiles/ data/week1_n2000/ --set dev

Reads data/profiles/<set>/profiles.jsonl and the batch's truth: labels/*.json (true
monthly income, customer type, true employer), transactions.parquet (EMI and bounce
rows) and manifest.csv (layout, hard-case tags). A split is loaded with load_split
(manifest hash checked; "test" refused unless STATEMENTSCREEN_OPEN_TEST=1); an evaluation
batch (--eval) with load_eval.

Truth beyond the labels:
  EMIs        distinct lenders (canonical_counterparty) of loan_payment rows; the total is
              the sum of each lender's median payment
  bounces     the number of bounce_charge rows (each bounce has one)

Income error is the absolute percentage error |estimate - truth| / truth, reported two
ways: over statements with an estimate (and the share without one), and with every
missing estimate counted as a 100% error. Statements whose true income is 0 are counted
separately, never divided by.

Everything is computed from one row per statement (statement_table), so the overall
numbers and every breakdown (layout, customer type, hard-case tag) are computed the same
way, and a bootstrap can resample statements directly. Writes
reports/week5_profile_<set>.json.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from statementscreen.eval_batches import load_eval
from statementscreen.pattern_scoring import EMPLOYER_FUZZY, normalise_employer, prf
from statementscreen.splits import SPLITS_DIR, SplitError, load_split

TYPES = ("salaried", "gig", "self_employed")
WORST = 20


# ---------------------------------------------------------------- one row per statement


def read_profiles(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        profiles = [json.loads(line) for line in f if line.strip()]
    return {p["statement_id"]: p for p in profiles}


def truth_table(batch: Path, ids: list[str]) -> pl.DataFrame:
    """One row per statement: true income, type, employer, EMIs, bounces, layout, tags."""
    tx = (pl.scan_parquet(batch / "transactions.parquet")
          .filter(pl.col("statement_id").is_in(ids))
          .select("statement_id", "category", "canonical_counterparty", "amount_paise")
          .collect())  # fmt: skip
    loans = (tx.filter(pl.col("category") == "loan_payment")
             .group_by("statement_id", "canonical_counterparty")
             .agg(pl.col("amount_paise").abs().median().alias("usual"))
             .group_by("statement_id")
             .agg(pl.len().alias("t_emi_count"),
                  pl.col("usual").sum().round().cast(pl.Int64).alias("t_emi_total")))  # fmt: skip
    bounces = (tx.filter(pl.col("category") == "bounce_charge").group_by("statement_id")
               .agg(pl.len().alias("t_bounces")))  # fmt: skip
    manifest = {r["statement_id"]: r for r in pl.read_csv(
        batch / "manifest.csv", infer_schema=False).iter_rows(named=True)}  # fmt: skip
    records = []
    for sid in ids:
        label = json.loads((batch / "labels" / f"{sid}.json").read_text(encoding="utf-8"))
        records.append({"statement_id": sid, "layout": manifest[sid]["layout"],
                        "tags": manifest[sid]["hard_case_tags"] or "",
                        "t_type": label["customer_type"],
                        "t_income": label["true_monthly_income_paise"],
                        "t_employer": label["true_employer"]})  # fmt: skip
    table = pl.DataFrame(records, infer_schema_length=None)
    table = table.join(loans, on="statement_id", how="left").join(bounces, on="statement_id",
                                                                   how="left")  # fmt: skip
    return table.with_columns(pl.col("t_emi_count", "t_emi_total", "t_bounces").fill_null(0))


def statement_table(profiles: dict[str, dict], truth: pl.DataFrame) -> pl.DataFrame:
    """Truth joined to what each profile says, one row per statement."""
    missing = sorted(set(truth["statement_id"]) - set(profiles))
    if missing:
        raise ValueError(f"{len(missing)} statements have no profile, e.g. {missing[0]}")
    rows = []
    for t in truth.iter_rows(named=True):
        p = profiles[t["statement_id"]]
        f = {k: v["value"] for k, v in p["findings"].items()}
        est, true = f["estimated_monthly_income_paise"], t["t_income"]
        ape = abs(est - true) / true if est is not None and true else None
        employer = f["employer"]
        both = bool(t["t_employer"]) and bool(employer)
        a, b = normalise_employer(t["t_employer"]), normalise_employer(employer)
        rows.append(
            {
                **t,
                "p_income": est,
                "ape": ape,
                "income_reason": p["findings"]["estimated_monthly_income_paise"]["reason"],
                "p_has_salary": bool(f["has_salary"]),
                "t_has_salary": t["t_employer"] is not None,
                "p_employer": employer,
                "employer_exact": both and a == b,
                "employer_fuzzy": both and fuzz.token_sort_ratio(a, b) >= EMPLOYER_FUZZY,
                "p_type": f["customer_type"],
                "p_emi_count": f["emi_count"],
                "p_emi_total": f["emi_total_monthly_paise"],
                "p_bounces": f["bounced_emis"],
                "warnings": " | ".join(p["warnings"]),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------- metrics


def income_metrics(table: pl.DataFrame) -> dict:
    """Income error, both ways: over estimated statements, and missing = 100% error."""
    positive = table.filter(pl.col("t_income") > 0)
    estimated = positive.filter(pl.col("ape").is_not_null())
    apes = estimated["ape"].to_list()
    all_apes = apes + [1.0] * (positive.height - estimated.height)

    def summary(values: list[float]) -> dict:
        if not values:
            return {"statements": 0, "median_ape": None, "within_5pct": None,
                    "within_10pct": None}  # fmt: skip
        return {"statements": len(values), "median_ape": statistics.median(values),
                "within_5pct": sum(v <= 0.05 for v in values) / len(values),
                "within_10pct": sum(v <= 0.10 for v in values) / len(values)}  # fmt: skip

    return {
        "estimated_only": summary(apes),
        "missing_as_100pct": summary(all_apes),
        "share_without_estimate": ((positive.height - estimated.height) / positive.height
                                   if positive.height else None),
        "true_income_zero": table.height - positive.height,
    }  # fmt: skip


def metrics(table: pl.DataFrame) -> dict:
    """Every profile metric over the statements of `table`."""
    if table.height == 0:
        return {"statements": 0}
    hs = prf(int((table["p_has_salary"] & table["t_has_salary"]).sum()),
             int((table["p_has_salary"] & ~table["t_has_salary"]).sum()),
             int((~table["p_has_salary"] & table["t_has_salary"]).sum()))  # fmt: skip
    salaried = table.filter(pl.col("t_has_salary"))
    confusion = {t: dict(Counter(table.filter(pl.col("t_type") == t)["p_type"].to_list()))
                 for t in TYPES if (table["t_type"] == t).any()}  # fmt: skip
    emi_error = (table["p_emi_total"] - table["t_emi_total"]).abs()
    return {
        "statements": table.height,
        "income": income_metrics(table),
        "has_salary": hs,
        "employer": {
            "statements_with_true_employer": salaried.height,
            "exact_match_rate": salaried["employer_exact"].mean() if salaried.height else None,
            "fuzzy_match_rate": salaried["employer_fuzzy"].mean() if salaried.height else None,
        },
        "customer_type": {"accuracy": (table["p_type"] == table["t_type"]).mean(),
                          "confusion_true_to_predicted": confusion},
        "emi": {"count_exact_rate": (table["p_emi_count"] == table["t_emi_count"]).mean(),
                "total_monthly_abs_error_paise": {"mean": emi_error.mean(),
                                                  "median": emi_error.median(),
                                                  "max": emi_error.max()},
                "total_monthly_exact_rate": (emi_error == 0).mean()},
        "bounces": {"count_exact_rate": (table["p_bounces"] == table["t_bounces"]).mean(),
                    "statements_with_true_bounce": int((table["t_bounces"] > 0).sum())},
    }  # fmt: skip


def breakdowns(table: pl.DataFrame) -> dict:
    tags = sorted({t for ts in table["tags"] for t in ts.split("|") if t})
    by_tag = {"none": metrics(table.filter(pl.col("tags") == ""))}
    for tag in tags:
        by_tag[tag] = metrics(table.filter(pl.col("tags").str.split("|").list.contains(tag)))
    return {
        "by_layout": {k: metrics(table.filter(pl.col("layout") == k))
                      for k in sorted(table["layout"].unique())},
        "by_customer_type": {k: metrics(table.filter(pl.col("t_type") == k))
                             for k in sorted(table["t_type"].unique())},
        "by_hard_case": by_tag,
    }  # fmt: skip


def worst(table: pl.DataFrame, n: int = WORST) -> list[dict]:
    """The n statements with the largest income error (a missing estimate counts as 100%)."""
    error = pl.col("ape").fill_null(1.0).alias("error")
    scored = table.filter(pl.col("t_income") > 0).with_columns(error)
    cols = ["statement_id", "t_type", "p_type", "t_income", "p_income", "error", "tags",
            "income_reason", "warnings"]  # fmt: skip
    ordered = scored.sort(["error", "statement_id"], descending=[True, False])
    return ordered.head(n).select(cols).to_dicts()


def build_report(profiles_dir: Path, batch: Path, ids: list[str]) -> tuple[dict, pl.DataFrame]:
    table = statement_table(read_profiles(profiles_dir / "profiles.jsonl"),
                            truth_table(batch, ids))  # fmt: skip
    report = {"overall": metrics(table), "breakdown": breakdowns(table),
              "worst_income_errors": worst(table)}  # fmt: skip
    return report, table


# ---------------------------------------------------------------- CLI


def _pct(x: float | None) -> str:
    return "  n/a  " if x is None else f"{x:7.2%}"


def print_overall(name: str, o: dict) -> None:
    inc = o["income"]
    e, m = inc["estimated_only"], inc["missing_as_100pct"]
    print(f"  {name}: {o['statements']} statements")
    print(f"    income, estimated only ({e['statements']}): median APE {_pct(e['median_ape'])}, "
          f"within 5% {_pct(e['within_5pct'])}, within 10% {_pct(e['within_10pct'])}")  # fmt: skip
    print(f"    income, missing = 100% ({m['statements']}): median APE {_pct(m['median_ape'])}, "
          f"within 5% {_pct(m['within_5pct'])}, within 10% {_pct(m['within_10pct'])}; "
          f"without estimate {_pct(inc['share_without_estimate'])}")  # fmt: skip
    hs, emp = o["has_salary"], o["employer"]
    print(f"    has_salary P {_pct(hs['precision'])} R {_pct(hs['recall'])}; employer exact "
          f"{_pct(emp['exact_match_rate'])} fuzzy {_pct(emp['fuzzy_match_rate'])}")  # fmt: skip
    ct = o["customer_type"]
    print(f"    customer type accuracy {_pct(ct['accuracy'])}; true -> predicted "
          f"{ct['confusion_true_to_predicted']}")  # fmt: skip
    emi, b = o["emi"], o["bounces"]
    emi_mae = emi["total_monthly_abs_error_paise"]["mean"]
    count, total = _pct(emi["count_exact_rate"]), _pct(emi["total_monthly_exact_rate"])
    print(f"    EMI count exact {count}, total exact {total}, mean abs error {emi_mae:.0f} paise")
    n_bounce = b["statements_with_true_bounce"]
    print(f"    bounce count exact {_pct(b['count_exact_rate'])} ({n_bounce} with a bounce)")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.score_profile",
        description="Score income profiles against the truth.",
    )
    p.add_argument("profiles", type=Path, help="root folder written by statementscreen.profile")
    p.add_argument("batch", type=Path, help="generated batch folder (truth)")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--set", dest="split", help="train or dev (test is closed)")
    which.add_argument("--eval", dest="eval_name", help="a registered evaluation batch")
    p.add_argument("--out", type=Path, help="default: reports/week5_profile_<set>.json")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    args = p.parse_args(argv)
    name = args.eval_name or args.split
    out = args.out or Path(f"reports/week5_profile_{name}.json")
    try:
        manifest = args.batch / "manifest.csv"
        if args.eval_name:
            ids = load_eval(args.eval_name, manifest, args.splits_dir)
        else:
            ids = load_split(args.split, manifest, args.splits_dir)
        report, _ = build_report(args.profiles / name, args.batch, ids)
    except (SplitError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    run = json.loads((args.profiles / name / "profiles_run.json").read_text(encoding="utf-8"))
    report = {"set": name, "batch": args.batch.name, "profiles_run": run, **report}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")  # fmt: skip
    print(f"Wrote {out}")
    print_overall(name, report["overall"])
    print(f"  worst income errors (of {len(report['worst_income_errors'])} listed):")
    for w in report["worst_income_errors"][:5]:
        print(f"    {w['statement_id']} {w['t_type']}->{w['p_type']} true {w['t_income']} "
              f"estimated {w['p_income']} error {w['error']:.0%}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
