"""Classifier features: built from pattern outputs alone, never from the rules' decisions."""

import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from statementscreen import features as ft
from statementscreen.splits import OPEN_TEST_ENV

SMALL = ft.TfidfConfig(sample_rows=1000, min_df=1, max_features=500)


def _statement(sid: str, narrations: list[str], flag: bool = False, style: str = "known"):
    """A statement's pattern rows and group features. Rows alternate debit / credit; the
    first half is group g1, the second g2. flag/style fill the excluded columns."""
    n = len(narrations)
    amounts = [(i + 1) * 10_000 * (1 if i % 2 else -1) for i in range(n)]
    rows = pl.DataFrame({
        "statement_id": [sid] * n, "row": list(range(n)),
        "posted": [date(2025, 1, 1 + i) for i in range(n)], "amount_paise": amounts,
        "direction": ["credit" if a > 0 else "debit" for a in amounts],
        "channel": ["NEFT"] * n, "style": [style] * n, "cut": [i == 0 for i in range(n)],
        "narration": narrations, "group_id": ["g1" if i < n // 2 else "g2" for i in range(n)],
        "is_salary": [flag] * n, "is_emi": [flag] * n, "is_bounce": [flag] * n,
        "is_rent": [flag] * n, "flag_reason": ["because" if flag else ""] * n,
    })  # fmt: skip
    groups = pl.DataFrame({
        "statement_id": [sid, sid], "group_id": ["g1", "g2"], "n_txn": [n // 2, n - n // 2],
        "median_amount_paise": [10_000, 20_000], "dom_medoid": [1, 31],
        "months_present": [1, 1], "month_coverage": [1.0, 1.0], "txn_per_month": [2.0, 2.0],
        "gap_mean_days": [1.0, 1.0], "gap_std_days": [0.0, 0.0],
        "dom_mean_deviation": [0.5, 0.5], "dom_share_on_time": [1.0, 1.0],
        "dom_arc_span": [1, 1], "amount_cv": [0.1, 0.1], "amount_share_near_median": [1.0, 0.5],
    })  # fmt: skip
    return rows, groups


def _patterns_folder(root: Path, flag: bool = False, style: str = "known") -> Path:
    """data/patterns/-like folder with ONLY rows.parquet and group_features.parquet."""
    sets = {
        "train": [_statement("stmt_0001", ["NEFT KKQQ PAYROLL 123", "UPI KKQQ 9", "KKQQ A",
                                           "KKQQ B"], flag, style),
                  _statement("stmt_0002", ["KKQQ C", "KKQQ D"], flag, style)],
        "dev": [_statement("stmt_0003", ["ZZJJ ZZJJ 77", "ZZJJ X"], flag, style)],
    }  # fmt: skip
    for name, statements in sets.items():
        (root / name).mkdir(parents=True)
        pl.concat([r for r, _ in statements]).write_parquet(root / name / "rows.parquet")
        pl.concat([g for _, g in statements]).write_parquet(root / name / "group_features.parquet")
    return root


def _build(tmp_path: Path, patterns: Path, name: str, out: str = "features"):
    vec, meta = ft.fit_tfidf(patterns, SMALL)
    ft.build_set(patterns, name, tmp_path / out, vec, meta)
    parts = list(ft.load_set(tmp_path / out, name))
    return pl.concat([d for d, _ in parts]), [t for _, t in parts], vec


def test_works_on_pattern_outputs_alone_and_never_reads_truth(tmp_path, monkeypatch) -> None:
    patterns = _patterns_folder(tmp_path / "patterns")
    assert {p.name for p in patterns.rglob("*") if p.is_file()} == {
        "rows.parquet", "group_features.parquet"}  # fmt: skip
    opened: list[str] = []
    real_read, real_scan = pl.read_parquet, pl.scan_parquet

    def read(source, *a, **k):
        opened.append(Path(source).name)
        return real_read(source, *a, **k)

    def scan(source, *a, **k):
        opened.append(Path(source).name)
        return real_scan(source, *a, **k)

    monkeypatch.setattr(pl, "read_parquet", read)
    monkeypatch.setattr(pl, "scan_parquet", scan)
    dense, _, _ = _build(tmp_path, patterns, "dev")
    assert dense.height == 2
    forbidden = {"transactions.parquet", "manifest.csv", "group_rules.parquet",
                 "statement_results.parquet"}  # fmt: skip
    assert set(opened) <= {"rows.parquet", "group_features.parquet"} | {
        f"{i:05d}.parquet" for i in range(10)}  # fmt: skip
    assert not set(opened) & forbidden


def test_no_feature_comes_from_the_rules_or_the_style(tmp_path) -> None:
    plain = _patterns_folder(tmp_path / "plain", flag=False, style="known")
    flagged = _patterns_folder(tmp_path / "flagged", flag=True, style="unknown")
    a, ta, _ = _build(tmp_path, plain, "train", "fa")
    b, tb, _ = _build(tmp_path, flagged, "train", "fb")
    assert a.columns == ft.KEYS + ft.DENSE_FEATURES
    assert not any(c in ft.EXCLUDED or "style" in c or "flag" in c for c in a.columns)
    assert a.equals(b)  # flipping every rule flag and the style changes nothing
    assert all((x != y).nnz == 0 for x, y in zip(ta, tb, strict=True))


def test_tfidf_is_fitted_on_train_rows_only(tmp_path) -> None:
    patterns = _patterns_folder(tmp_path / "patterns")
    vec, meta = ft.fit_tfidf(patterns, SMALL)
    assert meta["fitted_on"] == "train" and meta["train_rows"] == 6
    assert "kq" in vec.vocabulary_  # train-only text is in the vocabulary
    assert not any("zj" in g or "jj" in g for g in vec.vocabulary_)  # dev-only text isn't
    (patterns / "train" / "rows.parquet").unlink()  # without train, there is nothing to fit
    with pytest.raises(FileNotFoundError):
        ft.fit_tfidf(patterns, SMALL)


def test_feature_values_on_a_hand_built_statement(tmp_path) -> None:
    dense, text, _ = _build(tmp_path, _patterns_folder(tmp_path / "patterns"), "train")
    s = dense.filter(pl.col("statement_id") == "stmt_0001").sort("row")
    assert s["row"].to_list() == [0, 1, 2, 3]
    assert s["position"].to_list() == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert s["is_credit"].to_list() == [0, 1, 0, 1]
    assert s["channel_neft"].to_list() == [1, 1, 1, 1] and s["channel_upi"].sum() == 0
    assert s["cut"].to_list() == [1, 0, 0, 0]
    assert s["log_amount"][0] == pytest.approx(math.log1p(100.0))  # INR 100
    assert s["day_of_month"].to_list() == [1, 2, 3, 4]
    # rows 0-1 are group g1 (median INR 100, usual day 1); rows 2-3 g2 (INR 200, day 31)
    assert s["amount_to_group_median"].to_list() == pytest.approx([1, 2, 1.5, 2])
    assert s["day_from_group_medoid"].to_list() == [0, 1, 3, 4]  # day 3 is 3 from day 31
    assert s["grp_share_near_median"].to_list() == pytest.approx([1, 1, 0.5, 0.5])
    assert sum(t.shape[0] for t in text) == dense.height


def test_digits_are_replaced_by_one_symbol() -> None:
    assert ft.normalise("NEFT CR-VSTK0A1B2C3 JAN25") == "neft cr-vstk#a#b#c# jan##"
    assert ft.normalise(None) == ""


def test_the_test_split_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(OPEN_TEST_ENV, raising=False)
    assert ft.main([str(tmp_path), "--sets", "test", "--out", str(tmp_path / "f")]) == 2
    with pytest.raises(ValueError, match="closed"):
        list(ft.load_set(tmp_path, "test"))
