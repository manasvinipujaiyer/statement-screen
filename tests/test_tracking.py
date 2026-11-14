"""Local MLflow tracking: always a temp folder in tests, never mlruns/."""

import json
import warnings

import pytest

from statementscreen import tracking
from statementscreen.patterns.config import DEFAULT

FLAG = {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "tp": 1, "fp": 0, "fn": 1}
REPORT = {
    "split": "dev", "batch": "tiny", "statements": 2, "config": DEFAULT.as_dict(),
    "overall": {
        "grouping": {"precision": 0.99, "recall": 0.98, "f1": 0.985, "employer_recall": None},
        "transactions": {f: FLAG for f in ("is_salary", "is_emi", "is_bounce", "is_rent")},
        "has_salary": FLAG,
        "employer": {"exact_match_rate": 0.9, "fuzzy_match_rate": 0.95},
    },
    "false_salary_on_non_salaried": {"gig": {"flagged_has_salary": 0},
                                     "self_employed": {"flagged_has_salary": 0}},
    "patterns_run": {"seconds_per_statement": {"mean": 0.01}, "peak_memory_mb": 300.0},
}  # fmt: skip


def test_params_hold_every_threshold() -> None:
    p = tracking.params(REPORT)
    assert p["split"] == "dev"
    assert p["config.group_threshold"] == str(DEFAULT.group_threshold)
    assert p["config.salary_min_coverage"] == str(DEFAULT.salary_min_coverage)
    assert len([k for k in p if k.startswith("config.")]) == len(DEFAULT.as_dict())


def test_metrics_skip_missing_values() -> None:
    m = tracking.headline_metrics(REPORT)
    assert m["is_rent.recall"] == 0.5 and m["false_salary.gig.statements"] == 0.0
    assert "grouping.employer_recall" not in m  # None is left out, not logged as 0


def test_a_run_is_logged_to_the_given_folder(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(REPORT))
    run_id, _ = tracking.log_run(REPORT, report_path, tmp_path / "mlruns_test",
                              experiment="test_experiment")  # fmt: skip
    assert run_id and any((tmp_path / "mlruns_test").rglob("report.json"))


def test_a_deprecation_warning_stops_tracking_before_logging(tmp_path, monkeypatch) -> None:
    import mlflow

    real = mlflow.set_experiment

    def warn_then_set(*args, **kwargs):
        warnings.warn("The file store is deprecated; use a database backend",
                      FutureWarning, stacklevel=2)  # fmt: skip
        return real(*args, **kwargs)

    monkeypatch.setattr(mlflow, "set_experiment", warn_then_set)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(REPORT))
    with pytest.raises(tracking.TrackingRefused, match="deprecated"):
        tracking.log_run(REPORT, report_path, tmp_path / "refused", experiment="refused")
    assert not any((tmp_path / "refused").rglob("report.json"))  # nothing logged


def test_store_is_a_local_sqlite_file_inside_the_folder(tmp_path) -> None:
    uri = tracking.tracking_uri(tmp_path / "mlruns")
    assert uri.startswith("sqlite:///") and uri.endswith("/mlruns/mlflow.db")
    assert str(tmp_path.resolve()) in uri


def test_only_store_deprecations_refuse() -> None:
    assert tracking.about_the_store(
        "The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will "
        "not receive further updates."
    )  # MLflow 3.18's message
    assert not tracking.about_the_store(
        "The ``noload`` loader strategy is deprecated and will be removed in a future release."
    )  # SQLAlchemy's, about its own query option: reported, not refused
