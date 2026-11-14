"""Log a pattern-scoring run to MLflow, LOCALLY only.

Everything lives in one local folder (default mlruns/, which is git-ignored); nothing is
sent anywhere:

    mlruns/mlflow.db     the tracking store: a local SQLite file (no server)
    mlruns/artifacts/    the logged files (reports)

Why SQLite and not the plain-folder store: MLflow 3.18 refuses the filesystem backend
("maintenance mode ... migrate to a database backend"); decided 2026-11-08 to use a local
SQLite file rather than opt back in to the frozen store.

Before anything is logged, every warning and log message MLflow emits while opening the
store is captured. If any mentions deprecation, maintenance mode or removal (or opening
the store raises), nothing is logged: TrackingRefused is raised with the messages, so a
person decides what to do instead of the code quietly switching backends.

Each run records: the split, batch and statement count; every PatternConfig setting (the
grouping threshold and all rule thresholds); the headline metrics; the report file, plus
the sweep reports when they exist.
"""

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

DEFAULT_DIR = Path("mlruns")
EXPERIMENT = "week3_patterns"
WORRYING = ("deprecat", "maintenance mode", "no longer supported", "will be removed")
STORE_WORDS = ("store", "backend", "filesystem", "tracking", "mlruns")


def about_the_store(message: str) -> bool:
    """A deprecation-type message about the tracking store itself (e.g. MLflow's 'The
    filesystem tracking backend ... is in maintenance mode'). Other deprecation notices,
    such as SQLAlchemy's about its own query options, are reported, not refused."""
    m = message.lower()
    return any(w in m for w in WORRYING) and any(w in m for w in STORE_WORDS)


def tracking_uri(tracking_dir: Path) -> str:
    """sqlite:////absolute/path/mlruns/mlflow.db"""
    return f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}"


class TrackingRefused(RuntimeError):
    """MLflow warned about the local store; nothing was logged."""


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def headline_metrics(report: dict) -> dict[str, float]:
    """The numbers worth comparing across runs, flattened for MLflow."""
    o = report["overall"]
    m: dict[str, float | None] = {}
    for k in ("precision", "recall", "f1", "employer_recall"):
        m[f"grouping.{k}"] = o["grouping"][k]
    for flag, s in o["transactions"].items():
        for k in ("precision", "recall", "f1", "tp", "fp", "fn"):
            m[f"{flag}.{k}"] = s[k]
    for k in ("precision", "recall", "f1"):
        m[f"has_salary.{k}"] = o["has_salary"][k]
    m["employer.exact_match_rate"] = o["employer"]["exact_match_rate"]
    m["employer.fuzzy_match_rate"] = o["employer"]["fuzzy_match_rate"]
    for kind, fs in report["false_salary_on_non_salaried"].items():
        m[f"false_salary.{kind}.statements"] = fs["flagged_has_salary"]
    run = report.get("patterns_run", {})
    if run:
        m["seconds_per_statement.mean"] = run["seconds_per_statement"]["mean"]
        m["peak_memory_mb"] = run["peak_memory_mb"]
    scoring = report.get("scoring_run", {})
    if scoring:
        m["scoring.seconds_per_statement"] = scoring["seconds_per_statement"]
        m["scoring.peak_memory_mb"] = scoring["peak_memory_mb"]
    return {k: float(v) for k, v in m.items() if v is not None}


def params(report: dict) -> dict[str, str]:
    p = {"split": report["split"], "batch": report["batch"],
         "statements": str(report["statements"])}  # fmt: skip
    p |= {f"config.{k}": str(v) for k, v in sorted(report["config"].items())}
    return p


def log_run(report: dict, report_path: Path, tracking_dir: Path = DEFAULT_DIR,
            extra_artifacts: Iterable[Path] = (),
            experiment: str = EXPERIMENT) -> tuple[str, list[str]]:  # fmt: skip
    """Log one run; returns (run id, other warnings seen). Raises TrackingRefused if MLflow
    warns about the store first."""
    collect = _Collect()
    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logger.addHandler(collect)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import mlflow
            from mlflow.exceptions import MlflowException

            tracking_dir.mkdir(parents=True, exist_ok=True)
            try:
                mlflow.set_tracking_uri(tracking_uri(tracking_dir))
                if mlflow.get_experiment_by_name(experiment) is None:
                    artifacts = (tracking_dir / "artifacts").resolve().as_uri()
                    mlflow.create_experiment(experiment, artifact_location=artifacts)
                mlflow.set_experiment(experiment)
            except MlflowException as e:
                raise TrackingRefused(f"MLflow refused the local store: {e}") from e
        messages = list(dict.fromkeys([str(w.message) for w in caught] + collect.messages))
        worrying = [m for m in messages if about_the_store(m)]
        if worrying:
            raise TrackingRefused("MLflow warned about the local store, so nothing was "
                                  "logged:\n  " + "\n  ".join(worrying))  # fmt: skip
        with mlflow.start_run(run_name=f"{report['split']}") as run:
            mlflow.log_params(params(report))
            mlflow.log_metrics(headline_metrics(report))
            mlflow.log_artifact(str(report_path))
            for extra in extra_artifacts:
                if extra.exists():
                    mlflow.log_artifact(str(extra), artifact_path="sweeps")
            if messages:  # not about the store, but never hidden
                mlflow.set_tag("setup_warnings", " | ".join(messages)[:5000])
            return run.info.run_id, messages
    finally:
        mlflow_logger.removeHandler(collect)
