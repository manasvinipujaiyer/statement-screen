"""The last registered step (tamper/log_run.py): what it logs, and what it refuses."""

import json
from pathlib import Path

from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import log_run as lr
from statementscreen.tamper import preregister as pre
from statementscreen.tracking import TrackingRefused

REPORT = {
    "rule_version": "v2",
    "detection": {
        "inplace_edit": {
            "n": 100,
            "with_history": {
                "tampered": {
                    "count": 99,
                    "total": 100,
                    "rate": 0.99,
                    "text": "99/100",
                    "ci95": [0.97, 1.0],
                },
                "labels": {"no evidence of tampering": 1, "suspicious": 0, "tampered": 99},
            },
            "first_check": {"tampered": {"arithmetic:SUMMARY_TOTALS": 99}},
        }
    },
    "pooled_detection": None,
    "overlay_holdout": {"n": 100, "flag": True},
    "false_alarms": {"resaved_genuine": None, "genuine": {"test": {"n": 300, "rate": 0.0}}},
    "misses": [{"item": "copy_0001"}],
}


def test_every_number_in_the_scored_sections_becomes_one_metric():
    m = lr.report_metrics(REPORT)
    assert m["detection.inplace_edit.n"] == 100.0
    assert m["detection.inplace_edit.with_history.tampered.count"] == 99.0
    assert m["detection.inplace_edit.with_history.tampered.rate"] == 0.99
    assert m["detection.inplace_edit.with_history.tampered.ci95_lo"] == 0.97
    assert m["detection.inplace_edit.with_history.tampered.ci95_hi"] == 1.0
    assert m["detection.inplace_edit.with_history.labels.no_evidence_of_tampering"] == 1.0
    assert m["detection.inplace_edit.first_check.tampered.arithmetic_SUMMARY_TOTALS"] == 99.0
    assert m["overlay_holdout.n"] == 100.0 and m["false_alarms.genuine.test.n"] == 300.0


def test_text_booleans_none_and_sections_outside_the_list_are_not_metrics():
    m = lr.report_metrics(REPORT)
    assert not any(k.endswith(".text") or k.endswith(".flag") for k in m)
    assert not any(k.startswith(("misses", "pooled_detection")) for k in m)
    assert not any("resaved_genuine" in k for k in m)
    assert all(isinstance(v, float) for v in m.values())


def _files(tmp_path: Path, report: dict | None = None) -> dict:
    report = REPORT if report is None else report
    paths = {
        "report": tmp_path / "report.json",
        "csv": tmp_path / "copies.csv",
        "prereg": tmp_path / "prereg.json",
    }
    paths["report"].write_text(json.dumps(report), encoding="utf-8")
    paths["csv"].write_text("copy,type\ncopy_0001,inplace_edit\n", encoding="utf-8")
    data = pre.build(head=pre.git("rev-parse", "HEAD"), dirty="")
    paths["prereg"].write_text(json.dumps(data), encoding="utf-8")
    return paths


def _argv(p: dict, *extra: str) -> list[str]:
    return [
        *("--report", str(p["report"])),
        *("--csv", str(p["csv"])),
        *("--preregistration", str(p["prereg"])),
        *("--tracking-dir", str(p["report"].parent / "mlruns")),
        *extra,
    ]


def test_the_run_records_the_commit_the_fingerprint_and_the_files(tmp_path, monkeypatch):
    p = _files(tmp_path)
    seen = {}

    def fake_log(params, metrics, artifacts, tracking_dir):
        seen.update(params=params, metrics=metrics, artifacts=artifacts)
        return "run123", []

    monkeypatch.setattr(lr, "log", fake_log)
    assert lr.main(_argv(p)) == 0
    params = seen["params"]
    assert params["frozen_commit"] == json.loads(p["prereg"].read_text())["frozen_commit"]
    assert params["rule_version"] == "v2"
    assert {f"fingerprint.{k}" for k in fp.PARTS} <= set(params)
    assert params["report_sha256"] == lr.sha256(p["report"])
    assert params["copies_csv_sha256"] == lr.sha256(p["csv"])
    assert seen["metrics"] == lr.report_metrics(REPORT)
    assert seen["artifacts"] == [p["report"], p["csv"], p["prereg"]]


def test_it_reads_the_report_and_changes_nothing(tmp_path, monkeypatch):
    p = _files(tmp_path)
    before = {k: v.read_bytes() for k, v in p.items()}
    monkeypatch.setattr(lr, "log", lambda *a, **k: ("run123", []))
    assert lr.main(_argv(p)) == 0
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_changed_detector_or_rule_version_refuses_to_log(tmp_path, monkeypatch, capsys):
    def must_not_log(*args, **kwargs):
        raise AssertionError("logged despite a changed detector or rule version")

    monkeypatch.setattr(lr, "log", must_not_log)
    p = _files(tmp_path)
    data = json.loads(p["prereg"].read_text())
    data["detector_fingerprint"]["model"] = "a different model"
    p["prereg"].write_text(json.dumps(data), encoding="utf-8")
    assert lr.main(_argv(p)) == 2
    assert "model differs" in capsys.readouterr().err

    (tmp_path / "again").mkdir()
    p2 = _files(tmp_path / "again", {**REPORT, "rule_version": "v1"})
    assert lr.main(_argv(p2)) == 2
    assert "rule version" in capsys.readouterr().err


def test_a_missing_report_or_preregistration_is_refused(tmp_path, capsys):
    p = _files(tmp_path)
    p["report"].unlink()
    assert lr.main(_argv(p)) == 2 and "score first" in capsys.readouterr().err
    p = _files(tmp_path)
    p["prereg"].unlink()
    assert lr.main(_argv(p)) == 2 and "no pre-registration" in capsys.readouterr().err


def test_a_report_with_no_metrics_is_refused(tmp_path, capsys):
    p = _files(tmp_path, {"rule_version": "v2"})
    assert lr.main(_argv(p)) == 2
    assert "no metrics" in capsys.readouterr().err


def test_a_store_warning_exits_3_and_the_report_is_untouched(tmp_path, monkeypatch, capsys):
    p = _files(tmp_path)

    def refuse(*args, **kwargs):
        raise TrackingRefused("MLflow warned about the local store")

    monkeypatch.setattr(lr, "log", refuse)
    before = p["report"].read_bytes()
    assert lr.main(_argv(p)) == 3
    assert "MLflow:" in capsys.readouterr().err and p["report"].read_bytes() == before


def test_metrics_are_logged_in_chunks(tmp_path, monkeypatch):
    calls = []

    class FakeRun:
        info = type("I", (), {"run_id": "r1"})()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeMlflow:
        def start_run(self, run_name):
            return FakeRun()

        def log_params(self, params):
            pass

        def log_metrics(self, metrics):
            calls.append(len(metrics))

        def log_artifact(self, path):
            pass

        def set_tag(self, key, value):
            pass

    monkeypatch.setattr(lr, "_open_store", lambda tracking_dir, experiment: (FakeMlflow(), []))
    metrics = {f"m{n}": float(n) for n in range(1203)}
    run_id, _ = lr.log({"a": "b"}, metrics, [], tmp_path)
    assert run_id == "r1" and calls == [500, 500, 203]
