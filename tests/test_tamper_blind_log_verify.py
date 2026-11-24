"""Logging the blind report to MLflow, and rebuilding the 200 PDFs from the published key."""

import json
from pathlib import Path

import pytest

from statementscreen.tamper import blind_log as bl
from statementscreen.tamper import blind_prereg as bpr
from statementscreen.tamper import blind_verdicts as bv
from statementscreen.tamper import blind_verify as bvf
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.blind_common import pdf_set_sha256, sha256_bytes
from statementscreen.tamper.copy import CopyError
from statementscreen.tracking import TrackingRefused

HEAD = "c" * 40
KEY_SHA = "a" * 64
SET_SHA = "b" * 64


# ---------------------------------------------------------------- blind_log


def _report(**over) -> dict:
    ci = {"count": 2, "total": 2, "rate": 1.0, "text": "2/2 (100.0%)", "ci95": [0.5, 1.0]}
    report = {
        "post_hoc": False,
        "rule_version": "v2",
        "preregistration": {"frozen_commit": HEAD},
        "inputs": {"key_sha256": KEY_SHA, "pdf_set_sha256": SET_SHA, "verdicts_sha256": "?"},
        "mix": {"genuine": 4, "page_swap": 2, "total": 6},
        "detection": {"page_swap": {"n": 2, "tampered": ci}},
        "false_alarms": {"overall": {"tampered": {**ci, "count": 0}}},
        "precision_of_tampered": ci,
        "recall_of_tampered": ci,
        "confusion_verdict_by_truth": {"page_swap": {"tampered": 2, "no evidence of tampering": 0}},
        "known_limits": ["not a number"],
    }
    return {**report, **over}


@pytest.fixture
def logged(tmp_path, monkeypatch):
    """A report, a verdict file and a pre-registration that agree, with MLflow replaced by a
    recorder. Returns the paths, the pre-registration and the recorded calls."""
    verdicts = tmp_path / "verdicts.csv"
    verdicts.write_text("name\nblind_0001\n", encoding="utf-8")
    report = _report()
    report["inputs"]["verdicts_sha256"] = sha256_bytes(verdicts.read_bytes())
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    prereg = {
        "frozen_commit": HEAD,
        "rule_version": "v2",
        "detector_fingerprint": {"code": "c", "config": "d", "model": "m", "uv_lock": "u"},
        "sealed": {"key_sha256": KEY_SHA, "pdf_set_sha256": SET_SHA},
    }
    calls: list[dict] = []

    def record(params, metrics, artifacts, tracking_dir, run_name):
        calls.append({"params": params, "metrics": metrics, "artifacts": list(artifacts),
                      "run_name": run_name})  # fmt: skip
        return "run123", []

    monkeypatch.setattr(bl, "log", record)
    monkeypatch.setattr(bl.bpr, "load_and_verify", lambda p: prereg)
    return {"report": path, "verdicts": verdicts, "prereg": prereg, "calls": calls,
            "tmp": tmp_path}  # fmt: skip


def _log_argv(w: dict) -> list[str]:
    tracking = str(w["tmp"] / "mlruns")
    return [
        *("--report", str(w["report"])),
        *("--verdicts", str(w["verdicts"])),
        *("--preregistration", "unused.json"),
        *("--tracking-dir", tracking),
    ]


def test_every_number_in_the_report_becomes_one_metric_and_nothing_else_does():
    metrics = bl.report_metrics(_report())
    assert metrics["mix.page_swap"] == 2.0
    assert metrics["detection.page_swap.tampered.count"] == 2.0
    assert metrics["detection.page_swap.tampered.ci95_lo"] == 0.5
    assert metrics["detection.page_swap.tampered.ci95_hi"] == 1.0
    assert metrics["confusion_verdict_by_truth.page_swap.no_evidence_of_tampering"] == 0.0
    assert all(isinstance(v, float) for v in metrics.values())
    assert not any("known_limits" in name for name in metrics)


def test_the_run_is_logged_with_its_parameters_and_artifacts(logged, capsys):
    assert bl.main(_log_argv(logged)) == 0
    (call,) = logged["calls"]
    assert call["run_name"] == "blind_once"
    assert call["params"]["frozen_commit"] == HEAD and call["params"]["key_sha256"] == KEY_SHA
    assert call["params"]["fingerprint.model"] == "m" and call["params"]["post_hoc"] == "False"
    assert [Path(a).name for a in call["artifacts"]] == [
        "report.json",
        "verdicts.csv",
        "unused.json",
    ]
    assert "MLflow run run123" in capsys.readouterr().out


def test_a_post_hoc_report_is_logged_under_its_own_name(logged):
    report = json.loads(logged["report"].read_text(encoding="utf-8"))
    logged["report"].write_text(json.dumps({**report, "post_hoc": True}), encoding="utf-8")
    assert bl.main(_log_argv(logged)) == 0
    assert logged["calls"][0]["run_name"] == "blind_post_hoc"


def _rewrite(w: dict, **changes) -> None:
    report = json.loads(w["report"].read_text(encoding="utf-8"))
    w["report"].write_text(json.dumps({**report, **changes}), encoding="utf-8")


def test_a_report_from_another_key_pdf_set_commit_or_verdict_file_is_not_logged(logged, capsys):
    inputs = _report()["inputs"]
    cases = [
        ({"inputs": {**inputs, "key_sha256": "d" * 64}}, "key_sha256"),
        ({"inputs": {**inputs, "pdf_set_sha256": "d" * 64}}, "pdf_set_sha256"),
        ({"preregistration": {"frozen_commit": "e" * 40}}, "different pre-registration"),
        ({"inputs": {**inputs, "verdicts_sha256": "d" * 64}}, "not the verdict file"),
    ]
    for changes, text in cases:
        _rewrite(logged, **changes)
        assert bl.main(_log_argv(logged)) == 2
        assert text in capsys.readouterr().err
        _rewrite(logged, inputs=inputs | {"verdicts_sha256": sha256_bytes(
            logged["verdicts"].read_bytes())}, preregistration={"frozen_commit": HEAD})  # fmt: skip
    assert logged["calls"] == []


def test_nothing_is_logged_for_a_missing_report_a_refused_preregistration_or_no_metrics(
    logged, monkeypatch, capsys
):
    _rewrite(logged, mix=None, detection=None, false_alarms=None, precision_of_tampered=None,
             recall_of_tampered=None, confusion_verdict_by_truth=None)  # fmt: skip
    assert bl.main(_log_argv(logged)) == 2 and "no metrics" in capsys.readouterr().err
    logged["report"].unlink()
    assert bl.main(_log_argv(logged)) == 2 and "score first" in capsys.readouterr().err

    def stale(path):
        raise fp.StaleFindings("model differs")

    monkeypatch.setattr(bl.bpr, "load_and_verify", stale)
    _rewrite_new = logged["tmp"] / "other.json"
    _rewrite_new.write_text(json.dumps(_report()), encoding="utf-8")
    argv = _log_argv(logged)
    argv[1] = str(_rewrite_new)
    assert bl.main(argv) == 2 and "model differs" in capsys.readouterr().err
    assert logged["calls"] == []


def test_the_real_log_writes_one_run_to_a_temporary_store(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("x", encoding="utf-8")
    store = tmp_path / "mlruns"
    run_id, seen = bl.log({"frozen_commit": HEAD}, {"mix.total": 6.0}, [artifact], store)
    assert len(run_id) == 32 and isinstance(seen, list)
    assert store.exists()


def test_a_store_warning_exits_3_without_changing_the_report(logged, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise TrackingRefused("the store warns")

    monkeypatch.setattr(bl, "log", refuse)
    before = logged["report"].read_bytes()
    assert bl.main(_log_argv(logged)) == 3
    assert "the store warns" in capsys.readouterr().err
    assert logged["report"].read_bytes() == before


# ---------------------------------------------------------------- blind_verify


def _fake_copy(pdf, tamper_type, seed, donor=None, allow_held_out=False):
    assert allow_held_out
    tail = f"|{tamper_type}|{seed}|{donor.name if donor else ''}".encode()
    return Path(pdf).read_bytes() + tail, {}


# (blind name, source folder, source id, truth, copy seed, donor)
SPEC = [
    ("blind_0001", "default_v1", "stmt_0001", "genuine", None, None),
    ("blind_0002", "default_v1", "stmt_0002", "page_swap", 11, "stmt_0001"),
    ("blind_0003", "karan_v1", "stmt_0001", "inplace_edit", 12, None),
    ("blind_0004", "karan_v1", "stmt_0002", "genuine", None, None),
]
SUBS = {"default_v1": "blind_default_v1", "karan_v1": "blind_karan_v1"}


@pytest.fixture
def rebuilt(tmp_path, monkeypatch):
    """Stand-in sources, a published key, committed verdicts and a pre-registration that agree,
    with tamper_copy replaced by a deterministic fake."""
    monkeypatch.setattr(bvf, "tamper_copy", _fake_copy)
    src = tmp_path / "blind_src"
    for folder in SUBS:
        (src / folder / "pdfs").mkdir(parents=True)
        (src / folder / "manifest.csv").write_text(f"statement_id\n{folder}\n", encoding="utf-8")
        for sid in ("stmt_0001", "stmt_0002"):
            (src / folder / "pdfs" / f"{sid}.pdf").write_bytes(f"%PDF {folder} {sid}".encode())
    entries = {}
    for name, folder, sid, truth, seed, donor in SPEC:
        source = src / folder / "pdfs" / f"{sid}.pdf"
        if truth == "genuine":
            data = source.read_bytes()
        else:
            data, _ = _fake_copy(source, truth, seed, src / folder / "pdfs" / f"{donor}.pdf"
                                 if donor else None, allow_held_out=True)  # fmt: skip
        entries[name] = {"sub_batch": SUBS[folder], "source": sid, "truth": truth,
                         "copy_seed": seed, "donor": donor,
                         "source_sha256": sha256_bytes(source.read_bytes()),
                         "pdf_sha256": sha256_bytes(data)}  # fmt: skip
    key = {
        "pdfs": entries,
        "pdf_set_sha256": pdf_set_sha256({n: e["pdf_sha256"] for n, e in entries.items()}),
        "sources": {
            sub: {
                "folder": folder,
                "manifest_sha256": bvf.sha256_file(src / folder / "manifest.csv"),
            }
            for folder, sub in SUBS.items()
        },  # fmt: skip
    }
    key_text = json.dumps(key, sort_keys=True) + "\n"
    paths = {"key": tmp_path / "key.json", "verdicts": tmp_path / "verdicts.csv",
             "prereg": tmp_path / "prereg.json", "src": src, "entries": entries}  # fmt: skip
    paths["key"].write_text(key_text, encoding="utf-8")
    paths["prereg"].write_text(
        json.dumps({"sealed": {"key_sha256": sha256_bytes(key_text.encode())}}), encoding="utf-8"
    )
    _write_verdicts(paths, {n: e["pdf_sha256"] for n, e in entries.items()})
    return paths


def _write_verdicts(paths: dict, hashes: dict) -> None:
    paths["verdicts"].unlink(missing_ok=True)
    paths["verdicts"].with_suffix(".sha256").unlink(missing_ok=True)
    rows = [{**dict.fromkeys(bv.COLUMNS, ""), "name": n, "pdf_sha256": h}
            for n, h in sorted(hashes.items())]  # fmt: skip
    bv.write_verdicts(rows, paths["verdicts"])


def _verify_argv(p: dict) -> list[str]:
    return ["--key", str(p["key"]), "--verdicts", str(p["verdicts"]),
            "--blind-src", str(p["src"]), "--preregistration", str(p["prereg"])]  # fmt: skip


def test_every_pdf_is_rebuilt_byte_for_byte(rebuilt, capsys):
    assert bvf.main(_verify_argv(rebuilt)) == 0
    out = capsys.readouterr().out
    assert "Rebuilt 4 of 4 PDFs; 4 identical" in out and "byte for byte" in out


def test_a_source_that_is_not_the_one_the_key_hashed_is_listed(rebuilt, capsys):
    src = rebuilt["src"] / "default_v1" / "pdfs" / "stmt_0002.pdf"
    src.write_bytes(b"something else")
    assert bvf.main(_verify_argv(rebuilt)) == 1
    out = capsys.readouterr().out
    assert "blind_0002 (page_swap): the source PDF is not the one the key hashed" in out


def test_a_tamper_that_is_not_reproducible_is_listed_by_truth_and_never_hidden(
    rebuilt, monkeypatch, capsys
):
    def drifting(pdf, tamper_type, seed, donor=None, allow_held_out=False):
        data, key = _fake_copy(pdf, tamper_type, seed, donor, allow_held_out)
        return data + b"!", key

    monkeypatch.setattr(bvf, "tamper_copy", drifting)
    assert bvf.main(_verify_argv(rebuilt)) == 1
    out = capsys.readouterr().out
    assert "Rebuilt 4 of 4 PDFs; 2 identical" in out
    assert "'inplace_edit': 2" in out and "'page_swap': 2" in out  # key and verdict file each
    assert "the rebuilt set's hash differs from the key's PDF-set hash" in out
    assert "blind_0001" not in out and "blind_0004" not in out  # the genuine ones are exact


def test_a_pdf_that_cannot_be_rebuilt_is_listed(rebuilt, monkeypatch, capsys):
    def failing(pdf, tamper_type, seed, donor=None, allow_held_out=False):
        if tamper_type == "page_swap":
            raise CopyError("no donor page")
        return _fake_copy(pdf, tamper_type, seed, donor, allow_held_out)

    monkeypatch.setattr(bvf, "tamper_copy", failing)
    assert bvf.main(_verify_argv(rebuilt)) == 1
    assert "blind_0002 (page_swap): could not rebuild: CopyError: no donor page" in (
        capsys.readouterr().out
    )


def test_a_verdict_file_made_on_other_pdfs_is_listed(rebuilt, capsys):
    hashes = {n: e["pdf_sha256"] for n, e in rebuilt["entries"].items()}
    hashes["blind_0003"] = "0" * 64
    _write_verdicts(rebuilt, hashes)
    assert bvf.main(_verify_argv(rebuilt)) == 1
    out = capsys.readouterr().out
    assert "blind_0003 (inplace_edit): the rebuilt PDF differs from the committed verdict" in out


def test_a_manifest_that_is_not_the_one_the_key_hashed_is_listed(rebuilt, capsys):
    (rebuilt["src"] / "karan_v1" / "manifest.csv").write_text("changed\n", encoding="utf-8")
    assert bvf.main(_verify_argv(rebuilt)) == 1
    assert "is not the manifest the key hashed" in capsys.readouterr().out


def test_a_key_that_is_not_the_preregistered_one_is_refused(rebuilt, capsys):
    rebuilt["prereg"].write_text(json.dumps({"sealed": {"key_sha256": "0" * 64}}), "utf-8")
    assert bvf.main(_verify_argv(rebuilt)) == 2
    assert "not the key that was pre-registered" in capsys.readouterr().err


def test_the_verify_command_is_named_as_the_preregistration_says():
    assert any("blind_verify" in c and bpr.PUBLISHED_KEY in c for c in bpr.COMMANDS)
