"""The Week 7 blind scoring: the numbers, and every refusal that must leave the key unread."""

import json
from pathlib import Path

from statementscreen.tamper import blind_prereg as bpr
from statementscreen.tamper import blind_score as bs
from statementscreen.tamper import blind_verdicts as bv
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.blind_common import KEY_NAME, pdf_hashes, pdf_set_sha256, sha256_bytes
from statementscreen.tamper.detect.verdict import NONE, RULE_VERSION, SUSPICIOUS, TAMPERED
from statementscreen.tamper.scoring import COULD_NOT_CHECK

DIALECT = {"default": "blind_default_v1", "karan_v1": "blind_karan_v1"}
# (truth, verdict, dialect, layout, finding codes, checks not run)
SPEC = [
    ("genuine", NONE, "default", "columns", "", ""),
    ("genuine", SUSPICIOUS, "karan_v1", "drcr", "ANOMALY_SCORE", ""),
    ("genuine", TAMPERED, "default", "columns", "BALANCE_BREAK", ""),
    ("genuine", NONE, "karan_v1", "columns", "", ""),
    ("page_swap", TAMPERED, "default", "columns", "PAGE_SWAP", ""),
    ("page_swap", TAMPERED, "karan_v1", "drcr", "PAGE_SWAP", ""),
    ("rerender_consistent", NONE, "default", "columns", "", ""),
    ("inplace_edit", COULD_NOT_CHECK, "karan_v1", "drcr", "", "arithmetic"),
]


def _world(tmp_path: Path, monkeypatch, spec=SPEC, key_edit=None, rows_edit=None) -> dict:
    """PDFs, a verdict file, a key and a pre-registration that agree with each other, with the
    git checks patched so that every file counts as pushed. Edits make them disagree."""
    pdfs = tmp_path / "blind" / "pdfs"
    pdfs.mkdir(parents=True)
    for k in range(1, len(spec) + 1):
        (pdfs / f"blind_{k:04d}.pdf").write_bytes(f"pdf {k}".encode())
    hashes = pdf_hashes(pdfs)
    entries, rows = {}, []
    for k, (truth, verdict, dialect, layout, codes, unrun) in enumerate(spec, start=1):
        name = f"blind_{k:04d}"
        entries[name] = {"sub_batch": DIALECT[dialect], "source": f"stmt_{k:04d}", "layout": layout,
                         "truth": truth, "pdf_sha256": hashes[name]}  # fmt: skip
        rows.append({"name": name, "verdict": verdict, "verdict_without_history": verdict,
                     "finding_codes": codes, "checks_not_run": unrun, "reasons": f"why {k}",
                     "pdf_sha256": hashes[name]})  # fmt: skip
    key = {
        "pdfs": entries,
        "pdf_set_sha256": pdf_set_sha256(hashes),
        "sources": {
            DIALECT[d]: {"main_seed": s, "dialect": d}
            for d, s in (("default", 7101), ("karan_v1", 7102))
        },  # fmt: skip
        "redraws": [],
        "build_errors": [],
    }
    if key_edit:
        key_edit(key)
    if rows_edit:
        rows_edit(rows)
    key_path = tmp_path / "blind" / KEY_NAME
    key_text = json.dumps(key, sort_keys=True) + "\n"
    key_path.write_text(key_text, encoding="utf-8")
    verdicts = tmp_path / "reports" / "verdicts.csv"
    bv.write_verdicts(rows, verdicts)
    prereg = {
        "frozen_commit": "c" * 40,
        "rule_version": RULE_VERSION,
        "detector_fingerprint": {"code": "x"},
        "sealed": {
            "key_sha256": sha256_bytes(key_text.encode()),
            "pdf_set_sha256": pdf_set_sha256(hashes),
            "pdfs": len(spec),
            "sources": {
                DIALECT[d]: {"main_seed": s} for d, s in (("default", 7101), ("karan_v1", 7102))
            },  # fmt: skip
        },
        "known_limits": ["a limit"],
    }
    monkeypatch.setattr(bs.bpr, "ROOT", tmp_path)
    monkeypatch.setattr(bs.bpr, "load_and_verify", lambda path: prereg)
    monkeypatch.setattr(bs.pre, "pushed_text", lambda p, root=None: Path(p).read_text("utf-8"))
    monkeypatch.setattr(bs.pre, "has_git_history", lambda path, root=None: False)
    return {"pdfs": pdfs, "key": key_path, "verdicts": verdicts, "root": tmp_path,
            "out": tmp_path / "reports" / "week7_blind_report.json", "prereg": prereg}  # fmt: skip


def _argv(w: dict, **extra) -> list[str]:
    values = {"--key": w["key"], "--verdicts": w["verdicts"], "--pdfs": w["pdfs"],
              "--preregistration": "unused.json", "--out": w["out"], **extra}  # fmt: skip
    return [str(x) for pair in values.items() for x in pair]


def _refused(w: dict, capsys, text: str, **extra) -> None:
    assert bs.main(_argv(w, **extra)) == 2
    assert text in capsys.readouterr().err
    assert not Path(extra.get("--out", w["out"])).exists()


# ---------------------------------------------------------------- the numbers


def test_the_report_counts_the_mix_the_detection_and_the_false_alarms(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    assert bs.main(_argv(w)) == 0
    r = json.loads(w["out"].read_text(encoding="utf-8"))
    assert r["post_hoc"] is False and r["inputs"]["pdfs"] == 8
    assert r["mix"]["genuine"] == 4 and r["mix"]["tampered_total"] == 4 and r["mix"]["total"] == 8
    assert r["mix"]["page_swap"] == 2 and r["mix"]["overlay"] == 0
    swap = r["detection"]["page_swap"]
    assert swap["tampered"]["text"] == "2/2 (100.0%)" and swap["tampered"]["ci95"] == [1.0, 1.0]
    assert r["detection"]["rerender_consistent"]["labels"][NONE] == 1
    assert r["detection"]["inplace_edit"]["could_not_check"]["count"] == 1
    assert r["detection"]["overlay"]["tampered"]["text"] == "0/0"
    assert r["detection"]["inplace_edit"]["checks_not_run"] == {"arithmetic": 1}
    fa = r["false_alarms"]
    assert fa["overall"]["tampered"]["text"] == "1/4 (25.0%)"
    assert fa["overall"]["tampered_or_suspicious"]["count"] == 2
    assert fa["by_dialect"]["default"]["tampered"]["count"] == 1
    assert fa["by_dialect"]["karan_v1"]["tampered_or_suspicious"]["count"] == 1
    assert fa["flagged_by_the_anomaly_check_alone"] == 1 and len(fa["cases"]) == 2
    assert [c["name"] for c in r["genuine_called_tampered"]] == ["blind_0003"]
    assert r["recall_of_tampered"]["text"] == "2/4 (50.0%)"
    assert r["precision_of_tampered"]["text"] == "2/3 (66.7%)"
    assert r["confusion_verdict_by_truth"]["page_swap"][TAMPERED] == 2
    assert r["confusion_verdict_by_truth"]["inplace_edit"][COULD_NOT_CHECK] == 1
    assert {m["outcome"] for m in r["misses"]} == {"not flagged", COULD_NOT_CHECK}
    assert r["could_not_check"]["tampered"]["count"] == 1
    assert r["by_layout"]["genuine"]["drcr"]["n"] == 1
    assert r["known_limits"] == ["a limit"]


def test_the_report_sets_the_blind_counts_next_to_weeks_6_numbers(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    assert bs.main(_argv(w)) == 0
    cmp = json.loads(w["out"].read_text(encoding="utf-8"))["comparison_with_week6"]
    assert cmp["per_type"]["page_swap"]["blind_tampered"] == "2/2 (100.0%)"
    assert cmp["per_type"]["page_swap"]["week6_test_tampered"] == "100/100"
    assert cmp["per_type"]["rerender_consistent"]["week6_test_flagged"] == "14/100"
    assert cmp["genuine"]["week6_test"]["tampered_or_suspicious"] == "7/300"
    assert "not targets" in cmp["note"]


def test_the_week_6_numbers_are_the_ones_the_preregistration_states():
    text = " ".join(bpr.EXPECTATIONS)
    for t in ("97/100", "0/100", "14/100", "0/300", "7/300", "5/300"):
        assert t in text
    texts = {v for v in bs.WEEK6_TEST["tampered"].values()}
    assert texts == {"100/100", "97/100", "0/100"}


def test_the_precision_interval_is_exact_when_every_call_is_right_and_bounded_otherwise(
    tmp_path, monkeypatch
):
    right = [("genuine", NONE, "default", "columns", "", "")] + [
        ("page_swap", TAMPERED, "default", "columns", "", "")
    ] * 3
    w = _world(tmp_path / "a", monkeypatch, right)
    assert bs.main(_argv(w)) == 0
    p = json.loads(w["out"].read_text(encoding="utf-8"))["precision_of_tampered"]
    assert p["text"] == "3/3 (100.0%)" and p["ci95"] == [1.0, 1.0]
    w = _world(tmp_path / "b", monkeypatch)  # SPEC: 2 of the 3 'tampered' calls are right
    assert bs.main(_argv(w)) == 0
    p = json.loads(w["out"].read_text(encoding="utf-8"))["precision_of_tampered"]
    lo, hi = p["ci95"]
    assert 0.0 <= lo <= hi <= 1.0 and p["resamples_with_no_tampered_call"] >= 0


def test_precision_has_no_interval_when_nothing_was_called_tampered(tmp_path, monkeypatch):
    spec = [("genuine", NONE, "default", "columns", "", ""),
            ("page_swap", NONE, "default", "columns", "", "")]  # fmt: skip
    w = _world(tmp_path, monkeypatch, spec)
    assert bs.main(_argv(w)) == 0
    p = json.loads(w["out"].read_text(encoding="utf-8"))["precision_of_tampered"]
    assert p["text"] == "0/0" and p["ci95"] is None and p["rate"] is None


# ---------------------------------------------------------------- refusals before the key


def test_a_key_whose_hash_differs_is_refused_before_it_is_parsed(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    w["key"].write_text("not json at all", encoding="utf-8")
    _refused(w, capsys, "differs from the pre-registered one")


def test_a_key_edited_after_sealing_is_refused(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    text = w["key"].read_text(encoding="utf-8").replace("page_swap", "genuine", 1)
    w["key"].write_text(text, encoding="utf-8")
    _refused(w, capsys, "differs from the pre-registered one")


def test_a_verdict_file_that_differs_from_the_pushed_one_is_refused(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    w["key"].unlink()  # if the key were read first, the error would say so
    real = bs.pre.pushed_text
    monkeypatch.setattr(
        bs.pre, "pushed_text", lambda p, root=None: "other\n" if p == w["verdicts"] else real(p)
    )
    _refused(w, capsys, "differs from the pushed version")


def test_a_verdict_file_that_was_never_pushed_is_refused(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    w["key"].unlink()

    def not_pushed(p, root=None):
        raise bs.pre.PreregError("not on the pushed branch: commit and push it")

    monkeypatch.setattr(bs.pre, "pushed_text", not_pushed)
    _refused(w, capsys, "not on the pushed branch")


def test_a_hash_file_that_does_not_match_the_verdicts_is_refused(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    w["key"].unlink()
    w["verdicts"].with_suffix(".sha256").write_text(f"{'0' * 64}  verdicts.csv\n", "utf-8")
    _refused(w, capsys, "does not match the SHA-256")


def test_pdfs_that_are_not_the_sealed_set_are_refused(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    w["key"].unlink()
    (w["pdfs"] / "blind_0001.pdf").write_bytes(b"changed")  # check_pdf_set is not patched
    _refused(w, capsys, "not the sealed set")


def test_verdict_rows_for_other_pdfs_are_refused(tmp_path, monkeypatch, capsys):
    def wrong_hash(rows):
        rows[0]["pdf_sha256"] = "0" * 64

    w = _world(tmp_path, monkeypatch, rows_edit=wrong_hash)
    w["key"].unlink()
    _refused(w, capsys, "verdict row's PDF hash is not the sealed PDF's")


def test_a_detector_that_differs_from_week_6s_is_refused(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    w["key"].unlink()

    def stale(path):
        raise fp.StaleFindings("these findings were made by a different detector: model differs")

    monkeypatch.setattr(bs.bpr, "load_and_verify", stale)
    _refused(w, capsys, "model differs")


def test_a_key_that_disagrees_with_the_pdfs_is_refused(tmp_path, monkeypatch, capsys):
    def wrong(key):
        key["pdfs"]["blind_0002"]["pdf_sha256"] = "0" * 64

    w = _world(tmp_path, monkeypatch, key_edit=wrong)  # its hash is pre-registered, so it passes
    _refused(w, capsys, "key's PDF hashes differ")


def test_a_key_with_other_source_seeds_or_unknown_truths_is_refused(tmp_path, monkeypatch, capsys):
    def seeds(key):
        key["sources"]["blind_default_v1"]["main_seed"] = 1

    _refused(_world(tmp_path / "a", monkeypatch, key_edit=seeds), capsys, "source seeds differ")

    def truth(key):
        key["pdfs"]["blind_0001"]["truth"] = "invented"

    _refused(_world(tmp_path / "b", monkeypatch, key_edit=truth), capsys, "unknown truth")


# ---------------------------------------------------------------- write once, and post-hoc


def test_a_report_is_written_once_and_a_second_run_needs_post_hoc(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    assert bs.main(_argv(w)) == 0
    assert bs.main(_argv(w)) == 2
    assert "never overwritten" in capsys.readouterr().err
    again = tmp_path / "reports" / "second.json"
    _refused(w, capsys, "a first report already exists", **{"--out": again})
    post = tmp_path / "reports" / "week7_blind_report_post_hoc.json"
    assert bs.main(_argv(w, **{"--out": post})) == 0
    assert json.loads(post.read_text(encoding="utf-8"))["post_hoc"] is True
    assert "(post-hoc)" in capsys.readouterr().out


def test_a_post_hoc_report_needs_a_first_report(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    post = tmp_path / "reports" / "week7_blind_report_post_hoc.json"
    _refused(w, capsys, "needs a first report", **{"--out": post})


def test_a_path_git_has_recorded_cannot_be_written_again(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    monkeypatch.setattr(bs.pre, "has_git_history", lambda path, root=None: True)
    _refused(w, capsys, "never overwritten")
