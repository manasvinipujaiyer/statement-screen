"""Planning tests: no PDFs are created here, so they're fast enough for n=2000."""

from collections import Counter

import pytest

from statementscreen.batch import (
    MAX_MONTHS,
    MIN_MONTHS,
    BatchConfig,
    parse_mix,
    plan_batch,
    plan_statement,
)
from statementscreen.generate import OutputFolderError, main, prepare_output_folder
from statementscreen.synth.hard_cases import HARD_CASES, SALARIED_ONLY
from statementscreen.synth.render import LAYOUT_NAMES

BIG = plan_batch(BatchConfig(seed=42), 2000)


def test_default_mix_within_3_points_at_n_2000() -> None:
    counts = Counter(p.customer_type for p in BIG)
    for customer_type, target in {"salaried": 0.6, "gig": 0.2, "self_employed": 0.2}.items():
        assert abs(counts[customer_type] / 2000 - target) <= 0.03, (customer_type, counts)


def test_layouts_split_evenly_and_both_appear() -> None:
    counts = Counter(p.layout for p in BIG)
    assert set(counts) == set(LAYOUT_NAMES)
    assert all(abs(c / 2000 - 1 / len(LAYOUT_NAMES)) <= 0.03 for c in counts.values())


def test_every_statement_is_6_to_12_months_and_ends_by_dec_2025() -> None:
    assert {p.months for p in BIG} == set(range(MIN_MONTHS, MAX_MONTHS + 1))
    for p in BIG:
        y, m = divmod(p.start.month - 1 + p.months - 1, 12)
        assert (p.start.year + y, m + 1) <= (2025, 12)
        assert p.start.day == 1 and p.start.year >= 2024


def test_hard_case_rate_and_salary_tags_only_on_salaried() -> None:
    salaried = [p for p in BIG if p.customer_type == "salaried"]
    for p in BIG:
        if p.customer_type != "salaried":
            assert not SALARIED_ONLY & set(p.hard_cases), p
    for case in HARD_CASES:
        pool = salaried if case in SALARIED_ONLY else BIG
        rate = sum(case in p.hard_cases for p in pool) / len(pool)
        assert abs(rate - 0.2) <= 0.03, (case, rate)


def test_plans_do_not_depend_on_n() -> None:
    config = BatchConfig(seed=7)
    assert plan_batch(config, 5) == plan_batch(config, 10)[:5]
    assert plan_statement(config, 3) == plan_batch(config, 10)[2]


def test_different_main_seed_gives_different_plans() -> None:
    a, b = plan_batch(BatchConfig(seed=1), 20), plan_batch(BatchConfig(seed=2), 20)
    assert [p.seed for p in a] != [p.seed for p in b]
    assert len({p.seed for p in a}) == 20  # every statement gets its own seed


def test_custom_mix_is_respected() -> None:
    only_gig = plan_batch(BatchConfig(seed=3, mix=parse_mix("gig=1")), 200)
    assert {p.customer_type for p in only_gig} == {"gig"}
    half = plan_batch(BatchConfig(seed=3, mix=parse_mix("salaried=0.5,self_employed=0.5")), 2000)
    counts = Counter(p.customer_type for p in half)
    assert counts["gig"] == 0
    assert abs(counts["salaried"] / 2000 - 0.5) <= 0.03


@pytest.mark.parametrize(
    "bad",
    ["salaried=0.6,gig=0.2", "salaried=0.6,gig=0.2,self_employed=0.3", "pilot=1",
     "salaried=1.2,gig=-0.2", "salaried", "salaried=abc", "gig=0.5,gig=0.5"],
)  # fmt: skip
def test_invalid_mix_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_mix(bad)


def test_cli_rejects_bad_mix_and_bad_n(capsys) -> None:
    assert main(["--n", "5", "--mix", "salaried=0.9", "--dry-run"]) != 0
    assert main(["--n", "0", "--dry-run"]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_dry_run_writes_nothing(tmp_path, capsys) -> None:
    out = tmp_path / "batch"
    assert main(["--n", "30", "--seed", "1", "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert "Planned 30 statements" in capsys.readouterr().out


def test_non_empty_output_folder_is_refused_without_overwrite(tmp_path) -> None:
    (tmp_path / "manifest.csv").write_text("old")
    with pytest.raises(OutputFolderError):
        prepare_output_folder(tmp_path, overwrite=False)
    prepare_output_folder(tmp_path, overwrite=True)  # our own file: replaced
    assert not (tmp_path / "manifest.csv").exists()


def test_overwrite_never_deletes_files_it_did_not_write(tmp_path) -> None:
    (tmp_path / "my_thesis.docx").write_text("precious")
    (tmp_path / "pdfs").mkdir()
    with pytest.raises(OutputFolderError):
        prepare_output_folder(tmp_path, overwrite=True)
    assert (tmp_path / "my_thesis.docx").read_text() == "precious"
    assert (tmp_path / "pdfs").exists()  # nothing removed when refusing
