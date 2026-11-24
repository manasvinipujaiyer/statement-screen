"""The freshness check on the Week 7 blind sources: small generated batches, no data/ needed."""

import csv
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from statementscreen.tamper import blind_fresh as bf

ROW = tuple[str, str, int, int]  # posted, narration, amount, balance


def _batch(folder: Path, statements: dict[str, tuple[int, list[ROW]]]) -> Path:
    """A batch folder: manifest.csv (statement id, seed) and transactions.parquet."""
    folder.mkdir(parents=True)
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["statement_id", "seed"])
        writer.writerows([sid, seed] for sid, (seed, _) in statements.items())
    rows = [
        {
            "statement_id": sid,
            "row": n,
            "posted": date.fromisoformat(posted),
            "description": narration,
            "amount_paise": amount,
            "balance_paise": balance,
        }
        for sid, (_, txns) in statements.items()
        for n, (posted, narration, amount, balance) in enumerate(txns)
    ]
    pl.DataFrame(rows).write_parquet(folder / "transactions.parquet")
    return folder


def _txns(tag: str, base: int = 100) -> list[ROW]:
    return [
        ("2025-01-05", f"UPI/{tag}/one", base, 1000 + base),
        ("2025-01-09", f"NEFT/{tag}/two", base + 7, 1000 + 2 * base + 7),
    ]


@pytest.fixture
def world(tmp_path):
    """data/ with one earlier batch, and two new batches elsewhere in it."""
    data = tmp_path / "data"
    earlier = _batch(data / "week1", {"stmt_0001": (11, _txns("old", 100))})
    new_a = _batch(data / "blind_src" / "default_v1", {"stmt_0001": (21, _txns("a", 200))})
    new_b = _batch(data / "blind_src" / "karan_v1", {"stmt_0001": (31, _txns("b", 300))})
    return {"data": data, "earlier": earlier, "a": new_a, "b": new_b}


def _run(world, *extra: str) -> int:
    return bf.main([str(world["a"]), str(world["b"]), "--data", str(world["data"]), *extra])


def test_batches_with_nothing_in_common_are_fresh(world, capsys):
    assert _run(world) == 0
    out = capsys.readouterr().out
    assert "2 new statements in 2 batches" in out
    assert "Compared with 1 earlier batches, 1 statements: week1 (1)" in out
    assert "Matches: seed 0, content 0, money (same dates, amounts and balances) 0" in out
    assert "FRESH" in out and "NOT FRESH" not in out


def test_a_statement_seed_seen_before_is_a_match(world, capsys):
    _batch(world["data"] / "week2", {"stmt_0009": (21, _txns("other", 900))})
    assert _run(world) == 1
    out = capsys.readouterr().out
    assert "seed: default_v1/stmt_0001 = week2/stmt_0009" in out and "NOT FRESH" in out


def test_the_same_content_under_another_seed_is_a_match(world, capsys):
    _batch(world["data"] / "week2", {"stmt_0050": (77, _txns("a", 200))})
    assert _run(world) == 1
    out = capsys.readouterr().out
    assert "content: default_v1/stmt_0001 = week2/stmt_0050" in out
    assert "money: default_v1/stmt_0001 = week2/stmt_0050" in out


def test_the_same_dates_amounts_and_balances_in_another_dialect_is_a_money_match(world, capsys):
    twin = [(p, f"other words {n}", a, b) for n, (p, _, a, b) in enumerate(_txns("a", 200))]
    _batch(world["data"] / "week2", {"stmt_0050": (77, twin)})
    assert _run(world) == 1
    out = capsys.readouterr().out
    assert "Matches: seed 0, content 0, money (same dates, amounts and balances) 1" in out


def test_the_two_new_batches_are_compared_with_each_other(world, capsys):
    third = world["data"] / "blind_src" / "third"
    _batch(third, {"stmt_0001": (21, _txns("c", 500))})  # the seed of batch a
    assert bf.main([str(world["a"]), str(third), "--data", str(world["data"])]) == 1
    assert "seed: third/stmt_0001 = default_v1/stmt_0001" in capsys.readouterr().out


def test_a_duplicate_inside_one_new_batch_is_a_match(world, capsys):
    twin = _txns("z", 600)  # content no earlier batch holds
    _batch(world["data"] / "blind_src" / "dup", {"s1": (41, twin), "s2": (42, twin)})
    assert bf.main([str(world["data"] / "blind_src" / "dup"), "--data", str(world["data"])]) == 1
    assert "content: dup/s2 = dup/s1" in capsys.readouterr().out


def test_earlier_batches_are_found_one_and_two_levels_down_and_the_new_ones_are_left_out(world):
    deep = _batch(world["data"] / "nested" / "week3", {"stmt_0001": (51, _txns("deep", 400))})
    found, skipped = bf.find_batches(world["data"], [world["a"], world["b"]])
    assert set(found) == {world["earlier"].resolve(), deep.resolve()} and skipped == []


def test_a_manifest_without_transactions_is_listed_as_skipped_not_hidden(world, capsys):
    half = world["data"] / "half"
    half.mkdir()
    (half / "manifest.csv").write_text("statement_id,seed\nstmt_0001,5\n", encoding="utf-8")
    assert _run(world) == 0
    assert "Skipped, no transactions.parquet: half" in capsys.readouterr().out


def test_no_earlier_batch_at_all_is_an_error_not_a_pass(tmp_path, capsys):
    only = _batch(tmp_path / "data" / "new", {"stmt_0001": (1, _txns("x"))})
    assert bf.main([str(only), "--data", str(tmp_path / "data")]) == 2
    assert "proves nothing" in capsys.readouterr().err


def test_a_new_batch_missing_its_files_is_an_error(world, capsys):
    (world["b"] / "transactions.parquet").unlink()
    assert _run(world) == 2
    assert "transactions.parquet not found" in capsys.readouterr().err


def test_a_manifest_and_transactions_that_disagree_are_an_error(world, capsys):
    with (world["a"] / "manifest.csv").open("a", newline="", encoding="utf-8") as f:
        f.write("stmt_0002,99\n")
    assert _run(world) == 2
    assert "name different statements" in capsys.readouterr().err


def test_the_output_never_shows_a_narration_or_an_amount(world, capsys):
    _batch(world["data"] / "week2", {"stmt_0050": (77, _txns("a", 200))})
    _run(world)
    out = capsys.readouterr().out
    assert "UPI/" not in out and "NEFT/" not in out and "1200" not in out
