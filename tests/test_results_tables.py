"""The README's results block is generated from reports/, and cannot drift from them."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "results_tables.py"
    spec = importlib.util.spec_from_file_location("results_tables", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["results_tables"] = module
    spec.loader.exec_module(module)
    return module


rt = _load()


def _read_independently(report: str, path: str):
    """Walk the report with a separate, plain reader (not the generator's own)."""
    node = json.loads((ROOT / "reports" / report).read_text(encoding="utf-8"))
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def all_cells() -> list:
    cells = []
    for table in rt.tables():
        for row in table.rows:
            cells += row[1:]
    for sentence in rt.sentences():
        cells += sentence.cells
    return cells


def test_the_readme_block_equals_what_the_script_prints():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert rt.START in text and rt.END in text, "the README has no results markers"
    start = text.index(rt.START)
    end = text.index(rt.END) + len(rt.END)
    assert text[start:end] == rt.build_block(), (
        "README results block differs: run  uv run python scripts/results_tables.py --write"
    )


def test_no_table_cell_is_typed_by_hand():
    for table in rt.tables():
        for row in table.rows:
            assert isinstance(row[0], str)
            for item in row[1:]:
                assert isinstance(item, rt.Cell), f"hand-typed cell in {table.header[0]!r}"
                assert item.sources, "a cell with no source"


def test_every_source_is_a_real_value_in_a_committed_report():
    for c in all_cells():
        for report, path in c.sources:
            assert (ROOT / "reports" / report).is_file(), report
            value = _read_independently(report, path)
            assert value is None or isinstance(value, (int, float)), (report, path)


def test_every_displayed_number_is_the_formatted_value_from_its_report():
    for c in all_cells():
        values = [_read_independently(report, path) for report, path in c.sources]
        assert c.text() == rt.FORMATS[c.fmt](*values), c.sources


def test_every_caption_names_the_reports_its_cells_come_from():
    for table in rt.tables():
        reports = {report for row in table.rows for item in row[1:] for report, _ in item.sources}
        for report in reports:
            assert f"reports/{report}" in table.caption, (report, table.caption[:40])


@pytest.mark.parametrize(
    ("value", "shown"),
    [(1, "100.00%"), (0, "0.00%"), (0.99999, "99.99%"), (0.000001, "0.01%"),
     (0.8595, "85.95%"), (0.986667, "98.67%"), (0.125, "12.50%"), (None, "n/a")],
)  # fmt: skip
def test_a_percentage_never_rounds_into_perfection_or_zero(value, shown):
    assert rt.pct(value) == shown


def test_f1_and_signed_differences_follow_the_same_guards():
    assert rt.f1(1) == "1.0000" and rt.f1(0.99996) == "0.9999" and rt.f1(0.00001) == "0.0001"
    assert rt.f1(0.38903) == "0.3890" and rt.f1(None) == "n/a"
    assert rt.signed(-0.6836) == "-0.6836" and rt.signed(0.297) == "+0.2970"
    assert rt.signed(0) == "0.0000" and rt.signed(0.00001) == "+<0.0001"
