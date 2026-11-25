"""Every relative link and anchor in README.md and docs/ must resolve."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
CODE_SPAN = re.compile(r"`[^`\n]*`")
EXTERNAL = ("http://", "https://", "mailto:")


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped (letters, digits,
    spaces, hyphens and underscores stay), spaces become hyphens."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def prose_lines(text: str):
    """The lines outside fenced code blocks, with their line numbers (from 1)."""
    inside = False
    for number, line in enumerate(text.split("\n"), start=1):
        if FENCE.match(line):
            inside = not inside
            continue
        if not inside:
            yield number, line


def anchors(text: str) -> set[str]:
    """Every heading anchor in a Markdown text; a repeated heading gets -1, -2, ..."""
    seen: dict[str, int] = {}
    found = set()
    for _, line in prose_lines(text):
        match = HEADING.match(line)
        if not match:
            continue
        base = slug(match.group(2))
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.add(base if count == 0 else f"{base}-{count}")
    return found


def broken_links(path: Path) -> list[str]:
    """The relative links in one Markdown file that do not resolve, one message each."""
    problems = []
    text = path.read_text(encoding="utf-8")
    for number, line in prose_lines(text):
        for target in LINK.findall(CODE_SPAN.sub("", line)):
            if target.startswith(EXTERNAL):
                continue
            file_part, _, anchor = target.partition("#")
            destination = path if not file_part else (path.parent / file_part).resolve()
            where = f"{path.relative_to(ROOT)}:{number}: {target}"
            if not destination.exists():
                problems.append(f"{where} (no such file)")
            elif anchor and destination.suffix == ".md":
                if anchor not in anchors(destination.read_text(encoding="utf-8")):
                    problems.append(f"{where} (no such heading)")
    return problems


def test_the_anchor_rule_matches_how_github_names_headings():
    assert slug("Pattern detection") == "pattern-detection"
    assert slug('What "monthly income" means') == "what-monthly-income-means"
    assert slug("Results: test and shift (scored once)") == "results-test-and-shift-scored-once"
    assert slug("Run it in Docker (Colima, no Docker Desktop)") == (
        "run-it-in-docker-colima-no-docker-desktop"
    )
    assert slug("Erratum: Week 5") == "erratum-week-5"


def test_a_repeated_heading_gets_a_numbered_anchor_and_code_blocks_are_not_headings():
    text = "# A\n\n## Known limits\n\n```\n# not a heading\n```\n\n## Known limits\n"
    assert anchors(text) == {"a", "known-limits", "known-limits-1"}


def test_a_missing_file_and_a_missing_heading_are_both_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    (tmp_path / "other.md").write_text("# Real heading\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "[ok](other.md#real-heading) [bad file](gone.md) [bad anchor](other.md#nope)\n"
        "[web](https://example.com/x#y) `[in code](gone.md)`\n"
        "```\n[in a block](gone.md)\n```\n",
        encoding="utf-8",
    )
    problems = broken_links(page)
    assert len(problems) == 2
    assert "gone.md (no such file)" in problems[0]
    assert "other.md#nope (no such heading)" in problems[1]


def markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_and_anchor_resolves(path):
    assert broken_links(path) == []


def test_the_readme_links_to_every_page_in_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pages = sorted((ROOT / "docs").glob("*.md"))
    assert [p.name for p in pages if f"docs/{p.name}" not in readme] == []
