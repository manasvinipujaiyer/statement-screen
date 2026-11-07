"""Words and lines, as read from a PDF page.

pdfplumber gives each word's box in points from the TOP-LEFT of the page: x0/x1 are its
left/right edges and ``top`` grows downward. A *line* is a run of words whose tops are
within a small tolerance of each other, sorted left to right.
"""

import re
from dataclasses import dataclass

LINE_TOLERANCE = 2.0  # points: words this close vertically are on the same line

_FOOTER = re.compile(r"Page \d+ of \d+$")


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    x1: float
    top: float

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2


Line = list[Word]


def to_words(raw: list[dict]) -> list[Word]:
    return [Word(w["text"], float(w["x0"]), float(w["x1"]), float(w["top"])) for w in raw]


def group_lines(words: list[Word], tolerance: float = LINE_TOLERANCE) -> list[Line]:
    """Cluster words into lines by their top edge, then sort each line left to right."""
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: (w.top, w.x0)):
        if lines and abs(w.top - lines[-1][0].top) <= tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def text_of(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def is_footer(line: Line) -> bool:
    """The page footer: 'SYNTHETIC DOCUMENT ... Page X of Y'."""
    text = text_of(line)
    return text.startswith("SYNTHETIC DOCUMENT") or bool(_FOOTER.search(text))
