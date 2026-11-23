"""What every tamper check returns: a list of findings.

A finding names what was seen (``code``), which check group saw it, where (page and row,
when it applies), how much it counts (``strength``) and why, in plain words.

Strengths:
    strong  the PDF contradicts itself in a way an unedited statement can't (e.g. a
            balance that doesn't follow from the one before it)
    weak    unusual, or a sign the file went through an editing tool, but not proof
    info    listed so a reader sees it, never counted toward a verdict (e.g. a check
            that couldn't run, or a producer that differs from the creator)

The detector works from the PDF alone: no check imports the generator (synth.*), the
tamper generator (tamper.copy, tamper.rebuild) or reads an answer key, label, manifest
or transactions.parquet.
"""

from dataclasses import asdict, dataclass

STRENGTHS = ("strong", "weak", "info")
GROUPS = ("parse", "arithmetic", "structure", "layout", "history", "anomaly")


@dataclass(frozen=True)
class Finding:
    code: str
    group: str
    page: int | None  # 1-based, None when the finding is about the whole statement
    row: int | None  # 0-based transaction position, as the parser numbers rows
    strength: str
    reason: str

    def __post_init__(self) -> None:
        if self.strength not in STRENGTHS:
            raise ValueError(f"strength must be one of {STRENGTHS}, got {self.strength!r}")
        if self.group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {self.group!r}")
        if not self.reason.strip():
            raise ValueError(f"{self.code}: a finding needs a reason")

    def to_dict(self) -> dict:
        return asdict(self)


def rupees(paise: int) -> str:
    """12345678 -> 'INR 1,23,456.78' (Indian grouping, as the statements print it)."""
    sign = "-" if paise < 0 else ""
    whole, frac = divmod(abs(paise), 100)
    digits = str(whole)
    if len(digits) > 3:
        head, groups = digits[:-3], [digits[-3:]]
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        digits = ",".join([head, *groups])
    return f"INR {sign}{digits}.{frac:02d}"


def where(page: int | None, row: int | None) -> str:
    """'page 2, transaction 14: ' (row is 0-based, shown 1-based), or '' if neither."""
    parts = []
    if page is not None:
        parts.append(f"page {page}")
    if row is not None:
        parts.append(f"transaction {row + 1}")
    return (", ".join(parts) + ": ") if parts else ""
