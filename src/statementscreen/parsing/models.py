"""The parser's standard output.

One ``ParsedRow`` per transaction, one ``ParsedStatement`` per PDF (header and summary
fields, the rows, and every issue found). Money is integer paise: credits positive,
debits negative. Fields that a layout doesn't print are None, never invented.
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class ParsedRow:
    statement_id: str
    row: int  # 0-based position in the statement
    posted: date
    value_date: date | None
    ref: str | None
    narration: str
    amount_paise: int  # signed: credit +, debit -
    balance_paise: int
    page: int  # 1-based page the row starts on
    layout: str


@dataclass(frozen=True)
class Issue:
    """A problem found while parsing or checking. Never fixed silently, always listed."""

    statement_id: str
    code: str  # e.g. ORPHAN_LINE, BALANCE_CHAIN (see checks.py for the self-checks)
    page: int | None
    row: int | None
    detail: str


@dataclass
class StatementHeader:
    """Header and summary fields, where the layout prints them."""

    account_number: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    opening_balance_paise: int | None = None
    closing_balance_paise: int | None = None
    total_debits_paise: int | None = None
    total_credits_paise: int | None = None
    dr_count: int | None = None
    cr_count: int | None = None


@dataclass
class ParsedStatement:
    statement_id: str
    layout: str | None
    pages: int
    status: str  # "ok" | "error"
    error: str | None = None
    header: StatementHeader = field(default_factory=StatementHeader)
    rows: list[ParsedRow] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def issue(self, code: str, page: int | None, row: int | None, detail: str) -> None:
        self.issues.append(Issue(self.statement_id, code, page, row, detail))
