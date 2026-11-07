# StatementScreen

Takes an Indian bank-statement PDF and returns a verified income profile, a tamper
verdict, and a plain-language reason for every finding, using classical ML and pattern
recognition (no LLMs).

> **All data in this repo is synthetic.** Real statements are used only for private
> evaluation, stay on my laptop, and are never published.
>
> The banks, IFSC-style codes, UPI handles, employers and merchants in the synthetic data
> are fictional, and every generated PDF carries a "SYNTHETIC DOCUMENT" footer.

## Development setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # create .venv and install dependencies
uv run pytest        # run tests
uv run ruff check .  # lint
uv run ruff format . # format
```

## Generating a batch of synthetic statements

```bash
# Preview the plan without writing anything
uv run python -m statementscreen.generate --n 2000 --dry-run

# Generate 50 statements, then check them from the written files alone
uv run python -m statementscreen.generate --n 50 --seed 42 --out data/synthetic/
uv run python -m statementscreen.verify data/synthetic/
```

| Option | Default | Meaning |
|---|---|---|
| `--n` | (required) | Number of statements, 1-9999 |
| `--seed` | 42 | Main seed. Statement *i* gets its own seed from `SeedSequence(seed, spawn_key=(i,))`, so it is the same whatever `--n` is |
| `--mix` | `salaried=0.6,gig=0.2,self_employed=0.2` | Customer-type shares; must sum to 1 |
| `--hard-case-rate` | 0.2 | Chance of each applicable hard case per statement |
| `--out` | `data/synthetic` | Output folder. A non-empty folder is refused |
| `--overwrite` | off | Replace a previous batch in `--out`. Only files this tool writes are removed |
| `--dry-run` | off | Print the plan and stop |

Each statement is 6-12 months of an Indian savings account (INR), rendered on A4 in one
of two layouts: `columns` (DD/MM/YYYY, separate withdrawal and deposit columns) or
`drcr` (DD-Mon-YY, one amount column with Dr/Cr). Hard cases: `salary_date_shift`,
`employer_rename`, `bonus_month`, `missed_salary` (salaried only) and `emi_bounce`.

Outputs, written one statement at a time:

| File | Contents |
|---|---|
| `pdfs/stmt_NNNN.pdf` | The rendered statement |
| `labels/stmt_NNNN.json` | Ground truth: customer type, true monthly income, true employer (null for gig and self-employed), hard-case tags and details |
| `transactions.parquet` | Every transaction with `statement_id`, money as integer paise, and `category`, `canonical_counterparty`, `is_salary`, `is_emi`, `is_bounce` |
| `manifest.csv` | One row per statement: id, customer type, layout, months, transaction count, tags, seed, generator version |
| `run_summary.json` | Counts, verify's reconciliation rate, timing, disk size and peak memory |

The same command produces byte-identical PDFs, labels, manifest and Parquet file
(`run_summary.json` differs, because it records timings). `verify` recomputes every
running balance and checks that all outputs agree, naming any statement that fails.

### Week 1 reference run

`--n 2000 --seed 42` on a MacBook Air (M2, 8 GB), from
[`reports/week1_run_summary.json`](reports/week1_run_summary.json):
2,000 statements and 1,177,258 transactions in 133 s (0.066 s per statement), 129 MB on
disk, 356 MB peak memory, and 100% balance reconciliation.

## Parsing

Reads the statement PDFs back into transactions with pdfplumber. The parser works from
the PDF alone: it never reads the labels, Parquet, manifest or generator. Only the
separate `score` command compares its output with the ground truth.

```bash
# Parse a batch (one PDF at a time), then score it against the batch's truth
uv run python -m statementscreen.parse data/week1_n2000/pdfs/ --out data/parsed_2000/
uv run python -m statementscreen.score data/parsed_2000/ data/week1_n2000/
```

- **Layout detection** comes from the table-header text. A PDF matching neither layout
  gets an "unknown layout" error rather than a guess.
- **Money** is parsed as `Decimal` and stored as integer paise, with no floats.
- **Self-checks** are separate named flags and never silently fix anything:
  `BALANCE_CHAIN`, `SUMMARY_IDENTITY`, `SUMMARY_TOTALS`, `SUMMARY_COUNTS`, `DATE_ORDER`,
  `DATE_OUT_OF_PERIOD`, `CLOSING_MISMATCH`, `CHECK_NOT_RUN`.

`parse` writes `rows.parquet`, `statements.parquet`, `issues.parquet` and
`parse_run.json`; `score` writes `reports/week2_parse_report.json`.

### Week 2 reference run

All 2,000 statements of the Week 1 reference batch, from
[`reports/week2_parse_report.json`](reports/week2_parse_report.json):

| | columns | drcr |
|---|---|---|
| Statements | 1,015 | 985 |
| Rows (truth = parsed) | 594,261 | 582,997 |
| Exact rows (date, amount, balance) | 100% | 100% |
| Layout detected correctly | 100% | 100% |
| Narration exact, ignoring whitespace | 100% | 82.4% |
| Narration cut short by the layout | 0% | 17.6% |

Target: 99.5% exact rows per layout. Self-check flags: 0. Parse issues: 0.
2,000 PDFs took 1,739 s (0.87 s per PDF, p95 1.44 s) with 260 MB peak memory.

No narration was wrong. The drcr layout truncates long narrations when rendering, so the
full text is not in the PDF. Before any drcr code existed, forcing the columns rules onto
drcr PDFs scored **0%** exact rows (`first_contact`, 103 statements).

These are clean, text-based PDFs from this project's own renderer, so 100% shows that
the parser handles the synthetic layouts. It says nothing yet about real banks'
statements.

## License

MIT. See [LICENSE](LICENSE).
