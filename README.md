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

## Pattern detection

Groups each statement's counterparties and flags salary, EMIs, bounces and rent with
simple, explainable rules. Every flag carries a plain-language reason, for example
`salary: monthly NEFT credit in 12 of 12 months, amount always INR 80,028, paid on day
28-30`. Pattern detection works on the **parser output only**; it never reads the
labels, transactions.parquet, the manifest or generator code. Only the separate
`score_patterns` command compares it with the ground truth.

```bash
# Once: lock a 70/15/15 train/dev/test split by statement (seed 42), stratified by layout
# and customer type. Writes splits/split_v1.csv and .json (with the manifest's SHA-256).
uv run python -m statementscreen.splits data/week1_n2000/manifest.csv

# Clean, group and flag one split (train: features for Week 4; dev: for scoring)
uv run python -m statementscreen.patterns data/parsed_2000/ --split train --out data/patterns/
uv run python -m statementscreen.patterns data/parsed_2000/ --split dev --out data/patterns/

# Score dev against the truth: writes reports/week3_patterns_dev.json and logs an MLflow run
uv run python -m statementscreen.score_patterns data/patterns/ data/week1_n2000/ --split dev
```

How it works, one statement at a time:

1. **Cleaning:** each narration gets a channel (UPI, NEFT, RTGS, IMPS, NACH, BBPS, ATM,
   CARD, INTEREST, CHARGE, REVERSAL or OTHER), a direction and a cleaned counterparty,
   picked by field position for 29 known narration styles. References, UPI handles,
   IFSC-style codes, dates, filler words and legal forms are removed. Unknown styles fall
   back to a keyword channel and are counted, never hidden. The account holder's own name
   is inferred from the rows (the NEFT/RTGS holder slot and "self transfer" notes) and
   replaced by `self`.
2. **Grouping:** within each statement and direction, RapidFuzz similarity plus character
   n-gram TF-IDF cosine, agglomerative clustering at a distance threshold of 0.45 (best
   pairwise F1 in a dev sweep, `reports/week3_grouping_sweep.json`). A credit group is never
   merged with a debit group; a cut-off name joins its full form only when unambiguous.
3. **Features** per group: months present and coverage, transactions per month, gap mean
   and SD, circular day-of-month (the 31st and the 1st are 1 day apart), amount CV and the
   share within 5% of the median. Saved to `data/patterns/<split>/group_features.parquet`.
4. **Rules:** salary (a monthly bank-transfer credit of a stable amount; a bonus, an extra
   credit in a month or one shifted or missed month don't break it; an employer rename is
   one stream named by the original employer), EMI (a monthly NACH debit of a fixed
   amount), bounce (a returned EMI, the debit it returned, and the linked charge and GST)
   and rent (a monthly transfer of at least INR 1,500 to one person). Every threshold is in
   `PatternConfig`; the rule thresholds tuned on dev are in `reports/week3_rules_sweep.json`.

Splits are loaded by name. A loader refuses a manifest whose hash doesn't match, and the
test split is refused unless `STATEMENTSCREEN_OPEN_TEST=1` is set. Examples were looked at
on train only and thresholds tuned on dev only.

### Week 3 dev results

> **Synthetic dev results.** These come from the 300 dev statements of the synthetic
> reference batch, the split every threshold was tuned on. **The test set is still
> closed**, so these are not a held-out estimate, and they say nothing yet about real
> statements.

From [`reports/week3_patterns_dev.json`](reports/week3_patterns_dev.json):

| | Precision | Recall |
|---|---|---|
| Grouping (pairwise, vs `canonical_counterparty`) | 99.93% | 99.11% |
| `is_salary` (1,612 true rows) | 100.00% | 99.75% |
| `is_emi` (1,462) | 100.00% | 100.00% |
| `is_bounce` (180) | 100.00% | 100.00% |
| `is_rent` (1,891) | 100.00% | 99.63% |
| `has_salary` (185 salaried statements) | 100.00% | 100.00% |

Employer match on salaried statements: 95.68% exact and 97.84% fuzzy (normalised names,
RapidFuzz 90+). False salary on gig statements: 0 of 63; on self-employed: 0 of 52. The
report also breaks every number down by layout, customer type and hard-case tag, and
lists misses and false flags with the rule's reason or why it didn't fire.

Speed on a MacBook Air (M2, 8 GB): patterns took 0.0136 s per statement on both dev
(4.4 s, 363 MB peak) and train (20.3 s for 1,400 statements, 608 MB peak); scoring dev took
0.2 s. Every narration style in train and dev was recognised.

Known limits, from the dev misses:

- **Shifted salaries next to another credit:** 4 salary rows were left out as "an extra
  credit in its month", the cost of not counting a bonus equal to the salary.
- **One rent group** paid in 4 of 7 months (57%) stays below the 60% coverage rule.
- **Employer names cut off in every row** (e.g. `shakapu technol`): when the drcr layout
  cuts every narration of an employer at the same point, the full name is not in the PDF.
- **Employer recall in grouping is 94.48%,** because an employer's old and new names stay
  in separate groups; the salary rule joins them, not the grouping.
- The holder's name is only recognised where a holder slot or a "self" note shows it.

**MLflow** runs are logged locally to `mlruns/` (git-ignored) as a SQLite file, because
MLflow 3.18 refuses the plain-folder store. Each run records the split, every threshold,
the headline metrics and the report. To browse them:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
# open http://127.0.0.1:5000, experiment week3_patterns; press Ctrl+C to stop
```

**Licences:** MLflow brings in three dependencies that are not MIT, BSD or Apache:
certifi (MPL-2.0, a weak copyleft licence that only applies to changes to certifi's own
files), and matplotlib and typing-extensions (permissive, PSF-based licences). All three
are used unmodified.

## License

MIT. See [LICENSE](LICENSE).
