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
| `--narration-dialect` | `default` | Narration formats: `default` (output byte-identical to before dialects existed) or a file in `synth/dialects/`, e.g. `karan_v1`. Changes narrations only; labels, amounts and balances are the same |

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

## Classifier vs rules

Week 4 asks whether a classical model beats the Week 3 rules. On the reference batch it
can't show much: the rules already score 99.6-100% on dev. So the comparison also uses a
**shift batch** written in narration formats the code had never seen.

> **What these numbers are.** All data is synthetic. **Dev was used for tuning** (model
> settings and decision thresholds). **The shift batch was scored once**, after every
> setting was frozen; one later re-run, which only changed how an undefined F1 is
> reported, is labelled post-hoc below. Its narration formats were written by **an
> independent collaborator who had not seen the code**. **The test split was opened in
> Week 5** for the profile and label scoring (see Income profile below). Nothing here
> says anything about real bank statements.

```bash
# 1. The shift batch: 26 narration formats (src/statementscreen/synth/dialects/karan_v1.txt)
uv run python -m statementscreen.generate --n 300 --seed 2026 --narration-dialect karan_v1 \
    --out data/week4_shift_n300/
uv run python -m statementscreen.dialect_report data/week4_shift_n300/
uv run python -m statementscreen.parse data/week4_shift_n300/pdfs/ --out data/parsed_shift_n300/
uv run python -m statementscreen.score data/parsed_shift_n300/ data/week4_shift_n300/ \
    --out reports/week4_shift_parse_report.json

# 2. Register it as a whole evaluation batch, then run the frozen Week 3 rules on it
uv run python -m statementscreen.eval_batches register data/week4_shift_n300/ \
    data/parsed_shift_n300/ --name shift_karan_v1 --reference data/week1_n2000/manifest.csv
uv run python -m statementscreen.patterns data/parsed_shift_n300/ --eval shift_karan_v1 \
    --out data/patterns/
uv run python -m statementscreen.score_patterns data/patterns/ data/week4_shift_n300/ \
    --eval shift_karan_v1

# 3. Features, training (tuned on dev, then frozen), and the one-time comparison
uv run python -m statementscreen.features data/patterns/ --sets train dev shift_karan_v1 \
    --out data/features/
uv run python -m statementscreen.train data/features/ data/week1_n2000/ --out data/models/
uv run python -m statementscreen.compare --dev-only   # dry run: never reads the shift batch
uv run python -m statementscreen.compare              # refuses a second shift scoring
```

**The shift batch.** Each of Karan's 16 kinds (salary, EMI, EMI return, bounce charge,
rent, UPI, ATM, bills, card, interest, refund, gig, business, transfers) replaces the
narration of one generator event; advance tax and GST payments keep today's text.
Default batches stay byte-identical (a recorded fingerprint is checked in the tests).
300 statements (seed 2026) parse at **100% exact rows** in both layouts. Of the
177,425 narrations Karan's lines wrote, **89.80%** start with one of our 29 known
openings, but only **21.59%** are fully recognised by the Week 3 cleaning: 9 lines have
a known opening, 17 a new one (all 0% recognised), and 78.22% of parsed rows get an
unknown or variant style (0% in Week 3). The evaluation batch is registered with a
fingerprint of its parsed rows and refuses the reference batch, or any statement of it,
so the split guard can't be bypassed.

**The model.** One LightGBM binary model per label, on 30 row and group features (amount,
day, position, channel, direction, recurrence measures of the counterparty group) plus a
2,000-feature character 2-4-gram TF-IDF of the narration (digits replaced, fitted on
train rows only). The rule flags, their reasons, `group_rules` and the cleaning style
are never features. Every positive train row plus a fixed-seed 10% of the rest (weight
10, so probabilities stay calibrated): 104,340 of 825,318 train rows. Eight settings per
label, chosen and thresholded on dev F1 (threshold in the middle of the gap that gives
it), deterministic, 2 threads. Training took 154.1 s and 1,427 MB peak; the models are
97-704 KB each.

### Results

95% intervals come from a paired bootstrap over statements (1,000 resamples, seed 42).
From [`reports/week4_classifier.json`](reports/week4_classifier.json):

| F1 | Dev: rules | Dev: model | Model minus rules (95% CI) | Shift: rules | Shift: model | Model minus rules (95% CI) |
|---|---|---|---|---|---|---|
| `is_salary` | 0.9988 | 0.9994 | [-0.0008, +0.0020] includes 0 | 0.9991 | **0.3890** | **[-0.6836, -0.5440]** |
| `is_emi` | 1.0000 | 1.0000 | [0, 0] includes 0 | 1.0000 | 1.0000 | [0, 0] includes 0 |
| `is_bounce` | 1.0000 | 0.9205 | **[-0.1381, -0.0393]** | 0.0000 * | **0.3731** | **[+0.2970, +0.4587]** |
| `is_rent` | 0.9981 | 1.0000 | [0, +0.0057] includes 0 | 0.9959 | 0.9956 | [-0.0007, 0] includes 0 |
| `has_salary` | 1.0000 | 1.0000 | [0, 0] includes 0 | 1.0000 | **0.4079** | **[-0.6593, -0.5279]** |

| Dev-to-shift F1 drop | Rules | Model |
|---|---|---|
| `is_salary` | -0.0003 | 0.6104 |
| `is_emi` | 0.0000 | 0.0000 |
| `is_bounce` | 1.0000 * | 0.5474 |
| `is_rent` | 0.0023 | 0.0044 |

False salary flags: on gig statements 0 for both methods on dev (of 63) and shift (of
56); on self-employed statements 0 for both on dev (of 52), but on the shift batch rules
0 and **model 57 of 59**.

\* The rules flag no bounce row on the shift batch (0 of 172 found), so their precision
is undefined. The first, one-time scoring printed their F1 and drop as n/a; a post-hoc
re-run, appended to the report, reports F1 as 2tp / (2tp + fp + fn) = 0 (the definition
the bootstrap already used). Every other number was identical in both runs.

**What it shows.**

- On dev (same formats as training) the model ties the rules on salary, EMI, rent and
  has_salary, and is worse on bounces.
- On the shift batch the model beats the rules only where they have no fallback: it
  finds half the bounce rows, the rules none.
- **On salary the model collapses.** Every salary our generator writes is a NEFT credit,
  so "channel is NEFT" became near-certain evidence of salary in training; Karan's
  `SAL CREDIT ...` and `IMPS/.../SALARY FOR ...` salaries are not NEFT, and the
  explanations of the misses put "channel is not NEFT" first. His business credits
  (`NEFT CR-...-INV 678...`) lose the `INV123` text pattern of ours, and are flagged as
  salary on the strength of the amount and "NEFT CR". This is shortcut learning from a
  generator with too little variety, which is what the shift batch was built to expose.
- The rules held up because they rely on recurrence (same counterparty, stable amount,
  monthly rhythm) and accept any bank-transfer channel; they only failed where they
  depend on exact formats (bounces) or on reading a name (exact employer match fell from
  95.68% to 61.08%).

The report also breaks every number down by layout, hard-case tag and, on the shift
batch, by Karan format group (known vs new opening), and explains up to 30 model errors
per set with the top three `pred_contrib` features in plain words. MLflow runs are in the
`week4_classifier` experiment (`compare` and `compare_post_hoc`).

Speed and memory on a MacBook Air (M2, 8 GB): features 0.0008 s per statement (train
34.1 s, 1,502 MB peak, from fitting the TF-IDF); training 154.1 s; the comparison 2.6 s.

**Licences:** LightGBM is MIT. On macOS it needs the OpenMP runtime (`brew install
libomp`, part of LLVM, Apache-2.0 with LLVM exceptions), installed on the machine, not
added to this repo.

## Income profile

### What "monthly income" means

`true_monthly_income` in `labels/*.json` is the **average regular income per month that
actually arrived**: salary, gig payouts or payments from clients and customers, added up
over the statement and divided by the **number of calendar months the statement covers**.
Bonuses, money from family, self-transfers, refunds and interest are never counted, because
a lender shouldn't count them as stable income.

| Customer | Counted | So... |
|---|---|---|
| Salaried | salary credits that arrived | a missed month lowers the average; a late salary and a salary under the employer's new name still count; a bonus never counts |
| Gig | platform payouts that arrived | weeks without a payout lower it |
| Self-employed | payments from clients and customers, by any channel | money from family doesn't count, even though it can look the same |

> **What these numbers are.** All data is synthetic. **Dev was used for development**
> (two estimation variants and one bug fix were chosen on it). **The test split and the
> shift batch were each scored once**, after a pre-registration
> ([`reports/week5_preregistration.json`](reports/week5_preregistration.json)) was committed
> and pushed: the frozen commit, hashes of every source file, config, model and split, and
> the exact metrics. Checks against real statements are still to come.

### How the profile is built

The profile reads **only the parser and pattern outputs** (never the truth, the manifest or
generator code) and uses the Week 3 rules, not the Week 4 model. Every finding has a value
and a plain-language reason:

- **Salary:** has_salary, employer, typical monthly salary (median of the salary credits),
  months paid, missed months, and the payday spread on a 31-day circle.
- **Other regular income:** credit groups present in at least 3 months that are not
  salary, not the holder's own transfers and not interest, refunds or charges. A group
  paying 3+ times a month with gaps of 5-10.5 days is "weekly payouts" (gig).
- **Estimated monthly income** = (salary + other regular income) / statement months. For
  a statement estimated as self-employed, one-off credits from people are added too
  (customers who pay once). No estimate when there is no recurring income.
- **Customer type** (salaried, gig, self-employed or unknown), EMIs (count, total, lenders),
  bounces (counted once each, plus the charges), rent, the EMI-to-income ratio (or "not
  computed" and why), and warnings: history under 6 months (with low confidence), an
  employer name change, no recurring income, bounced EMIs.

```bash
uv run python -m statementscreen.profile data/patterns/ --set dev --parsed data/parsed_2000/
uv run python -m statementscreen.score_profile data/profiles/ data/week1_n2000/ --set dev
uv run python -m statementscreen.profile.pdf data/week1_n2000/pdfs/stmt_0002.pdf   # one PDF
```

`profile_pdf(path)` runs parse, patterns and profile on one PDF with no split, manifest or
truth (the Week 7 service will wrap it): 0.786 s and 254 MB peak for `stmt_0002`. An
abridged profile of that synthetic dev statement:

```json
{"statement_id": "stmt_0002", "statement_months": 10, "warnings": [],
 "findings": {
  "has_salary": {"value": true, "reason": "monthly NEFT credit in 10 of 10 months, amount always INR 36,195, paid on day 29-31; 1 other credit(s) from this payer not counted (bonus, an extra credit in a month, or a different amount)"},
  "employer": {"value": "mitika logistics", "reason": "employer of record of the salary credits: ..."},
  "estimated_monthly_income_paise": {"value": 3619500, "reason": "INR 36,195 a month: salary INR 36,195 a month (10 credits over 10 months), over 10 months"},
  "payday_spread": {"value": {"usual_day": 30, "from": 29, "to": 31, "span_days": 2}, "reason": "paid between day 29 and day 31 of the month (2 days apart on a 31-day circle)"},
  "customer_type": {"value": "salaried", "reason": "a monthly salary from 'mitika logistics'"},
  "emi_to_income_ratio": {"value": 0.0, "reason": "INR 0 of EMIs a month / INR 36,195 of income = 0%"}}}
```

**Chosen on dev.** Counting only recurring groups put 82.00% of dev incomes within 10%;
adding one-off credits for self-employed statements put 98.67% there, so that variant was
kept. One dev gig statement paid weekly but skipped weeks (mean gap 9.15 days) and was
called self-employed; the weekly gap limit was raised from 9 to 10.5 days, which fixed it
without moving any self-employed statement.

### Results: dev, test and shift

From [`reports/week5_test_and_shift.json`](reports/week5_test_and_shift.json), with 95%
bootstrap intervals over statements (1,000 resamples, seed 42). 300 statements each.

| Profile | Dev | Test (scored once) | Shift (scored once) |
|---|---|---|---|
| Income within 5% | 94.00% [91.33, 96.67] | 94.00% [91.00, 96.67] | 93.00% [90.00, 95.67] |
| Income within 10% | 99.00% [97.99, 100] | **97.00% [95.00, 98.67]** | 98.33% [96.67, 99.67] |
| Income median APE; no estimate | 0.00%; 0.00% | 0.00%; 0.00% | 0.00%; 0.00% |
| has_salary precision / recall | 100% / 100% | 100% / 100% | 100% / 100% |
| Employer exact match | 95.68% [92.44, 98.40] | 95.14% [91.93, 98.01] | **61.08% [53.80, 68.26]** |
| Employer fuzzy match | 97.84% [95.35, 99.48] | 97.30% [94.87, 99.44] | **66.49% [59.03, 72.97]** |
| Customer-type accuracy | 100% | 100% | 100% |
| EMI count / total exact | 100% / 100% | 100% / 100% | 100% / 100% |
| Bounce count exact | 100% | 100% | **77.33% [72.66, 82.00]** |

| Label F1, rules / model | Dev | Test | Shift |
|---|---|---|---|
| `is_salary` | 99.88% / 99.94% | 99.82% / 99.94% | 99.91% / 38.90% |
| `is_emi` | 100% / 100% | 100% / 100% | 100% / 100% |
| `is_bounce` | 100% / 92.05% | 100% / 82.00% | 0.00% / 37.31% |
| `is_rent` | 99.81% / 100% | 100% / 100% | 99.59% / 99.56% |

**Did dev overstate performance?** On one pre-registered metric: income within 10% was
99.00% on dev, above the whole test interval (97.00%, [95.00%, 98.67%]). Every other dev
value is inside the test interval. The model's bounce F1 also fell, from 92.05% on dev to
82.00% on test. The shift batch's label metrics repeat Week 4's frozen models.

### Stress tests and the layout hold-out (dev only)

From [`reports/week5_stress.json`](reports/week5_stress.json). Each perturbation changes
the parsed rows with a fixed seed per statement; the truth is not changed.

| Perturbation | Income within 10% | has_salary recall | Employer exact | Type accuracy | Expected sign |
|---|---|---|---|---|---|
| None (baseline) | 99.00% | 100.00% | 95.68% | 100.00% | |
| Drop one month | 42.67% | 98.92% | 94.59% | 99.00% | 183 of 185 still salaried, all 183 show the missed month |
| Cut to 3 months | 71.33% | 89.73% | 85.95% | 86.33% | all 300 warn of a short history; none "normal" confidence |
| Typos in names | 98.67% | 99.46% | 92.97% | 99.67% | |
| Lower-case narrations | 99.00% | 100.00% | 95.68% | 100.00% | no change |
| Narrations cut to 20 chars | 79.00% | 94.59% | 0.00% | 73.00% | |
| Dates moved up to 3 days | 91.67% | 89.19% | 84.32% | 93.33% | |
| Amounts x 1.1 | 68.33% | 100.00% | 95.68% | 100.00% | all 300 estimates are 1.1x the baseline |

has_salary precision stayed 100% under every perturbation. Every perturbation was
deterministic, the profile JSON was byte-identical across two runs, and an empty, a
debits-only and a single-month statement each returned a profile with warnings.

Layout hold-out of the Week 4 models (retrained on one layout with the frozen settings and
scored on the other layout's dev statements; no retuning), F1:

| | Columns to drcr: hold-out / both | Drcr to columns: hold-out / both |
|---|---|---|
| `is_salary` | 0.9994 / 1.0000 | 0.9945 / 0.9988 |
| `is_emi` | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `is_bounce` | 0.7813 / 0.8808 | 0.8786 / 0.9686 |
| `is_rent` | 0.9697 / 1.0000 | 1.0000 / 1.0000 |

### Known limits

- **Bounces in unseen formats:** the rules recognise none of the shift batch's bounce
  narrations, so bounce counts are right for only 77.33% of shift statements.
- **Employer names in unseen formats** are read through the generic fallback: 61.08% exact.
- **Date jitter** of up to 3 days costs 10.81 points of salary recall, because the salary
  rule expects most payments within 3 days of the usual day.
- **A shifted salary next to another credit** is left out by the Week 3 rule, so a few
  salaried incomes are 10-17% low.
- **Money from family** is counted as customer income for self-employed statements,
  because one-off transfers from people look the same.

## Tamper detection

`uv run python -m statementscreen.tamper.run <pdfs> --out <folder>` runs five checks on each
PDF, one file at a time, and `scoring.py` turns the findings into one verdict per statement
with a reason for each finding: **no evidence of tampering**, **suspicious**, **tampered**,
or **could not check**. The checks are arithmetic (balance chain, page carry-over, summary
totals), structure, file history (for example incremental updates), layout and font
(against baselines measured on genuine train statements), and an IsolationForest anomaly
score. Strong findings give "tampered". The anomaly score is weak: alone it gives
only "suspicious", never "tampered" (verdict rule v2; v1 vs v2 was settled on dev by a
rule written down beforehand, in `reports/week6_rule_versions.json`). "Could not check"
means no evidence was found but an applicable check did not run; it is never counted as
"no evidence".

**How it was tested.** Tampered copies of statements are built with answer keys the
detector never reads (`python -m statementscreen.tamper`): re-render with inconsistent or
with consistent arithmetic, in-place digit edit (plain, and with the balances recalculated),
page swap, page removal, and a held-out **overlay** type (white box with new text) that dev
never saw. `resaved_genuine` is a no-change control. Everything was frozen and written down
in [`reports/week6_preregistration.json`](reports/week6_preregistration.json) (detector
fingerprint, hashes, seeds, the six commands, what to expect) and pushed before the test
split's copies were built. The scoring step refuses to run if anything differs. The commands
ran once each, in order, on the 300 test statements (800 copies: 100 per type) and on the
300 genuine shift statements.

### Results: test and shift (scored once)

From [`reports/week6_tamper_test.json`](reports/week6_tamper_test.json) and the per-copy
table [`reports/week6_tamper_test_copies.csv`](reports/week6_tamper_test_copies.csv), with
95% bootstrap intervals over source statements. 100 copies per type.

| Tamper type | Tampered | Tampered or suspicious |
|---|---|---|
| rerender_inconsistent | 100/100 | 100/100 |
| rerender_consistent | **0/100** | **14/100 [8%, 21%]** |
| inplace_edit | 100/100 | 100/100 |
| inplace_edit_consistent | 97/100 [93%, 100%] | 97/100 |
| page_swap | 100/100 | 100/100 |
| page_remove | 100/100 | 100/100 |
| overlay (held out) | 100/100 | 100/100 |

| Genuine PDFs | Tampered | Tampered or suspicious | Without the anomaly check |
|---|---|---|---|
| resaved_genuine (100) | 0 | 2/100 | |
| Genuine test (300) | 0/300 | 7/300 (2.3%) | 0/300 flagged |
| Genuine shift (300) | 0/300 | 5/300 (1.7%) | 0/300 flagged |

"Could not check" was 0 in every group and no check failed to run on any PDF.

**What this says.** Arithmetic and file-structure edits are caught every time, including
overlay, which the detector was never tuned on. **A fraud with correct arithmetic is mostly
not caught:** every `rerender_consistent` copy (salary raised, every balance and total
recalculated, re-rendered by the project's own renderer) got no "tampered" verdict, and only
14% were even flagged, all by the anomaly score alone. The 103 misses are those 100 copies
and 3 `inplace_edit_consistent` copies. Every flag on a genuine statement came from the
anomaly score, and none of those is "tampered".

### Known limits

- Everything here is synthetic. The copies are made by this project's own tamper code and
  the layouts by its own generator, so these rates say nothing about real forgeries.
- The anomaly threshold and the layout and font baselines came from train and dev, so dev
  and train numbers for them are in-sample; only test and shift are independent.
- Two anomaly features (`income_to_debits`, `salary_to_spending`) depend indirectly on the
  Week 3 salary rule, through the profile. The anomaly check can never give "tampered" alone.
- Rule v2 was adopted on dev with a thin margin (rerender_consistent 0/60 to 7/60), and all
  of its gain on that type comes from the in-sample anomaly threshold.
- The generator always prints every summary label, so a missing label here means it could
  not be read. Real statements may omit totals by design; that will need a per-layout list
  of expected labels.
- The shift batch's flag rate (1.7%) was not given a pre-stated number; the anomaly model
  never saw that distribution.
- `log_run` does not prevent a second MLflow entry for the same report; the committed report
  is the record of truth. The full list of limits is in the pre-registration.

### Erratum: Week 5

Week 5's report text said the test split and shift batch were "not looked at before
scoring". That was wrong for the shift batch: it was scored in Week 4. The test statements
were opened in Week 5 for the profile and label scoring. The Week 6 detector had never been
run on either, and no tampered copies of them existed before step 6. Week 5's committed
report files are unchanged.

## Service and blind test

### The service

`POST /screen` takes one PDF and returns the income profile (Week 5's `profile_pdf`), the
tamper verdict (Week 6's detector), the processing time and the code version (git commit and
detector fingerprint). `GET /health` says the service is up, whether the audit log is sound,
and the versions. `GET /audit/{request_id}` returns the audit row of one request.

> **There is no authentication. This service is for local use only.** It binds `127.0.0.1`
> by default. Anyone who can reach the port can screen PDFs, read audit rows and add rows.

**The detector is unchanged.** The service's detector code, config and model are identical
to Week 6's. What changed in the dependency files is `pyproject.toml` (the dependency lists)
and `uv.lock`, because they gained the service's dependencies: FastAPI (MIT), uvicorn
(BSD-3-Clause) and python-multipart (Apache-2.0), plus httpx and httpcore (both
BSD-3-Clause) in the dev group for the test client. The `uv.lock` diff showed no removed
line, so no package that was already locked changed version. The fingerprint's `uv_lock`
part is the only part that changed. A test checks the
code, config and model parts of the live fingerprint against Week 6's record
(`reports/week6_preregistration.json`); the `uv_lock` part cannot match any more and is
tested with a stand-in. `/health` shows the four parts:

```
code 3dd23628…  config 5b332b4b…  model d1a8aff5…  uv_lock d44f6245…
```

#### Run it locally

```bash
uv sync
uv run python -m statementscreen.service          # http://127.0.0.1:8000
curl -F "file=@statement.pdf;type=application/pdf" http://127.0.0.1:8000/screen
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/audit/<request_id>
```

| Environment variable | Default | Meaning |
|---|---|---|
| `STATEMENTSCREEN_HOST` | `127.0.0.1` | Address to bind |
| `STATEMENTSCREEN_PORT` | `8000` | Port |
| `STATEMENTSCREEN_AUDIT_DB` | `data/audit/audit.db` | The SQLite audit log (inside git-ignored `data/`) |
| `STATEMENTSCREEN_GIT_COMMIT` | from `git rev-parse HEAD` | Commit to report. `+dirty` is appended to a commit read from git when the working tree has uncommitted changes. The Docker image has no `.git`, so the commit is passed in as a build argument |

#### Run it in Docker (Colima, no Docker Desktop)

```bash
colima start --cpu 2 --memory 2
COMMIT="$(git rev-parse HEAD)$([ -n "$(git status --porcelain)" ] && echo +dirty)"
docker build -t statementscreen:w7 \
  --build-arg UV_VERSION="$(uv --version | cut -d' ' -f2)" \
  --build-arg GIT_COMMIT="$COMMIT" .
docker volume create statementscreen-audit
docker run -d --name statementscreen -p 127.0.0.1:8000:8000 \
  -v statementscreen-audit:/audit statementscreen:w7
curl http://127.0.0.1:8000/health
docker stop statementscreen && docker rm statementscreen && colima stop
```

The image is `python:3.12-slim` with `uv sync --frozen --no-dev` from `uv.lock`, the source,
and `models/week6_iforest.pkl`. It runs as a non-root user (`app`), exposes port 8000, has a
healthcheck on `/health` (503 when the audit log is not sound), and keeps the audit database
in `/audit`, which should be a volume. The `.dockerignore` keeps `data/`, `mlruns/`,
`.review/`, `.claude/`, `.git/` and `reports/` out of the build. No OpenMP runtime is
installed in the image (LightGBM would need one), and the container still screened PDFs, so
the service path does not need LightGBM's runtime. The image keeps the
repository's folder layout, because the fingerprint finds `uv.lock` and the model by their
place relative to the source. Inside the container the code, config and model hashes were
the same as Week 6's (the `/health` output is in
[`reports/week7_service_measurements.txt`](reports/week7_service_measurements.txt)). No CI
job builds the image.

#### Example

A generated, genuine, synthetic salaried statement, abridged (`…` marks cut text; hashes
are shortened). The commit shows `+dirty` because that image was built from a working tree
with uncommitted changes.

```json
{"request_id": "d884b9b844d64b0980f7b71cde8faebe", "status": "ok", "error_code": null,
 "readable": true,
 "profile": {"statement_id": "d884b9b844d64b0980f7b71cde8faebe", "profile_version": 1,
  "statement_months": 6, "period": ["2025-01-01", "2025-06-30"], "warnings": [],
  "findings": {
   "has_salary": {"value": true, "reason": "monthly NEFT credit in 6 of 6 months, amount always INR 84,944, paid on day 28-31"},
   "estimated_monthly_income_paise": {"value": 8494400, "reason": "INR 84,944 a month: salary INR 84,944 a month (6 credits over 6 months), over 6 months"},
   "customer_type": {"value": "salaried", "reason": "a monthly salary from 'kekeri infote'"},
   "emi_count": {"value": 1, "reason": "…"}, "bounced_emis": {"value": 0, "reason": "no returned EMI or charge"}}},
 "tamper": {"verdict": "no evidence of tampering", "summary": "no strong or weak finding",
  "reasons": [{"strength": "info", "code": "PRODUCER_CREATOR_DIFFER", "count": 1, "reason": "…(listed only, never counted) [not counted]"}],
  "finding_codes": [], "checks_not_run": [], "rule_version": "v2"},
 "processing_seconds": 3.656,
 "version": {"git_commit": "5f77e703f94832113f11f5bfa7c2261c5f053cdf+dirty",
  "detector_fingerprint": {"code": "3dd23628…", "config": "5b332b4b…", "model": "d1a8aff5…", "uv_lock": "d44f6245…"},
  "rule_version": "v2", "profile_version": 1}}
```

The profile's `statement_id` equals the `request_id`: the service names the stored file
after the request and never uses the client's file name. A tampered copy of the same
statement (a re-render with inconsistent arithmetic) came back as `"tampered"` with finding
codes `BALANCE_CHAIN` and `SUMMARY_IDENTITY`, and a synthetic gig-worker statement as
`"no evidence of tampering"`, customer type `gig`, no salary.

#### What is checked, and what is refused

The upload must start with `%PDF`, be at most 10 MB and at most 50 pages. It is copied to a
private temp folder (named after the request id), hashed, screened, and the folder is
deleted in every case, including errors. One PDF is copied, checked and screened at a time
(a second request waits its turn; receiving a request's body happens before that wait), and
the screening and the audit write run off the event loop. Every refusal is a JSON body with
the request id, `"status": "error"`, an error code and no stack trace, and every refusal is
recorded in the audit log:

| Case | HTTP | `error_code` |
|---|---|---|
| Not a PDF (including empty) | 400 | `NOT_A_PDF` |
| Over 10 MB | 413 | `TOO_LARGE` |
| Over 50 pages | 422 | `TOO_MANY_PAGES` |
| No `file` field | 422 | `NO_FILE` |
| Body that is not valid multipart | 400 or 422 | `BAD_REQUEST` (or `NO_FILE`) |
| A crash outside the profile and tamper parts | 500 | `INTERNAL_ERROR` |
| The audit row could not be written | 500 | `AUDIT_FAILED` (body is only `request_id` and `error_code`) |

A PDF that parses badly is not an error: the answer is 200 with `"readable": false`, the
profile carries its own "could not read the statement" warning, and the verdict follows
Week 6's rules (a lone parse failure is `suspicious`; "could not check" means no evidence
was found but a check that applies did not run, and is never reported as clean). If the
profile part or the tamper part crashes, the response is still 200 with the part that
worked, `"status": "error"` and `PROFILE_CRASHED` or `TAMPER_CRASHED` (joined with a comma if
both), the same code as in the audit row. Only the exception's class name is logged, never
its message, because a message could contain statement text.

#### The audit log

One SQLite row per request, whatever the outcome (stdlib `sqlite3`). It keeps: request id,
UTC timestamp, the PDF's SHA-256, the declared and the actual size in bytes, page count,
verdict, finding codes, has_salary, estimated monthly income (paise), customer type, EMI
count, bounce count, processing time, status (`ok` or `error`) and error code, git commit and
detector fingerprint.

**It never keeps** the PDF, its name, the account holder's name, an account number, an
employer, a counterparty or any narration. There is no column for any of them and no
free-text column at all: verdicts, customer types and error codes come from fixed sets,
finding codes must match a strict pattern, and any value that does not fit is stored as
NULL, not as text. A test builds a statement full of a made-up holder, account number,
employer and narrations and searches the raw bytes of the database file for them. Another
does the same for a generated PDF's holder name, account number, employer and narrations
after requests through the service.

The service only inserts. `BEFORE UPDATE` and `BEFORE DELETE` triggers abort anything else,
and lookups use a read-only connection. The file is mode 0600.

**It fails closed.** If the row cannot be written, the client gets `AUDIT_FAILED` and no
result, so every result a client sees has a row (the `AUDIT_FAILED` response itself has
none, because its write failed). The withheld result is not
written anywhere else, and there are no retries beyond SQLite's 5-second busy timeout.
The service refuses to start unless the database opens for writing, is at schema version 1
and has both triggers (an existing table is never repaired), and `/health` reports 503 if
that stops being true.

#### Limits of the service

- **No authentication, local use only.** Anyone who can reach the port can add junk rows.
- **One PDF at a time.** Copying, page counting and screening take turns, and other
  requests queue behind them. Receiving a request's body does not wait, so several uploads
  can be in flight at once.
- **One audit row per `POST /screen` request.** Requests to other paths (an unknown path,
  `GET /screen`) get the framework's own 404 or 405 and no row.
- **Size limit and chunked uploads.** An oversized `Content-Length` is refused before the
  body is read. A request with no `Content-Length` (a chunked upload) cannot be refused that
  way, so its bytes are counted as they arrive and it is refused with a 413 once more than
  10 MB plus 64 KiB of multipart framing have arrived. The exact 10 MB limit on the file
  itself is checked afterwards, while the file is copied. Until then FastAPI and Starlette
  receive and parse the whole body and spool each file part to a temporary file, so up to
  10 MB + 64 KiB per request can be on disk before any check on the file runs. Some clients
  (curl, for one) may show a broken pipe instead of the 413.
- **Page count.** If the page count cannot be read, the 50-page limit cannot be enforced for
  that PDF (the size limit still applies); the PDF is still screened and gets the
  unreadable-PDF answer.
- **The triggers guard against code mistakes, not against a person.** Anyone who can write
  the database file can drop a trigger. Rows are never deleted by the service and there is
  no retention policy.
- **The image is large** because the project's main dependencies (MLflow, LightGBM, Faker,
  ReportLab) are installed with it; trimming them would change `pyproject.toml` and
  `uv.lock`.
- **The tests** show a Starlette deprecation warning that its `TestClient` prefers `httpx2`.
  `httpx2` is not an approved dependency, so it is left alone.
- Everything measured here is on synthetic PDFs.

#### Speed and size

20 synthetic PDFs (15 genuine, 5 tampered copies), the same set both times (set hash
`23e66d4f…006a`), posted one at a time with `scripts/service_latency.py`, one run each.
Latency is as the client sees it. Docker ran in a Colima VM with 2 CPUs and 2 GB on an arm64
Mac; the local run was on the same Mac without Docker. All 20 requests returned 200 in both
runs, with 5 tampered and 15 "no evidence of tampering" verdicts.

| | Docker | Local |
|---|---|---|
| p50 | 2.834 s | 2.806 s |
| p95 (nearest rank) | 4.249 s | 4.093 s |
| Fastest / slowest | 2.015 s / 5.390 s | 1.928 s / 5.987 s |
| First request | 2.834 s | 2.657 s |

The two runs differ by about 1% at p50 and 4% at p95, well inside the spread between the
fastest and slowest request, so this does not show either is faster. Memory was not
measured. The image: 646,472,434 bytes by `docker image inspect` (arm64); Docker's listing
shows 646 MB content size and 2.91 GB disk usage. These were measured before the review
fixes (the lock now also covers copying and page counting, audit writes moved to worker
threads, and the request body is counted as it arrives); the screening code and the
detector did not change, and nothing was re-measured. The raw output of these commands is in
[`reports/week7_service_measurements.txt`](reports/week7_service_measurements.txt).

### The blind tamper test

**What it is.** A mix of genuine and tampered PDFs was built and sealed first, the answer
key was locked away, and the frozen Week 6 detector was run on the 200 PDFs without it.
Only after the verdicts were committed and pushed was the key opened and the verdicts
scored against it.

**The order, with each step written down before the next:**

1. The 200 PDFs were built (each source genuine with probability 0.4, otherwise one of
   seven tamper types) and `blind_seal` sealed the answer key; the key, backup and PDF-set
   hashes go into the pre-registration.
2. The pre-registration ([`reports/week7_blind_preregistration.json`](reports/week7_blind_preregistration.json):
   detector fingerprint, hashes of the code and artefacts, the commands, what to expect)
   was pushed, frozen at commit `2c129d8c…`.
3. `blind_detect` ran the detector; `blind_verdicts` wrote
   [`reports/week7_blind_verdicts.csv`](reports/week7_blind_verdicts.csv), which was committed
   and pushed **before** the key was opened (SHA-256 `01d474b7…dafb`).
4. `blind_score` opened the key once. It refused unless the pushed verdict copy, the key hash
   from the pre-registration and the PDF-set hash all matched, and wrote
   [`reports/week7_blind_report.json`](reports/week7_blind_report.json).
5. `blind_log` logged the run to MLflow (experiment `week7_blind`; `mlruns/` is git-ignored).
6. The key was published as [`reports/week7_blind_key.json`](reports/week7_blind_key.json),
   and `blind_verify` rebuilt all 200 PDFs from it. Re-run after the service changes, its
output is saved in [`reports/week7_blind_verify.txt`](reports/week7_blind_verify.txt).

The commands are in the pre-registration. The `blind_*` commands that check the
pre-registration (`blind_detect`, `blind_verdicts`, `blind_score`, `blind_log`) now refuse to
run on this branch, by design: it hashes everything under `src/statementscreen`, and the
service code and `uv.lock` changed after the run. `blind_verify` does not check it and still
works. The results are in the committed files above.

**Disclosure: the detector ran under the first pre-registration.** It ran earlier than
planned, because a chained command started it before the review of the pre-registration
push had finished. The review then led to changes, and the pre-registration was regenerated;
the committed file is the regenerated one. The two differ in text, docstrings, a default
path and tests only, and the detector fingerprint (code, config, model) is Week 6's in both.
The run was not repeated. The expectations in the pre-registration were written before any
verdict existed.

#### Results

From [`reports/week7_blind_report.json`](reports/week7_blind_report.json): 200 PDFs, 79
genuine and 121 tampered, 95% bootstrap intervals over PDFs (1,000 resamples, seed 2026).
"Could not check" was 0 and no PDF failed to build.

| Tamper type | n | Tampered | Tampered or suspicious | Week 6 test, tampered |
|---|---|---|---|---|
| rerender_inconsistent | 17 | 17 (100.0%) | 17 | 100/100 |
| inplace_edit | 21 | 21 (100.0%) | 21 | 100/100 |
| page_swap | 28 | 28 (100.0%) | 28 | 100/100 |
| page_remove | 11 | 11 (100.0%) | 11 | 100/100 |
| overlay | 15 | 15 (100.0%) | 15 | 100/100 |
| inplace_edit_consistent | 13 | 12 (92.3%) | 12 | 97/100 |
| rerender_consistent | 16 | **0 (0.0%)** | **2 (12.5%)** | 0/100 (14/100 flagged) |
| **All tampered** | 121 | **104 (86.0%) [79.3, 92.6]** | 106 (87.6%) [81.0, 93.4] | |
| **Genuine** | 79 | **0 (0.0%)** | 0 | 0/300 (7/300 flagged) |

Of the 104 PDFs called "tampered", all 104 were tampered (precision 104/104). The 17 not
called tampered are 16 `rerender_consistent` (14 not flagged, 2 `suspicious` from the
anomaly score alone) and 1 `inplace_edit_consistent`. Genuine PDFs: 0/41 flagged with the
default narrations and 0/38 with the `karan_v1` narrations. The numbers are the same with
file-history findings removed. 11 `rerender_consistent` draws had to be redrawn because the
source had no salary for the Week 3 rules to find; they are listed in the report.

**What this says.** It matches what was written down beforehand: the edit types are caught
almost every time, **a fraud with correct arithmetic is mostly not caught**
(`rerender_consistent` 0/16 tampered), and no genuine PDF was called tampered. Nothing
was tuned to these numbers, and the pre-registered expectation for genuine PDFs flagged as
"suspicious" was about 2%; here it was 0 of 79.

#### What the blind test does and does not show

- **The blind run used our own tamper code.** Every tampered PDF was made by this project's
  tamper generator, and every genuine one by its synthetic generator. No outside person
  edited a PDF. These rates say nothing about real forgeries.
- **The seal proves the key was not changed after it was sealed, not that nobody could have
  read it.**
- File sizes may differ between genuine and tampered PDFs; the seal hides the mix from the
  scoring process, not from someone inspecting the files.
- `data/blind_src/` stays on disk because the rebuild-and-verify check needs it; once the
  key is published, anyone can rebuild everything anyway. Genuine PDFs are byte copies of
  files there, so hashing a blind PDF against them identifies the genuine ones before any
  scoring. Nothing in the pipeline does this, and the seal does not prevent it.
- Per-type counts are small, so most intervals are wide, and a group where every PDF was
  caught has a zero-width bootstrap interval that is narrower than the real uncertainty.
- Genuine PDFs are not re-saved, and the other tamper types are written by pypdf, so
  file-structure cues might separate them from genuine PDFs; the report also gives the
  numbers without file-history findings, and they do not change. Re-save false alarms were
  measured in Week 6, not here.
- The anomaly threshold and the layout and font baselines came from Week 6's train and dev
  statements, so the blind run is independent of them only in the sense that its PDFs are
  new. `blind_fresh` found no seed or content shared with an earlier batch before sealing;
  its output was read by hand and is not committed, so that is stated here, not checkable
  from the repository.
- The full list of limits is in `known_limits` in the report and the pre-registration;
  Week 6's limits apply too.

## License

MIT. See [LICENSE](LICENSE).
