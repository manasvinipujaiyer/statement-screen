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
> independent collaborator who had not seen the code**. **The test split is still
> closed.** Nothing here says anything about real bank statements.

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

## License

MIT. See [LICENSE](LICENSE).
