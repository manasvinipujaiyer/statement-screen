# Income profile

Back to the [README](../README.md).

## What "monthly income" means

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
> ([`reports/week5_preregistration.json`](../reports/week5_preregistration.json)) was committed
> and pushed: the frozen commit, hashes of every source file, config, model and split, and
> the exact metrics. Real statements are out of scope for this project: the gap between
> synthetic and real statements is unmeasured, a deliberate limit.

## How the profile is built

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
truth (the Week 7 service will wrap it): 0.786 s and 254 MB peak for `stmt_0002` (from the run's own output, not committed). An
abridged profile of that synthetic dev statement (also from the run's own output, not committed):

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
kept. One dev gig statement paid weekly but skipped weeks (mean gap 9.15 days) (from the run's own output, not committed) and was
called self-employed; the weekly gap limit was raised from 9 to 10.5 days, which fixed it
without moving any self-employed statement.

## Results: dev, test and shift

From [`reports/week5_test_and_shift.json`](../reports/week5_test_and_shift.json), with 95%
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

## Stress tests and the layout hold-out (dev only)

From [`reports/week5_stress.json`](../reports/week5_stress.json). Each perturbation changes
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

## Known limits

- **Bounces in unseen formats:** the rules recognise none of the shift batch's bounce
  narrations, so bounce counts are right for only 77.33% of shift statements.
- **Employer names in unseen formats** are read through the generic fallback: 61.08% exact.
- **Date jitter** of up to 3 days costs 10.81 points of salary recall, because the salary
  rule expects most payments within 3 days of the usual day.
- **A shifted salary next to another credit** is left out by the Week 3 rule, so a few
  salaried incomes are 10-17% low.
- **Money from family** is counted as customer income for self-employed statements,
  because one-off transfers from people look the same.
