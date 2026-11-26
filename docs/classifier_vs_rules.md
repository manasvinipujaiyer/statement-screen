# Classifier vs rules

Back to the [README](../README.md).

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

## Results

95% intervals come from a paired bootstrap over statements (1,000 resamples, seed 42).
From [`reports/week4_classifier.json`](../reports/week4_classifier.json):

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
34.1 s, 1,502 MB peak, from fitting the TF-IDF) (from the run's own output, not committed); training 154.1 s; the comparison 2.6 s.

**Licences:** LightGBM is MIT. On macOS it needs the OpenMP runtime (`brew install
libomp`, part of LLVM, Apache-2.0 with LLVM exceptions), installed on the machine, not
added to this repo.
