# Evaluation protocol

Back to the [README](../README.md).

This page puts the evaluation rules in one place. It is a summary: every row points to the
page that explains the stage and to the committed report that holds the numbers, and it
adds no result of its own. All data is synthetic, from this project's own generator, and
none of these stages used real statements.

## Rules that apply to every stage

- **Splits are loaded by name.** The 70/15/15 train/dev/test split by statement (seed 42,
  stratified by layout and customer type) is locked in `splits/`. A loader refuses a
  manifest whose hash does not match. The test split is refused unless
  `STATEMENTSCREEN_OPEN_TEST=1` is set. See [Pattern detection](patterns.md).
- **Dev is for tuning, so dev numbers are in-sample** for anything that was tuned on it:
  thresholds, model settings, estimation variants and the tamper rule version. Dev numbers
  are never presented as independent.
- **One-time scoring.** A held-out set is scored once, after everything is frozen. Where a
  later re-run was made, it is labelled post-hoc, and the shift-batch comparison command
  refuses a second scoring. See [Classifier vs rules](classifier_vs_rules.md).
- **Pre-registration.** For the income profile, the tamper detector and the blind run, the
  frozen commit, hashes of the code and data, the commands and the expectations were
  written to a file in `reports/`, committed and pushed before the held-out data was
  scored. The scoring step refuses to run if anything differs.
- **Errata and disclosures stay in the record.** See the
  [erratum on Week 5](tamper_detection.md#erratum-week-5) and the
  [disclosure about the first pre-registration](blind_test.md).

## Stage by stage

| Stage | Data | Tuned on | Scored once | Pre-registration | Report |
|---|---|---|---|---|---|
| Parser ([parsing](parsing.md)) | The 2,000-statement reference batch | Not stated; scored on the whole reference batch | Not described as one-time | None described | `reports/week2_parse_report.json` |
| Patterns ([patterns](patterns.md)) | Split `split_v1` | Dev: grouping threshold, rule thresholds | No: dev only, test split closed at the time | None described | `reports/week3_patterns_dev.json` |
| Classifier vs rules ([page](classifier_vs_rules.md)) | Dev, and the 300-statement shift batch (narration formats written by an independent collaborator who had not seen the code) | Dev: model settings, decision thresholds | Shift batch, after all settings were frozen; one post-hoc re-run labelled as such | Not described on that page | `reports/week4_classifier.json` |
| Income profile ([page](income_profile.md)) | Dev, test split, shift batch | Dev: two estimation variants and one bug fix | Test and shift, each once | `reports/week5_preregistration.json` | `reports/week5_test_and_shift.json`; stress tests and layout hold-out (dev only): `reports/week5_stress.json` |
| Tamper detection ([page](tamper_detection.md)) | Tampered copies of dev and test statements; genuine test and shift statements; the overlay type held out | Train and dev: anomaly threshold, layout and font baselines. Dev: the verdict rule version | Test copies and genuine test and shift, once each, in order | `reports/week6_preregistration.json` | `reports/week6_tamper_test.json`, `reports/week6_tamper_test_copies.csv`; rule versions: `reports/week6_rule_versions.json` |
| Blind tamper run ([page](blind_test.md)) | 200 sealed PDFs: 79 genuine, 121 tampered | Nothing was tuned on it: the Week 6 detector was frozen | The key was opened once, after the verdicts were committed and pushed | `reports/week7_blind_preregistration.json` | `reports/week7_blind_report.json`; rebuild check: `reports/week7_blind_verify.txt` |

## What this protocol does not cover

Every tampered copy comes from this project's own tamper code, and every statement from its
own generator. The protocol keeps results from being tuned to, but it cannot show how the
code behaves on real statements or on forgeries made by someone else. See
[limits, errata and disclosures](limits.md).
