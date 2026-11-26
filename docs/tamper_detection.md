# Tamper detection

Back to the [README](../README.md).

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
in [`reports/week6_preregistration.json`](../reports/week6_preregistration.json) (detector
fingerprint, hashes, seeds, the six commands, what to expect) and pushed before the test
split's copies were built. The scoring step refuses to run if anything differs. The commands
ran once each, in order, on the 300 test statements (800 copies: 100 per type) and on the
300 genuine shift statements.

## Results: test and shift (scored once)

From [`reports/week6_tamper_test.json`](../reports/week6_tamper_test.json) and the per-copy
table [`reports/week6_tamper_test_copies.csv`](../reports/week6_tamper_test_copies.csv), with
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

## Known limits

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

## Erratum: Week 5

Week 5's report text said the test split and shift batch were "not looked at before
scoring". That was wrong for the shift batch: it was scored in Week 4. The test statements
were opened in Week 5 for the profile and label scoring. The Week 6 detector had never been
run on either, and no tampered copies of them existed before step 6. Week 5's committed
report files are unchanged.
