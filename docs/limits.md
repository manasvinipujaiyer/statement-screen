# Limits, errata and disclosures

Back to the [README](../README.md).

This page lists every limit, erratum and disclosure from Weeks 1 to 7, in four groups: data,
evaluation, the tamper detector (including the blind test) and the service. Nothing here is
softened or shortened from the page that first stated it; this page only gathers them. Each
entry says where it is explained, and names the committed report behind any figure.
Entries are numbered so the table at the end can show where each source limit went.

All data is synthetic, from this project's own generator. **No real statement has been
evaluated.** Every result in this repository is on synthetic statements, and how far they
carry over to real statements has not been measured.

## Data and the generator

- **D1. Everything is synthetic.** The banks, IFSC-style codes, UPI handles, employers and
  merchants are fictional, and every generated PDF carries a "SYNTHETIC DOCUMENT" footer.
  Real statements are used only for private evaluation, stay on one laptop and are never
  published; none has been evaluated yet. ([README](../README.md))
- **D2. The parser was scored on clean, text-based PDFs** from this project's own renderer,
  so 100% exact rows shows that the parser handles the synthetic layouts. It says nothing yet
  about real banks' statements. ([Parsing](parsing.md#week-2-reference-run),
  [`reports/week2_parse_report.json`](../reports/week2_parse_report.json))
- **D3. The drcr layout truncates long narrations when rendering,** so the full text is not
  in the PDF: 17.6% of drcr narrations are cut short by the layout (none are wrong).
  ([Parsing](parsing.md#week-2-reference-run),
  [`reports/week2_parse_report.json`](../reports/week2_parse_report.json))
- **D4. The generator has too little variety.** Every salary it writes is a NEFT credit,
  which is what the Week 4 model learned as a shortcut (see E8). The shift batch was built
  to expose this. ([Classifier vs rules](classifier_vs_rules.md#results))
- **D5. The generator always prints every summary label,** so a missing label here means it
  could not be read. Real statements may omit totals by design; that will need a per-layout
  list of expected labels. ([Tamper detection](tamper_detection.md#known-limits),
  [`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **D6. The shift batch's narration formats are unfamiliar to the code:** of the 177,425
  narrations the collaborator's lines wrote, 89.80% start with one of the code's known openings,
  but only 21.59% are fully recognised by the Week 3 cleaning. The formats were written by
  an independent collaborator who had not seen the code.
  ([Classifier vs rules](classifier_vs_rules.md),
  [`reports/week4_dialect_karan_v1.json`](../reports/week4_dialect_karan_v1.json))
- **D7. The holder's name is only recognised** where a holder slot or a "self" note shows it.
  ([Pattern detection](patterns.md#week-3-dev-results))
- **D8. Employer names cut off in every row** (for example `shakapu technol`): when the drcr
  layout cuts every narration of an employer at the same point, the full name is not in the
  PDF. ([Pattern detection](patterns.md#week-3-dev-results))

## Evaluation

- **E1. Dev was used for tuning, so dev numbers are in-sample** for what was tuned on it: the
  Week 3 grouping and rule thresholds, the Week 4 model settings and decision thresholds, two
  Week 5 estimation variants and one bug fix, and the Week 6 verdict rule version. Dev results
  are never presented as independent. ([Evaluation protocol](evaluation_protocol.md),
  [`reports/week4_classifier.json`](../reports/week4_classifier.json),
  [`reports/week5_test_and_shift.json`](../reports/week5_test_and_shift.json))
- **E2. The Week 3 results are synthetic dev results:** from the 300 dev statements, the split
  every threshold was tuned on. The test set was still closed, so they are not a held-out
  estimate and say nothing about real statements.
  ([Pattern detection](patterns.md#week-3-dev-results),
  [`reports/week3_patterns_dev.json`](../reports/week3_patterns_dev.json))
- **E3. The test split was opened in Week 5** for the profile and label scoring, after the
  Week 4 text said it was still closed. ([Classifier vs rules](classifier_vs_rules.md),
  [Income profile](income_profile.md))
- **E4. Erratum, Week 5.** Week 5's report text said the test split and shift batch were "not
  looked at before scoring". That was wrong for the shift batch: it was scored in Week 4. The
  test statements were opened in Week 5 for the profile and label scoring. The Week 6
  detector had never been run on either, and no tampered copies of them existed before step
  6. Week 5's committed report files are unchanged.
  ([Tamper detection](tamper_detection.md#erratum-week-5))
- **E5. The shift batch was scored once, after every setting was frozen; one later re-run is
  post-hoc.** It only changed how an undefined F1 is reported: the rules flag no bounce row on
  the shift batch (0 of 172 found), so their precision is undefined; the first scoring printed
  their F1 and drop as n/a and the post-hoc re-run reports F1 as 2tp / (2tp + fp + fn) = 0.
  Every other number was identical in both runs.
  ([Classifier vs rules](classifier_vs_rules.md#results),
  [`reports/week4_classifier.json`](../reports/week4_classifier.json))
- **E6. Dev overstated one pre-registered metric.** Income within 10% was 99.00% on dev,
  above the whole test interval (97.00%, [95.00%, 98.67%]). Every other dev value is inside
  the test interval. The model's bounce F1 also fell, from 92.05% on dev to 82.00% on test.
  ([Income profile](income_profile.md#results-dev-test-and-shift),
  [`reports/week5_test_and_shift.json`](../reports/week5_test_and_shift.json))
- **E7. The stress tests and the layout hold-out are dev only.** The shift batch's label
  metrics repeat Week 4's frozen models.
  ([Income profile](income_profile.md#stress-tests-and-the-layout-hold-out-dev-only),
  [`reports/week5_stress.json`](../reports/week5_stress.json))
- **E8. On unseen formats the Week 4 model collapses on salary.** Shift F1 was 0.3890 for the
  model against 0.9991 for the rules, and on self-employed statements the model flagged false
  salary on 57 of 59 where the rules flagged 0. It had learned that a NEFT credit means
  salary, shortcut learning from a generator with too little variety (D4).
  ([Classifier vs rules](classifier_vs_rules.md#results),
  [`reports/week4_classifier.json`](../reports/week4_classifier.json))
- **E9. The rules fail where they depend on exact formats or on reading a name.** They
  recognise none of the shift batch's bounce narrations, so bounce counts are right for only
  77.33% of shift statements, and employer exact match fell from 95.68% to 61.08%.
  ([Income profile](income_profile.md#known-limits),
  [`reports/week5_test_and_shift.json`](../reports/week5_test_and_shift.json))
- **E10. Other income-profile limits.** Date jitter of up to 3 days costs 10.81 points of
  salary recall ([`reports/week5_stress.json`](../reports/week5_stress.json)), because the
  salary rule expects most payments within 3 days of the usual day. A shifted salary next to
  another credit is left out by the Week 3 rule, so a few salaried incomes are 10-17% low (from the run's own output, not committed).
  Money from family is counted as customer income for self-employed statements, because
  one-off transfers from people look the same.
  ([Income profile](income_profile.md#known-limits))
- **E11. Week 3 dev misses.** 4 salary rows were left out as "an extra credit in its month",
  the cost of not counting a bonus equal to the salary; one rent group paid in 4 of 7 months
  (57%) stays below the 60% coverage rule; employer recall in grouping is 94.48%, because an
  employer's old and new names stay in separate groups and the salary rule joins them, not
  the grouping. ([Pattern detection](patterns.md#week-3-dev-results),
  [`reports/week3_patterns_dev.json`](../reports/week3_patterns_dev.json))
- **E12. Only the "Results at a glance" block in the README is generated and tested** against
  the reports. The result tables in these pages were written from the reports in earlier weeks
  and moved word for word; they are not machine-checked.
- **E13. Some figures come from run output that was not committed.** They are marked "from
  the run's own output, not committed" where they appear: the train figures for pattern
  detection ([Pattern detection](patterns.md#week-3-dev-results)), the feature-building
  figures ([Classifier vs rules](classifier_vs_rules.md#results)), the `stmt_0002` timing,
  the abridged profile and the 9.15-day mean gap ([Income profile](income_profile.md)), and
  the abridged service example ([The service](service.md#example)).
- **E14. Licences and tooling.** MLflow brings in three dependencies that are not MIT, BSD or
  Apache: certifi (MPL-2.0, a weak copyleft licence that only applies to changes to
  certifi's own files), and matplotlib and typing-extensions (permissive, PSF-based licences);
  all three are used unmodified. LightGBM is MIT and needs the OpenMP runtime on macOS
  (`brew install libomp`, Apache-2.0 with LLVM exceptions), installed on the machine, not
  added to this repo. MLflow 3.18 refuses the plain-folder store, so runs are logged to a
  local SQLite file (`mlruns/`, git-ignored). The service adds FastAPI (MIT), uvicorn
  (BSD-3-Clause) and python-multipart (Apache-2.0), and httpx and httpcore (BSD-3-Clause) in
  the dev group. ([Pattern detection](patterns.md), [Classifier vs rules](classifier_vs_rules.md),
  [The service](service.md))
- **E15. `log_run` does not prevent a second MLflow entry for the same report;** the committed
  report is the record of truth. ([Tamper detection](tamper_detection.md#known-limits),
  [`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **E16. Real statements are not yet evaluated.** See the top of this page.

## Tamper detection

- **T1. Everything here is synthetic.** The copies are made by this project's own tamper code
  and the layouts by its own generator, so these rates say nothing about real forgeries.
  ([Tamper detection](tamper_detection.md#known-limits))
- **T2. A tampered copy with correct arithmetic is mostly not caught.** Every `rerender_consistent`
  copy (salary raised, every balance and total recalculated, re-rendered by the project's own
  renderer) got no "tampered" verdict in the Week 6 test (0/100 tampered, 14/100 flagged, all
  by the anomaly score alone), and 0/16 were called tampered on the blind run (2/16 flagged).
  The 103 misses in the Week 6 test are those 100 copies and 3 `inplace_edit_consistent`
  copies. ([Tamper detection](tamper_detection.md#results-test-and-shift-scored-once),
  [`reports/week6_tamper_test.json`](../reports/week6_tamper_test.json),
  [`reports/week7_blind_report.json`](../reports/week7_blind_report.json))
- **T3. In-sample baselines.** The anomaly threshold and the layout and font baselines came
  from train and dev, so dev and train numbers for them are in-sample; only test and shift
  are independent. The anomaly check's false-alarm rate on genuine dev and train is in-sample
  (its threshold and model came from those statements), and the layout tolerances and font
  baseline were measured on genuine train, so the zero strong findings on genuine dev and
  train are not independent evidence.
  ([`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T4. Two anomaly features** (`income_to_debits`, `salary_to_spending`) depend indirectly on
  the Week 3 salary rule, through the profile. The anomaly check can never give "tampered"
  alone. ([Tamper detection](tamper_detection.md#known-limits))
- **T5. Rule v2 was adopted on dev with a thin margin:** `rerender_consistent` rose from 0/60
  to 7/60, 11.7 points against 10 needed, and all of v2's gain on that type comes from the
  anomaly score alone, whose threshold is in-sample on dev. Even so, 53/60 `rerender_consistent`
  dev copies get no flag, and the 6/300 suspicious on genuine dev is the anomaly threshold's
  2% by construction, not an independent estimate.
  ([`reports/week6_rule_versions.json`](../reports/week6_rule_versions.json),
  [`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T6. The shift batch's flag rate (1.7%) was not given a pre-stated number;** the anomaly
  model never saw that distribution. ([Tamper detection](tamper_detection.md#known-limits),
  [`reports/week6_tamper_test.json`](../reports/week6_tamper_test.json))
- **T7. The in-table character-overlap maximum on genuine PDFs was never recorded;** only that
  no genuine dev or train PDF produced a strong finding, so the margin to the 0.30 threshold
  is not shown. This measurement cannot change a verdict.
  ([`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T8. There is no test for a header mismatch on the drcr layout** (only on columns). A
  missing test cannot change a verdict.
  ([`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T9. Populations are identified by statement id and by the PDF hash recorded by each run,
  not checked against the generator;** shift PDFs copied to another folder would be caught
  only through their overlap with the test ids.
  ([`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T10. The rebuild check covered only the 300 dev statements;** it was never run on the test
  statements, so a test source may fail to build and be replaced (recorded, at most 5).
  ([`reports/week6_rebuild_check_dev.json`](../reports/week6_rebuild_check_dev.json),
  [`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T11. Nothing at step 6 can be hashed before it exists:** a deleted and rebuilt
  `data/tamper/test` is not detected before scoring, and a build that fails and is re-run
  once uses the same seed.
  ([`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T12. The stand-ins in the test suite** (the remote branch, the upstream commit) mean the
  full remote flow is exercised only on a throwaway repository, not end to end.
  ([`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T13. The pooled all-six-types rate hides the `rerender_consistent` gap and is not a
  headline.** Dev copies are 60 per type and test is 100 per type, so test intervals are
  narrower. ([`reports/week6_preregistration.json`](../reports/week6_preregistration.json))
- **T14. Verdict rule details.** Info findings are listed and never counted; weak findings
  count by distinct code, not by instance; `PARSE_FAILED` stays a weak finding, and the third
  rule is what makes it suspicious. On dev, one `inplace_edit_consistent` miss was an edit on
  the last page of a multi-page statement saved incrementally (only `INCREMENTAL_UPDATE`
  fired), and one `resaved_genuine` false alarm was a statement the anomaly check flags,
  re-saved incrementally. ([`reports/week6_rule_versions.json`](../reports/week6_rule_versions.json))
- **T15. The anomaly check's false-alarm rate on genuine dev is in-sample:** its threshold was
  set on these statements, so about 2% of them exceed it by construction. Under a rule that
  flags an anomaly score alone, that 2% is the "suspicious" rate on genuine dev; it is not
  an independent estimate.
  ([`reports/week6_rule_comparison_dev.json`](../reports/week6_rule_comparison_dev.json))

### The blind test

- **B1. Everything is synthetic, and every tampered PDF comes from this project's own tamper
  code.** The blind run used our own tamper code. No outside person edited a PDF.
  ([The blind test](blind_test.md#what-the-blind-test-does-and-does-not-show))
- **B2. The seal proves the key was not changed after it was sealed, not that nobody could
  have read it.** ([The blind test](blind_test.md#what-the-blind-test-does-and-does-not-show))
- **B3. File sizes may differ between genuine and tampered PDFs; the seal hides the mix from
  the scoring process, not from someone inspecting the files.**
  ([The blind test](blind_test.md#what-the-blind-test-does-and-does-not-show))
- **B4. `data/blind_src/` stays on disk because the rebuild-and-verify check needs it; once
  the key is published, anyone can rebuild everything anyway.** A genuine PDF is a byte copy
  of a file there, so hashing a blind PDF against those files identifies the genuine ones
  before any scoring. Nothing in the pipeline does this, and the seal does not prevent it.
  ([The blind test](blind_test.md#what-the-blind-test-does-and-does-not-show))
- **B5. Genuine PDFs are byte copies of their sources and are not re-saved,** because the two
  re-render types are not written by pypdf, so no re-save step could hide the difference from
  every tampered copy. The other tampered types are written by pypdf, so file-structure cues
  may separate them from genuine PDFs; the report gives the numbers without file-history
  findings, and they do not change. Re-save false alarms are not measured here (Week 6
  measured them). ([`reports/week7_blind_report.json`](../reports/week7_blind_report.json))
- **B6. A source that cannot take its drawn type gets another type,** so the seven types are
  not exactly uniform: `rerender_consistent`, which needs a salary, is probably the rarest.
  11 `rerender_consistent` draws were redrawn because the source had no salary for the Week 3
  rules to find; they are listed in the report.
  ([`reports/week7_blind_report.json`](../reports/week7_blind_report.json))
- **B7. Per-type counts are small,** so most intervals are wide, and a group that is all hits
  has a zero-width bootstrap interval that is narrower than the real uncertainty.
  ([The blind test](blind_test.md#what-the-blind-test-does-and-does-not-show))
- **B8. The bootstrap resamples PDFs.** A `page_swap` donor is another blind source, so a
  donor's own PDF and a swapped copy are not strictly independent.
  ([`reports/week7_blind_report.json`](../reports/week7_blind_report.json))
- **B9. A lone anomaly score gives "suspicious", never "tampered"** (verdict rule v2), and two
  anomaly features depend indirectly on the Week 3 salary rule (Week 6 limits).
  ([`reports/week7_blind_report.json`](../reports/week7_blind_report.json))
- **B10. The anomaly threshold and the layout and font baselines came from Week 6's train and
  dev statements,** so the blind run is independent of them only in the sense that its PDFs
  are new. Before sealing, `blind_fresh` (a read-only audit that reads the earlier batches
  under `data/` and never the key) found no seed or content shared between the blind sources
  and any earlier batch; its output was read by hand and is not committed, so this is stated
  here, not checkable from the repository.
  ([The blind test](blind_test.md#what-the-blind-test-does-and-does-not-show))
- **B11. `blind_log` does not prevent a second MLflow entry for the same report;** the
  committed report is the record of truth (see E15).
- **B12. Week 6's other known limits apply** (T1 to T15), and the full lists are `known_limits`
  in [`reports/week7_blind_report.json`](../reports/week7_blind_report.json) and
  [`reports/week7_blind_preregistration.json`](../reports/week7_blind_preregistration.json).
- **B13. Disclosure: the detector ran under the first pre-registration.** It ran earlier than
  planned, because a chained command started it before the review of the pre-registration
  push had finished. The review then led to changes, and the pre-registration was regenerated;
  the committed file is the regenerated one. The two differ in text, docstrings, a default
  path and tests only, and the detector fingerprint (code, config, model) is Week 6's in both.
  The run was not repeated. The expectations in the pre-registration were written before any
  verdict existed. ([The blind test](blind_test.md))
- **B14. The `blind_*` commands that check the pre-registration** (`blind_detect`,
  `blind_verdicts`, `blind_score`, `blind_log`) now refuse to run on this branch, by design:
  it hashes everything under `src/statementscreen`, and the service code and `uv.lock`
  changed after the run. `blind_verify` does not check it and still works. The results are in
  the committed files. ([The blind test](blind_test.md))
- **B15. The blind sources were generated from seeds 7101 (default narrations) and 7102
  (`karan_v1`).** The by-hand comparison of the generator's output at Week 7 step 2 (two runs,
  identical) is not recorded in the repository, and `blind_verify` does not regenerate the
  sources: it checks them against the key's hashes. (The module notes of
  `src/statementscreen/tamper/blind_verify.py`)

## The service

- **S1. No authentication, local use only.** It binds `127.0.0.1` by default. Anyone who can
  reach the port can screen PDFs, read audit rows and add junk rows.
  ([The service](service.md#limits-of-the-service))
- **S2. One PDF at a time.** Copying, page counting and screening take turns, and other
  requests queue behind them. Receiving a request's body does not wait, so several uploads
  can be in flight at once. ([The service](service.md#limits-of-the-service))
- **S3. One audit row per `POST /screen` request.** Requests to other paths (an unknown path,
  `GET /screen`) get the framework's own 404 or 405 and no row. The `AUDIT_FAILED` response
  itself has no row, because its write failed.
  ([The service](service.md#limits-of-the-service))
- **S4. Size limit and chunked uploads.** An oversized `Content-Length` is refused before the
  body is read. A request with no `Content-Length` (a chunked upload) cannot be refused that
  way, so its bytes are counted as they arrive and it is refused with a 413 once more than
  10 MB plus 64 KiB of multipart framing have arrived. The exact 10 MB
  limit on the file itself is checked afterwards, while the file is copied. Until then FastAPI
  and Starlette receive and parse the whole body and spool each file part to a temporary
  file, so up to 10 MB + 64 KiB per request can be on disk before any check on the file runs.
  Some clients (curl, for one) may show a broken pipe instead of the 413.
  ([The service](service.md#limits-of-the-service))
- **S5. Page count.** If the page count cannot be read, the 50-page limit cannot be enforced
  for that PDF (the size limit still applies); the PDF is still screened and gets the
  unreadable-PDF answer. ([The service](service.md#limits-of-the-service))
- **S6. The audit triggers guard against code mistakes, not against a person.** Anyone who can
  write the database file can drop a trigger. Rows are never deleted by the service and there
  is no retention policy. ([The service](service.md#limits-of-the-service))
- **S7. The image is large** because the project's main dependencies (MLflow, LightGBM, Faker,
  ReportLab) are installed with it; trimming them would change `pyproject.toml` and
  `uv.lock`. No CI job builds the image. ([The service](service.md))
- **S8. The tests show a Starlette deprecation warning** that its `TestClient` prefers
  `httpx2`. `httpx2` is not an approved dependency, so it is left alone.
  ([The service](service.md#limits-of-the-service))
- **S9. Speed and size were measured on synthetic PDFs, one run each,** before the review fixes
  (the lock now also covers copying and page counting, audit writes moved to worker threads,
  and the request body is counted as it arrives); the screening code and the detector did not
  change, and nothing was re-measured. Memory was not measured. The Docker image was built
  from a working tree with uncommitted changes, so its recorded commit ends in `+dirty`.
  ([The service](service.md#speed-and-size),
  [`reports/week7_service_measurements.txt`](../reports/week7_service_measurements.txt))
- **S10. The detector is unchanged:** the service's detector code, config and model are
  identical to Week 6's. What changed in the dependency files is `pyproject.toml` (the
  dependency lists) and `uv.lock`; the fingerprint's `uv_lock` part is the only part that
  changed. ([The service](service.md))

## Where each source limit went

| Source | Entries |
|---|---|
| [Parsing](parsing.md): the 100% note and the drcr truncation | D2, D3 |
| [Pattern detection](patterns.md): the dev-results note, the five known limits, licences, MLflow | E2, E11, D7, D8, E14 |
| [Classifier vs rules](classifier_vs_rules.md): the what-numbers-are note, the footnote, what it shows, licences | E1, E3, E5, E8, E9, D4, D6, E14 |
| [Income profile](income_profile.md): the what-numbers-are note, the dev-overstated paragraph, the stress note, the five known limits | E1, E6, E7, E9, E10 |
| [Tamper detection](tamper_detection.md): the seven known limits, the erratum | T1, T3, T4, T5, T6, D5, E15, E4 |
| [The service](service.md): the nine limits, the detector-unchanged paragraph, the speed note | S1 to S10 |
| [The blind test](blind_test.md): the eight limits, the disclosure, the refusing commands | B1 to B15 |
| `reports/week4_classifier.json` notes, `reports/week5_test_and_shift.json` notes | E1, E3, E5, E7, D1, D6 |
| `reports/week6_preregistration.json` known_limits, items 1 to 15 | T3 (1, 5), T5 (2, 3), T4 (4), T7 (6), T8 (7), T9 (8), T10 (9), T11 (10), T12 (11), E15 (12), D5 (13), T13 (14, 15) |
| `reports/week6_rule_versions.json` and `reports/week6_rule_comparison_dev.json` notes | T14, T5, T15 |
| `reports/week7_blind_report.json` and `reports/week7_blind_preregistration.json` known_limits, items 1 to 13 | B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B4, B11, B12 |
| Added this week | E12, E13 |
