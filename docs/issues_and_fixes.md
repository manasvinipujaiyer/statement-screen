# Issues found along the way, and how they were handled

Back to the [README](../README.md).

This page lists the mistakes, corrections and setbacks recorded in this repository while the
project was built, with what was done about each and where the full record is. It adds no
result of its own: every item points to the page or committed report that holds the details.
The standing limits of the finished project are in [limits.md](limits.md); this page is about
what went wrong on the way and how it was dealt with.

## Wrong text in an earlier report

- **Week 5 report text about unseen data (an erratum).**
  - *What happened:* Week 5's report text said the test split and the shift batch were "not
    looked at before scoring". That was wrong for the shift batch, which had been scored in
    Week 4. The test statements were opened in Week 5.
  - *How it was handled:* the erratum was written down in the tamper-detection page and in
    [limits.md](limits.md#evaluation). Week 5's committed report files were not edited.
  - *Where:* [Tamper detection, erratum](tamper_detection.md#erratum-week-5).
- **The test split was described as closed after it had been opened.**
  - *What happened:* the Week 4 text said the test split was still closed. It was opened in
    Week 5 for the profile and label scoring.
  - *How it was handled:* recorded as a limit, with both pages kept as they were.
  - *Where:* [Classifier vs rules](classifier_vs_rules.md),
    [Income profile](income_profile.md), [limits.md](limits.md#evaluation).

## A number that could not be defined the first time

- **F1 for the rules on bounces, Week 4 shift batch.**
  - *What happened:* the rules flagged no bounce row on the shift batch, so their precision
    was undefined, and the first, one-time scoring printed their F1 as n/a.
  - *How it was handled:* one later re-run, labelled post-hoc, reports F1 as
    2tp / (2tp + fp + fn). It changed only how that undefined value is shown; every other
    number was identical in both runs.
  - *Where:* [Classifier vs rules](classifier_vs_rules.md#results),
    [`reports/week4_classifier.json`](../reports/week4_classifier.json).

## Results that came out worse than the tuning suggested

- **Dev overstated one pre-registered metric.** Income within 10% was higher on dev than
  anywhere in the test interval, and the model's bounce F1 fell from dev to test. Both are
  reported as they came out; dev numbers are described as in-sample throughout.
  Where: [Income profile](income_profile.md#results-dev-test-and-shift).
- **The Week 4 model learned a shortcut.** It treated a NEFT credit as salary and collapsed
  on the shift batch's unfamiliar narration formats, while the rules held up on salary. The
  shift batch was built to expose exactly this. Where:
  [Classifier vs rules](classifier_vs_rules.md#results).

## A pre-registration that was regenerated

- **The Week 7 blind detector ran under the first version of its pre-registration.**
  - *What happened:* a chained command started the detector before the review of the
    pre-registration push had finished. The review then led to changes, and the
    pre-registration was regenerated.
  - *How it was handled:* the committed file is the regenerated one. The two differ in text,
    docstrings, a default path and tests only, and the detector fingerprint (code, config,
    model) is Week 6's in both. The run was not repeated, and the disclosure was added to the
    blind-test page.
  - *Where:* [The blind test](blind_test.md), [limits.md](limits.md#the-blind-test).

## Problems found by review and fixed (Week 7 service)

- **What changed after review.** The lock now also covers copying and page counting, audit
  writes moved to worker threads, and the request body is counted as it arrives. The
  screening code and the detector did not change.
- **What that cost.** The latency and image-size measurements were taken before these fixes
  and were not repeated, and the service page says so.
- **A decision on audit failures.** If the audit row cannot be written, the client gets
  `AUDIT_FAILED` and no result, and the withheld result is not written anywhere else.
- **Where:** [The service](service.md#speed-and-size),
  [`reports/week7_service_measurements.txt`](../reports/week7_service_measurements.txt).

## Tooling and Docker

- **MLflow 3.18 refuses the plain-folder store,** so runs are logged to a local SQLite file
  (`mlruns/`, git-ignored). Where: [limits.md](limits.md#evaluation) (E14).
- **The Docker image has no `.git` folder,** so the git commit cannot be read inside the
  container and is passed in as a build argument. Where: [The service](service.md).
- **The image has no OpenMP runtime,** which LightGBM would need. The container still screened
  PDFs, so the service path does not need it. Where: [The service](service.md).
- **The image is large,** because the project's main dependencies are installed with it, and
  no CI job builds it. Where: [The service](service.md#limits-of-the-service).

## Figures that are not in a committed report

- **Some numbers come from run output that was not committed.** They are marked "from the run's
  own output, not committed" where they appear, and listed in
  [limits.md](limits.md#evaluation) (E13), so nobody mistakes them for checked figures.
- **Only the "Results at a glance" block in the README is generated and tested** against the
  reports. The result tables in the older pages were written from the reports in earlier weeks
  and are not machine-checked (E12).
