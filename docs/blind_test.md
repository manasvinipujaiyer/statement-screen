# The blind tamper test

Back to the [README](../README.md).

**What it is.** A mix of genuine and tampered PDFs was built and sealed first, the answer
key was locked away, and the frozen Week 6 detector was run on the 200 PDFs without it.
Only after the verdicts were committed and pushed was the key opened and the verdicts
scored against it.

**The order, with each step written down before the next:**

1. The 200 PDFs were built (each source genuine with probability 0.4, otherwise one of
   seven tamper types) and `blind_seal` sealed the answer key; the key, backup and PDF-set
   hashes go into the pre-registration.
2. The pre-registration ([`reports/week7_blind_preregistration.json`](../reports/week7_blind_preregistration.json):
   detector fingerprint, hashes of the code and artefacts, the commands, what to expect)
   was pushed, frozen at commit `2c129d8c…`.
3. `blind_detect` ran the detector; `blind_verdicts` wrote
   [`reports/week7_blind_verdicts.csv`](../reports/week7_blind_verdicts.csv), which was committed
   and pushed **before** the key was opened (SHA-256 `01d474b7…dafb`).
4. `blind_score` opened the key once. It refused unless the pushed verdict copy, the key hash
   from the pre-registration and the PDF-set hash all matched, and wrote
   [`reports/week7_blind_report.json`](../reports/week7_blind_report.json).
5. `blind_log` logged the run to MLflow (experiment `week7_blind`; `mlruns/` is git-ignored).
6. The key was published as [`reports/week7_blind_key.json`](../reports/week7_blind_key.json),
   and `blind_verify` rebuilt all 200 PDFs from it. Re-run after the service changes, its
output is saved in [`reports/week7_blind_verify.txt`](../reports/week7_blind_verify.txt).

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

## Results

From [`reports/week7_blind_report.json`](../reports/week7_blind_report.json): 200 PDFs, 79
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

## What the blind test does and does not show

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
