"""Detector settings, each with where it came from. Changing one is a new rule version.

Layout tolerances (approved 2026-11-23): genuine train PDFs measured 0 for every value
(reports/week6_layout_baseline.json: 1,400 statements, 1,650,636 money cells), so these
are safety margins above 0, not fitted values. The one non-zero train value, an overlap
share of 1.0 in genuine stmt_1832, is a drcr page-1 header where a long address runs
into the IFSC label; overlaps outside the transaction table are therefore weak.
"""

ALIGNMENT_TOLERANCE_PT = 0.5  # AMOUNT_ALIGNMENT: right edge off the column's usual edge
SPACING_TOLERANCE_PT = 0.5  # DIGIT_SPACING: largest minus smallest gap inside one number
OVERLAP_SHARE = 0.30  # CHAR_OVERLAP: overlap / the narrower character's width
