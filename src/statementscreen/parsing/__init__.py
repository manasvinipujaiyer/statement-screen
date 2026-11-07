"""Parse statement PDFs back into tables, from the PDF alone.

The parser never reads the generator's outputs (transactions.parquet, labels, the
manifest) or its code: everything it knows comes from the words printed on the page.
Only the separate scoring step compares the result with the ground truth.
"""
