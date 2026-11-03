"""Generate one synthetic statement and save it as a PDF in data/samples/.

Usage:
    uv run python scripts/render_statement.py                 # seed 42, classic layout
    uv run python scripts/render_statement.py 7 ledger         # seed 7, ledger layout
    uv run python scripts/render_statement.py 7 all            # seed 7, one PDF per layout
"""

import sys

from statementscreen.synth import generate_statement
from statementscreen.synth.render import LAYOUT_NAMES, render_statement

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
choice = sys.argv[2] if len(sys.argv) > 2 else "classic"
layouts = LAYOUT_NAMES if choice == "all" else (choice,)

statement = generate_statement(seed=seed)
for layout in layouts:
    path = render_statement(statement, f"data/samples/statement_{seed}_{layout}.pdf", layout)
    print(f"Wrote {path}")
