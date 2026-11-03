"""Generate one synthetic statement and save it as a PDF in data/samples/.

Usage:
    uv run python scripts/render_statement.py          # seed 42
    uv run python scripts/render_statement.py 7        # any other seed
"""

import sys

from statementscreen.synth import generate_statement
from statementscreen.synth.render import render_statement

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
path = render_statement(generate_statement(seed=seed), f"data/samples/statement_{seed}.pdf")
print(f"Wrote {path}")
