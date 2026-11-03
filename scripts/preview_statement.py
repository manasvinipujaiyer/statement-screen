"""Print one synthetic statement to the terminal.

Usage:
    uv run python scripts/preview_statement.py          # seed 42
    uv run python scripts/preview_statement.py 7        # any other seed
"""

import sys

from statementscreen.synth import generate_statement

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
s = generate_statement(seed=seed)

print(f"{s.bank_name}  |  {s.account_holder}  {s.account_number}")
print(f"Period: {s.period_start} to {s.period_end}")
print(f"Opening balance: {s.opening_balance}\n")

for t in s.transactions:
    print(f"{t.posted}  {t.description:<58}{t.amount:>10}{t.balance:>11}  [{t.category}]")

print(f"\nClosing balance: {s.closing_balance}")
print(f"Ground truth (hidden from the PDF): {s.income}")
