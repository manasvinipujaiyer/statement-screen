"""Print one synthetic statement to the terminal.

Usage:
    uv run python scripts/preview_statement.py                    # seed 42, salaried, 6 months
    uv run python scripts/preview_statement.py 7 gig              # seed 7, gig worker
    uv run python scripts/preview_statement.py 7 self_employed 3  # ... over 3 months
    uv run python scripts/preview_statement.py 7 salaried 6 bonus_month,missed_salary
"""

import sys

from statementscreen.synth import generate_statement

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
customer_type = sys.argv[2] if len(sys.argv) > 2 else "salaried"
months = int(sys.argv[3]) if len(sys.argv) > 3 else 6
hard_cases = sys.argv[4].split(",") if len(sys.argv) > 4 else []
s = generate_statement(seed, customer_type, months, hard_cases=hard_cases)

print(f"{s.bank_name}  |  {s.account_holder}  {s.account_number}")
print(f"Period: {s.period_start} to {s.period_end}")
print(f"Opening balance: {s.opening_balance}\n")

for t in s.transactions:
    flag = f"  <-- {','.join(t.hard_cases)}" if t.hard_cases else ""
    print(f"{t.posted}  {t.description:<58}{t.amount:>10}{t.balance:>11}  [{t.category}]{flag}")

print(f"\nClosing balance: {s.closing_balance}   ({len(s.transactions)} transactions)")
print(f"Ground truth (hidden from the PDF): {s.income}")
if s.hard_cases:
    print("Hard cases:")
    for case, detail in s.hard_case_details.items():
        print(f"  {case:<18} {detail}")
