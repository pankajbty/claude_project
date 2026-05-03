#!/usr/bin/env python3
"""
Expense Tracker — add, list, summarize, and delete personal expenses.
Data is stored in expenses.csv in the current directory.
"""

import csv
import os
import sys
from datetime import date, datetime
from typing import Optional

DATA_FILE = "expenses.csv"
FIELDNAMES = ["id", "date", "amount", "category", "description"]

CATEGORIES = [
    "food", "transport", "housing", "utilities",
    "health", "entertainment", "shopping", "education", "other"
]


# ── Storage helpers ────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, newline="") as f:
        return list(csv.DictReader(f))


def _save(rows: list[dict]) -> None:
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _next_id(rows: list[dict]) -> int:
    return max((int(r["id"]) for r in rows), default=0) + 1


# ── Commands ──────────────────────────────────────────────────────────────────

def add(amount: float, category: str, description: str, on: Optional[str] = None) -> None:
    category = category.lower()
    if category not in CATEGORIES:
        print(f"Unknown category '{category}'. Choose from: {', '.join(CATEGORIES)}")
        sys.exit(1)

    expense_date = on or date.today().isoformat()
    try:
        datetime.strptime(expense_date, "%Y-%m-%d")
    except ValueError:
        print("Date must be YYYY-MM-DD format.")
        sys.exit(1)

    rows = _load()
    rows.append({
        "id": _next_id(rows),
        "date": expense_date,
        "amount": f"{amount:.2f}",
        "category": category,
        "description": description,
    })
    _save(rows)
    print(f"Added: {expense_date}  {category:<14}  ${amount:.2f}  —  {description}")


def list_expenses(
    category: Optional[str] = None,
    month: Optional[str] = None,   # YYYY-MM
    limit: int = 50,
) -> None:
    rows = _load()
    if not rows:
        print("No expenses recorded yet.")
        return

    if category:
        rows = [r for r in rows if r["category"] == category.lower()]
    if month:
        rows = [r for r in rows if r["date"].startswith(month)]

    rows = rows[-limit:]  # show most recent N

    print(f"\n{'ID':>4}  {'Date':<12}  {'Category':<14}  {'Amount':>8}  Description")
    print("─" * 64)
    for r in rows:
        print(f"{r['id']:>4}  {r['date']:<12}  {r['category']:<14}  ${float(r['amount']):>7.2f}  {r['description']}")
    print("─" * 64)
    total = sum(float(r["amount"]) for r in rows)
    print(f"{'Total':>33}  ${total:>7.2f}  ({len(rows)} expenses)\n")


def summary(month: Optional[str] = None) -> None:
    rows = _load()
    if not rows:
        print("No expenses recorded yet.")
        return

    label = "all time"
    if month:
        rows = [r for r in rows if r["date"].startswith(month)]
        label = month

    if not rows:
        print(f"No expenses found for {label}.")
        return

    totals: dict[str, float] = {}
    for r in rows:
        totals[r["category"]] = totals.get(r["category"], 0) + float(r["amount"])

    grand = sum(totals.values())
    print(f"\nExpense summary — {label}")
    print("─" * 38)
    for cat, amt in sorted(totals.items(), key=lambda x: -x[1]):
        bar = "█" * int(amt / grand * 20)
        print(f"  {cat:<14}  ${amt:>8.2f}  {bar}")
    print("─" * 38)
    print(f"  {'TOTAL':<14}  ${grand:>8.2f}\n")


def delete(expense_id: int) -> None:
    rows = _load()
    new_rows = [r for r in rows if int(r["id"]) != expense_id]
    if len(new_rows) == len(rows):
        print(f"No expense with id {expense_id}.")
        sys.exit(1)
    _save(new_rows)
    print(f"Deleted expense #{expense_id}.")


# ── CLI ───────────────────────────────────────────────────────────────────────

USAGE = """
Expense Tracker
───────────────
Usage:
  python expense_tracker.py add <amount> <category> "<description>" [--date YYYY-MM-DD]
  python expense_tracker.py list [--category <cat>] [--month YYYY-MM] [--limit N]
  python expense_tracker.py summary [--month YYYY-MM]
  python expense_tracker.py delete <id>
  python expense_tracker.py categories

Categories:
  food, transport, housing, utilities, health, entertainment, shopping, education, other

Examples:
  python expense_tracker.py add 12.50 food "Lunch at cafe"
  python expense_tracker.py add 45.00 transport "Uber" --date 2026-04-30
  python expense_tracker.py list --month 2026-05
  python expense_tracker.py summary --month 2026-05
  python expense_tracker.py delete 3
"""


def _get_flag(args: list[str], flag: str) -> Optional[str]:
    try:
        idx = args.index(flag)
        return args[idx + 1]
    except (ValueError, IndexError):
        return None


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return

    cmd = args[0]

    if cmd == "categories":
        print("Available categories:", ", ".join(CATEGORIES))

    elif cmd == "add":
        if len(args) < 4:
            print("Usage: add <amount> <category> \"<description>\" [--date YYYY-MM-DD]")
            sys.exit(1)
        try:
            amount = float(args[1])
        except ValueError:
            print("Amount must be a number.")
            sys.exit(1)
        add(amount, args[2], args[3], on=_get_flag(args, "--date"))

    elif cmd == "list":
        list_expenses(
            category=_get_flag(args, "--category"),
            month=_get_flag(args, "--month"),
            limit=int(_get_flag(args, "--limit") or 50),
        )

    elif cmd == "summary":
        summary(month=_get_flag(args, "--month"))

    elif cmd == "delete":
        if len(args) < 2:
            print("Usage: delete <id>")
            sys.exit(1)
        try:
            delete(int(args[1]))
        except ValueError:
            print("ID must be an integer.")
            sys.exit(1)

    else:
        print(f"Unknown command: {cmd}")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
