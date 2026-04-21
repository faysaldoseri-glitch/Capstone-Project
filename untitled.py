"""
import_history.py – Import your full SMS history into the budget database.

Usage:  python import_history.py

Reads generated_sms_data_1_year.csv (your real spending history),
parses each purchase, categorizes it using the app's own parser,
and saves everything into the database.

This gives the ML prediction model proper historical context.
"""

import pandas as pd
from pathlib import Path

from db import init_db, save_transactions, load_transactions, clear_transactions
from parser import categorize, extract_item_name, normalize_arabic
from config import ML_LABEL_MAP


def import_csv(csv_path: str = "generated_sms_data_1_year.csv", clear_first: bool = False):
    """Import the full CSV history into the database."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ File not found: {csv_path}")
        return

    # Initialize database
    init_db()

    if clear_first:
        print("🗑  Clearing existing transactions...")
        clear_transactions()

    # Check what's already in the database
    existing = load_transactions()
    existing_count = len(existing)
    print(f"📊 Current database has {existing_count} transactions")

    # Load CSV
    df = pd.read_csv(csv_path)
    purchases = df[df["SMS Type"] == "Purchase"].copy()
    print(f"📄 Found {len(purchases)} purchases in {csv_path}")

    if purchases.empty:
        print("❌ No purchases found in CSV")
        return

    # Parse dates
    purchases["date"] = pd.to_datetime(purchases["date"], format="%d-%m-%y", dayfirst=True)

    # Build transaction rows using the app's own categorization
    rows = []
    for _, row in purchases.iterrows():
        merchant = str(row.get("Merchant", "")).strip()
        body = str(row.get("body", "")).strip()
        amount = float(row["Amount (BHD)"])

        # Use the app's categorizer for consistency
        category = categorize(merchant, body)

        rows.append({
            "date": row["date"],
            "amount_bhd": round(amount, 3),
            "merchant": merchant[:40] if merchant else "Unknown",
            "category": category,
            "raw_sms": body[:200] if body else "",
            "item_name": merchant[:50] if merchant else "",
            "qty": 1,
            "source_type": "sms",
            "store_name": merchant[:40] if merchant else "",
            "currency": "BHD",
        })

    import_df = pd.DataFrame(rows)
    import_df["date"] = pd.to_datetime(import_df["date"])

    # Avoid duplicates: check if dates already exist in database
    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        existing_dates = set(existing["date"].dt.date.unique())
        import_dates = set(import_df["date"].dt.date.unique())
        overlap = existing_dates & import_dates

        if overlap:
            print(f"⚠️  Found {len(overlap)} overlapping dates with existing data")
            print(f"   Skipping rows from those dates to avoid duplicates")
            import_df = import_df[~import_df["date"].dt.date.isin(overlap)]

    if import_df.empty:
        print("ℹ️  Nothing new to import (all dates already in database)")
        return

    # Save to database
    n = save_transactions(import_df)
    print(f"✅ Imported {n} transactions into the database")

    # Summary
    final = load_transactions()
    final["date"] = pd.to_datetime(final["date"])
    monthly = final.groupby(final["date"].dt.to_period("M")).agg(
        total=("amount_bhd", "sum"),
        count=("amount_bhd", "count"),
    )
    print(f"\n📊 Database now has {len(final)} total transactions")
    print(f"\nMonthly breakdown:")
    for period, row in monthly.iterrows():
        print(f"  {period}: {row['total']:.1f} BHD ({int(row['count'])} transactions)")


if __name__ == "__main__":
    import sys

    clear = "--clear" in sys.argv
    if clear:
        confirm = input("⚠️  This will DELETE all existing transactions first. Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            sys.exit(0)

    import_csv(clear_first=clear)