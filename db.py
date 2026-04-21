"""
db.py – Database layer for the Bahrain Budget App.

Handles all SQLite operations: schema creation, transaction storage,
retrieval, and cleanup.  Every public function opens and closes its
own connection so callers never worry about leaks.
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

# ── Configuration ────────────────────────────────────────────────────
DB_PATH = Path("budget_demo.db")

SCHEMA_COLUMNS = {
    # column_name: (type_declaration, default_for_new_rows)
    "date":        ("TEXT",                None),
    "amount_bhd":  ("REAL",                None),
    "merchant":    ("TEXT",                ""),
    "category":    ("TEXT",                "أخرى"),
    "raw_sms":     ("TEXT",                ""),
    "item_name":   ("TEXT",                None),
    "qty":         ("INTEGER DEFAULT 1",   1),
    "source_type": ("TEXT DEFAULT 'sms'",  "sms"),
    "store_name":  ("TEXT",                None),
    "currency":    ("TEXT DEFAULT 'BHD'",  "BHD"),
}

ORDERED_COLS = list(SCHEMA_COLUMNS.keys())


# ── Helpers ──────────────────────────────────────────────────────────
@contextmanager
def _connect():
    """Context‑managed SQLite connection with WAL mode for performance."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _existing_columns(cursor, table: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


# ── Public API ───────────────────────────────────────────────────────
def init_db() -> None:
    """Create the transactions table (and migrate new columns if needed)."""
    with _connect() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT,
                amount_bhd  REAL,
                merchant    TEXT,
                category    TEXT,
                raw_sms     TEXT
            )
        """)

        existing = _existing_columns(cur, "transactions")
        migrate = {
            "item_name":   "TEXT",
            "qty":         "INTEGER DEFAULT 1",
            "source_type": "TEXT DEFAULT 'sms'",
            "store_name":  "TEXT",
            "currency":    "TEXT DEFAULT 'BHD'",
        }
        for col, typedef in migrate.items():
            if col not in existing:
                cur.execute(f"ALTER TABLE transactions ADD COLUMN {col} {typedef}")


def save_transactions(df: pd.DataFrame) -> int:
    """
    Append a DataFrame of transactions to the database.
    Returns the number of rows written.
    """
    if df is None or df.empty:
        return 0

    data = df.copy()

    # Ensure every expected column exists with sensible defaults
    for col, (_, default) in SCHEMA_COLUMNS.items():
        if col not in data.columns:
            data[col] = default

    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    with _connect() as conn:
        data[ORDERED_COLS].to_sql("transactions", conn, if_exists="append", index=False)

    return len(data)


def load_transactions() -> pd.DataFrame:
    """Return all transactions ordered newest‑first, with parsed dates."""
    query = f"""
        SELECT {', '.join(ORDERED_COLS)}
        FROM transactions
        ORDER BY date DESC, id DESC
    """
    with _connect() as conn:
        df = pd.read_sql_query(query, conn)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    return df


def load_transactions_with_id() -> pd.DataFrame:
    """Return all transactions with their database ID for editing."""
    query = f"""
        SELECT id, {', '.join(ORDERED_COLS)}
        FROM transactions
        ORDER BY date DESC, id DESC
    """
    with _connect() as conn:
        df = pd.read_sql_query(query, conn)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    return df


def update_transaction(row_id: int, merchant: str, category: str, item_name: str = "") -> None:
    """Update the merchant, category, and item name for a specific transaction."""
    with _connect() as conn:
        conn.execute(
            "UPDATE transactions SET merchant = ?, category = ?, item_name = ? WHERE id = ?",
            (merchant, category, item_name, row_id),
        )


def delete_transaction(row_id: int) -> None:
    """Delete a single transaction by ID."""
    with _connect() as conn:
        conn.execute("DELETE FROM transactions WHERE id = ?", (row_id,))


def clear_transactions() -> None:
    """Delete every row in the transactions table."""
    with _connect() as conn:
        conn.execute("DELETE FROM transactions")