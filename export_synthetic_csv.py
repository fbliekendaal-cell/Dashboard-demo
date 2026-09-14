"""Export the synthetic SQLite tables to portable CSV files."""

from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "synthetic_health.db"
DATA_DIR = BASE_DIR / "data"


def export_all() -> None:
    """Export every table so the dashboard can run without SQLite."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name",
            connection,
        )["name"].tolist()
        for table_name in tables:
            frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
            frame.to_csv(DATA_DIR / f"{table_name}.csv", index=False)
            print(f"Exported {table_name}: {len(frame)} rows")


if __name__ == "__main__":
    export_all()