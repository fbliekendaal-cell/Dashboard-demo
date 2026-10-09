import sqlite3
from pathlib import Path

import pandas as pd


DEMO_DIR = Path(__file__).resolve().parents[1]
DB_PATH = DEMO_DIR / "synthetic_health.db"
CSV_DIR = DEMO_DIR / "data"


def test_demo_csv_exports_match_synthetic_sqlite_tables():
    with sqlite3.connect(DB_PATH) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        for (table_name,) in tables:
            csv_path = CSV_DIR / f"{table_name}.csv"
            assert csv_path.is_file(), f"CSV export ontbreekt voor tabel {table_name}"

            frame = pd.read_csv(csv_path)
            db_columns = [
                row[1]
                for row in connection.execute(f'PRAGMA table_info("{table_name}")')
            ]
            db_rows = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            assert list(frame.columns) == db_columns, (
                f"Kolommen in {csv_path.name} lopen niet gelijk met SQLite"
            )
            assert len(frame) == db_rows, (
                f"Aantal rijen in {csv_path.name} loopt niet gelijk met SQLite"
            )

    participants = pd.read_csv(CSV_DIR / "users_met_scores.csv")
    lifestyle_scores = pd.to_numeric(
        participants["rec_ls_lifestyle_score"], errors="coerce"
    )
    assert lifestyle_scores.notna().sum() >= 80
    assert lifestyle_scores.isna().any()

    orders = pd.read_csv(CSV_DIR / "orders.csv")
    assert "paid" in set(orders["status"].dropna())

    factors = pd.read_csv(CSV_DIR / "questionnaire_factors.csv")
    lifestyle_factor_ids = factors.loc[factors["slug"] == "lifestyle", "id"]
    assert not lifestyle_factor_ids.empty
    histories = pd.read_csv(CSV_DIR / "factor_score_histories.csv")
    assert histories["questionnaire_factor_id"].isin(lifestyle_factor_ids).any()
