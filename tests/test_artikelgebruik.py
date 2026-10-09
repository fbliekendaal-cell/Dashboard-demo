from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visualisaties


def test_empty_artikel_overzicht_retourneert_kolommen(monkeypatch, tmp_path):
    monkeypatch.setattr(visualisaties, "DB_URL", "sqlite://")
    monkeypatch.setattr(visualisaties, "create_engine", lambda *_args, **_kwargs: object())

    def fake_read_sql(query, engine):
        if "SELECT * FROM interactions" in query:
            return pd.DataFrame(columns=["id", "interactable_type", "interactable_id", "user_id", "type", "created_at", "updated_at"])
        if "SELECT * FROM content_translations" in query:
            return pd.DataFrame(columns=["id", "public_id", "content_id", "locale", "title", "body"])
        if "SELECT id, email, deleted_at FROM users" in query:
            return pd.DataFrame(columns=["id", "email", "deleted_at"])
        if "SELECT id, email FROM users" in query:
            return pd.DataFrame(columns=["id", "email"])
        if "SELECT * FROM content" in query:
            return pd.DataFrame(columns=["id", "public_id"])
        return pd.DataFrame()

    monkeypatch.setattr(visualisaties.pd, "read_sql", fake_read_sql)

    top_artikelen, df_views, samenvatting = visualisaties.maak_artikel_interacties_overzicht(tmp_path)

    assert list(top_artikelen.columns) == ["interactable_id", "title", "Views", "Unieke_lezers", "Laatste_view"]
    assert df_views.empty
    assert samenvatting["Totaal views"] == 0


def test_artikel_overzicht_filtert_via_participant_id_bridge(monkeypatch, tmp_path):
    monkeypatch.setattr(visualisaties, "DB_URL", "sqlite://")
    monkeypatch.setattr(visualisaties, "create_engine", lambda *_args, **_kwargs: object())

    def fake_read_sql(query, engine):
        query_str = str(query)
        if "SELECT id, interactable_type, interactable_id, user_id, type, created_at, updated_at FROM interactions" in query_str or "SELECT * FROM interactions" in query_str:
            return pd.DataFrame([
                {"id": 1, "interactable_type": "content", "interactable_id": 100, "user_id": 1, "type": "view", "created_at": "2024-01-01", "updated_at": "2024-01-01"},
                {"id": 2, "interactable_type": "content", "interactable_id": 200, "user_id": 2, "type": "view", "created_at": "2024-01-02", "updated_at": "2024-01-02"},
            ])
        if "SELECT content_id, locale, title FROM content_translations" in query_str or "SELECT * FROM content_translations" in query_str:
            return pd.DataFrame([
                {"content_id": 100, "locale": "nl_NL", "title": "Article A"},
                {"content_id": 200, "locale": "nl_NL", "title": "Article B"},
            ])
        if "SELECT id, email, deleted_at FROM users" in query_str:
            return pd.DataFrame([
                {"id": 1, "email": "user1@example.com", "deleted_at": pd.NA},
                {"id": 2, "email": "user2@example.com", "deleted_at": pd.NA},
            ])
        if "SELECT id, email FROM users" in query_str:
            return pd.DataFrame([
                {"id": 1, "email": "user1@example.com"},
                {"id": 2, "email": "user2@example.com"},
            ])
        return pd.DataFrame()

    monkeypatch.setattr(visualisaties.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(visualisaties, "_laad_participant_user_bridge", lambda db_url=None: pd.DataFrame([{"participant_id": 16910, "user_id": 1}]))

    top_artikelen, df_views, samenvatting = visualisaties.maak_artikel_interacties_overzicht(tmp_path, user_ids=[16910])

    assert not df_views.empty
    assert samenvatting["Totaal views"] == 1
    assert top_artikelen.iloc[0]["title"] == "Article A"
    assert top_artikelen.iloc[0]["Unieke_lezers"] == 1


def test_participant_user_bridge_uses_csv_tables_in_demo_mode(monkeypatch):
    participants = pd.DataFrame({
        "id": [10, 20],
        "public_id": ["participant-10", "participant-20"],
    })
    my_clic = pd.DataFrame({
        "qe_participant_id": ["participant-10", "participant-20"],
        "user_id": [100, 200],
    })

    monkeypatch.setattr(visualisaties, "DATA_SOURCE", "csv")
    monkeypatch.setattr(
        visualisaties,
        "load_table_from_database",
        lambda table, _db_url: participants if table == "participants" else my_clic,
    )

    bridge = visualisaties._laad_participant_user_bridge("sqlite://")

    assert bridge.to_dict("records") == [
        {"participant_id": 10, "user_id": 100},
        {"participant_id": 20, "user_id": 200},
    ]
