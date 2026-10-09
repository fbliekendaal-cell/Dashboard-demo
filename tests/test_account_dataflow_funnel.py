import sys
from pathlib import Path
import pandas as pd
import pytest
import sqlalchemy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyses
import visualisaties
import data_ingestion


def test_analyse_account_dataflow_funnel_structuur_en_filtering(monkeypatch, tmp_path):
    monkeypatch.setattr(data_ingestion, "load_completions", lambda db_url: pd.DataFrame([
        {'participant_id': 1, 'questionnaire_id': 10},
        {'participant_id': 1, 'questionnaire_id': 10},  # participant 1 herhaald (qid 10)
        {'participant_id': 1, 'questionnaire_id': 20},  # participant 1 andere vl (qid 20)
        {'participant_id': 2, 'questionnaire_id': 10},
        {'participant_id': 2, 'questionnaire_id': 20},  # participant 2 andere vl (qid 20)
        {'participant_id': 3, 'questionnaire_id': 10},  # participant 3 alleen qid 10 1x
    ]))

    def fake_load_my_clic(db_url):
        return pd.DataFrame([
            {'participant_id': 1, 'rec_age_current': 30, 'rec_user_gender': 1, 'rec_ls_lifestyle_score': 7.5},
            {'participant_id': 2, 'rec_age_current': 40, 'rec_user_gender': 0, 'rec_ls_lifestyle_score': 8.0},
            {'participant_id': 3, 'rec_age_current': 50, 'rec_user_gender': 1, 'rec_ls_lifestyle_score': 6.0},
        ])

    monkeypatch.setattr(data_ingestion, "load_my_clic_participants_expanded", fake_load_my_clic)

    def fake_read_sql(query, engine):
        return pd.DataFrame([
            {'participant_id': 1, 'user_id': 100, 'user_name': 'Alice', 'first_name': 'Alice', 'last_name': 'A'},
            {'participant_id': 2, 'user_id': 200, 'user_name': 'Bob', 'first_name': 'Bob', 'last_name': 'B'},
            {'participant_id': 3, 'user_id': 300, 'user_name': 'Charlie', 'first_name': 'Charlie', 'last_name': 'C'},
        ])

    monkeypatch.setattr(analyses.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *_args, **_kwargs: object())

    # 1. Test zonder participant_ids filter
    funnel_df, meta = analyses.analyse_account_dataflow_funnel(tmp_path, "sqlite://")
    assert not funnel_df.empty
    assert len(funnel_df) == 8  # 5 basis + 2 parallel post-scores + 1 parallel naam

    keys = list(funnel_df['stap_key'])
    assert 'herhaald_parallel' in keys
    assert 'andere_vl_parallel' in keys

    herhaald_row = funnel_df[funnel_df['stap_key'] == 'herhaald_parallel'].iloc[0]
    andere_vl_row = funnel_df[funnel_df['stap_key'] == 'andere_vl_parallel'].iloc[0]

    assert herhaald_row['aantal'] == 1  # Alleen participant 1
    assert andere_vl_row['aantal'] == 2  # Participant 1 en 2

    # 2. Test met participant_ids filter
    funnel_df_sub, meta_sub = analyses.analyse_account_dataflow_funnel(tmp_path, "sqlite://", participant_ids=[1])
    assert not funnel_df_sub.empty
    sub_herhaald = funnel_df_sub[funnel_df_sub['stap_key'] == 'herhaald_parallel'].iloc[0]
    sub_andere = funnel_df_sub[funnel_df_sub['stap_key'] == 'andere_vl_parallel'].iloc[0]
    assert sub_herhaald['aantal'] == 1
    assert sub_andere['aantal'] == 1
    assert meta_sub['totaal_geregistreerd'] == 1


def test_maak_account_dataflow_funnel_plot_kleuren_en_stappen(monkeypatch, tmp_path):
    monkeypatch.setattr(data_ingestion, "load_completions", lambda db_url: pd.DataFrame([
        {'participant_id': 1, 'questionnaire_id': 10},
        {'participant_id': 1, 'questionnaire_id': 10},
    ]))
    monkeypatch.setattr(data_ingestion, "load_my_clic_participants_expanded", lambda db_url: pd.DataFrame([
        {'participant_id': 1, 'rec_age_current': 30, 'rec_user_gender': 1, 'rec_ls_lifestyle_score': 7.5},
    ]))
    monkeypatch.setattr(analyses.pd, "read_sql", lambda query, engine: pd.DataFrame([
        {'participant_id': 1, 'user_id': 100, 'user_name': 'Alice', 'first_name': 'Alice', 'last_name': 'A'},
    ]))
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(visualisaties, "DB_URL", "sqlite://")

    fig, funnel_df, meta = visualisaties.maak_account_dataflow_funnel_plot(tmp_path, "sqlite://")
    assert not funnel_df.empty
    assert len(fig.data[0].y) == 7
    assert len(fig.data[0].marker.color) == 7
