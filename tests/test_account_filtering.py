import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import account_filtering
from account_filtering import (
    filter_interne_accounts,
    interne_email_mask,
    interne_localparts,
    laad_interne_account_ids,
    reset_interne_accounts_cache,
)


USERS = [
    {"id": 1, "email": "medewerker@smarthealth.works"},
    {"id": 2, "email": "Tester@SmartHealth.NL"},
    {"id": 3, "email": "klant@example.com"},
]


def setup_function(_):
    """Zorg dat elke test met een schone cache start."""
    reset_interne_accounts_cache()


def _monkeypatch_users(monkeypatch, users_rows, engine_calls=None):
    """Patch create_engine/pd.read_sql zodat `users` fake data bevat."""

    class _FakeEngine:
        pass

    def fake_create_engine(url):
        if engine_calls is not None:
            engine_calls.append(url)
        return _FakeEngine()

    def fake_read_sql(query, engine, *args, **kwargs):
        if "FROM users" in str(query):
            return pd.DataFrame(users_rows)
        return pd.DataFrame()

    monkeypatch.setattr(account_filtering, "create_engine", fake_create_engine)
    monkeypatch.setattr(account_filtering.pd, "read_sql", fake_read_sql)


def test_interne_email_mask_herkent_domeinen_case_insensitive():
    emails = pd.Series([
        "A@smarthealth.works",
        "  b@SMARTHEALTH.nl  ",
        "c@example.com",
        pd.NA,
    ])
    mask = interne_email_mask(emails)
    assert list(mask) == [True, True, False, False]


def test_interne_localparts_geeft_unieke_namen():
    parts = interne_localparts(pd.Series([
        "x@smarthealth.nl",
        "y@smarthealth.works",
        "z@example.com",
    ]))
    assert parts == {"x", "y"}


def test_filter_op_emailkolom_verwijdert_domein_en_overeenkomende_naam(monkeypatch):
    _monkeypatch_users(monkeypatch, USERS)

    df = pd.DataFrame([
        {"email": "tester@smarthealth.nl", "waarde": 1},
        {"email": "tester@gmail.com", "waarde": 2},  # zelfde naam, ander domein -> weg
        {"email": "klant@example.com", "waarde": 3},
        {"email": pd.NA, "waarde": 4},
    ])
    uit = filter_interne_accounts(df, db_url="sqlite://")

    assert sorted(uit["waarde"]) == [3, 4]


def test_filter_op_user_id_verwijdert_interne_accounts(monkeypatch):
    _monkeypatch_users(monkeypatch, USERS)

    df = pd.DataFrame({"user_id": [1, 2, 3, 4], "score": [10, 99, 20, 30]})
    uit = filter_interne_accounts(df, db_url="sqlite://")

    assert sorted(uit["user_id"]) == [3, 4]


def test_filter_via_bridge_bij_alleen_participant_id(monkeypatch):
    _monkeypatch_users(monkeypatch, USERS)

    def fake_bridge(db_url=None):
        return pd.DataFrame([
            {"participant_id": 100, "user_id": 1},
            {"participant_id": 200, "user_id": 3},
        ])

    df = pd.DataFrame({"participant_id": [100, 200, 300], "bmi": [45.0, 24.0, 22.0]})
    uit = filter_interne_accounts(df, db_url="sqlite://", bridge_loader=fake_bridge)

    assert sorted(uit["participant_id"]) == [200, 300]


def test_zonder_herkenningskolommen_ongewijzigd_terug(monkeypatch):
    _monkeypatch_users(monkeypatch, USERS)

    df = pd.DataFrame({"foo": [1, 2]})
    uit = filter_interne_accounts(df, db_url="sqlite://")
    pd.testing.assert_frame_equal(uit, df)


def test_cache_wordt_gebruikt_tot_refresh(monkeypatch):
    calls = []
    _monkeypatch_users(monkeypatch, USERS, engine_calls=calls)

    filter_interne_accounts(pd.DataFrame({"user_id": [1]}), db_url="sqlite://")
    filter_interne_accounts(pd.DataFrame({"user_id": [1]}), db_url="sqlite://")
    assert len(calls) == 1  # tweede keer vanuit cache

    laad_interne_account_ids("sqlite://", refresh=True)
    assert len(calls) == 2


def test_leeg_of_none_dataframe_veel_afgehandeld():
    leeg = pd.DataFrame()
    assert filter_interne_accounts(leeg).empty
    assert filter_interne_accounts(None) is None