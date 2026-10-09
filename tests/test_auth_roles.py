"""Tests voor de is_admin() rolcheck in auth.py (zichtbaarheid beheerderspagina's, o.a. de Testpagina)."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auth


def _met_sessie(monkeypatch, username=None):
    """Vervang st.session_state door een simpele dict-omgeving met optionele gebruiker."""
    monkeypatch.setattr(auth, "st", SimpleNamespace(session_state={"auth_user": username}))


def test_auth_uit_betekent_admin_dev_mode(monkeypatch):
    """Zonder authenticatie (locale/dev-omgeving) is iedereen admin."""
    monkeypatch.setattr(auth, "load_auth_config", lambda: {"enabled": False})
    _met_sessie(monkeypatch, None)
    assert auth.is_admin() is True


def test_role_admin_geeft_toegang(monkeypatch):
    monkeypatch.setattr(auth, "load_auth_config", lambda: {"enabled": True, "credentials": {"usernames": {
        "admin": {"role": "admin"},
    }}})
    _met_sessie(monkeypatch, "admin")
    assert auth.is_admin() is True


def test_role_admin_is_hoofdletterongevoelig(monkeypatch):
    monkeypatch.setattr(auth, "load_auth_config", lambda: {"enabled": True, "credentials": {"usernames": {
        "admin": {"role": "ADMIN"},
    }}})
    _met_sessie(monkeypatch, "admin")
    assert auth.is_admin() is True


def test_admin_vlag_ook_toegestaan(monkeypatch):
    """Alternatieve notatie `admin: true` werkt ook."""
    monkeypatch.setattr(auth, "load_auth_config", lambda: {"enabled": True, "credentials": {"usernames": {
        "admin": {"admin": True},
    }}})
    _met_sessie(monkeypatch, "admin")
    assert auth.is_admin() is True


def test_geen_role_betekent_geen_admin(monkeypatch):
    monkeypatch.setattr(auth, "load_auth_config", lambda: {"enabled": True, "credentials": {"usernames": {
        "joep": {"totp_secret": "KRSXG5DSORUGK3TP"},
    }}})
    _met_sessie(monkeypatch, "joep")
    assert auth.is_admin() is False


def test_andere_role_betekent_geen_admin(monkeypatch):
    monkeypatch.setattr(auth, "load_auth_config", lambda: {"enabled": True, "credentials": {"usernames": {
        "joep": {"role": "analist"},
    }}})
    _met_sessie(monkeypatch, "joep")
    assert auth.is_admin() is False


def test_niet_ingelogd_betekent_geen_admin(monkeypatch):
    monkeypatch.setattr(auth, "load_auth_config", lambda: {"enabled": True, "credentials": {"usernames": {
        "admin": {"role": "admin"},
    }}})
    _met_sessie(monkeypatch, None)
    assert auth.is_admin() is False


def test_onbekende_gebruiker_betekent_geen_admin(monkeypatch):
    monkeypatch.setattr(auth, "load_auth_config", lambda: {"enabled": True, "credentials": {"usernames": {}}})
    _met_sessie(monkeypatch, "spook")
    assert auth.is_admin() is False


def test_fout_in_config_fail_closed(monkeypatch):
    """Bij een fout in de config behandelen we de gebruiker als niet-admin."""
    def kapot():
        raise RuntimeError("config onleesbaar")
    monkeypatch.setattr(auth, "load_auth_config", kapot)
    _met_sessie(monkeypatch, "admin")
    assert auth.is_admin() is False
