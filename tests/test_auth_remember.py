"""Tests voor het vertrouwd-apparaat-token (onthoud 2FA) in auth.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auth


def test_token_roundtrip():
    token = auth._maak_vertrouw_token("admin")
    assert auth._verifieer_vertrouw_token(token) == "admin"


def test_provisioning_uri_bevat_secret_en_issuer(monkeypatch):
    monkeypatch.setattr(
        auth,
        "load_auth_config",
        lambda: {"enabled": True, "credentials": {"usernames": {
            "admin": {"totp_secret": "JBSWY3DPEHPK3PXP"},
        }}},
    )
    uri = auth.geef_totp_provisioning_uri("admin")
    assert uri and uri.startswith("otpauth://totp/")
    assert "JBSWY3DPEHPK3PXP" in uri
    assert "Smart%20Health%20Dashboard" in uri


def test_provisioning_uri_none_zonder_secret(monkeypatch):
    monkeypatch.setattr(
        auth,
        "load_auth_config",
        lambda: {"enabled": True, "credentials": {"usernames": {
            "admin": {},
        }}},
    )
    assert auth.geef_totp_provisioning_uri("admin") is None


def test_getampererd_token_wordt_geweigerd():
    token = auth._maak_vertrouw_token("admin")
    b64, sig = token.rsplit(".", 1)
    # payload wijzigen maar handtekening behouden
    nep = b64[:-2] + "XY"
    assert auth._verifieer_vertrouw_token(f"{nep}.{sig}") is None
    # handtekening wijzigen
    assert auth._verifieer_vertrouw_token(f"{b64}.{'0' * len(sig)}") is None
    # rommel input
    assert auth._verifieer_vertrouw_token("onzin") is None
    assert auth._verifieer_vertrouw_token("") is None


def test_verlopen_token_wordt_geweigerd(monkeypatch):
    token = auth._maak_vertrouw_token("admin")
    # tijd verder zetten voorbij de trust-periode
    echte_time = auth.time.time
    monkeypatch.setattr(auth.time, "time", lambda: echte_time() + auth.AUTH_TRUST_DAYS * 86400 + 10)
    assert auth._verifieer_vertrouw_token(token) is None


def test_lege_username_wordt_geweigerd():
    token = auth._maak_vertrouw_token("")
    assert auth._verifieer_vertrouw_token(token) is None


def test_token_is_gedragen_voor_anderen_account():
    """Token van gebruiker A mag niet als B valideren — username zit in payload."""
    token_a = auth._maak_vertrouw_token("admin")
    assert auth._verifieer_vertrouw_token(token_a) == "admin"
    # een ander account kan dezelfde payload niet forgen zonder sleutel:
    # controleer dat signature check de payload bindt
    b64, sig = token_a.rsplit(".", 1)
    import base64, json
    padding = "=" * (-len(b64) % 4)
    data = json.loads(base64.urlsafe_b64decode(b64 + padding))
    data["u"] = "kwaadwillende"
    nieuwe_payload = json.dumps(data, separators=(",", ":"))
    nieuwe_b64 = base64.urlsafe_b64encode(nieuwe_payload.encode()).decode().rstrip("=")
    assert auth._verifieer_vertrouw_token(f"{nieuwe_b64}.{sig}") is None
