from __future__ import annotations
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime
from pathlib import Path

import pyotp
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml

from config import (
    AUTH_CONFIG_FILE,
    AUTH_COOKIE_EXPIRY_DAYS,
    AUTH_COOKIE_KEY,
    AUTH_ENABLED_ENV,
    AUTH_LOG_FILE,
    AUTH_RATE_LIMIT_MAX_ATTEMPTS,
    AUTH_RATE_LIMIT_PERIOD_MINUTES,
    AUTH_SESSION_TIMEOUT_MINUTES,
    AUTH_TRUST_DAYS,
    setup_logger,
)


auth_logger = setup_logger("auth")
if not auth_logger.handlers:
    file_handler = logging.FileHandler(AUTH_LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    auth_logger.addHandler(file_handler)
    auth_logger.setLevel(logging.INFO)


def _env_flag(env_name: str) -> bool:
    return os.getenv(env_name, "").strip().lower() in {"1", "true", "yes", "on"}


# ── Vertrouwd-apparaat cookie ("onthoud 2FA") ─────────────────────────────────
# Probleem: Streamlit start bij elke paginaherlading een nieuwe sessie, waardoor
# auth_ok/auth_2fa_verified uit session_state verdwijnen en de gebruiker
# wachtwoord + 2FA opnieuw moet invullen.
# Oplossing: na succesvolle 2FA een HMAC-ondertekend cookie zetten. Bij een
# nieuwe sessie herstelt require_login() de sessie vanuit dit cookie, tot
# AUTH_TRUST_DAYS voorbij is (sliding renewal halverwege de periode).

_TRUST_COOKIE_NAME = "sh_remember_2fa"
_EPHEMERAL_SECRET: bytes | None = None


def _trust_secret() -> bytes:
    """Ondertekeningssleutel: expliciet geconfigureerde key, anders een per-proces willekeurige sleutel."""
    global _EPHEMERAL_SECRET
    key = (AUTH_COOKIE_KEY or "").strip()
    if not key:
        try:
            key = str(load_auth_config().get("cookie", {}).get("key") or "").strip()
        except Exception:
            key = ""
    if not key or key == "VERANDER_DIT_NAAR_EEN_LANG_WILLEKEURIG_GEHEIM":
        # Geen echte key geconfigureerd: gebruik een willekeurig geheim dat
        # per serverproces geldt (cookies overleven een herstart dan niet).
        if _EPHEMERAL_SECRET is None:
            _EPHEMERAL_SECRET = secrets.token_hex(32).encode()
            auth_logger.warning(
                "AUTH_COOKIE_KEY is niet (goed) geconfigureerd; vertrouwde "
                "apparaten verliezen hun onthouden login bij een serverherstart."
            )
        return _EPHEMERAL_SECRET
    return key.encode()


def _maak_vertrouw_token(username: str) -> str:
    nu = int(time.time())
    exp = nu + int(AUTH_TRUST_DAYS * 86400)
    payload = json.dumps({"u": username, "iat": nu, "exp": exp}, separators=(",", ":"))
    b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    sig = hmac.new(_trust_secret(), b64.encode(), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _verifieer_vertrouw_token(token: str) -> str | None:
    """Geef de username terug als het token geldig en niet verlopen is, anders None."""
    try:
        b64, sig = token.rsplit(".", 1)
        verwacht = hmac.new(_trust_secret(), b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, verwacht):
            return None
        padding = "=" * (-len(b64) % 4)
        data = json.loads(base64.urlsafe_b64decode(b64 + padding))
        if int(data.get("exp", 0)) < time.time():
            return None
        gebruiker = str(data.get("u", "")).strip()
        return gebruiker or None
    except Exception:
        return None


def _lees_cookie(naam: str) -> str | None:
    try:
        return st.context.cookies.get(naam)
    except Exception:
        return None


def _zet_cookie(naam: str, waarde: str, max_age_s: int) -> None:
    """
    Zet (of wis met max_age_s=0) een browsercookie. Streamlit biedt geen
    server-side cookie-API, dus dit loopt via een klein same-origin scriptje.
    """
    cookie_str = f"{naam}={waarde}; path=/; max-age={int(max_age_s)}; SameSite=Lax"
    components.html(
        "<script>document.cookie=" + json.dumps(cookie_str) + ";</script>",
        height=0,
    )


def _verleng_vertrouw_cookie_indien_nodig(username: str) -> None:
    """Sliding renewal: vernieuw zodra minder dan de helft van de periode resteert."""
    try:
        token = _lees_cookie(_TRUST_COOKIE_NAME) or ""
        b64, _sig = token.rsplit(".", 1)
        padding = "=" * (-len(b64) % 4)
        data = json.loads(base64.urlsafe_b64decode(b64 + padding))
        rest = int(data.get("exp", 0)) - time.time()
        if rest < AUTH_TRUST_DAYS * 86400 / 2:
            _zet_cookie(_TRUST_COOKIE_NAME, _maak_vertrouw_token(username), int(AUTH_TRUST_DAYS * 86400))
    except Exception:
        pass


def _load_yaml_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        auth_logger.error(f"Kan auth-config niet laden: {exc}")
        return {}


def generate_bcrypt_password_hashes(passwords: list[str]) -> list[str]:
    """Generate bcrypt hashes for streamlit-authenticator credentials."""
    return stauth.Hasher.hash_list(passwords)


def generate_totp_secret() -> str:
    """Genereer een TOTP secret voor Google/Microsoft/Duo Authenticator."""
    return pyotp.random_base32()


def geef_totp_provisioning_uri(username: str, issuer: str = "Smart Health Dashboard") -> str | None:
    """
    Geef de otpauth://-inschrijvingslink voor deze gebruiker. Handig om de
    TOTP-secret in te voeren in een authenticator-app (Google, Microsoft,
    Duo Mobile, 1Password, enz.) zonder typefouten.
    Geeft None terug als de gebruiker geen TOTP-secret heeft.
    """
    user = _get_allowed_usernames(load_auth_config()).get(username, {})
    secret = user.get("totp_secret")
    if not secret or not isinstance(secret, str):
        return None
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=issuer)


def load_auth_config() -> dict:
    config = _load_yaml_file(AUTH_CONFIG_FILE)
    env_enabled_str = os.getenv(AUTH_ENABLED_ENV)

    if not config:
        if not AUTH_CONFIG_FILE.exists():
            auth_logger.warning(
                f"Authentication config ontbreekt: {AUTH_CONFIG_FILE}. "
                "Auth staat uit tenzij SMART_HEALTH_AUTH_ENABLED=true wordt gezet."
            )
        enabled = _env_flag(AUTH_ENABLED_ENV) if env_enabled_str is not None else False
        return {"enabled": enabled, "credentials": {"usernames": {}}}

    if env_enabled_str is not None:
        # De omgevingsvariabele heeft voorrang op de YAML-configuratie,
        # zodat auth per omgeving aan/uit kan zonder de YAML aan te passen.
        config["enabled"] = _env_flag(AUTH_ENABLED_ENV)
    elif "enabled" not in config:
        config["enabled"] = bool(config.get("credentials", {}).get("usernames"))

    if config.get("enabled") and "credentials" not in config:
        config["credentials"] = {"usernames": {}}

    return config


def _get_allowed_usernames(config: dict) -> dict:
    return config.get("credentials", {}).get("usernames", {}) or {}


def is_admin() -> bool:
    """
    Geef True terug als de huidige gebruiker admin-rechten heeft.

    Regels:
    - Auth uitgeschakeld (locale/dev-omgeving): True, zodat ontwikkelaars
      alle pagina's (o.a. de Testpagina) blijven zien.
    - Ingelogde gebruiker met `role: admin` of `admin: true` in
      auth_config.yaml: True.
    - In alle andere gevallen: False (fail-closed).
    """
    try:
        config = load_auth_config()
        if not config.get("enabled", False):
            return True
        username = st.session_state.get("auth_user")
        if not username:
            return False
        user = _get_allowed_usernames(config).get(username, {}) or {}
        if str(user.get("role", "")).strip().lower() == "admin":
            return True
        return bool(user.get("admin", False))
    except Exception:
        auth_logger.warning(
            "is_admin(): fout bij bepalen van rol; behandel als niet-admin.",
            exc_info=True,
        )
        return False


def _get_authenticator(config: dict) -> stauth.Authenticate:
    credentials = config.get("credentials", {})
    cookie = config.get("cookie", {})
    preauthorized = config.get("preauthorized", {})
    cookie_name = cookie.get("name", "smart_health_auth")
    key = AUTH_COOKIE_KEY or cookie.get("key") or "change-me-secret-key"
    expiry = float(cookie.get("expiry_days", AUTH_COOKIE_EXPIRY_DAYS))
    return stauth.Authenticate(
        credentials,
        cookie_name,
        key,
        cookie_expiry_days=expiry,
        preauthorized=preauthorized,
    )


def _current_timestamp() -> float:
    return time.time()


def _is_locked_out() -> bool:
    lockout_until = st.session_state.get("auth_lockout_until")
    return isinstance(lockout_until, (int, float)) and _current_timestamp() < lockout_until


def _record_failed_attempt(username: str | None = None) -> None:
    now = _current_timestamp()
    timestamps = st.session_state.get("auth_failed_timestamps", [])
    window = AUTH_RATE_LIMIT_PERIOD_MINUTES * 60
    timestamps = [ts for ts in timestamps if now - ts <= window]
    timestamps.append(now)
    st.session_state["auth_failed_timestamps"] = timestamps

    if len(timestamps) >= AUTH_RATE_LIMIT_MAX_ATTEMPTS:
        lockout_seconds = min(300, window)
        st.session_state["auth_lockout_until"] = now + lockout_seconds
        auth_logger.warning(
            json.dumps(
                {
                    "event": "lockout",
                    "username": username,
                    "reason": "rate_limit_triggered",
                    "until": datetime.utcfromtimestamp(now + lockout_seconds).isoformat() + "Z",
                }
            )
        )


def _clear_failed_attempts() -> None:
    st.session_state["auth_failed_timestamps"] = []
    st.session_state["auth_lockout_until"] = None


def _log_auth_event(username: str | None, status: str, message: str) -> None:
    auth_logger.info(
        json.dumps(
            {
                "event": "login_attempt",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "username": username,
                "status": status,
                "message": message,
            }
        )
    )


def _verify_totp(username: str, code: str) -> bool:
    config = load_auth_config()
    user = _get_allowed_usernames(config).get(username, {})
    secret = user.get("totp_secret")
    if not secret or not isinstance(secret, str):
        return False

    try:
        totp = pyotp.TOTP(secret)
        return totp.verify(code.strip(), valid_window=1)
    except Exception:
        return False


def logout() -> None:
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None
    st.session_state["auth_2fa_verified"] = False
    st.session_state["auth_pending_2fa"] = False
    st.session_state["auth_last_activity"] = None
    _clear_failed_attempts()
    # Expliciet afmelden vergeet ook het vertrouwde apparaat.
    _zet_cookie(_TRUST_COOKIE_NAME, "", 0)

    config = load_auth_config()
    try:
        authenticator = _get_authenticator(config)
        authenticator.logout("Logout", "sidebar")
    except Exception:
        pass


def _session_expired() -> bool:
    last = st.session_state.get("auth_last_activity")
    if not last:
        return False
    return _current_timestamp() - float(last) > AUTH_SESSION_TIMEOUT_MINUTES * 60


def _render_2fa_form(username: str) -> None:
    st.title("Beveiligde 2FA login")
    st.subheader("Voer de code uit uw Authenticator-app in")
    st.caption("Gebruik Google Authenticator of Microsoft Authenticator om de 6-cijferige code te genereren.")

    with st.form("totp_form", clear_on_submit=False):
        totp_code = st.text_input("2FA-code", type="password")
        submitted = st.form_submit_button("Verifiëren")

    if submitted:
        if _verify_totp(username, totp_code):
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = username
            st.session_state["auth_2fa_verified"] = True
            st.session_state["auth_pending_2fa"] = False
            st.session_state["auth_last_activity"] = _current_timestamp()
            _clear_failed_attempts()
            _log_auth_event(username, "success", "2FA geverifieerd")
            # Onthoud dit apparaat: herladen van het dashboard vereist geen
            # nieuwe 2FA zolang het token geldig is (AUTH_TRUST_DAYS).
            _zet_cookie(_TRUST_COOKIE_NAME, _maak_vertrouw_token(username), int(AUTH_TRUST_DAYS * 86400))
            st.success("2FA succesvol geverifieerd. Toegang verleend.")
        else:
            _record_failed_attempt(username)
            _log_auth_event(username, "failed", "Ongeldige 2FA-code")
            st.error("Ongeldige 2FA-code. Probeer opnieuw.")
    else:
        if st.session_state.get("auth_pending_2fa"):
            st.info("U bent succesvol geauthenticeerd met gebruikersnaam/wachtwoord. Voltooi nu 2FA.")

    if st.session_state.get("auth_ok"):
        return

    st.stop()


def require_login() -> None:
    config = load_auth_config()
    if not config.get("enabled", False):
        st.session_state["auth_ok"] = True
        return

    usernames = _get_allowed_usernames(config)
    if not usernames:
        st.error(
            "Authentication is ingeschakeld, maar er zijn geen gebruikers geconfigureerd. "
            "Kopieer auth_config.example.yaml naar auth_config.yaml en voeg gebruikers toe met bcrypt-hashes."
        )
        st.stop()

    # Herstel de sessie vanuit het vertrouwd-apparaat-cookie. Noodzakelijk
    # omdat een paginaherlading (F5) de Streamlit-sessie en dus alle
    # session_state wist; zonder dit cookie moet 2FA elke keer opnieuw.
    if not (st.session_state.get("auth_ok") and st.session_state.get("auth_2fa_verified")):
        _token = _lees_cookie(_TRUST_COOKIE_NAME)
        if _token:
            _herstelde_gebruiker = _verifieer_vertrouw_token(_token)
            if _herstelde_gebruiker and _herstelde_gebruiker in usernames:
                st.session_state["auth_ok"] = True
                st.session_state["auth_user"] = _herstelde_gebruiker
                st.session_state["auth_2fa_verified"] = True
                st.session_state["auth_pending_2fa"] = False
                st.session_state["auth_last_activity"] = _current_timestamp()
                _log_auth_event(_herstelde_gebruiker, "success", "Sessie hersteld via vertrouwd apparaat")
                _verleng_vertrouw_cookie_indien_nodig(_herstelde_gebruiker)
                return

    if _session_expired():
        logout()
        st.warning("Uw sessie is verlopen door inactiviteit. Log opnieuw in.")

    if st.session_state.get("auth_ok") and st.session_state.get("auth_2fa_verified"):
        st.session_state["auth_last_activity"] = _current_timestamp()
        return

    if _is_locked_out():
        lockout_until = st.session_state.get("auth_lockout_until")
        wait_seconds = int(lockout_until - _current_timestamp())
        st.error(f"Te veel mislukte pogingen. Probeer het over {wait_seconds} seconden opnieuw.")
        st.stop()

    authenticator = _get_authenticator(config)
    authenticator.login(location="main", key="Login")

    authentication_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    if not authentication_status:
        st.stop()

    if username not in usernames:
        _log_auth_event(username, "failed", "Onbekende gebruiker")
        st.error("Gebruiker niet toegestaan.")
        st.stop()

    st.session_state["auth_user"] = username
    st.session_state["auth_pending_2fa"] = True
    st.session_state["auth_last_activity"] = _current_timestamp()
    _log_auth_event(username, "pending", "Wacht op 2FA")

    _render_2fa_form(username)

    if st.session_state.get("auth_ok") and st.session_state.get("auth_2fa_verified"):
        return

    st.stop()
