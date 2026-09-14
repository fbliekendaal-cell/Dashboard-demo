from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pyotp
import streamlit as st
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
    """Genereer een TOTP secret voor Google/Microsoft Authenticator."""
    return pyotp.random_base32()


def load_auth_config() -> dict:
    config = _load_yaml_file(AUTH_CONFIG_FILE)
    env_enabled_str = os.getenv(AUTH_ENABLED_ENV)

    if not config:
        if not AUTH_CONFIG_FILE.exists():
            auth_logger.info(
                f"Authentication config ontbreekt: {AUTH_CONFIG_FILE}. "
                "Standaard uitgeschakeld voor demo."
            )
        enabled = _env_flag(AUTH_ENABLED_ENV) if env_enabled_str is not None else False
        return {"enabled": enabled, "credentials": {"usernames": {}}}

    if env_enabled_str is not None:
        config["enabled"] = _env_flag(AUTH_ENABLED_ENV)
    elif "enabled" not in config:
        config["enabled"] = bool(config.get("credentials", {}).get("usernames"))

    if config.get("enabled") and "credentials" not in config:
        config["credentials"] = {"usernames": {}}

    return config


def _get_allowed_usernames(config: dict) -> dict:
    return config.get("credentials", {}).get("usernames", {}) or {}


def is_admin() -> bool:
    """Return whether the current demo user has administrator access."""
    try:
        config = load_auth_config()
        if not config.get("enabled", False):
            return True
        username = st.session_state.get("auth_user")
        if not username:
            return False
        user = _get_allowed_usernames(config).get(username, {}) or {}
        return str(user.get("role", "")).strip().lower() == "admin" or bool(user.get("admin", False))
    except Exception:
        auth_logger.warning("is_admin(): fout bij bepalen van rol", exc_info=True)
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
