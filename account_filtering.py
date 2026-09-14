"""Central filtering of internal Smart Health accounts."""

from __future__ import annotations

import logging
import time

import pandas as pd

try:
    from sqlalchemy import create_engine
except ImportError:  # pragma: no cover
    create_engine = None

logger = logging.getLogger(__name__)
INTERNAL_EMAIL_SUFFIXES: tuple[str, ...] = ("@smarthealth.works", "@smarthealth.nl")
INTERNE_ACCOUNTS_CACHE_TTL = 300
_CACHE: dict[str, tuple[float, tuple[set[int], set[str]]]] = {}


def _norm_email_series(emails: pd.Series) -> pd.Series:
    return emails.astype(str).str.strip().str.lower()


def interne_email_mask(emails: pd.Series) -> pd.Series:
    norm = _norm_email_series(emails)
    mask = pd.Series(False, index=emails.index)
    for suffix in INTERNAL_EMAIL_SUFFIXES:
        mask |= norm.str.endswith(suffix, na=False)
    return mask


def interne_localparts(emails: pd.Series) -> set[str]:
    norm = _norm_email_series(emails)
    parts = norm[interne_email_mask(norm)].str.split('@').str[0]
    return {part for part in parts if part and part != 'nan'}


def reset_interne_accounts_cache() -> None:
    _CACHE.clear()


def laad_interne_account_ids(db_url: str | None = None, ttl: int = INTERNE_ACCOUNTS_CACHE_TTL, refresh: bool = False) -> tuple[set[int], set[str]]:
    if not isinstance(db_url, str) or not db_url:
        return set(), set()
    now = time.monotonic()
    cached = _CACHE.get(db_url)
    if not refresh and cached is not None and (ttl <= 0 or now - cached[0] <= ttl):
        return cached[1]
    user_ids: set[int] = set()
    localparts: set[str] = set()
    try:
        if create_engine is None:
            raise RuntimeError("sqlalchemy niet beschikbaar")
        df_users = pd.read_sql("SELECT id, email FROM users", create_engine(db_url))
        if not df_users.empty and {'id', 'email'}.issubset(df_users.columns):
            mask = interne_email_mask(df_users['email'])
            user_ids = set(pd.to_numeric(df_users.loc[mask, 'id'], errors='coerce').dropna().astype(int))
            localparts = interne_localparts(df_users.loc[mask, 'email'])
    except Exception as exc:
        logger.warning("Kon interne Smart Health accounts niet laden: %s", exc)
    _CACHE[db_url] = (now, (user_ids, localparts))
    return user_ids, localparts


def filter_interne_accounts(df: pd.DataFrame, db_url: str | None = None, *, user_id_col: str = 'user_id', email_col: str = 'email', participant_id_col: str = 'participant_id', bridge_loader=None) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is None else df.copy()
    mask_eruit = pd.Series(False, index=df.index)
    if email_col in df.columns:
        norm = _norm_email_series(df[email_col])
        mask_eruit |= interne_email_mask(norm)
        _, localparts = laad_interne_account_ids(db_url)
        if localparts:
            mask_eruit |= norm.str.split('@').str[0].isin(localparts)
    if user_id_col in df.columns:
        intern_ids, _ = laad_interne_account_ids(db_url)
        if intern_ids:
            mask_eruit |= pd.to_numeric(df[user_id_col], errors='coerce').isin(intern_ids)
    if not mask_eruit.any() and participant_id_col in df.columns and user_id_col not in df.columns and callable(bridge_loader):
        try:
            bridge = bridge_loader(db_url)
        except Exception:
            bridge = pd.DataFrame()
        if not bridge.empty and {'participant_id', 'user_id'}.issubset(bridge.columns):
            intern_ids, _ = laad_interne_account_ids(db_url)
            intern_pids = pd.to_numeric(bridge.loc[pd.to_numeric(bridge['user_id'], errors='coerce').isin(intern_ids), 'participant_id'], errors='coerce').dropna()
            mask_eruit |= pd.to_numeric(df[participant_id_col], errors='coerce').isin(intern_pids)
    return df[~mask_eruit].copy()
