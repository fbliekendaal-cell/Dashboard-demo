import pandas as pd
import numpy as np
import re
import unicodedata


def normalize_search_text(value) -> str:
    """Normalize searchable text for case- and accent-insensitive matching."""
    if value is None:
        return ''
    text = unicodedata.normalize('NFKD', str(value))
    text = ''.join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r'[_/.-]+', ' ', text.casefold())
    return re.sub(r'\s+', ' ', text).strip()


def search_variable_specs(specs, query):
    """Return variable specs matching every term in the query.

    Labels, descriptions, groups, column names and optional types are searched.
    Results are ranked by label, then by the order in which specs were supplied.
    """
    terms = normalize_search_text(query).split()
    if not terms:
        return []

    matches = []
    for position, spec in enumerate(specs):
        haystack = normalize_search_text(' '.join(
            str(spec.get(field, ''))
            for field in ('label', 'omschrijving', 'groep', 'kolom', 'type', 'eenheid')
        ))
        if all(term in haystack for term in terms):
            label = normalize_search_text(spec.get('label', ''))
            score = sum(term in label for term in terms)
            matches.append((-score, position, spec))

    return [spec for _, _, spec in sorted(matches)]

def get_numeric_clean(df, kolom):
    """Zet een kolom om naar numeriek, vervangt 'None' door NaN."""
    if df is None or df.empty or kolom not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[kolom].replace('None', np.nan), errors='coerce')

def filter_by_gender(df, gender):
    """Filtert op geslacht: 'man', 'vrouw', 'beide' of 'totaal'."""
    if df is None or df.empty or gender in ('beide', 'totaal'):
        return df
    # 1=man, 0=vrouw
    val = 1 if gender == 'man' else 0
    return df[pd.to_numeric(df['rec_user_gender'], errors='coerce') == val]

def calculate_kpi_count(df, kolom, waarde, op='eq'):
    """Berekent KPI tellingen (aantal, totaal, percentage)."""
    if df is None or df.empty or kolom not in df.columns:
        return 0, 0, 0.0
    s = pd.to_numeric(df[kolom], errors='coerce')
    n_geldig = int(s.notna().sum())
    if n_geldig == 0:
        return 0, 0, 0.0
    count = int(s.ge(waarde).sum() if op == 'ge' else s.eq(waarde).sum())
    pct = count / n_geldig * 100
    return count, n_geldig, pct

def aggregate_by_groups(df, groupby_cols, agg_dict):
    """Groepeert en aggregeert veilig voor analysemodules."""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        return df.groupby(groupby_cols, dropna=False).agg(agg_dict).reset_index()
    except Exception:
        return pd.DataFrame()

def export_to_csv(df):
    """Converteert dataframe naar CSV bytes voor download."""
    if df is None:
        return b""
    return df.to_csv(index=False).encode('utf-8')


def _maak_net_label(naam: str) -> str:
    """Converteert een interne kolomnaam naar een leesbaar label."""
    vervang = naam
    for prefix in ('rec_', 'ls_', 'med_', 'asr_', 'digital_detox_'):
        vervang = vervang.replace(prefix, '')
    return (
        vervang
        .replace('_', ' ')
        .replace('psqi', 'PSQI')
        .replace('dass', 'DASS')
        .replace('wai', 'WAI')
        .strip()
        .title()
    )
