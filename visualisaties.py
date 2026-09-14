from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import logging
from helpers import get_numeric_clean, filter_by_gender, normalize_to_scale

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from scipy import stats
from config import DB_URL
from data_ingestion import (
    generate_consolidated_scores_in_db,
    load_table_from_database,
    load_factor_score_histories,
    load_completions,
    load_users_met_scores,
    load_participants,
    load_my_clic_participants,
    load_my_clic_participants_expanded,
    load_store_employees,
    load_stores,
    load_orders,
    get_user_purchases,
    add_app_user_ids_and_addresses,
)
from helpers import _maak_net_label
from account_filtering import filter_interne_accounts, interne_email_mask
logger = logging.getLogger(__name__)

# Import variabelen EENMALIG hier, niet in functies!
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("Statsmodels niet geïnstalleerd. Trendlijnen in scatterplots zijn uitgeschakeld.")

try:
    import variabelen as _variabelen
    SLAAP_TEKST_MAP = getattr(_variabelen, 'SLAAP_TEKST_MAP', {})
    SLAAP_VOLGORDE = getattr(_variabelen, 'SLAAP_VOLGORDE', [])
    SLAAP_KLEUREN = getattr(_variabelen, 'SLAAP_KLEUREN', {})
    VARIABELEN_DICT = getattr(_variabelen, 'VARIABELEN_DICT', {})
    VARIABELEN_PER_GROEP = getattr(_variabelen, 'VARIABELEN_PER_GROEP', {})
    # Deze variabelen zijn gedefinieerd in analyses.py, niet in variabelen.py
    from analyses import LEEFSTIJL_SCORES, RUWE_WAARDEN
    logger.info("✓ Variabelen module imported")
except ImportError as e:
    SLAAP_TEKST_MAP = {}
    SLAAP_VOLGORDE = []
    SLAAP_KLEUREN = {}
    VARIABELEN_DICT = {}
    VARIABELEN_PER_GROEP = {}
    LEEFSTIJL_SCORES = {}
    RUWE_WAARDEN = {}
    logger.error(f"Error importing variabelen: {e}")

# Globale fontgrootte voor alle Plotly grafieken
pio.templates["smarthealth"] = go.layout.Template(
    layout=go.Layout(
        font=dict(size=14),
        title=dict(font=dict(size=16)),
        xaxis=dict(title=dict(font=dict(size=13)), tickfont=dict(size=12)),
        yaxis=dict(title=dict(font=dict(size=13)), tickfont=dict(size=12)),
        legend=dict(font=dict(size=13)),
    )
)
pio.templates.default = "plotly+smarthealth"

from sqlalchemy import create_engine
from kleuren import (
    HOOFD_KLEUR, GENDER_COLORS, GENDER_LABELS,
    RISICO_COLORS, HEARTRISK_LABELS, STRESS_LABELS,
    BMI_LABELS, BMI_COLORS,
)
from i18n import tr


# ── Helpers ───────────────────────────────────────────────────────────────────
def _empty_plot(message: str, height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14, color='gray'),
    )
    fig.update_layout(template='plotly_white', height=height)
    return fig


def _laad_users_tabel(base_pad: Path) -> pd.DataFrame:
    """Laad users tabel uit database."""
    return load_table_from_database('participants', DB_URL) # Deze functie is al correct


def _laad_actieve_users(base_pad: Path, include_deleted: bool = False) -> pd.DataFrame:
    """Laad users en voeg jaar toe. Optioneel inclusief verwijderde accounts."""
    df_users = _laad_users_tabel(base_pad) # Roep de database-ladende functie aan
    if df_users is None or df_users.empty:
        return pd.DataFrame(columns=['id', 'user_id', 'created_at', 'deleted_at'])

    if 'id' not in df_users.columns:
        for fallback_column in ('user_id', 'participant_id'):
            if fallback_column in df_users.columns:
                df_users = df_users.copy()
                df_users['id'] = df_users[fallback_column]
                break
        else:
            logger.warning("Users-data mist een ID-kolom; platformgroei wordt overgeslagen.")
            return pd.DataFrame(columns=['id', 'user_id', 'created_at', 'deleted_at'])

    if not include_deleted and 'deleted_at' in df_users.columns:
        df_users = df_users[df_users['deleted_at'].isna()].copy()

    if 'created_at' in df_users.columns:
        df_users['jaar'] = pd.to_datetime(df_users['created_at'], errors='coerce').dt.year
    else:
        df_users['jaar'] = pd.NA
    return df_users


def _numeriek(df, kolom):
    """Zet een kolom om naar int, dropt NaN rijen."""
    if kolom not in df.columns:
        return pd.Series(dtype='int64')
    s = pd.to_numeric(df[kolom], errors='coerce')
    return s.dropna().astype(int)


def _has_valid_db_url(db_url: str | None) -> bool:
    return isinstance(db_url, str) and "://" in db_url


def _laad_actieve_stores(base_pad: Path, include_deleted: bool = True) -> pd.DataFrame:
    stores_df = load_stores(DB_URL).copy()
    if stores_df.empty:
        return pd.DataFrame(columns=['id', 'name'])
    if 'id' not in stores_df.columns:
        stores_df['id'] = pd.NA
    if 'name' not in stores_df.columns:
        stores_df['name'] = 'Onbekend'
    stores_df['id'] = pd.to_numeric(stores_df['id'], errors='coerce')
    if not include_deleted and 'deleted_at' in stores_df.columns:
        stores_df['deleted_at'] = stores_df['deleted_at'].replace({'NULL': pd.NA, 'None': pd.NA, '': pd.NA})
        stores_df = stores_df[stores_df['deleted_at'].isna()].copy()
    if not include_deleted and 'is_active' in stores_df.columns:
        actieve_mask = ~stores_df['is_active'].astype(str).str.strip().str.lower().isin({'0', 'false', 'nan', 'none'})
        stores_df = stores_df[actieve_mask].copy()
    return stores_df.dropna(subset=['id']).drop_duplicates(subset='id')


def _laad_actieve_store_links(base_pad: Path) -> pd.DataFrame:
    opdrachtgever_links = _laad_opdrachtgever_links(DB_URL)
    if not opdrachtgever_links.empty:
        return (
            opdrachtgever_links[['user_id', 'store_id']]
            .dropna(subset=['user_id', 'store_id'])
            .sort_values(['user_id', 'store_id'])
            .drop_duplicates(subset=['user_id'], keep='first')
        )

    df_store = load_store_employees(DB_URL).copy()
    if df_store.empty or not {'user_id', 'store_id'}.issubset(df_store.columns):
        return pd.DataFrame(columns=['user_id', 'store_id'])

    df_store['user_id'] = pd.to_numeric(df_store['user_id'], errors='coerce')
    df_store['store_id'] = pd.to_numeric(df_store['store_id'], errors='coerce')
    if 'deleted_at' in df_store.columns:
        df_store['deleted_at'] = df_store['deleted_at'].replace({'NULL': pd.NA, 'None': pd.NA, '': pd.NA})
        df_store = df_store[df_store['deleted_at'].isna()].copy()
    if 'archived_at' in df_store.columns:
        df_store['archived_at'] = df_store['archived_at'].replace({'NULL': pd.NA, 'None': pd.NA, '': pd.NA})
        df_store = df_store[df_store['archived_at'].isna()].copy()

    geldige_store_ids = set(_laad_actieve_stores(base_pad, include_deleted=True)['id'].dropna().astype(int))
    if geldige_store_ids:
        df_store = df_store[df_store['store_id'].isin(geldige_store_ids)].copy()

    return (
        df_store[['user_id', 'store_id']]
        .dropna(subset=['user_id', 'store_id'])
        .drop_duplicates(subset=['user_id'], keep='first')
    )

def _laad_opdrachtgever_links(db_url: str | None = None) -> pd.DataFrame:
    """
    Koppel QE participants aan echte opdrachtgevers.

    De betrouwbare route in deze database is:
    public.participants.public_id -> smart_health.my_clic_participants.qe_participant_id
    -> smart_health.store_employees.user_id -> public.stores.id/name.
    """
    db_url = db_url or DB_URL
    if not _has_valid_db_url(db_url):
        return pd.DataFrame(columns=['participant_id', 'user_id', 'store_id', 'store_name'])

    query = text("""
        SELECT DISTINCT
            p.id AS participant_id,
            m.user_id AS user_id,
            s.id AS store_id,
            COALESCE(NULLIF(s.name, ''), CONCAT('Store ', s.id)) AS store_name
        FROM public.participants p
        JOIN smart_health.my_clic_participants m
          ON m.qe_participant_id = p.public_id
        JOIN smart_health.store_employees se
          ON se.user_id = m.user_id
        JOIN public.stores s
          ON s.id = se.store_id
        WHERE se.deleted_at IS NULL
          AND s.id IS NOT NULL
    """)
    try:
        engine = create_engine(db_url)
        df_links = pd.read_sql(query, engine)
    except Exception:
        # Fallback voor databases zonder schema-prefixen (bijv. SQLite in de
        # demo/testomgeving): zelfde join zonder 'public.'/'smart_health.'.
        fallback_query = text("""
            SELECT DISTINCT
                p.id AS participant_id,
                m.user_id AS user_id,
                s.id AS store_id,
                COALESCE(NULLIF(s.name, ''), 'Store ' || s.id) AS store_name
            FROM participants p
            JOIN my_clic_participants m
              ON m.qe_participant_id = p.public_id
            JOIN store_employees se
              ON se.user_id = m.user_id
            JOIN stores s
              ON s.id = se.store_id
            WHERE se.deleted_at IS NULL
              AND s.id IS NOT NULL
        """)
        try:
            engine = create_engine(db_url)
            df_links = pd.read_sql(fallback_query, engine)
        except Exception as e:
            logger.warning(f"Kon opdrachtgeverkoppeling niet laden: {e}")
            return pd.DataFrame(columns=['participant_id', 'user_id', 'store_id', 'store_name'])

    for col in ['participant_id', 'user_id', 'store_id']:
        if col in df_links.columns:
            df_links[col] = pd.to_numeric(df_links[col], errors='coerce')

    df_links = (
        df_links
        .dropna(subset=['participant_id', 'store_id'])
        .drop_duplicates(subset=['participant_id', 'store_id'])
        .sort_values(['participant_id', 'store_id'])
    )

    dubbele_participants = df_links.duplicated(subset=['participant_id'], keep=False).sum()
    if dubbele_participants:
        logger.warning(
            "Meerdere opdrachtgeverkoppelingen gevonden voor %s participant-rijen; "
            "eerste store_id per participant wordt gebruikt.",
            dubbele_participants,
        )

    return df_links.drop_duplicates(subset=['participant_id'], keep='first')


def _dashboard_score_to_history_slug(score_slug: str) -> str:
    """Vertaal dashboardkolommen naar factorhistorie-slugs voor opdrachtgevertrends."""
    return {
        'rec_med_bmi': 'bmi',
        'rec_heartrisk': 'heartrisk',
        'rec_framingham_non_invasive': 'heartrisk',
        'rec_ls_stress_sum': 'stress',
        'rec_ls_sleep_psqi_sum': 'sleep',
        'rec_ls_alcohol_total_per_week': 'alcohol',
        'rec_ls_score_alcohol': 'alcohol',
        'rec_ls_score_exercise': 'exercise',
        'rec_ls_exercise_physical_activity_minutes_total': 'exercise',
        'rec_ls_exercise_steps_per_day': 'steps',
        'rec_ls_nutrition_fruit_fruit_per_day': 'fruit',
        'rec_ls_score_fruit': 'fruit',
        'rec_ls_vegetables_gram_per_day': 'vegetables',
        'rec_ls_score_vegetables': 'vegetables',
        'rec_ls_nutrition_sugar_per_day': 'sugar',
        'rec_ls_score_sugar': 'sugar',
        'rec_ls_nutrition_saturated_fat_per_day': 'fat',
        'rec_ls_score_saturated_fat': 'fat',
        'rec_ls_nutrition_natrium_per_day': 'salt',
        'rec_ls_score_natrium': 'salt',
        'rec_smoking_answer': 'smoking',
        # De huidige scorehistorie bevat geen aparte 'lifestyle'-slug.
        # Gebruik de beschikbare 'wellbeing'-historie als bruikbare fallback
        # voor de trendweergave van de dashboard-leefstijlscore.
        'rec_ls_lifestyle_score': 'wellbeing',
        'rec_resilience_score': 'resilience',
        'rec_wellbeing_score': 'wellbeing',
        'rec_self_efficacy_score': 'selfefficacy',
        'rec_dass_stress_score': 'dass_stress',
        'rec_dass_anxiety_score': 'dass_anxiety',
        'rec_dass_depression_score': 'dass_depression',
        'rec_asr_job_satisfaction_score': 'job_satisfaction',
        'rec_asr_working_attitude_score': 'working_attitude',
        'rec_asr_workload_score': 'workload',
        'rec_asr_work_ability_score': 'work_life_balance',
        'rec_asr_wai_score': 'work_life_balance',
        'rec_asr_burn_out_score': 'health',
        'rec_asr_personal_competences_score': 'competences',
        'rec_asr_minor_mental_complaints_score': 'mental_complaints',
    }.get(score_slug, score_slug)



def _tel_percentages(series, labels: dict) -> pd.DataFrame:
    counts = series.value_counts(normalize=True).mul(100).reset_index()
    counts.columns = ['code', 'percentage']
    counts['label'] = counts['code'].map(labels)
    counts = counts.dropna(subset=['label'])
    return counts


def laad_data(base_pad: Path, db_url: str | None = None) -> pd.DataFrame:
    """
    Centrale functie voor het inladen van de geconsolideerde gebruikersdata.
    Probeert eerst de database (users_met_scores), daarna lokale fallback.
    """
    from config import DATA_SOURCE

    if not isinstance(db_url, str):
        from config import DB_URL
        db_url = DB_URL

    if DATA_SOURCE == 'csv':
        try:
            df = load_users_met_scores(db_url)
            if df.empty:
                return df
            if 'user_id' in df.columns:
                df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
                df = df.drop_duplicates(subset='user_id', keep='first')
            return df.reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"CSV-ingestie mislukt ({exc}).")
            return pd.DataFrame()

    if not db_url:
        logger.warning("Geen database URL geconfigureerd, probeer lokale fallback.")
        try:
            return load_users_met_scores(DB_URL)
        except Exception:
            return pd.DataFrame()

    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        # Controleer of de tabel bestaat, anders proberen op te bouwen
        if 'users_met_scores' not in table_names:
            logger.warning("De tabel 'users_met_scores' ontbreekt. Poging tot genereren...")
            ok, _ = generate_consolidated_scores_in_db(engine)
            if not ok:
                return load_users_met_scores(db_url)

        df = pd.read_sql("SELECT * FROM users_met_scores", engine)
        df = df.drop_duplicates(subset='user_id', keep='first').reset_index(drop=True)
        
        # Garandeer numerieke user_id
        df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')

        # Interne Smart Health (test)accounts uit alle platformdata filteren
        df = filter_interne_accounts(df, db_url=db_url)
        return df
    except Exception as e:
        logger.warning(f"Database-ingestie mislukt ({e}), probeer lokale fallback.")
        try:
            return load_users_met_scores(db_url or DB_URL)
        except Exception:
            return pd.DataFrame()


# ── Demografisch ──────────────────────────────────────────────────────────────
def maak_geslacht_plot(df: pd.DataFrame, lang: str = 'nl') -> go.Figure:
    s = _numeriek(df, 'rec_user_gender')
    gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
    counts = _tel_percentages(s, gender_labels)
    kleur_map = {v: GENDER_COLORS[k] for k, v in gender_labels.items()}
    fig = px.bar(
        counts, x='label', y='percentage',
        color='label',
        color_discrete_map=kleur_map,
        labels={'label': '', 'percentage': tr('Percentage (%)', lang)},
        title=tr('Verdeling geslacht', lang),
        category_orders={'label': [tr('Man', lang), tr('Vrouw', lang)]},
    )
    fig.update_layout(showlegend=False)
    return fig


def maak_leeftijd_plot(df: pd.DataFrame, geslacht: str = 'beide', lang: str = 'nl') -> go.Figure:
    if 'rec_age_current' not in df.columns:
        return _empty_plot(tr('Geen leeftijdsdata beschikbaar.', lang))
    df2 = df.copy()
    df2['rec_age_current'] = pd.to_numeric(df2['rec_age_current'], errors='coerce')
    df2 = df2.dropna(subset=['rec_age_current'])

    if geslacht == 'beide' and 'rec_user_gender' in df2.columns:
        gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
        df2['geslacht_label'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(gender_labels)
        df2 = df2.dropna(subset=['geslacht_label'])
        kleur_map = {gender_labels[1]: GENDER_COLORS['Man'], gender_labels[0]: GENDER_COLORS['Vrouw']}
        fig = px.histogram(
            df2, x='rec_age_current',
            color='geslacht_label',
            color_discrete_map=kleur_map,
            barmode='overlay',
            opacity=0.75,
            labels={'rec_age_current': tr('Leeftijd', lang), 'count': tr('Aantal', lang), 'geslacht_label': tr('Geslacht', lang)},
            title=tr('Leeftijdsverdeling naar geslacht', lang),
            category_orders={'geslacht_label': [tr('Man', lang), tr('Vrouw', lang)]},
        )
        fig.update_layout(legend_title_text=tr('Geslacht', lang))
    else:
        fig = px.histogram(
            df2, x='rec_age_current',
            nbins=30,
            color_discrete_sequence=[HOOFD_KLEUR],
            labels={'rec_age_current': tr('Leeftijd', lang), 'count': tr('Aantal', lang)},
            title=tr('Leeftijdsverdeling', lang),
        )

    return fig


# ── Cardiovasculair risico ─────────────────────────────────────────────────────
def maak_heartrisk_plot(df: pd.DataFrame, lang: str = 'nl') -> go.Figure:
    s = _numeriek(df, 'rec_heartrisk_cat')
    labels = {k: tr(v, lang) for k, v in HEARTRISK_LABELS.items()}
    counts = _tel_percentages(s, labels)
    kleur_map = {v: RISICO_COLORS[k] for k, v in labels.items()}
    fig = px.bar(
        counts, x='label', y='percentage',
        color='label',
        color_discrete_map=kleur_map,
        labels={'label': '', 'percentage': tr('Percentage (%)', lang)},
        title=tr('Cardiovasculair risico', lang),
        category_orders={'label': [tr('Laag', lang), tr('Matig', lang), tr('Hoog', lang)]},
    )
    fig.update_layout(showlegend=False)
    return fig


def maak_heartrisk_naar_geslacht_plot(df: pd.DataFrame, lang: str = 'nl', geslacht: str = 'beide') -> go.Figure:
    if 'rec_heartrisk_cat' not in df.columns:
        return _empty_plot(tr('Geen heartrisk-data beschikbaar.', lang))
    df2 = df.copy()
    df2['heartrisk_code'] = pd.to_numeric(df2['rec_heartrisk_cat'], errors='coerce')
    risk_labels = {k: tr(v, lang) for k, v in HEARTRISK_LABELS.items()}
    df2['risico_label'] = df2['heartrisk_code'].map(risk_labels)
    df2 = df2.dropna(subset=['risico_label'])

    if geslacht == 'beide' and 'rec_user_gender' in df2.columns:
        gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
        df2['geslacht_label'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(gender_labels)
        df2 = df2.dropna(subset=['geslacht_label'])
        counts = df2.groupby(['geslacht_label', 'risico_label']).size().reset_index(name='aantal')
        totaal = counts.groupby('geslacht_label')['aantal'].transform('sum')
        counts['percentage'] = counts['aantal'] / totaal * 100

        kleur_map = {gender_labels[1]: GENDER_COLORS['Man'], gender_labels[0]: GENDER_COLORS['Vrouw']}
        fig = px.bar(
            counts, x='risico_label', y='percentage',
            color='geslacht_label',
            barmode='group',
            color_discrete_map=kleur_map,
            labels={'risico_label': tr('Risicocategorie', lang), 'percentage': tr('Percentage (%)', lang), 'geslacht_label': tr('Geslacht', lang)},
            title=tr('Cardiovasculair risico naar geslacht', lang),
            category_orders={
                'risico_label':   [tr('Laag', lang), tr('Matig', lang), tr('Hoog', lang)],
                'geslacht_label': [tr('Man', lang), tr('Vrouw', lang)],
            },
        )
    else:
        counts = df2['risico_label'].value_counts(normalize=True).mul(100).reset_index()
        counts.columns = ['risico_label', 'percentage']
        kleur_map = {tr(v, lang): RISICO_COLORS[k] for k, v in HEARTRISK_LABELS.items()}
        fig = px.bar(
            counts, x='risico_label', y='percentage',
            color='risico_label',
            color_discrete_map=kleur_map,
            labels={'risico_label': tr('Risicocategorie', lang), 'percentage': tr('Percentage (%)', lang)},
            title=tr('Cardiovasculair risico', lang),
            category_orders={'risico_label': [tr('Laag', lang), tr('Matig', lang), tr('Hoog', lang)]},
        )
        fig.update_layout(showlegend=False)

    return fig


# ── BMI ────────────────────────────────────────────────────────────────────────
def maak_bmi_plot(df: pd.DataFrame, geslacht: str = 'beide', lang: str = 'nl') -> go.Figure:
    if 'rec_med_bmi_cat' not in df.columns:
        return _empty_plot(tr('Geen BMI-data beschikbaar.', lang))

    df2 = df.copy()
    df2['bmi_cat'] = pd.to_numeric(df2['rec_med_bmi_cat'], errors='coerce')
    df2 = df2.dropna(subset=['bmi_cat'])

    if df2.empty:
        return _empty_plot(tr('Geen BMI-data beschikbaar.', lang))

    bmi_labels = {k: tr(v, lang) for k, v in BMI_LABELS.items()}
    bmi_order = [bmi_labels[k] for k in sorted(BMI_LABELS.keys())]

    if geslacht == 'beide' and 'rec_user_gender' in df2.columns:
        gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
        df2['geslacht_label'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(gender_labels)
        df2 = df2.dropna(subset=['geslacht_label'])
        if df2.empty:
            return _empty_plot(tr('Geen BMI-data beschikbaar.', lang))

        counts = (
            df2.groupby(['geslacht_label', 'bmi_cat'], observed=True)
            .size()
            .reset_index(name='aantal')
        )
        totals = counts.groupby('geslacht_label')['aantal'].transform('sum')
        counts['percentage'] = counts['aantal'] / totals * 100
        counts['label'] = counts['bmi_cat'].map(bmi_labels)
        fig = go.Figure()
        for code, kleur in [(1, GENDER_COLORS[1]), (0, GENDER_COLORS[0])]:
            naam = gender_labels[code]
            subset = counts[counts['geslacht_label'] == naam].set_index('label').reindex(bmi_order).fillna(0)
            fig.add_trace(go.Bar(
                x=subset.index,
                y=subset['percentage'],
                name=naam,
                marker_color=kleur,
            ))
        fig.update_layout(
            barmode='group',
            legend_title_text=tr('Geslacht', lang),
            title=tr('BMI categorie verdeling naar geslacht', lang),
            xaxis_title='',
            yaxis_title=tr('Percentage (%)', lang),
        )
        return fig

    counts = _tel_percentages(df2['bmi_cat'], BMI_LABELS)
    kleur_map = {v: BMI_COLORS[k] for k, v in BMI_LABELS.items()}
    fig = px.bar(
        counts, x='label', y='percentage',
        color='label',
        color_discrete_map=kleur_map,
        labels={'label': '', 'percentage': tr('Percentage (%)', lang)},
        title=tr('BMI categorie verdeling', lang),
        category_orders={'label': bmi_order},
    )
    fig.update_layout(showlegend=False)
    return fig


# ── Stress ─────────────────────────────────────────────────────────────────────
def maak_stress_plot(df: pd.DataFrame, geslacht: str = 'beide', lang: str = 'nl') -> go.Figure:
    if 'rec_ls_stress_cat' not in df.columns:
        return _empty_plot(tr('Geen stressdata beschikbaar.', lang))

    df2 = df.copy()
    df2['stress_cat'] = pd.to_numeric(df2['rec_ls_stress_cat'], errors='coerce')
    df2 = df2.dropna(subset=['stress_cat'])
    stress_labels = {k: tr(v, lang) for k, v in STRESS_LABELS.items()}
    stress_order = [stress_labels[k] for k in sorted(STRESS_LABELS.keys())]

    if geslacht == 'beide' and 'rec_user_gender' in df2.columns:
        gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
        df2['geslacht_label'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(gender_labels)
        df2 = df2.dropna(subset=['geslacht_label'])
        if df2.empty:
            return _empty_plot(tr('Geen stressdata beschikbaar.', lang))

        counts = (
            df2.groupby(['geslacht_label', 'stress_cat'], observed=True)
            .size()
            .reset_index(name='aantal')
        )
        totals = counts.groupby('geslacht_label')['aantal'].transform('sum')
        counts['percentage'] = counts['aantal'] / totals * 100
        counts['label'] = counts['stress_cat'].map(stress_labels)
        fig = go.Figure()
        for code, kleur in [(1, GENDER_COLORS[1]), (0, GENDER_COLORS[0])]:
            naam = gender_labels[code]
            subset = counts[counts['geslacht_label'] == naam].set_index('label').reindex(stress_order).fillna(0)
            fig.add_trace(go.Bar(
                x=subset.index,
                y=subset['percentage'],
                name=naam,
                marker_color=kleur,
            ))
        fig.update_layout(
            barmode='group',
            legend_title_text=tr('Geslacht', lang),
            title=tr('Stresscategorie verdeling naar geslacht', lang),
            xaxis_title='',
            yaxis_title=tr('Percentage (%)', lang),
        )
        return fig

    counts = _tel_percentages(df2['stress_cat'], stress_labels)
    kleur_map = {v: RISICO_COLORS[k] for k, v in stress_labels.items()}
    fig = px.bar(
        counts, x='label', y='percentage',
        color='label',
        color_discrete_map=kleur_map,
        labels={'label': '', 'percentage': tr('Percentage (%)', lang)},
        title=tr('Stresscategorie verdeling', lang),
        category_orders={'label': stress_order},
    )
    fig.update_layout(showlegend=False)
    return fig


# ── Leefstijlscore ─────────────────────────────────────────────────────────────
def maak_leefstijl_score_plot(df: pd.DataFrame, geslacht: str = 'beide', lang: str = 'nl') -> go.Figure:
    """Leefstijlscore categorieën (Slecht/Matig/Goed) met consistente kleuren.

    Dezelfde afkapwaarden en kleuren gelden in álle geslachtsmodi: Slecht = rood,
    Matig = oranje, Goed = groen. Bij 'beide' wordt per geslacht gegroepeerd met
    de vaste geslachtkleuren (blauw/roze) en categorieën op de x-as.
    """
    if 'rec_ls_lifestyle_score' not in df.columns:
        return _empty_plot(tr('Geen leefstijlscore beschikbaar.', lang))
    df2 = df.copy()
    df2['rec_ls_lifestyle_score'] = pd.to_numeric(df2['rec_ls_lifestyle_score'], errors='coerce')
    df2 = df2.dropna(subset=['rec_ls_lifestyle_score'])

    # Eenduidige categorie-indeling en kleuren voor alle modi
    bins = [0, 4, 7, 10]
    cat_labels = [tr('Slecht', lang), tr('Matig', lang), tr('Goed', lang)]
    cat_colors = {
        tr('Slecht', lang): RISICO_COLORS[2],   # rood
        tr('Matig', lang): RISICO_COLORS[1],    # oranje
        tr('Goed', lang): RISICO_COLORS[0],     # groen
    }
    df2['categorie'] = pd.cut(
        df2['rec_ls_lifestyle_score'],
        bins=bins,
        labels=cat_labels,
        include_lowest=True,
        right=True,
    )
    df2 = df2.dropna(subset=['categorie'])

    if geslacht == 'beide' and 'rec_user_gender' in df.columns:
        gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
        df2['geslacht_label'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(gender_labels)
        df2 = df2.dropna(subset=['geslacht_label'])
        if df2.empty:
            return _empty_plot(tr('Geen leefstijldata beschikbaar voor deze selectie.', lang))
        counts = (
            df2.groupby(['geslacht_label', 'categorie'], observed=True)
            .size()
            .reset_index(name='aantal')
        )
        totals = counts.groupby('geslacht_label')['aantal'].transform('sum').replace(0, pd.NA)
        counts['percentage'] = (counts['aantal'] / totals * 100).fillna(0)
        fig = go.Figure()
        for code in (1, 0):
            naam = gender_labels[code]
            subset = counts[counts['geslacht_label'] == naam].set_index('categorie').reindex(cat_labels).fillna(0)
            fig.add_trace(go.Bar(
                x=list(subset.index),
                y=subset['percentage'],
                name=naam,
                marker_color=GENDER_COLORS.get(code, '#999999'),
            ))
        fig.update_layout(
            barmode='group',
            legend_title_text=tr('Geslacht', lang),
            title=tr('Leefstijlscore categorieën naar geslacht', lang),
            xaxis_title=tr('Categorie', lang),
            yaxis_title=tr('Percentage (%)', lang),
        )
        return fig
    else:
        if df2.empty:
            return _empty_plot(tr('Geen leefstijldata beschikbaar voor deze selectie.', lang))
        counts = df2['categorie'].value_counts().reindex(cat_labels).fillna(0).reset_index()
        counts.columns = ['categorie', 'aantal']
        counts['percentage'] = counts['aantal'] / counts['aantal'].sum() * 100
        fig = px.bar(
            counts, x='categorie', y='percentage',
            color='categorie',
            color_discrete_map=cat_colors,
            labels={'categorie': tr('Categorie', lang), 'percentage': tr('Percentage (%)', lang)},
            title=tr('Leefstijlscore categorieën', lang),
            category_orders={'categorie': cat_labels},
        )
        fig.update_layout(showlegend=False)
        return fig


# ── Longitudinale data ─────────────────────────────────────────────────────────
def laad_longitudinale_data(base_pad: Path, db_url: str) -> pd.DataFrame:
    df_history    = load_factor_score_histories(db_url or DB_URL)
    if df_history.empty:
        return pd.DataFrame(columns=[
            'id', 'participant_id', 'questionnaire_factor_id', 'completion_id',
            'score_value', 'score_category_value', 'completion_created_at',
            'created_at', 'updated_at', 'slug', 'user_gender', 'store_id', 'jaar'
        ])
    df_completions = load_completions(db_url or DB_URL)
    df_scores_latest = load_my_clic_participants_expanded(db_url or DB_URL)
    # Ensure participant_id column exists
    if 'participant_id' not in df_scores_latest.columns:
        df_scores_latest = df_scores_latest.rename(columns={'user_id': 'participant_id'})
    df_scores_latest = df_scores_latest.drop_duplicates(subset='participant_id', keep='first')
    df_scores_latest = df_scores_latest[df_scores_latest['participant_id'].notna()]
    df_store_employees = _laad_actieve_store_links(base_pad)  # Used for store_id linking
    # Gebruik participants als primaire bron en my_clic_participants als fallback/bridge.
    df_participants = load_participants(db_url or DB_URL)
    try:
        df_my_clic = load_my_clic_participants(db_url or DB_URL)
    except Exception:
        df_my_clic = pd.DataFrame()

    # Load questionnaire factors
    df_factors = None
    if _has_valid_db_url(db_url):
        try:
            engine = create_engine(db_url)
            df_factors = pd.read_sql("SELECT id, slug FROM questionnaire_factors", engine)
        except Exception:
            df_factors = None
    if df_factors is None or not _has_valid_db_url(db_url):
        # Harmonized factor mapping met deduplicatie
        # - Wellbeing vragen (23-27) → 1 "Wellbeing" factor
        # - Resilience vragen (28-29) → 1 "Resilience" factor
        # Fallback mapping voor lokale analyse zonder questionnaire_factors uit de DB.
        # Deze ids zijn afgeleid uit de bestaande scorehistorie en huidige kolomdistributies.
        df_factors = pd.DataFrame([
            {'id': 1, 'slug': 'bmi'}, {'id': 2, 'slug': 'alcohol'}, {'id': 3, 'slug': 'exercise'},
            {'id': 4, 'slug': 'fruit'}, {'id': 5, 'slug': 'vegetables'}, {'id': 6, 'slug': 'smoking'},
            {'id': 7, 'slug': 'stress'}, {'id': 8, 'slug': 'sugar'}, {'id': 9, 'slug': 'fat'},
            {'id': 10, 'slug': 'salt'}, {'id': 11, 'slug': 'blood_pressure'}, {'id': 12, 'slug': 'diabetes'},
            {'id': 13, 'slug': 'resilience'}, {'id': 14, 'slug': 'wellbeing'}, {'id': 15, 'slug': 'selfefficacy'},
            {'id': 16, 'slug': 'dass_stress'}, {'id': 17, 'slug': 'dass_anxiety'}, {'id': 18, 'slug': 'dass_depression'},
            {'id': 19, 'slug': 'bmi'}, {'id': 20, 'slug': 'age'}, {'id': 21, 'slug': 'alcohol'},
            {'id': 22, 'slug': 'smoking'}, {'id': 23, 'slug': 'wellbeing'}, {'id': 24, 'slug': 'wellbeing'},
            {'id': 25, 'slug': 'wellbeing'}, {'id': 26, 'slug': 'wellbeing'}, {'id': 27, 'slug': 'wellbeing'},
            {'id': 28, 'slug': 'resilience'}, {'id': 29, 'slug': 'resilience'}, {'id': 32, 'slug': 'digital_detox_stress'},
            {'id': 33, 'slug': 'selfefficacy'}, {'id': 34, 'slug': 'digital_detox_addiction_risk'},
            {'id': 35, 'slug': 'menopause_somatic'}, {'id': 36, 'slug': 'menopause_psychological'},
            {'id': 37, 'slug': 'menopause_genitourinary'},
        ])

    # Defensieve kolomverwerking voor score historie
    if 'completion_created_at' in df_history.columns:
        df_history['completion_created_at'] = pd.to_datetime(df_history['completion_created_at'], errors='coerce')
    if 'score_value' in df_history.columns:
        df_history['score_value'] = pd.to_numeric(df_history['score_value'], errors='coerce')
    if 'questionnaire_factor_id' in df_history.columns:
        df_history['questionnaire_factor_id'] = pd.to_numeric(df_history['questionnaire_factor_id'], errors='coerce')
    df_factors['id'] = pd.to_numeric(df_factors['id'], errors='coerce')

    # Koppel slug via factor id
    df = df_history.merge(df_factors[['id', 'slug']], left_on='questionnaire_factor_id', right_on='id', how='left')

    # Voeg BMI-waarden uit users_met_scores toe als aanvullende longitudinaledata.
    # Dit voegt BMI-uit andere metingen toe (inclusief berekening via gewicht/lengte) zodat
    # bmi-trends vollediger worden weergegeven.
    df_bmi_from_latest = pd.DataFrame()
    if {'participant_id', 'rec_med_bmi'}.issubset(df_scores_latest.columns):
        df_bmi_from_latest = df_scores_latest[['participant_id', 'rec_med_bmi']].copy()
        df_bmi_from_latest['participant_id'] = pd.to_numeric(df_bmi_from_latest['participant_id'], errors='coerce')
        df_bmi_from_latest['score_value'] = pd.to_numeric(df_bmi_from_latest['rec_med_bmi'], errors='coerce')
        df_bmi_from_latest = df_bmi_from_latest.dropna(subset=['participant_id', 'score_value']).drop_duplicates('participant_id')

        # Get latest completion date for each participant to use as 'completion_created_at' for these BMI records
        latest_completion_dates = df_completions.groupby('participant_id')['created_at'].max().reset_index()
        latest_completion_dates.rename(columns={'created_at': 'completion_created_at'}, inplace=True)

        df_bmi_from_latest = df_bmi_from_latest.merge(latest_completion_dates, on='participant_id', how='left')

        # Assign other required columns to match df_history structure
        df_bmi_from_latest['id'] = pd.NA # No specific ID for these synthetic history records
        df_bmi_from_latest['questionnaire_factor_id'] = pd.NA # No specific factor ID
        df_bmi_from_latest['completion_id'] = pd.NA # No specific completion ID
        df_bmi_from_latest['slug'] = 'bmi'
        df_bmi_from_latest['score_category_value'] = pd.NA
        df_bmi_from_latest['created_at'] = df_bmi_from_latest['completion_created_at']
        df_bmi_from_latest['updated_at'] = df_bmi_from_latest['completion_created_at']

        needed_cols = [
            'id', 'participant_id', 'questionnaire_factor_id', 'completion_id',
            'score_value', 'score_category_value', 'completion_created_at',
            'created_at', 'updated_at', 'slug'
        ]
        df_bmi_from_latest = df_bmi_from_latest[needed_cols] # Select and order columns

        df = pd.concat([df, df_bmi_from_latest], ignore_index=True, sort=False)

    # Koppel gender via de geconsolideerde latest scores.
    gender_per_participant = pd.DataFrame(columns=['participant_id', 'user_gender'])
    if {'participant_id', 'rec_user_gender'}.issubset(df_scores_latest.columns):
        gender_per_participant = (
            df_scores_latest[['participant_id', 'rec_user_gender']]
            .rename(columns={'rec_user_gender': 'user_gender'})
            .dropna(subset=['participant_id'])
            .drop_duplicates(subset='participant_id', keep='first')
        )
    gender_per_participant['user_gender'] = pd.to_numeric(gender_per_participant['user_gender'], errors='coerce')

    df = df.merge(gender_per_participant, on='participant_id', how='left')
    
    # Koppel store_id: participant_id -> user_id -> store_id (via store_employees)
    # Initialiseer store_id als NA om KeyErrors in latere functies te voorkomen.
    if 'store_id' not in df.columns:
        df['store_id'] = pd.NA

    # Normaliseer store_employees: sommige versies gebruiken employee_id ipv user_id
    if 'employee_id' in df_store_employees.columns and 'user_id' not in df_store_employees.columns:
        df_store_employees = df_store_employees.rename(columns={'employee_id': 'user_id'})

    participant_store_maps: list[pd.DataFrame] = []
    if 'user_id' in df_store_employees.columns and 'store_id' in df_store_employees.columns:
        df_store_link = df_store_employees[['user_id', 'store_id']].copy()
        df_store_link['user_id'] = pd.to_numeric(df_store_link['user_id'], errors='coerce').astype('Int64')
        df_store_link['store_id'] = pd.to_numeric(df_store_link['store_id'], errors='coerce').astype('Int64')
        df_store_link = df_store_link.dropna(subset=['user_id', 'store_id']).drop_duplicates()

        # Route 1: participants bevat direct user_id.
        if {'id', 'user_id'}.issubset(df_participants.columns):
            direct_map = df_participants[['id', 'user_id']].copy()
            direct_map['id'] = pd.to_numeric(direct_map['id'], errors='coerce')
            direct_map['user_id'] = pd.to_numeric(direct_map['user_id'], errors='coerce')
            direct_map = ( # type: ignore
                direct_map.dropna(subset=['id', 'user_id'])
                .merge(df_store_link, on='user_id', how='left')
                .rename(columns={'id': 'participant_id'})
            )
            participant_store_maps.append(direct_map[['participant_id', 'store_id']])

        # Route 2: participants.public_id -> my_clic.public_id -> user_id.
        if not df_my_clic.empty and {'id', 'public_id'}.issubset(df_participants.columns) and {'public_id', 'user_id'}.issubset(df_my_clic.columns):
            public_map = df_participants[['id', 'public_id']].copy()
            public_map = public_map.dropna(subset=['id', 'public_id'])
            my_clic_public = df_my_clic[['public_id', 'user_id']].copy()
            my_clic_public['user_id'] = pd.to_numeric(my_clic_public['user_id'], errors='coerce')
            public_map = ( # type: ignore
                public_map.merge(my_clic_public.dropna(subset=['public_id', 'user_id']), on='public_id', how='left')
                .merge(df_store_link, on='user_id', how='left')
                .rename(columns={'id': 'participant_id'})
            )
            participant_store_maps.append(public_map[['participant_id', 'store_id']])

        # Route 3: participants.pmo_id -> my_clic.id -> user_id.
        if not df_my_clic.empty and {'id', 'pmo_id'}.issubset(df_participants.columns) and {'id', 'user_id'}.issubset(df_my_clic.columns):
            pmo_map = df_participants[['id', 'pmo_id']].copy()
            pmo_map['pmo_id'] = pd.to_numeric(pmo_map['pmo_id'], errors='coerce')
            my_clic_id = df_my_clic[['id', 'user_id']].copy()
            my_clic_id['id'] = pd.to_numeric(my_clic_id['id'], errors='coerce')
            my_clic_id['user_id'] = pd.to_numeric(my_clic_id['user_id'], errors='coerce')
            pmo_map = ( # type: ignore
                pmo_map.dropna(subset=['id', 'pmo_id'])
                .merge(my_clic_id.dropna(subset=['id', 'user_id']), left_on='pmo_id', right_on='id', how='left')
                .merge(df_store_link, on='user_id', how='left')
                .rename(columns={'id_x': 'participant_id'})
            )
            participant_store_maps.append(pmo_map[['participant_id', 'store_id']])

    participant_store_map = (
        pd.concat(participant_store_maps, ignore_index=True)
        if participant_store_maps
        else pd.DataFrame(columns=['participant_id', 'store_id'])
    )
    participant_store_map['participant_id'] = pd.to_numeric(participant_store_map['participant_id'], errors='coerce')
    participant_store_map['store_id'] = pd.to_numeric(participant_store_map['store_id'], errors='coerce')
    participant_store_map = participant_store_map.dropna(subset=['participant_id', 'store_id'])
    participant_store_map = participant_store_map.drop_duplicates(subset=['participant_id'], keep='first')

    if not participant_store_map.empty:
        df = df.merge(participant_store_map[['participant_id', 'store_id']], on='participant_id', how='left', suffixes=('', '_new'))
        if 'store_id_new' in df.columns:
            df['store_id'] = df['store_id_new'].combine_first(df['store_id'])
            df = df.drop(columns=['store_id_new'])

    # De longitudinale dataset wordt soms met een mix van datetijd- en stringwaarden samengesteld.
    # Normaliseer de kolom defensief voordat we .dt-accessors gebruiken.
    if 'completion_created_at' in df.columns:
        df['completion_created_at'] = pd.to_datetime(df['completion_created_at'], errors='coerce')
    df = df.dropna(subset=['completion_created_at'])
    df['jaar'] = df['completion_created_at'].dt.year
    # Use all available years (based on data), but keep 2019 and onwards
    jaren_geldig = sorted(df['jaar'].dropna().unique())
    jaren_geldig = [y for y in jaren_geldig if y >= 2019]
    return df[df['jaar'].isin(jaren_geldig)]


def bereken_verandering(df_history_geldig: pd.DataFrame, split_gender: bool = False) -> pd.DataFrame:
    df_sorted = df_history_geldig.sort_values(['participant_id', 'slug', 'completion_created_at'])
    df_eerste = (
        df_sorted.groupby(['participant_id', 'slug']).first().reset_index()
        [['participant_id', 'slug', 'score_value', 'completion_created_at']]
        .rename(columns={'score_value': 'eerste_score', 'completion_created_at': 'eerste_datum'})
    )
    df = df_sorted.merge(df_eerste, on=['participant_id', 'slug'], how='left')
    df['verandering'] = df['score_value'] - df['eerste_score']
    df = df[df['completion_created_at'] > df['eerste_datum']]
    df['dagen'] = (df['completion_created_at'] - df['eerste_datum']).dt.days
    df['periode'] = pd.cut(
        df['dagen'],
        bins=[0, 180, 365, 730, 1095, 99999],
        labels=['0-6 mnd', '6-12 mnd', '1-2 jaar', '2-3 jaar', '3+ jaar']
    )
    gender_group = []
    if split_gender and 'user_gender' in df.columns:
        df['geslacht_label'] = pd.to_numeric(df['user_gender'], errors='coerce').map(GENDER_LABELS)
        gender_group = ['geslacht_label']

    df_per_participant = (
        df.dropna(subset=['periode'])
        .groupby(['participant_id', 'slug', 'periode'] + gender_group, observed=False)
        .agg(verandering_per_participant=('verandering', 'mean'))
        .reset_index()
    )
    df_trend = df_per_participant.groupby(['periode', 'slug'] + gender_group, observed=False).agg(
        mean=('verandering_per_participant', 'mean'),
        count=('verandering_per_participant', 'count'),
        n_participanten=('participant_id', 'nunique'),
    ).reset_index()
    return df_trend[df_trend['n_participanten'] >= 1]


def _weighted_average(values, weights):
    """
    Bereken gewogen gemiddelde.
    
    Parameters:
    -----------
    values : array-like
        De waarden om gemiddelde van te berekenen
    weights : array-like
        De gewichten (bijv. aantal deelnemers per meetpunt)
    
    Returns:
    --------
    float
        Het gewogen gemiddelde
    """
    if len(values) == 0:
        return pd.NA
    values_arr = pd.to_numeric(values, errors='coerce')
    weights_arr = pd.to_numeric(weights, errors='coerce')
    # Verwijder missende of niet-positieve gewichten
    mask = ~(values_arr.isna() | weights_arr.isna()) & (weights_arr > 0)
    if not mask.any():
        return pd.NA
    total_weight = weights_arr[mask].sum()
    if total_weight <= 0:
        return pd.NA
    return (values_arr[mask] * weights_arr[mask]).sum() / total_weight


def get_available_stores(base_pad: Path, db_url: str) -> pd.DataFrame:
    """
    Get all stores that have score history data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame met store_id en store_name kolommen, sorted by store_name
    """
    # Load data with store_id
    df_history = laad_longitudinale_data(base_pad, db_url)
    
    # Get unique store IDs with data
    stores_with_data = df_history['store_id'].dropna().unique()
    
    if len(stores_with_data) == 0:
        return pd.DataFrame({'store_id': [], 'store_name': []})
    
    stores_df = load_stores(db_url or DB_URL)
    if stores_df.empty:
        return pd.DataFrame({'store_id': [], 'store_name': []})
        
    stores_df = stores_df[stores_df['id'].isin(stores_with_data)].copy()
    
    stores_df = stores_df.rename(columns={'id': 'store_id', 'name': 'store_name'})
    stores_df = stores_df.sort_values('store_name').reset_index(drop=True)
    
    return stores_df


def laad_store_average_scores(base_pad: Path) -> pd.DataFrame:
    """
    Laad gemiddelde scores per store uit parquet en koppel store-namen.
    Laad gemiddelde scores per store uit database en koppel store-namen.
    """
    df_scores = load_table_from_database('store_average_scores', DB_URL)

    if df_scores.empty:
        return pd.DataFrame(columns=['store_id', 'store_name', 'average', 'date', 'participants_count', 'score_slug'])
    
    df_stores = load_stores(DB_URL).copy()
    
    # Zorg dat id en name kolommen bestaan voordat we dropna doen
    if 'id' not in df_stores.columns: df_stores['id'] = pd.NA
    if 'name' not in df_stores.columns: df_stores['name'] = 'Onbekend'
    df_stores = df_stores.dropna(subset=['id']).drop_duplicates(subset='id')[['id', 'name']].copy()
    df_stores = df_stores.rename(columns={'id': 'store_id', 'name': 'store_name'})
    df_stores['store_id'] = pd.to_numeric(df_stores['store_id'], errors='coerce')

    # Defensieve verwerking van scores kolommen om KeyErrors te voorkomen
    for col in ['store_id', 'average', 'participants_count']:
        if col in df_scores.columns:
            df_scores[col] = pd.to_numeric(df_scores[col], errors='coerce')
        else:
            df_scores[col] = pd.NA

    if 'date' in df_scores.columns:
        # Probeer datums robuust te parsen. dayfirst=True helpt bij NL/database formaten
        df_scores['date'] = pd.to_datetime(df_scores['date'], errors='coerce', dayfirst=True)
    else:
        df_scores['date'] = pd.NaT

    if 'score_slug' not in df_scores.columns:
        df_scores['score_slug'] = pd.NA

    df_scores = df_scores.merge(df_stores, on='store_id', how='left')
    if 'store_name' in df_scores.columns:
        fallback_names = df_scores['store_id'].apply(
            lambda store_id: f"Store #{int(store_id)}" if pd.notna(store_id) else "Onbekend"
        )
        df_scores['store_name'] = df_scores['store_name'].fillna(fallback_names)

    # NIEUW: Koppel partners aan de scores om op opdrachtgeverniveau te kunnen aggregeren
    df_partners = load_table_from_database('partners', DB_URL)
    if not df_partners.empty and 'partner_id' in df_stores.columns:
        df_store_meta = df_stores.merge(
            df_partners[['id', 'name']].rename(columns={'id': 'partner_id', 'name': 'partner_name'}),
            on='partner_id', how='left'
        )
        df_scores = df_scores.merge(
            df_store_meta[['store_id', 'partner_name']], 
            on='store_id', how='left'
        )
        # Gebruik Partner naam als primaire groepering, fallback naar Store naam
        df_scores['store_name'] = df_scores['partner_name'].fillna(df_scores['store_name'])

    return df_scores.dropna(subset=['store_id', 'average', 'date'])


def _transform_store_score_values(df_scores: pd.DataFrame, score_slug: str) -> tuple[pd.DataFrame, str]:
    """
    Zet scorewaarden om naar beter interpreteerbare schaal waar nodig.
    Geeft getransformeerde dataframe en een suffix voor labels terug.
    """
    df_scores = df_scores.copy()
    label_suffix = ''

    if score_slug == 'rec_ls_exercise_steps_per_day':
        # Deze score is opgeslagen in duizenden stappen per dag.
        df_scores['average'] = df_scores['average'] * 1000

    return df_scores, label_suffix


def get_available_stores_from_average_scores(base_pad: Path) -> pd.DataFrame:
    """
    Geef stores terug waarvoor gemiddelde scorehistorie beschikbaar is.
    """
    df_scores = laad_store_average_scores(base_pad)
    stores_df = (
        df_scores[['store_id', 'store_name']]
        .drop_duplicates()
        .sort_values('store_name')
        .reset_index(drop=True)
    )
    return stores_df


def get_available_stores_with_score_changes(base_pad: Path, score_slug: str,
                                            min_deelnemers: int = 1) -> pd.DataFrame:
    """
    Geef alleen stores terug die voor de gekozen score minstens één bruikbaar
    veranderingspunt hebben na filtering op minimum deelnemers.
    """
    df_trends = _prepare_store_score_trends(
        base_pad=base_pad,
        score_slug=score_slug,
        min_deelnemers=min_deelnemers,
        store_id=None,
        aggregate_all=False,
    )
    if df_trends.empty:
        return pd.DataFrame(columns=['store_id', 'store_name'])

    geldige_store_ids = (
        df_trends.groupby(['store_id', 'store_name'])['gemiddelde_score']
        .apply(lambda s: s.notna().any())
        .reset_index(name='heeft_data')
    )
    geldige_store_ids = geldige_store_ids[geldige_store_ids['heeft_data']]

    return (
        geldige_store_ids[['store_id', 'store_name']]
        .sort_values('store_name')
        .reset_index(drop=True)
    )


def _prepare_store_score_trends(base_pad: Path, score_slug: str,
                                min_deelnemers: int = 1,
                                store_id: int = None,
                                aggregate_all: bool = False,
                                participant_ids: list | set | pd.Series | None = None) -> pd.DataFrame:
    history_slug = _dashboard_score_to_history_slug(score_slug)
    df_history = laad_longitudinale_data(base_pad, DB_URL)
    df_scores = df_history[df_history['slug'] == history_slug].copy()
    if participant_ids is not None and 'participant_id' in df_scores.columns:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        df_scores = df_scores[pd.to_numeric(df_scores['participant_id'], errors='coerce').isin(pids)]

    links = _laad_opdrachtgever_links(DB_URL)
    if not links.empty and 'participant_id' in df_scores.columns:
        df_scores = df_scores.drop(columns=[c for c in ['store_id', 'store_name'] if c in df_scores.columns])
        df_scores = df_scores.merge(
            links[['participant_id', 'store_id', 'store_name']],
            on='participant_id',
            how='inner',
        )
    else:
        # Zonder koppeling geen store-kolommen: vul ze als leeg zodat de
        # groepering hieronder niet op een KeyError stukloopt.
        for _kolom in ('store_id', 'store_name'):
            if _kolom not in df_scores.columns:
                df_scores[_kolom] = pd.NA

    if df_scores.empty:
        empty = pd.DataFrame(columns=[
            'store_id', 'store_name', 'jaar_maand', 'gemiddelde_score',
            'gem_deelnemers', 'n_dagen', 'gemiddelde_verandering'
        ])
        empty.attrs['label_suffix'] = ''
        return empty

    df_scores = df_scores.rename(columns={'score_value': 'average'})
    df_scores['participants_count'] = 1
    df_scores, label_suffix = _transform_store_score_values(df_scores, score_slug)

    if store_id is not None:
        df_scores = df_scores[df_scores['store_id'] == store_id]

    df_scores['average'] = pd.to_numeric(df_scores['average'], errors='coerce')
    df_scores['completion_created_at'] = pd.to_datetime(df_scores['completion_created_at'], errors='coerce')
    df_scores = df_scores.dropna(subset=['store_id', 'store_name', 'average', 'completion_created_at'])
    if df_scores.empty:
        # Voorkom KeyError door de kolom alsnog (leeg) aan te maken
        df_scores['gemiddelde_verandering'] = pd.Series(dtype=float)
        return df_scores

    df_scores['jaar_maand'] = df_scores['completion_created_at'].dt.to_period('M').dt.to_timestamp()

    df_per_deelnemer = (
        df_scores.groupby(['store_id', 'store_name', 'jaar_maand', 'participant_id'], as_index=False)
        .agg(
            score_per_deelnemer=('average', 'mean'),
            n_metingen=('average', 'count'),
            eerste_meting=('completion_created_at', 'min'),
        )
    )

    df_maand = (
        df_per_deelnemer.groupby(['store_id', 'store_name', 'jaar_maand'], as_index=False)
        .agg(
            gemiddelde_score=('score_per_deelnemer', 'mean'),
            n_deelnemers=('participant_id', 'nunique'),
            n_metingen=('n_metingen', 'sum'),
            n_dagen=('eerste_meting', lambda s: s.dt.date.nunique())
        )
    )

    # Bereken totaalgemiddelde over alle data voor de referentielijn
    df_totaal = df_per_deelnemer.groupby('jaar_maand').agg(
        totaal_gemiddelde=('score_per_deelnemer', 'mean')
    ).reset_index()

    min_deelnemers = max(1, int(min_deelnemers))
    df_maand = df_maand[df_maand['n_deelnemers'] >= min_deelnemers].copy()
    df_maand = df_maand.merge(df_totaal, on='jaar_maand', how='left')

    if df_maand.empty:
        return df_maand

    df_maand = df_maand.sort_values(['store_name', 'jaar_maand']).reset_index(drop=True)

    if df_maand.empty:
        return df_maand

    df_maand['gemiddelde_verandering'] = (
        df_maand
        .groupby('store_id')['gemiddelde_score']
        .diff()
        .fillna(0.0)
    )

    df_maand.attrs['label_suffix'] = label_suffix
    return df_maand


def _empty_store_plot(melding: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=melding, showarrow=False, font=dict(size=13))
    fig.update_layout(template='plotly_white', height=450)
    return fig


def maak_store_gemiddelde_scores_plot(base_pad: Path, score_slug: str,
                                      score_label: str, min_deelnemers: int = 1,
                                      store_id: int = None,
                                      richting_label: str = '',
                                      participant_ids: list | set | pd.Series | None = None) -> go.Figure:
    df_plot = _prepare_store_score_trends(
        base_pad=base_pad,
        score_slug=score_slug,
        min_deelnemers=min_deelnemers,
        store_id=store_id,
        participant_ids=participant_ids,
    )

    if df_plot.empty:
        return _empty_store_plot("Geen scoredata beschikbaar voor deze selectie.")

    label_suffix = df_plot.attrs.get('label_suffix', '')
    y_label = f'Gemiddelde {score_label.lower()}{label_suffix}'

    fig = px.line(
        df_plot,
        x='jaar_maand',
        y='gemiddelde_score',
        color='store_name',
        markers=True,
        custom_data=['gemiddelde_score', 'n_deelnemers', 'n_metingen', 'n_dagen'],
        labels={
            'jaar_maand': 'Periode',
            'gemiddelde_score': y_label,
            'store_name': 'Selectie',
        },
        title=f'{y_label} over tijd',
    )
    fig.update_traces(
        hovertemplate=(
            '<b>%{fullData.name}</b><br>' +
            'Periode: %{x|%b %Y}<br>' +
            'Maandgemiddelde: %{y:,.0f}<br>' if score_slug == 'rec_ls_exercise_steps_per_day'
            else 'Maandgemiddelde: %{y:.2f}<br>'
        ) + (
            'Werknemers: %{customdata[1]}<br>' +
            'Metingen: %{customdata[2]}<br>' +
            'Dagen met data: %{customdata[3]}<extra></extra>'
        )
    )
    if score_slug == 'rec_ls_exercise_steps_per_day':
        fig.update_yaxes(tickformat=',.0f')

    # Voeg referentielijn toe bij selectie van één bedrijf
    if store_id is not None and not df_plot.empty:
        fig.add_trace(go.Scatter(
            x=df_plot['jaar_maand'],
            y=df_plot['totaal_gemiddelde'],
            name=tr('Totaalgemiddelde (alle)', 'nl'),
            line=dict(color='gray', dash='dash', width=2),
            opacity=0.6,
            hovertemplate='Totaal: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend_title_text='Selectie',
    )
    return fig


def maak_store_gemiddelde_verandering_plot(base_pad: Path, score_slug: str,
                                           score_label: str, min_deelnemers: int = 1,
                                           store_id: int = None,
                                           richting_label: str = '',
                                           participant_ids: list | set | pd.Series | None = None) -> go.Figure:
    df_plot = _prepare_store_score_trends(
        base_pad=base_pad,
        score_slug=score_slug,
        min_deelnemers=min_deelnemers,
        store_id=store_id,
        participant_ids=participant_ids,
    )

    df_plot = df_plot.copy()
    if 'gemiddelde_verandering' not in df_plot.columns:
        return _empty_store_plot("Nog onvoldoende meetmomenten om verandering te tonen.")
    df_plot['gemiddelde_verandering'] = pd.to_numeric(df_plot['gemiddelde_verandering'], errors='coerce').fillna(0.0)
    if df_plot.empty:
        return _empty_store_plot("Nog onvoldoende meetmomenten om verandering te tonen.")

    label_suffix = df_plot.attrs.get('label_suffix', '')
    y_label = f'Gemiddelde verandering in {score_label.lower()}{label_suffix}'

    fig = px.line(
        df_plot,
        x='jaar_maand',
        y='gemiddelde_verandering',
        color='store_name',
        markers=True,
        custom_data=['gemiddelde_verandering', 'n_deelnemers', 'n_metingen', 'n_dagen'],
        labels={
            'jaar_maand': 'Periode',
            'gemiddelde_verandering': y_label,
            'store_name': 'Selectie',
        },
        title=f'{y_label} over tijd',
    )
    fig.update_traces(
        hovertemplate=(
            '<b>%{fullData.name}</b><br>' +
            'Periode: %{x|%b %Y}<br>' +
            'Maandverandering: %{y:,.0f}<br>' if score_slug == 'rec_ls_exercise_steps_per_day'
            else 'Maandverandering: %{y:.2f}<br>'
        ) + (
            'Werknemers: %{customdata[1]}<br>' +
            'Metingen: %{customdata[2]}<br>' +
            'Dagen met data: %{customdata[3]}<extra></extra>'
        )
    )
    if score_slug == 'rec_ls_exercise_steps_per_day':
        fig.update_yaxes(tickformat=',.0f')
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend_title_text='Selectie',
    )
    return fig


def maak_store_scoreverbetering_plot(base_pad: Path, db_url: str, score_type: str = 'job_satisfaction',
                                      maanden_window: int = 3, store_id: int = None) -> go.Figure:
    """
    Toont score trend over tijd per maand, optioneel gefilterd per store.
    
    Parameters:
    -----------
    base_pad : Path
        Pad naar de data directory
    db_url : str
        Database URL
    score_type : str
        Welke score weergeven (bijv. 'job_satisfaction', 'stress', 'wellbeing')
    maanden_window : int
        Venstergrootte voor rolling average in maanden
    store_id : int, optional
        Store ID om te filteren. Als None, tonen alle stores samen.
    
    Returns:
    --------
    go.Figure
        Plotly line chart met trend over tijd
    """
    # Longitudinale data laden
    df_history = laad_longitudinale_data(base_pad, db_url)
    
    # Alleen de gekozen score-type
    df_history = df_history[df_history['slug'] == score_type].copy()
    if len(df_history) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"❌ Geen data beschikbaar voor '{score_type}'.", 
                          showarrow=False, font=dict(size=12))
        return fig
    
    # Filter op store_id als opgegeven
    store_name = "alle deelnemers"
    if store_id is not None and store_id > 0:
        df_history = df_history[df_history['store_id'] == store_id]
        if len(df_history) == 0:
            fig = go.Figure()
            fig.add_annotation(text=f"❌ Geen data beschikbaar voor store {store_id} en score '{score_type}'.", 
                              showarrow=False, font=dict(size=12))
            return fig
        # Get store name from the database
        try:
            if _has_valid_db_url(db_url):
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    result = pd.read_sql(f"SELECT name FROM stores WHERE id = {store_id}", conn)
            else:
                result = load_stores(DB_URL)[['id', 'name']]
                result = result[result['id'] == store_id]
            if len(result) > 0:
                store_name = result['name'].iloc[0]
            else:
                store_name = f"Store {store_id}"
        except:
            store_name = f"Store {store_id}"
    
    # Score waarden numeriek maken en opschonen
    df_history['score_value'] = pd.to_numeric(df_history['score_value'], errors='coerce')
    df_history = df_history.dropna(subset=['score_value', 'completion_created_at'])
    
    if len(df_history) == 0:
        fig = go.Figure()
        fig.add_annotation(text="❌ Geen geldige score data.", 
                          showarrow=False, font=dict(size=12))
        return fig
    
    # Eerst per deelnemer per maand middelen, zodat frequente invullers niet zwaarder tellen.
    df_history['jaar_maand'] = df_history['completion_created_at'].dt.to_period('M')
    df_participant_maand = (
        df_history.groupby(['participant_id', 'jaar_maand'])
        .agg(score_per_participant=('score_value', 'mean'))
        .reset_index()
    )

    # Aggregatie per maand
    df_agg = df_participant_maand.groupby('jaar_maand').agg(
        score_mean=('score_per_participant', 'mean'),
        score_std=('score_per_participant', 'std'),
        n_metingen=('score_per_participant', 'count'),
        n_participants=('participant_id', 'nunique'),
    ).reset_index()
    
    # Filter: minimaal 2 deelnemers per maand
    df_agg = df_agg[df_agg['n_participants'] >= 2]
    
    if len(df_agg) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Onvoldoende data<br>Slechts {len(df_agg)} maanden met genoeg metingen", 
            showarrow=False, font=dict(size=12)
        )
        return fig
    
    # Timestamp voor plotting
    df_agg['timestamp'] = df_agg['jaar_maand'].dt.to_timestamp()
    df_agg = df_agg.sort_values('timestamp')
    
    # Voortschrijdend gemiddelde
    df_agg['score_smooth'] = df_agg['score_mean'].rolling(
        window=maanden_window, center=True, min_periods=1
    ).mean()
    df_agg['score_std_smooth'] = df_agg['score_std'].rolling(
        window=maanden_window, center=True, min_periods=1
    ).mean()
    
    # Plotly figure
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=df_agg['timestamp'],
        y=df_agg['score_smooth'],
        mode='lines+markers',
        name='Gemiddelde score',
        line=dict(width=3, color=HOOFD_KLEUR),
        marker=dict(size=8),
        hovertemplate=(
            '<b>Periode</b><br>' +
            'Datum: %{x|%b %Y}<br>' +
            'Gemiddelde: %{y:.2f}<br>' +
            'Metingen: %{customdata[0]}<br>' +
            'Deelnemers: %{customdata[1]}<extra></extra>'
        ),
        customdata=df_agg[['n_metingen', 'n_participants']].astype(int).values,
    ))
    
    # Std band
    fig.add_trace(go.Scatter(
        x=df_agg['timestamp'],
        y=df_agg['score_smooth'] + df_agg['score_std_smooth'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=df_agg['timestamp'],
        y=df_agg['score_smooth'] - df_agg['score_std_smooth'],
        mode='lines',
        line=dict(width=0),
        name='±1 Std Dev',
        fillcolor='rgba(232, 119, 34, 0.2)',
        fill='tonexty',
        hoverinfo='skip',
    ))
    
    # Labels
    score_labels = {
        'job_satisfaction': 'Werktevredenheid',
        'stress': 'Stressniveau',
        'wellbeing': 'Welzijn',
        'work_life_balance': 'Work-life balans',
        'workload': 'Werkdruk',
        'working_attitude': 'Werkhouding',
        'exercise': 'Beweging',
        'bmi': 'BMI',
        'dass_stress': 'DASS Stress',
        'dass_anxiety': 'DASS Angst',
        'dass_depression': 'DASS Depressie',
        'resilience': 'Veerkracht',
    }
    score_label_nl = score_labels.get(score_type, score_type)
    
    fig.update_layout(
        title=f'{score_label_nl} trend over tijd<br><sub>Voortschrijdend gemiddelde ({maanden_window} mnd), {store_name}</sub>',
        xaxis_title='Periode',
        yaxis_title=f'Gemiddelde {score_label_nl.lower()}',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    )
    
    return fig


def maak_bmi_beweging_plot(df: pd.DataFrame, geslacht: str = 'beide', lang: str = 'nl'):
    """
    Line plot van BMI en Bewegingsminuten per week over tijd.
    Gebruikt jaargemiddelden. Dubbele y-as: BMI links, beweging rechts.
    """
    df_bmi_exercise = df[df['slug'].isin(['bmi', 'exercise'])].copy()

    df_bmi_exercise['score_value'] = pd.to_numeric(df_bmi_exercise['score_value'], errors='coerce')
    df_bmi_exercise = df_bmi_exercise.dropna(subset=['score_value', 'completion_created_at', 'participant_id'])
    df_bmi_exercise['jaar'] = df_bmi_exercise['completion_created_at'].dt.year
    jaren_geldig = sorted(df_bmi_exercise['jaar'].dropna().unique())
    jaren_geldig = [y for y in jaren_geldig if y >= 2019]
    df_bmi_exercise = df_bmi_exercise[df_bmi_exercise['jaar'].isin(jaren_geldig)]

    if 'user_gender' in df_bmi_exercise.columns:
        df_bmi_exercise['geslacht_label'] = pd.to_numeric(df_bmi_exercise['user_gender'], errors='coerce').map({1: tr('Man', lang), 0: tr('Vrouw', lang)})

    if geslacht in ('man', 'vrouw') and 'geslacht_label' in df_bmi_exercise.columns:
        doelfilter = tr('Man', lang) if geslacht == 'man' else tr('Vrouw', lang)
        df_bmi_exercise = df_bmi_exercise[df_bmi_exercise['geslacht_label'] == doelfilter]

    df_bmi_exercise['score_value'] = pd.to_numeric(df_bmi_exercise['score_value'], errors='coerce')
    df_bmi_exercise = df_bmi_exercise.dropna(subset=['score_value', 'completion_created_at', 'participant_id'])
    df_bmi_exercise['jaar'] = df_bmi_exercise['completion_created_at'].dt.year
    jaren_geldig = sorted(df_bmi_exercise['jaar'].dropna().unique())
    jaren_geldig = [y for y in jaren_geldig if y >= 2019]
    df_bmi_exercise = df_bmi_exercise[df_bmi_exercise['jaar'].isin(jaren_geldig)]

    df_bmi_exercise = (
        df_bmi_exercise.groupby(['participant_id', 'jaar', 'slug'] + (['geslacht_label'] if 'geslacht_label' in df_bmi_exercise.columns else []))
        .agg(score_per_participant=('score_value', 'mean'))
        .reset_index()
    )

    if geslacht == 'beide' and 'geslacht_label' in df_bmi_exercise.columns:
        df_plot = (
            df_bmi_exercise.groupby(['jaar', 'slug', 'geslacht_label'])['score_per_participant']
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot_bmi = df_plot[df_plot['slug'] == 'bmi']
        if not df_plot_bmi.empty:
            pivot = df_plot_bmi.pivot(index='jaar', columns='geslacht_label', values='score_per_participant')
            pivot.plot(kind='bar', ax=ax, color={tr('Man', lang): '#1F6FBF', tr('Vrouw', lang): '#E67E22'})
            ax.set_ylabel('Gemiddelde BMI')
            ax.set_xlabel('Jaar')
            ax.set_title(f'BMI per jaar {tr("naar geslacht", lang)}')
            ax.legend(title=tr('Geslacht', lang))
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            return fig

    n_per_jaar = df_bmi_exercise.groupby('jaar')['participant_id'].nunique()
    df_pivot = df_bmi_exercise.groupby(['jaar', 'slug'])['score_per_participant'].mean().unstack()

    fig, ax_bmi = plt.subplots(figsize=(10, 5))
    ax_exercise = ax_bmi.twinx()

    if 'bmi' in df_pivot.columns:
        ax_bmi.plot(df_pivot.index, df_pivot['bmi'], marker='o',
                    color='salmon', linewidth=2, markersize=7, label='BMI')
        for jaar, val in df_pivot['bmi'].items():
            ax_bmi.annotate(f'{val:.1f}', (jaar, val),
                            textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=8, color='salmon')

    if 'exercise' in df_pivot.columns:
        ax_exercise.plot(df_pivot.index, df_pivot['exercise'], marker='o',
                         color='steelblue', linewidth=2, markersize=7, label='Bewegingsminuten/week')
        for jaar, val in df_pivot['exercise'].items():
            ax_exercise.annotate(f'{val:.0f}', (jaar, val),
                                 textcoords='offset points', xytext=(0, 8),
                                 ha='center', fontsize=8, color='steelblue')

    ax_bmi.set_xticks(df_pivot.index)
    ax_bmi.set_xticklabels([f"{int(j)}\n(n={n_per_jaar.get(j, 0)})" for j in df_pivot.index], fontsize=9)
    ax_bmi.set_xlabel('Jaar')
    ax_bmi.set_ylabel('Gemiddelde BMI', color='salmon')
    ax_exercise.set_ylabel('Gemiddelde bewegingsminuten per week', color='steelblue')
    ax_bmi.tick_params(axis='y', labelcolor='salmon')
    ax_exercise.tick_params(axis='y', labelcolor='steelblue')
    titel_suffix = {
        'man': f" — {tr('mannen', lang)}",
        'vrouw': f" — {tr('vrouwen', lang)}",
        'beide': '',
    }.get(geslacht, '')
    ax_bmi.set_title(f'BMI vs Beweging over tijd{titel_suffix}')
    ax_bmi.legend(loc='upper left', fontsize=9)
    ax_exercise.legend(loc='upper right', fontsize=9)
    ax_bmi.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig

def maak_gemiddelde_score_over_tijd_plot(df_long: pd.DataFrame, slug: str, lang: str = 'nl', geslacht: str = 'beide') -> go.Figure:
    """
    Toont de gemiddelde score van een specifieke factor over tijd, plus
    gewogen gemiddelden per jaar waarbij maanden met meer deelnemers zwaarder meetellen.
    """
    df_factor = df_long[df_long['slug'] == slug].copy()

    if geslacht == 'man':
        df_factor = df_factor[df_factor['user_gender'] == 1]
    elif geslacht == 'vrouw':
        df_factor = df_factor[df_factor['user_gender'] == 0]

    df_factor['completion_created_at'] = pd.to_datetime(df_factor['completion_created_at'], errors='coerce')
    df_factor = df_factor.dropna(subset=['score_value', 'completion_created_at'])

    if df_factor.empty:
        fig = go.Figure()
        fig.add_annotation(text=tr("Geen data beschikbaar voor deze factor en selectie.", lang), showarrow=False, font=dict(size=14))
        fig.update_layout(template='plotly_white', height=450)
        return fig

    # Aggregeer per maand - bereken eerst per participant, dan gemiddelde
    df_factor['jaar_maand'] = df_factor['completion_created_at'].dt.to_period('M').dt.to_timestamp()
    
    # Per participant per maand: gemiddelde score
    df_participant_maand = df_factor.groupby(['participant_id', 'jaar_maand']).agg(
        score_per_participant=('score_value', 'mean'),
    ).reset_index()

    split_gender = geslacht == 'beide' and 'user_gender' in df_factor.columns
    if split_gender:
        gender_map = {1: tr('Man', lang), 0: tr('Vrouw', lang)}
        participant_gender = df_factor[['participant_id', 'user_gender']].drop_duplicates('participant_id')
        participant_gender['geslacht_label'] = pd.to_numeric(participant_gender['user_gender'], errors='coerce').map(gender_map)
        df_participant_maand = df_participant_maand.merge(
            participant_gender[['participant_id', 'geslacht_label']], on='participant_id', how='left'
        ).dropna(subset=['geslacht_label'])
    
    # Aantal unieke deelnemers per maand
    group_columns = ['jaar_maand', 'geslacht_label'] if split_gender else ['jaar_maand']
    df_agg = df_participant_maand.groupby(group_columns).agg(
        gemiddelde_score=('score_per_participant', 'mean'),  # Gemiddelde van participantgemiddelden
        n_deelnemers=('participant_id', 'nunique'),
    ).reset_index()

    df_agg = df_agg[df_agg['n_deelnemers'] >= 1]

    if df_agg.empty:
        fig = go.Figure()
        fig.add_annotation(text=tr("Onvoldoende deelnemers per periode om een trend te tonen.", lang), showarrow=False, font=dict(size=14))
        fig.update_layout(template='plotly_white', height=450)
        return fig

    df_agg['jaar'] = df_agg['jaar_maand'].dt.year
    df_jaar = pd.DataFrame([
        {
            'jaar': jaar,
            'gewogen_gemiddelde': _weighted_average(group['gemiddelde_score'], group['n_deelnemers']),
            'start_datum': group['jaar_maand'].min(),
            'eind_datum': group['jaar_maand'].max(),
        }
        for jaar, group in df_agg.groupby('jaar')
    ])

    # Get factor label from variabelen.py or use slug
    factor_label = slug
    for group_vars in VARIABELEN_PER_GROEP.values():
        for var_spec in group_vars:
            if var_spec.get('kolom') == slug:
                factor_label = var_spec.get('label', slug)
                break
        if factor_label != slug:
            break
    
    # Translate label
    factor_label = tr(factor_label, lang)

    fig = px.line(
        df_agg,
        x='jaar_maand',
        y='gemiddelde_score',
        color='geslacht_label' if split_gender else None,
        color_discrete_map={tr('Man', lang): GENDER_COLORS['Man'], tr('Vrouw', lang): GENDER_COLORS['Vrouw']} if split_gender else None,
        markers=True,
        custom_data=['n_deelnemers'],
        labels={
            'jaar_maand': tr('Datum', lang),
            'gemiddelde_score': tr(f'Gemiddelde {factor_label.lower()}', lang),
        },
        title=tr(f'Gemiddelde {factor_label} over tijd naar geslacht' if split_gender else f'Gemiddelde {factor_label} over tijd', lang),
        color_discrete_sequence=[HOOFD_KLEUR] if not split_gender else None,
    )
    fig.update_traces(
        hovertemplate=(
            '<b>%{x|%b %Y}</b><br>' +
            tr('Gemiddelde', lang) + ': %{y:.2f}<br>' +
            tr('Deelnemers', lang) + ': %{customdata[0]}<extra></extra>'
        )
    )
    for i, jaar_row in df_jaar.iterrows():
        if split_gender:
            break
        gewogen_gemiddelde = jaar_row['gewogen_gemiddelde']
        if pd.isna(gewogen_gemiddelde):
            continue
        fig.add_trace(
            go.Scatter(
                x=[jaar_row['start_datum'], jaar_row['eind_datum']],
                y=[float(gewogen_gemiddelde), float(gewogen_gemiddelde)],
                mode='lines',
                name=tr('Gewogen gemiddelde per jaar', lang),
                line=dict(color='dimgray', dash='dash', width=2),
                hovertemplate=(
                    f"{jaar_row['jaar']}<br>" +
                    tr('Gewogen gemiddelde', lang) + ': %{y:.2f}<br>' +
                    tr('Gewogen op aantal deelnemers per maand', lang) +
                    '<extra></extra>'
                ),
                showlegend=(i == 0),
            )
        )
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        height=500,
    )
    return fig

def maak_scoreverandering_plot(df_trend: pd.DataFrame, lang: str = 'nl', geslacht: str = 'totaal'):
    alle_factoren = sorted(df_trend['slug'].dropna().unique())
    factor_labels = {
        'age': tr('Leeftijd', lang),
        'bmi': 'BMI',
        'alcohol': tr('Alcohol', lang),
        'exercise': tr('Beweging', lang),
        'fruit': tr('Fruit', lang),
        'vegetables': tr('Groenten', lang),
        'smoking': tr('Roken', lang),
        'stress': tr('Stress', lang),
        'sugar': tr('Suiker', lang),
        'fat': tr('Vet', lang),
        'salt': tr('Zout', lang), 
        'blood_pressure': tr('Bloeddruk', lang),
        'diabetes': tr('Diabetes', lang),
        'dass_stress': 'DASS Stress',
        'dass_anxiety': tr('DASS Angst', lang),
        'dass_depression': tr('DASS Depressie', lang),
        'sleep': tr('Slaap', lang),
        'steps': tr('Stappen', lang),
        'resilience': tr('Veerkracht', lang),
        'wellbeing': tr('Welzijn', lang),
        'selfefficacy': tr('Zelfeffectiviteit', lang),
        'health': tr('Gezondheid', lang),
        'digital_detox_stress': tr('Digitale detox stress', lang),
        'digital_detox_addiction_risk': tr('Smartphone verslaving', lang),
        'menopause_somatic': tr('Menopauze (somatisch)', lang),
        'menopause_psychological': tr('Menopauze (psychologisch)', lang),
        'menopause_genitourinary': tr('Menopauze (genitourinair)', lang),
    }
    # For unknown factors, use the slug as label
    for f in alle_factoren:
        if f not in factor_labels:
            factor_labels[f] = f.replace('_', ' ').title()
    
    lager_is_beter = [
        'alcohol', 'smoking', 'stress', 'bmi', 'sugar', 'fat', 'salt', 'blood_pressure', 'diabetes',
        'dass_stress', 'dass_anxiety', 'dass_depression', 'sleep',
        'digital_detox_stress', 'digital_detox_addiction_risk',
        'menopause_somatic', 'menopause_psychological', 'menopause_genitourinary',
    ]
    factoren_aanwezig = alle_factoren

    split_gender = geslacht == 'beide' and 'geslacht_label' in df_trend.columns
    cols = 3
    rows = (len(factoren_aanwezig) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
    axes = axes.flatten()

    for i, factor in enumerate(factoren_aanwezig):
        df_factor = df_trend[df_trend['slug'] == factor]
        if split_gender:
            for gender_code, kleur in ((tr('Man', lang), GENDER_COLORS['Man']), (tr('Vrouw', lang), GENDER_COLORS['Vrouw'])):
                subset = df_factor[df_factor['geslacht_label'] == gender_code]
                axes[i].plot(subset['periode'].astype(str), subset['mean'], marker='o', color=kleur, label=gender_code)
            axes[i].legend(fontsize=7)
        else:
            if factor in lager_is_beter:
                kleuren  = [RISICO_COLORS[0] if x <= 0 else RISICO_COLORS[2] for x in df_factor['mean']]
            else:
                kleuren  = [RISICO_COLORS[0] if x >= 0 else RISICO_COLORS[2] for x in df_factor['mean']]
            axes[i].bar(df_factor['periode'].astype(str), df_factor['mean'], color=kleuren)
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[i].set_title(f"{factor_labels.get(factor, factor)}", fontsize=9)
        axes[i].set_xlabel(tr('Periode na eerste meting', lang), fontsize=8)
        axes[i].set_ylabel(tr('Gemiddelde verandering', lang), fontsize=8)
        axes[i].tick_params(axis='x', rotation=45, labelsize=7)

    for j in range(len(factoren_aanwezig), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(tr('Scoreverandering ten opzichte van eerste meting per factor', lang), fontsize=14)
    plt.tight_layout()
    return fig


# ── Dieet score ────────────────────────────────────────────────────────────────
DIEET_SCORE_COLS = [
    'rec_ls_score_fruit', 'rec_ls_score_vegetables', 'rec_ls_score_sugar',
    'rec_ls_score_saturated_fat', 'rec_ls_score_alcohol', 'rec_ls_score_natrium'
]
DIEET_GRENZEN = [0, 2.0, 2.8, 3.4, 4.2, 5.0001]
DIEET_LABELS  = ['Slecht', 'Matig', 'Voldoende', 'Goed', 'Uitstekend']
DIEET_COLORS  = {
    'Slecht':     '#E74C3C',
    'Matig':      '#E87722',
    'Voldoende':  '#F4D03F',
    'Goed':       '#2ECC71',
    'Uitstekend': '#1A8A4A',
}


def voeg_dieet_score_toe(df: pd.DataFrame) -> pd.DataFrame:
    """Voegt dieet_score en dieet_cat kolommen toe aan het dataframe.

    Subscores horen op een 0-5 schaal te liggen. Waarden buiten dit bereik
    (bijv. ruwe grammen/mg invoer of foutieve metingen) worden als ongeldig
    behandeld, zodat extreme uitschieters de gemiddelden en visualisaties
    niet verstoren.
    """
    df = df.copy()
    beschikbaar = [c for c in DIEET_SCORE_COLS if c in df.columns]
    for c in beschikbaar:
        waarden = pd.to_numeric(df[c], errors='coerce')
        df[c] = waarden.where(waarden.between(0, 5))
    df['dieet_score'] = df[beschikbaar].mean(axis=1)
    df['dieet_cat'] = pd.cut(
        df['dieet_score'],
        bins=DIEET_GRENZEN,
        labels=DIEET_LABELS,
        include_lowest=True,
    )
    return df


def maak_dieet_verdeling_plot(df: pd.DataFrame, geslacht: str = 'totaal', lang: str = 'nl') -> go.Figure:
    """Staafdiagram van de verdeling van dieet categorieën."""
    df = voeg_dieet_score_toe(df)
    labels_t = [tr(label, lang) for label in DIEET_LABELS]
    if geslacht == 'beide' and 'rec_user_gender' in df.columns:
        gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
        df['geslacht_label'] = pd.to_numeric(df['rec_user_gender'], errors='coerce').map(gender_labels)
        df = df.dropna(subset=['dieet_cat', 'geslacht_label'])
        counts = df.groupby(['geslacht_label', 'dieet_cat'], observed=True).size().reset_index(name='aantal')
        counts['percentage'] = counts['aantal'] / counts.groupby('geslacht_label')['aantal'].transform('sum') * 100
        counts['categorie'] = counts['dieet_cat'].astype(str).map(lambda x: tr(x, lang))
        fig = px.bar(
            counts, x='categorie', y='percentage', color='geslacht_label', barmode='group',
            color_discrete_map={gender_labels[1]: GENDER_COLORS['Man'], gender_labels[0]: GENDER_COLORS['Vrouw']},
            labels={'categorie': '', 'percentage': tr('Percentage (%)', lang), 'geslacht_label': tr('Geslacht', lang)},
            title=tr('Verdeling dieetkwaliteit naar geslacht', lang),
            category_orders={'categorie': labels_t, 'geslacht_label': [gender_labels[1], gender_labels[0]]},
        )
        return fig

    counts = df['dieet_cat'].value_counts(normalize=True).mul(100).reset_index()
    counts.columns = ['categorie', 'percentage']
    kleur_map = {tr(label, lang): kleur for label, kleur in DIEET_COLORS.items()}
    counts['categorie'] = counts['categorie'].astype(str).map(lambda x: tr(x, lang))
    counts['categorie'] = pd.Categorical(counts['categorie'], categories=labels_t, ordered=True)
    counts = counts.sort_values('categorie')

    fig = px.bar(
        counts, x='categorie', y='percentage',
        color='categorie',
        color_discrete_map=kleur_map,
        labels={'categorie': '', 'percentage': tr('Percentage (%)', lang)},
        title=tr('Verdeling dieetkwaliteit', lang),
        category_orders={'categorie': labels_t},
    )
    fig.update_layout(showlegend=False)
    return fig


def maak_dieet_bmi_plot(df: pd.DataFrame, geslacht: str = 'totaal', lang: str = 'nl') -> go.Figure:
    """Gegroepeerd staafdiagram: dieet categorie vs BMI categorie."""
    df = voeg_dieet_score_toe(df)
    df['bmi_label'] = pd.to_numeric(df['rec_med_bmi_cat'], errors='coerce').map(
        {k: tr(v, lang) for k, v in BMI_LABELS.items()}
    )
    df2 = df.dropna(subset=['dieet_cat', 'bmi_label']).copy()
    df2['dieet_cat'] = df2['dieet_cat'].astype(str).map(lambda x: tr(x, lang))

    if geslacht == 'beide' and 'rec_user_gender' in df2.columns:
        gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
        df2['geslacht_label'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(gender_labels)
        df2 = df2.dropna(subset=['geslacht_label'])

    group_cols = ['geslacht_label', 'dieet_cat', 'bmi_label'] if geslacht == 'beide' and 'geslacht_label' in df2 else ['dieet_cat', 'bmi_label']
    counts = df2.groupby(group_cols).size().reset_index(name='aantal')
    totaal_group = ['geslacht_label', 'dieet_cat'] if 'geslacht_label' in counts.columns else ['dieet_cat']
    totaal = counts.groupby(totaal_group)['aantal'].transform('sum')
    counts['percentage'] = counts['aantal'] / totaal * 100
    counts['n_diet_cat'] = totaal.astype(int)
    dieet_labels_t = [tr(label, lang) for label in DIEET_LABELS]
    counts['dieet_cat'] = pd.Categorical(counts['dieet_cat'], categories=dieet_labels_t, ordered=True)
    counts = counts.sort_values('dieet_cat')

    bmi_volgorde = [tr(BMI_LABELS[k], lang) for k in sorted(BMI_LABELS.keys())]
    bmi_kleuren  = {tr(v, lang): BMI_COLORS[k] for k, v in BMI_LABELS.items()}
    n_per_dieet = (
        counts[[*totaal_group, 'n_diet_cat']]
        .drop_duplicates()
        .sort_values('dieet_cat')
    )
    x_labels = {
        tuple(row[col] for col in totaal_group): f"{row['dieet_cat']}<br>(n={int(row['n_diet_cat'])})"
        for _, row in n_per_dieet.iterrows()
    }
    counts['dieet_cat_label'] = counts.apply(lambda row: x_labels[tuple(row[col] for col in totaal_group)], axis=1)

    fig = px.bar(
        counts,
        x='dieet_cat_label', y='percentage',
        color='bmi_label',
        facet_col='geslacht_label' if 'geslacht_label' in counts.columns else None,
        barmode='stack',
        color_discrete_map=bmi_kleuren,
        custom_data=['aantal', 'n_diet_cat'],
        labels={
            'dieet_cat_label': tr('Dieetkwaliteit', lang),
            'percentage': tr('Percentage (%)', lang),
            'bmi_label': tr('BMI categorie', lang),
        },
        title=tr('BMI verdeling per dieetkwaliteit naar geslacht' if 'geslacht_label' in counts.columns else 'BMI verdeling per dieetkwaliteit', lang),
        category_orders={
            'dieet_cat_label': [
                x_labels[key]
                for key in x_labels
                if key[-1] in dieet_labels_t
            ],
            'bmi_label': bmi_volgorde,
        },
    )
    fig.update_traces(
        hovertemplate=(
            '<b>%{x}</b><br>' +
            tr('BMI categorie', lang) + ': %{fullData.name}<br>' +
            tr('Percentage', lang) + ': %{y:.1f}%<br>' +
            tr('Aantal', lang) + ': %{customdata[0]}<br>' +
            'n: %{customdata[1]}<extra></extra>'
        )
    )
    return fig


def maak_dieet_score_histogram(df: pd.DataFrame, geslacht: str = 'beide', lang: str = 'nl') -> go.Figure:
    """Verdeling van de ruwe dieet score, uitgesplitst naar geslacht."""
    df = voeg_dieet_score_toe(df)
    df2 = df.dropna(subset=['dieet_score'])
    if df2.empty:
        fig = go.Figure()
        fig.add_annotation(text=tr("Geen data beschikbaar voor deze selectie.", lang), showarrow=False)
        fig.update_layout(template='plotly_white', height=430)
        return fig

    if geslacht == 'beide':
        gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
        df2['geslacht_label'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(gender_labels)
        df2 = df2.dropna(subset=['geslacht_label'])
        kleur_map = {gender_labels[k]: GENDER_COLORS[k] for k in gender_labels}
        volgorde = [gender_labels[1], gender_labels[0]]
        summary = (
            df2.groupby('geslacht_label', as_index=False)
            .agg(
                gemiddelde=('dieet_score', 'mean'),
                mediaan=('dieet_score', 'median'),
                n=('dieet_score', 'count'),
            )
        )

        fig = px.box(
            df2,
            x='geslacht_label',
            y='dieet_score',
            color='geslacht_label',
            color_discrete_map=kleur_map,
            points='all',
            labels={
                'dieet_score': tr('Dieet score (0-5)', lang),
                'geslacht_label': tr('Geslacht', lang),
            },
            title=tr('Dieet score naar geslacht', lang),
            category_orders={'geslacht_label': volgorde},
        )
        fig.update_traces(
            jitter=0.28,
            pointpos=0,
            marker=dict(size=4, opacity=0.28),
            line=dict(width=2),
        )
        fig.add_trace(
            go.Scatter(
                x=summary['geslacht_label'],
                y=summary['gemiddelde'],
                mode='markers+text',
                marker=dict(color='black', size=11, symbol='diamond'),
                text=summary.apply(
                    lambda row: f"{row['gemiddelde']:.2f}", axis=1),
                textposition='top center',
                showlegend=False,
            )
        )
    else:
        fig = px.box(
            df2,
            y='dieet_score',
            color_discrete_sequence=[HOOFD_KLEUR],
            points='all',
            labels={'dieet_score': tr('Dieet score (0-5)', lang)},
            title=tr('Dieet score', lang),
        )
        fig.update_traces(
            jitter=0.28,
            pointpos=0,
            marker=dict(size=4, opacity=0.28),
            line=dict(width=2),
        )

    fig.update_layout(template='plotly_white', height=430)
    # Vaste y-as binnen de 0-5 schaal zodat de box leesbaar blijft
    fig.update_yaxes(range=[-0.25, 5.35], dtick=1)
    return fig


# ── Variabelen verkenner ──────────────────────────────────────────────────────
def _kleur_voor_bar(labels_dict: dict, kleur_richting: str) -> dict:
    """
    Genereert kleurenmap voor bar plots op basis van richting.
    eerste_goed: eerste = groen, laatste = rood
    laatste_goed: eerste = rood, laatste = groen
    midden_goed:  uitersten rood, midden groen
    """
    volgorde = [labels_dict[k] for k in sorted(labels_dict.keys())]
    n = len(volgorde)
    if n == 1:
        return {volgorde[0]: HOOFD_KLEUR}

    # Kleurgradiënt van groen naar rood of omgekeerd
    groen, oranje, rood = '#2ECC71', '#E87722', '#E74C3C'

    if kleur_richting == 'eerste_goed':
        if n == 2:   kleuren = [groen, rood]
        elif n == 3: kleuren = [groen, oranje, rood]
        else:
            import numpy as np
            kleuren = []
            for i in range(n):
                t = i / (n - 1)
                if t < 0.5:
                    kleuren.append(groen if t < 0.25 else oranje)
                else:
                    kleuren.append(rood if t > 0.75 else oranje)
    elif kleur_richting == 'laatste_goed':
        if n == 2:   kleuren = [rood, groen]
        elif n == 3: kleuren = [rood, oranje, groen]
        else:
            kleuren = []
            for i in range(n):
                t = i / (n - 1)
                if t > 0.75: kleuren.append(groen)
                elif t > 0.25: kleuren.append(oranje)
                else: kleuren.append(rood)
    elif kleur_richting == 'midden_goed':
        midden = n // 2
        kleuren = []
        for i in range(n):
            dist = abs(i - midden) / midden if midden > 0 else 0
            if dist < 0.3: kleuren.append(groen)
            elif dist < 0.6: kleuren.append(oranje)
            else: kleuren.append(rood)
    else:
        kleuren = [HOOFD_KLEUR] * n

    return {v: kleuren[i] for i, v in enumerate(volgorde)}


def maak_verkenner_plot(df: pd.DataFrame, variabele: dict, geslacht_filter: str = 'beide', lang: str = 'nl') -> go.Figure:
    """Genereert de beste plot voor een variabele uit het register."""
    import sys

    kolom           = variabele['kolom']
    label           = variabele['label']
    plot_type       = variabele['plot_type']
    gender_ok       = variabele.get('split_gender', False)
    labels_map      = variabele.get('labels', None)
    schaal          = variabele.get('schaal_factor', None)
    eenheid         = variabele.get('eenheid', '')
    referentie      = variabele.get('referentie', None)
    ref_label       = variabele.get('referentie_label', '')
    outlier_max     = variabele.get('outlier_max', None)
    vtype           = variabele.get('type', None)
    kleur_richting  = variabele.get('kleur_richting', 'eerste_goed')
    min_score       = variabele.get('min_score', None)
    max_score       = variabele.get('max_score', None)

    if kolom not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=tr("Kolom '{kolom}' niet beschikbaar.", lang, kolom=kolom),
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    df2 = df.copy()

    # Geslacht filter
    gender_labels = {k: tr(v, lang) for k, v in GENDER_LABELS.items()}
    df2['geslacht_label'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(gender_labels)
    if geslacht_filter == 'man':
        df2 = df2[df2['geslacht_label'] == gender_labels[1]]
    elif geslacht_filter == 'vrouw':
        df2 = df2[df2['geslacht_label'] == gender_labels[0]]
    elif geslacht_filter in {'beide', 'totaal', None}:
        df2 = df2.dropna(subset=['geslacht_label'])

    x_label = f"{label}" + (f" ({eenheid})" if eenheid else "")

    # ── Slaaptekst speciaal geval ──────────────────────────────────────────────
    if vtype == 'slaap_tekst':
        slaap_map_t = {k: tr(v, lang) for k, v in SLAAP_TEKST_MAP.items()}
        slaap_volgorde_t = [tr(v, lang) for v in SLAAP_VOLGORDE]
        slaap_kleuren_t = {tr(k, lang): v for k, v in SLAAP_KLEUREN.items()}
        df2['_label'] = df2[kolom].map(slaap_map_t)
        df2 = df2[df2['_label'].notna() & (df2['_label'] != 'None')]
        counts = df2['_label'].value_counts().reset_index()
        counts.columns = ['categorie', 'aantal']
        counts['pct'] = counts['aantal'] / counts['aantal'].sum() * 100
        counts['categorie'] = pd.Categorical(counts['categorie'], categories=slaap_volgorde_t, ordered=True)
        counts = counts.sort_values('categorie')
        fig = px.bar(counts, x='categorie', y='pct',
                     color='categorie', color_discrete_map=slaap_kleuren_t,
                     labels={'categorie': '', 'pct': tr('Percentage (%)', lang)},
                     title=label, category_orders={'categorie': slaap_volgorde_t})
        fig.update_layout(showlegend=False)
        return fig

    # ── Numerieke waarde ───────────────────────────────────────────────────────
    s = pd.to_numeric(df2[kolom], errors='coerce')

    # Schaalfactor toepassen — gebruik round om float precision issues te voorkomen
    if schaal:
        s = (s * schaal).round(1)

    # Outliers afkappen
    if outlier_max:
        masker = s <= outlier_max
        s  = s[masker]
        df2 = df2.loc[s.index]

    df2['_waarde'] = s

    # ── Histogram ─────────────────────────────────────────────────────────────
    if plot_type == 'histogram':
        df2 = df2.dropna(subset=['_waarde'])
        split_by_gender = gender_ok and geslacht_filter == 'beide'
        if split_by_gender:
            df2 = df2.dropna(subset=['geslacht_label'])
            kleur_map = {gender_labels[k]: GENDER_COLORS[k] for k in gender_labels}
            fig = px.histogram(
                df2, x='_waarde', color='geslacht_label',
                color_discrete_map=kleur_map,
                barmode='overlay', opacity=0.75, nbins=30,
                labels={'_waarde': x_label, 'count': tr('Aantal', lang), 'geslacht_label': tr('Geslacht', lang)},
                title=label,
                category_orders={'geslacht_label': [gender_labels[1], gender_labels[0]]},
            )
            # Vastzetten van bins op 0.5 voor scores
            if schaal or max_score:
                fig.update_traces(xbins=dict(
                    start=min_score if min_score is not None else 0.0,
                    end=max_score if max_score is not None else 10.0,
                    size=0.5
                ))
            fig.update_layout(legend_title_text=tr('Geslacht', lang))
        else:
            fig = px.histogram(
                df2, x='_waarde', nbins=30,
                color_discrete_sequence=[HOOFD_KLEUR],
                labels={'_waarde': x_label, 'count': tr('Aantal', lang)},
                title=label,
            )
        if referentie is not None:
            # Referentie staat altijd al op de weergaveschaal (0-10 of originele eenheid)
            # Nooit extra schaalfactor toepassen
            fig.add_vline(x=referentie, line_dash='dash', line_color='red',
                          annotation_text=ref_label, annotation_position='top right')

        # X-as bij 0 laten beginnen als schaal bekend is
        if min_score is not None and max_score is not None:
            fig.update_xaxes(range=[min_score, max_score])
        elif schaal is not None:
            # Geschaalde variabelen zijn altijd 0-10 tenzij expliciet anders ingesteld
            fig.update_xaxes(range=[0, 10])

    # ── Bar ───────────────────────────────────────────────────────────────────
    elif plot_type == 'bar':
        if labels_map:
            # Verwijder 'None' strings en converteer naar numeriek
            s_int = df2[kolom].replace('None', pd.NA)
            s_int = pd.to_numeric(s_int, errors='coerce')
            df2['_label'] = s_int.map(labels_map)
            df2 = df2.dropna(subset=['_label'])
            counts = df2['_label'].value_counts(normalize=True).mul(100).reset_index()
            counts.columns = ['categorie', 'percentage']
            counts['categorie'] = counts['categorie'].map(lambda x: tr(x, lang))
            aanwezige = set(counts['categorie'].astype(str).tolist())
            volgorde = [tr(labels_map[k], lang) for k in sorted(labels_map.keys()) if tr(labels_map[k], lang) in aanwezige]
            kleur_map = _kleur_voor_bar(
                {k: tr(v, lang) for k, v in labels_map.items() if tr(v, lang) in volgorde},
                kleur_richting
            )
            fig = px.bar(counts, x='categorie', y='percentage',
                         color='categorie', color_discrete_map=kleur_map,
                         labels={'categorie': '', 'percentage': tr('Percentage (%)', lang)},
                         title=label, category_orders={'categorie': volgorde})
        else:
            df2 = df2.dropna(subset=['_waarde'])
            counts = df2['_waarde'].value_counts(normalize=True).mul(100).reset_index()
            counts.columns = ['waarde', 'percentage']
            fig = px.bar(counts, x='waarde', y='percentage',
                         color_discrete_sequence=[HOOFD_KLEUR],
                         labels={'waarde': x_label, 'percentage': tr('Percentage (%)', lang)},
                         title=label)
        fig.update_layout(showlegend=False)

    # ── Pie ───────────────────────────────────────────────────────────────────
    elif plot_type == 'pie':
        if labels_map:
            s_int = pd.to_numeric(df2[kolom], errors='coerce')
            df2['_label'] = s_int.map(labels_map)
            df2 = df2.dropna(subset=['_label'])
            counts = df2['_label'].value_counts().reset_index()
            counts.columns = ['categorie', 'aantal']
            counts['categorie'] = counts['categorie'].map(lambda x: tr(x, lang))
            kleur_map = _kleur_voor_bar(
                {k: tr(v, lang) for k, v in labels_map.items() if tr(v, lang) in counts['categorie'].values},
                kleur_richting
            )
            fig = px.pie(counts, names='categorie', values='aantal',
                         title=label, color='categorie', color_discrete_map=kleur_map)
        else:
            df2 = df2.dropna(subset=['_waarde'])
            counts = df2['_waarde'].value_counts().reset_index()
            counts.columns = ['categorie', 'aantal']
            fig = px.pie(counts, names='categorie', values='aantal', title=label,
                         color_discrete_sequence=px.colors.qualitative.Set2)

    # ── Box ───────────────────────────────────────────────────────────────────
    elif plot_type == 'box':
        df2 = df2.dropna(subset=['_waarde'])
        split_by_gender = gender_ok and geslacht_filter == 'beide'
        if split_by_gender:
            df2 = df2.dropna(subset=['geslacht_label'])
            kleur_map = {gender_labels[k]: GENDER_COLORS[k] for k in gender_labels}
            fig = px.box(df2, x='geslacht_label', y='_waarde',
                         color='geslacht_label', color_discrete_map=kleur_map,
                         labels={'geslacht_label': tr('Geslacht', lang), '_waarde': x_label},
                         title=label, category_orders={'geslacht_label': [gender_labels[1], gender_labels[0]]})
            fig.update_layout(showlegend=False)
        else:
            fig = px.box(df2, y='_waarde', color_discrete_sequence=[HOOFD_KLEUR],
                         labels={'_waarde': x_label}, title=label)
    else:
        fig = go.Figure()

    return fig


# ── Platform groei ─────────────────────────────────────────────────────────────
def maak_platform_groei_plot(base_pad, participant_ids: list | set | pd.Series | None = None) -> go.Figure:
    """
    Toont nieuwe accounts en ingevulde vragenlijsten per jaar als gecombineerde grafiek.
    Aligned met account_activatie: "ingevuld" betekent "heeft scores in users_met_scores".
    """
    from pathlib import Path
    import pandas as pd

    base_pad = Path(base_pad)

    # Nieuwe accounts per jaar — zelfde definitie als accountactivatie
    df_users = _laad_actieve_users(base_pad, include_deleted=False)
    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        df_users['id_num'] = pd.to_numeric(df_users['id'], errors='coerce')
        df_users = df_users[df_users['id_num'].isin(pids)].copy()
    
    # Laad scores om actieve accounts te bepalen (consistent met account_activatie)
    df_scores_expanded = load_my_clic_participants_expanded(DB_URL)
    
    # Zorg dat participant_id kolom aanwezig is (dit is de juiste linkage!)
    if df_scores_expanded.empty:
        logger.warning("⚠️ df_scores_expanded is leeg - kan geen actieve accounts bepalen")
        actieve_ids = set()
    else: # Use participant_id for matching (this is the correct linkage!)
        df_scores_expanded = df_scores_expanded.drop_duplicates(subset='participant_id', keep='first')
        if 'latest_completion_at' in df_scores_expanded.columns:
            completed = df_scores_expanded[df_scores_expanded['latest_completion_at'].notna()]
            actieve_ids = set(pd.to_numeric(completed['participant_id'], errors='coerce').dropna().astype(int))
        else:
            actieve_ids = set(pd.to_numeric(df_scores_expanded['participant_id'], errors='coerce').dropna().astype(int))
    
    if 'id_num' not in df_users.columns:
        df_users['id_num'] = pd.to_numeric(df_users['id'], errors='coerce')
    df_users['jaar'] = pd.to_datetime(df_users['created_at'], errors='coerce').dt.year
    df_users['actief'] = df_users['id_num'].isin(actieve_ids)
    
    # Accounts per jaar (totaal en actief)
    accounts_per_jaar = (
        df_users[df_users['jaar'].between(2018, 2030)] # Filter years
        .groupby('jaar').size().reset_index(name='nieuwe_accounts')
    )

    # Actieve accounts per jaar (met ingevulde vragenlijst)
    actieve_per_jaar = (
        df_users[df_users['actief'] & df_users['jaar'].between(2018, 2030)] # Filter years
        .groupby('jaar').size().reset_index(name='ingevulde_vragenlijsten')
    )

    # Combineer
    df = accounts_per_jaar.merge(actieve_per_jaar, on='jaar', how='outer').sort_values('jaar')
    df['ingevulde_vragenlijsten'] = df['ingevulde_vragenlijsten'].fillna(0).astype(int)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=df['jaar'], y=df['nieuwe_accounts'],
        name='Nieuwe accounts',
        marker_color='#1F6FBF',
        opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['jaar'], y=df['ingevulde_vragenlijsten'],
        name='Ingevulde vragenlijsten',
        mode='lines+markers',
        line=dict(color='#E87722', width=2),
        marker=dict(size=8),
    ), secondary_y=True)

    fig.update_layout(
        title='Platformgroei: nieuwe accounts en ingevulde vragenlijsten per jaar',
        xaxis=dict(title='Jaar', tickmode='linear', dtick=1),
        yaxis=dict(title=dict(text='Nieuwe accounts', font=dict(color='#1F6FBF'))),
        yaxis2=dict(
            title=dict(text='Ingevulde vragenlijsten', font=dict(color='#E87722')),
            overlaying='y',
            side='right',
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='group',
    )
    fig.add_annotation(
        text='Alleen niet-verwijderde accounts. Aligned met accountactivatie: "ingevuld" = heeft scores in users_met_scores',
        xref='paper', yref='paper', x=0.01, y=1.12,
        showarrow=False, font=dict(size=12, color='dimgray')
    )
    return fig


def maak_update_verificatie_plot(df: pd.DataFrame, lang: str = 'nl') -> go.Figure:
    """
    Toont de cumulatieve groei van het aantal deelnemers over de tijd.
    Dit is een goede manier om te verifiëren of een data-import effect heeft gehad.
    """
    if df is None or df.empty or 'created_at' not in df.columns:
        return _empty_store_plot(tr("Geen tijddata beschikbaar voor controle.", lang))
    
    df_plot = df.copy()
    df_plot['created_at'] = pd.to_datetime(df_plot['created_at'], errors='coerce')
    df_plot = df_plot.dropna(subset=['created_at']).sort_values('created_at')
    
    # Filter op reële data (vanaf 2019)
    df_plot = df_plot[df_plot['created_at'].dt.year >= 2019]
    
    if df_plot.empty:
        return _empty_store_plot(tr("Geen data na 2019 gevonden.", lang))

    # Telling per dag
    df_plot['datum'] = df_plot['created_at'].dt.date
    dag_counts = df_plot.groupby('datum').size().reset_index(name='n_nieuw')
    dag_counts['totaal_cumulatief'] = dag_counts['n_nieuw'].cumsum()

    fig = px.line(
        dag_counts, x='datum', y='totaal_cumulatief',
        title=tr('Verificatie: Totaal aantal deelnemers over tijd (Cumulatief)', lang),
        labels={'datum': tr('Datum', lang), 'totaal_cumulatief': tr('Totaal aantal deelnemers', lang)},
    )
    fig.update_traces(line_shape='hv', line_color='#2ECC71') # Step plot in groen
    
    # Markeer de laatste update datum
    laatste_datum = dag_counts['datum'].max()
    max_count = int(dag_counts['totaal_cumulatief'].max()) if not dag_counts.empty else 0
    fig.add_annotation(
        x=laatste_datum, y=dag_counts['totaal_cumulatief'].max(),
        text=f"{tr('Huidige stand', lang)}: {max_count:,}",
        showarrow=True, arrowhead=1
    )
    
    fig.update_layout(template='plotly_white', height=450)
    return fig


# ── Account activatie ──────────────────────────────────────────────────────────
def maak_account_activatie_plot(base_pad, participant_ids: list | set | pd.Series | None = None) -> go.Figure:
    """
    Toont per jaar:
    - Nieuwe accounts aangemaakt
    - Daarvan hebben vragenlijst ingevuld (actief)
    - Daarvan hebben GEEN vragenlijst ingevuld (inactief)
    """
    from pathlib import Path
    import pandas as pd

    base_pad = Path(base_pad)

    df_users  = _laad_actieve_users(base_pad, include_deleted=False)
    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        df_users['id_num'] = pd.to_numeric(df_users['id'], errors='coerce')
        df_users = df_users[df_users['id_num'].isin(pids)].copy()
    df_scores_expanded = load_my_clic_participants_expanded(DB_URL)
    
    # Use participant_id for matching (this is the correct linkage!)
    if 'participant_id' not in df_scores_expanded.columns:
        logger.warning("⚠️ participant_id kolom niet gevonden in df_scores_expanded")
        df_scores_expanded['participant_id'] = pd.NA
    
    df_scores_expanded = df_scores_expanded.drop_duplicates(subset='participant_id', keep='first')

    if 'latest_completion_at' in df_scores_expanded.columns:
        completed = df_scores_expanded[df_scores_expanded['latest_completion_at'].notna()]
        actieve_ids = set(pd.to_numeric(completed['participant_id'], errors='coerce').dropna().astype(int))
    else:
        actieve_ids = set(pd.to_numeric(df_scores_expanded['participant_id'], errors='coerce').dropna().astype(int))

    id_column = 'id' if 'id' in df_users.columns else 'participant_id' if 'participant_id' in df_users.columns else None
    if id_column is None:
        logger.warning("⚠️ Geen id-kolom gevonden in users/participants voor accountactivatie")
        id_column = 'id'
        df_users['id'] = pd.NA

    df_users['id_num'] = pd.to_numeric(df_users[id_column], errors='coerce')
    df_users['actief'] = df_users['id_num'].isin(actieve_ids)

    # Zorg dat jaar-kolom bestaat en filter op plausibele jaren
    df_users['jaar'] = pd.to_datetime(df_users.get('created_at'), errors='coerce').dt.year

    # Aggregeer per jaar (beperk tot realistische jaren)
    groep = (
        df_users[df_users['jaar'].between(2018, 2030)]
        .groupby('jaar')
        .agg(
            totaal=('id', 'count'),
            actief=('actief', 'sum'),
        )
        .reset_index()
    )
    groep['inactief'] = groep['totaal'] - groep['actief']
    groep['pct_actief'] = (groep['actief'] / groep['totaal'] * 100).round(1)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=groep['jaar'], y=groep['actief'],
        name='Vragenlijst ingevuld',
        marker_color='#2ECC71',
    ))
    fig.add_trace(go.Bar(
        x=groep['jaar'], y=groep['inactief'],
        name='Geen vragenlijst ingevuld',
        marker_color='#E74C3C',
    ))
    fig.add_trace(go.Scatter(
        x=groep['jaar'], y=groep['pct_actief'],
        name='Activatiegraad (%)',
        mode='lines+markers',
        line=dict(color='#1F6FBF', width=2, dash='dot'),
        marker=dict(size=8),
        yaxis='y2',
    ))

    fig.update_layout(
        title='Accountactivatie per jaar',
        xaxis=dict(title='Jaar', tickmode='linear', dtick=1),
        yaxis=dict(title='Aantal accounts'),
        yaxis2=dict(
            title=dict(text='Activatiegraad (%)', font=dict(color='#1F6FBF')),
            overlaying='y',
            side='right',
            range=[0, 100],
        ),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.add_annotation(
        text='Actief = minstens 1 ingevulde vragenlijst met scoredata',
        xref='paper', yref='paper', x=0.01, y=1.12,
        showarrow=False, font=dict(size=12, color='dimgray')
    )
    return fig, groep


# ── Vragenlijst invulpercentage ────────────────────────────────────────────────
def maak_vragenlijst_plot(base_pad, db_url: str, participant_ids: list | set | pd.Series | None = None):
    """
    Toont per vragenlijst:
    - Wanneer deze voor het eerst werd ingevuld
    - Hoeveel unieke deelnemers hem hebben ingevuld
    - Invulpercentage ten opzichte van totaal aantal deelnemers
    """
    import json
    import pandas as pd
    from pathlib import Path
    base_pad = Path(base_pad)

    # Alleen de in deze visualisatie bekende vragenlijsten tonen.
    # De database kan extra technische/legacy records bevatten, die we hier bewust uitsluiten.
    known_questionnaire_ids = {1, 2, 3, 4, 5, 6, 7, 8}
    naam_map = {
        1: 'Smart Health Test',
        2: 'Smart Work Test',
        3: 'Resilience',
        4: 'Well-being',
        5: 'Positive Health',
        6: 'Self-efficacy',
        7: 'Negative emotions',
        8: 'Smartphone and stress',
    }
    # Dynamisch ophalen van vragenlijstnamen, maar alleen voor de bekende IDs uit deze visualisatie.
    naam_map = {q_id: naam for q_id, naam in naam_map.items() if q_id in known_questionnaire_ids}
    try:
        from sqlalchemy import create_engine
        engine = create_engine(db_url or DB_URL)
        df_q = pd.read_sql("SELECT id, internal_name, name::text AS name, slug FROM questionnaires", engine)
        for _, row in df_q.iterrows():
            q_id = int(row['id'])
            if q_id not in known_questionnaire_ids:
                continue
            name = row.get('internal_name') or row.get('name') or row.get('slug')

            # Filter "bloeddruk" en technische records die geen echte vragenlijst zijn
            if name:
                name_low = str(name).lower()
                if any(x in name_low for x in ['bloeddruk', 'blood_pressure', 'bloodpressure', 'test_']):
                    continue
                naam_map[q_id] = str(name)
    except Exception as e:
        logger.warning(f"Kon vragenlijstnamen niet uit DB laden: {e}")
        naam_map = {
            1: 'Smart Health Test', 2: 'Smart Work Test', 3: 'Resilience',
            4: 'Well-being', 5: 'Positive Health', 6: 'Self-efficacy',
            7: 'Negative emotions', 8: 'Smartphone and stress'
        }

    # Completions laden
    df_comp = load_completions(db_url or DB_URL)
    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        df_comp = df_comp[pd.to_numeric(df_comp['participant_id'], errors='coerce').isin(pids)]

    df_comp['created_at'] = pd.to_datetime(df_comp['created_at'], errors='coerce')
    n_totaal = int(df_comp['participant_id'].nunique())

    # Totaal unieke deelnemers
    # Filter completions: gebruik alleen echte vragenlijsten (geen bloeddruk etc)
    df_comp['questionnaire_id'] = pd.to_numeric(df_comp['questionnaire_id'], errors='coerce')
    df_comp = df_comp[df_comp['questionnaire_id'].isin(naam_map.keys())].copy()

    # Per vragenlijst aggregeren
    agg = df_comp.groupby('questionnaire_id').agg(
        eerste_invulling=('created_at', 'min'),
        laatste_invulling=('created_at', 'max'),
        n_ingevuld=('participant_id', 'nunique'),
    ).reset_index()

    # Voeg questionnaire naam toe
    agg['naam'] = agg['questionnaire_id'].map(naam_map)
    
    # Filter out legacy/unknown questionnaires (those without entries in naam_map)
    agg = agg[agg['naam'].notna()].copy()
    # Filter records zonder naam
    agg = agg.dropna(subset=['naam']).copy()
    
    deelnemers_per_vragenlijst = []
    for _, row in agg.iterrows():
        eerste_invulling = row['eerste_invulling']
        n_eligible = df_comp.loc[df_comp['created_at'] >= eerste_invulling, 'participant_id'].nunique()
        deelnemers_per_vragenlijst.append(n_eligible)
    agg['n_eligible'] = deelnemers_per_vragenlijst
    agg['pct'] = (agg['n_ingevuld'] / agg['n_eligible'] * 100).round(1)
    agg['eerste_jaar'] = agg['eerste_invulling'].dt.year

    # Sorteer op percentage
    agg = agg.sort_values('pct', ascending=True).reset_index(drop=True)

    # Kleur op basis van invulpercentage - nu met zwart voor >=50%
    from kleuren import VRAGENLIJST_COLORS
    kleuren = [
        VRAGENLIJST_COLORS[0] if p >= 50 else
        VRAGENLIJST_COLORS[1] if p >= 20 else
        VRAGENLIJST_COLORS[2]
        for p in agg['pct']
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg['pct'],
        y=agg['naam'],
        orientation='h',
        marker_color=kleuren,
        text=agg.apply(lambda r: f"{int(r['n_ingevuld']):,}/{int(r['n_eligible']):,} ({r['pct']}%) — sinds {r['eerste_jaar']:.0f}", axis=1),
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Bereik sinds eerste invulling: %{x:.1f}%<br>'
            'Deelnemers: %{customdata[2]}<br>'
            'Eligible deelnemers: %{customdata[3]}<br>'
            'Eerste: %{customdata[0]}<br>'
            'Laatste: %{customdata[1]}<extra></extra>'
        ),
        customdata=agg[['eerste_invulling', 'laatste_invulling', 'n_ingevuld', 'n_eligible']].values,
    ))

    fig.update_layout(
        title=f'Questionnaire Completion Percentage (total: {n_totaal:,} participants)',
        xaxis=dict(title='Completion percentage (%)', range=[0, 115]),
        yaxis=dict(title=''),
        height=max(400, len(agg) * 28),
        margin=dict(l=200, r=150),
        showlegend=False,
    )
    fig.add_vline(x=50, line_dash='dash', line_color='black',
                  annotation_text='50%', annotation_font_color='black')

    return fig, agg


# ── Kopers vs niet-kopers ──────────────────────────────────────────────────────
def maak_kopers_vergelijking_plot(base_pad: Path, db_url: str, participant_ids: list | set | pd.Series | None = None) -> tuple:
    """
    Vergelijkt gemiddelde scores van kopers vs niet-kopers.
    
    NOTE: This requires that order data can be mapped to participant IDs.
    If the mapping is not available (e.g., separate user systems), returns an empty figure.
    """
    # Gebruik de centrale koppelfunctie uit data_ingestion om aankoopdata te laden
    df_koppeling = get_user_purchases(base_pad, db_url)
    
    if df_koppeling.empty:
        # Empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Geen aankoopgegevens beschikbaar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14)
        )
        return fig, pd.DataFrame(columns=['Product', 'Aantal orders']), 0
    
    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        if 'participant_id' in df_koppeling.columns:
            df_koppeling = df_koppeling[pd.to_numeric(df_koppeling['participant_id'], errors='coerce').isin(pids)]

    # Filter for successful purchases
    if 'status' in df_koppeling.columns:
        success_statuses = {'completed', 'paid', 'success', 'shipped', 'delivered', 'voltooid', 'betaald'}
        df_koppeling = df_koppeling[df_koppeling['status'].astype(str).str.lower().isin(success_statuses)].copy()

    # Scores laden (gebruik de geconsolideerde data)
    df_scores_expanded = load_my_clic_participants_expanded(db_url or DB_URL)
    df_scores_expanded['participant_id'] = pd.to_numeric(df_scores_expanded['participant_id'], errors='coerce')
    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        df_scores_expanded = df_scores_expanded[df_scores_expanded['participant_id'].isin(pids)]
    df_scores_expanded = df_scores_expanded.drop_duplicates(subset='participant_id', keep='first')

    # Bepaal kopers op basis van participant_id of user_id, zodat koppeling ook werkt
    # wanneer de order- en scorebronnen verschillende identifieringsystemen gebruiken.
    kopers_participant = set(pd.to_numeric(df_koppeling.get('participant_id'), errors='coerce').dropna().astype(int))
    kopers_user = set(pd.to_numeric(df_koppeling.get('user_id'), errors='coerce').dropna().astype(int))

    if not kopers_participant and not kopers_user:
        # No valid participant mapping - cannot determine buyers
        fig = go.Figure()
        fig.add_annotation(
            text="Kan aankoopgegevens niet koppelen aan deelnemersgegevens.<br>De systemen gebruiken verschillende user-ID bereiken.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=12)
        )
        return fig, pd.DataFrame(columns=['Product', 'Aantal orders']), 0

    def _is_koper(row):
        pid = pd.to_numeric(row.get('participant_id'), errors='coerce')
        uid = pd.to_numeric(row.get('user_id'), errors='coerce')
        if pd.notna(pid) and int(pid) in kopers_participant:
            return True
        if pd.notna(uid) and int(uid) in kopers_user:
            return True
        return False

    df_scores_expanded['groep'] = df_scores_expanded.apply(_is_koper, axis=1).map(
        {True: 'Heeft product gekocht', False: 'Geen product gekocht'}
    )
    n_kopers = int(df_scores_expanded['groep'].eq('Heeft product gekocht').sum())

    # Indicatoren
    indicatoren = {
        'Leefstijlscore':  'rec_ls_lifestyle_score',
        'BMI':             'rec_med_bmi',
        'Stress':          'rec_ls_stress_sum',
        'Heartrisk score': 'rec_heartrisk',
        'Slaap (PSQI)':    'rec_ls_sleep_psqi_sum',
        'Veerkracht':      'rec_resilience_score',
        'Welzijn':         'rec_wellbeing_score',
        'Zelfeffectiviteit': 'rec_self_efficacy_score',
        'Werkvermogen (WAI)': 'rec_asr_wai_score',
        'Vitaliteit':       'rec_asr_vitality_score',
        'Werktevredenheid': 'rec_asr_job_satisfaction_score',
        'Werkdruk':         'rec_asr_workload_score',
        'Bewegingsscore':   'rec_ls_score_exercise',
        'Fruit score':      'rec_ls_score_fruit',
    }

    # Lager is beter voor deze indicatoren
    lager_beter = {'BMI', 'Stress', 'Heartrisk score', 'Slaap (PSQI)', 'Werkdruk'}

    rijen = []
    for label, kolom in indicatoren.items():
        if kolom not in df_scores_expanded.columns:
            continue
        s = pd.to_numeric(df_scores_expanded[kolom], errors='coerce')
        df_scores_expanded['_s'] = s
        for groep in ['Heeft product gekocht', 'Geen product gekocht']: # type: ignore
            gem = df_scores_expanded[df_scores_expanded['groep'] == groep]['_s'].mean()
            n   = df_scores_expanded[df_scores_expanded['groep'] == groep]['_s'].notna().sum()
            rijen.append({'Indicator': label, 'Groep': groep,
                          'Gemiddelde': round(gem, 2), 'N': n})

    df_plot = pd.DataFrame(rijen)

    # Normaliseer naar 0-100 schaal per indicator voor vergelijkbaarheid
    genorm = []
    for ind in df_plot['Indicator'].unique():
        sub = df_plot[df_plot['Indicator'] == ind].copy()
        min_v = sub['Gemiddelde'].min()
        max_v = sub['Gemiddelde'].max()
        bereik = max_v - min_v if max_v != min_v else 1
        sub['Verschil_%'] = ((sub['Gemiddelde'] - min_v) / bereik * 100).round(1)
        if ind in lager_beter:
            sub['Verschil_%'] = 100 - sub['Verschil_%']
        genorm.append(sub)
    df_plot = pd.concat(genorm)

    volgorde = ['Heeft product gekocht', 'Geen product gekocht']
    kleur_map = {
        'Heeft product gekocht': '#2ECC71',
        'Geen product gekocht': '#95A5A6',
    }
    indicator_volgorde = list(indicatoren.keys())

    fig = go.Figure()
    for groep in [g for g in volgorde if g in df_plot['Groep'].unique()]:
        sub = df_plot[df_plot['Groep'] == groep].copy()
        sub['Indicator'] = pd.Categorical(sub['Indicator'], categories=indicator_volgorde, ordered=True)
        sub = sub.sort_values('Indicator')
        fig.add_bar(
            name=groep,
            x=sub['Indicator'].astype(str),
            y=sub['Gemiddelde'],
            marker_color=kleur_map[groep],
            text=[f"{v:.2f}" for v in sub['Gemiddelde']],
            textposition='outside',
            offsetgroup=groep,
        )

    fig.update_layout(
        title='Gemiddelde scores: kopers vs niet-kopers',
        xaxis_title='',
        yaxis_title='Gemiddelde score',
        barmode='group', # type: ignore
        legend_title_text='',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), # type: ignore
    )

    # Producten top 10
    top_producten = df_koppeling['name'].value_counts().head(10).reset_index()
    top_producten = df_koppeling['name'].str.strip().value_counts().head(10).reset_index()
    top_producten.columns = ['Product', 'Aantal orders']

    return fig, top_producten, n_kopers


# ── Inactieve accounts analyse ────────────────────────────────────────────────
def maak_inactieve_accounts_plot(base_pad) -> tuple:
    """
    Analyseert inactieve accounts: aanmaakjaar, domeinen, kenmerken.
    """
    from pathlib import Path
    import pandas as pd

    base_pad = Path(base_pad)
    df_users  = _laad_actieve_users(base_pad, include_deleted=False)
    df_scores_expanded = load_my_clic_participants_expanded(DB_URL)
    # Ensure user_id column exists
    if 'user_id' not in df_scores_expanded.columns:
        df_scores_expanded = df_scores_expanded.rename(columns={'pmo_id': 'user_id'})
    df_scores_expanded = df_scores_expanded.drop_duplicates(subset='user_id', keep='first')

    for col in [ # type: ignore
        'completed_onboarding_at',
        'accepted_terms_at',
        'accepted_medical_data_processing_terms_at',
        'accepted_automated_data_processing_terms_at',
        'subscribed_to_service_mails_at',
        'subscribed_to_newsletter_at',
        'phone_number',
    ]:
        if col not in df_users.columns:
            df_users[col] = pd.NA

    # Gebruik robuuste numerieke conversie voor ID matching
    actieve_ids = set(pd.to_numeric(df_scores_expanded['user_id'], errors='coerce').dropna().astype(int))
    df_users_actief = df_users.copy()
    df_users_actief['actief'] = df_users_actief['id'].isin(actieve_ids)
    df_users_actief['id_num'] = pd.to_numeric(df_users_actief['id'], errors='coerce')
    df_users_actief['actief'] = df_users_actief['id_num'].isin(actieve_ids)
    df_users_actief = df_users_actief.dropna(subset=['id_num'])

    df_inactief = df_users_actief[~df_users_actief['actief']]
    df_actief   = df_users_actief[df_users_actief['actief']]

    # Plot 1: Aanmaakjaar verdeling actief vs inactief
    jaren = sorted(df_users_actief['jaar'].dropna().unique().astype(int))
    df_jaar = pd.DataFrame({
        'jaar':     jaren,
        'actief':   [int(df_actief['jaar'].eq(j).sum()) for j in jaren],
        'inactief': [int(df_inactief['jaar'].eq(j).sum()) for j in jaren],
    })

    fig_jaar = go.Figure()
    fig_jaar.add_trace(go.Bar(
        x=df_jaar['jaar'], y=df_jaar['actief'],
        name='Vragenlijst ingevuld',
        marker_color='#2ECC71',
    ))
    fig_jaar.add_trace(go.Bar(
        x=df_jaar['jaar'], y=df_jaar['inactief'],
        name='Nooit vragenlijst ingevuld',
        marker_color='#E74C3C',
    ))
    fig_jaar.update_layout(
        title='Accounts per aanmaakjaar: actief vs inactief',
        xaxis=dict(title='Aanmaakjaar', tickmode='linear', dtick=1),
        yaxis=dict(title='Aantal accounts'),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    # Samenvatting
    samenvatting = {
        'Totaal inactieve accounts': len(df_inactief),
        'Piek aanmaakjaar':          int(df_inactief['jaar'].mode().iloc[0]) if len(df_inactief) > 0 else None,
        'Gemiddeld jaar':            round(df_inactief['jaar'].mean(), 0),
        'Activatiegraad totaal (%)': round(len(df_actief) / len(df_users_actief) * 100, 1),
        'Onboarding afgerond (%)':   round(df_inactief['completed_onboarding_at'].notna().mean() * 100, 1),
        'Voorwaarden geaccepteerd (%)': round(df_inactief['accepted_terms_at'].notna().mean() * 100, 1),
        'Service mails aan (%)':     round(df_inactief['subscribed_to_service_mails_at'].notna().mean() * 100, 1),
        'Telefoonnummer bekend (%)': round(df_inactief['phone_number'].notna().mean() * 100, 1),
    }

    kenmerken = pd.DataFrame([
        {'Kenmerk': 'Onboarding afgerond', 'Aantal': int(df_inactief['completed_onboarding_at'].notna().sum())},
        {'Kenmerk': 'Voorwaarden geaccepteerd', 'Aantal': int(df_inactief['accepted_terms_at'].notna().sum())},
        {'Kenmerk': 'Medische toestemming', 'Aantal': int(df_inactief['accepted_medical_data_processing_terms_at'].notna().sum())},
        {'Kenmerk': 'Geautomatiseerde verwerking akkoord', 'Aantal': int(df_inactief['accepted_automated_data_processing_terms_at'].notna().sum())},
        {'Kenmerk': 'Service mails aan', 'Aantal': int(df_inactief['subscribed_to_service_mails_at'].notna().sum())},
        {'Kenmerk': 'Nieuwsbrief aan', 'Aantal': int(df_inactief['subscribed_to_newsletter_at'].notna().sum())},
        {'Kenmerk': 'Telefoonnummer bekend', 'Aantal': int(df_inactief['phone_number'].notna().sum())},
    ])
    kenmerken['Percentage (%)'] = (kenmerken['Aantal'] / max(len(df_inactief), 1) * 100).round(1)

    return fig_jaar, samenvatting, kenmerken


def maak_artikel_interacties_overzicht(base_pad, user_ids: list | set | pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Koppel content views aan gebruikers en artikeltitels.
    Haalt data uit database via 'interactions' tabel i.p.v. CSV bestanden.
    """
    base_pad = Path(base_pad)

    # Initialiseer lege dataframes. Data wordt direct uit de PostgreSQL database geladen.
    df_inter = pd.DataFrame()
    df_content_nl = pd.DataFrame()
    df_users = pd.DataFrame()
    empty_top_artikelen = pd.DataFrame(columns=['interactable_id', 'title', 'Views', 'Unieke_lezers', 'Laatste_view'])
    
    def _normalize_locale(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .str.replace('-', '_', regex=False)
            .fillna('')
        )

    # Consistente vorm voor lege resultaten, ook bij een lege/foutieve database.
    empty_top_artikelen = pd.DataFrame(columns=['interactable_id', 'title', 'Views', 'Unieke_lezers', 'Laatste_view'])

    try:
        from config import DB_URL
        if not DB_URL:
            raise ValueError("Geen database URL beschikbaar")
        
        engine = create_engine(DB_URL)
        
        # Laad data uit database tabellen
        try:
            df_inter_raw = pd.read_sql(
                "SELECT id, interactable_type, interactable_id, user_id, type, created_at, updated_at FROM interactions",
                engine
            )
        except Exception as e:
            logger.warning(f"Kon interactions niet laden met expliciete kolommen: {e}. Val terug op SELECT *")
            df_inter_raw = pd.read_sql("SELECT * FROM interactions", engine)

        if set(['id', 'interactable_type', 'interactable_id', 'user_id', 'type', 'created_at', 'updated_at']).issubset(df_inter_raw.columns):
            df_inter = df_inter_raw[['id', 'interactable_type', 'interactable_id', 'user_id', 'type', 'created_at', 'updated_at']].copy()
        elif len(df_inter_raw.columns) >= 7:
            df_inter = df_inter_raw.iloc[:, :7].copy()
            df_inter.columns = ['id', 'interactable_type', 'interactable_id', 'user_id', 'type', 'created_at', 'updated_at']
        else:
            logger.warning(f"Interactions tabel heeft slechts {len(df_inter_raw.columns)} kolommen (verwacht 7+)")
            df_inter = pd.DataFrame()

        if not df_inter.empty:
            df_inter = df_inter[
                (df_inter['interactable_type'].astype(str).str.strip() == 'content') &
                (df_inter['type'].astype(str).str.strip() == 'view')
            ].copy()
            logger.info(f"✓ Interactions geladen: {len(df_inter)} content views")
        
        try:
            df_content_nl_raw = pd.read_sql(
                "SELECT content_id, locale, title FROM content_translations",
                engine
            )
        except Exception as e:
            logger.warning(f"Kon content_translations niet laden met expliciete kolommen: {e}. Val terug op SELECT *")
            df_content_nl_raw = pd.read_sql("SELECT * FROM content_translations", engine)

        if set(['content_id', 'locale', 'title']).issubset(df_content_nl_raw.columns):
            df_content_nl = df_content_nl_raw[['content_id', 'locale', 'title']].copy()
        elif set(['id', 'public_id', 'content_id', 'locale', 'title']).issubset(df_content_nl_raw.columns):
            df_content_nl = df_content_nl_raw[['content_id', 'locale', 'title']].copy()
        elif len(df_content_nl_raw.columns) >= 5:
            df_content_nl = df_content_nl_raw.iloc[:, [2, 3, 4]].copy()
            df_content_nl.columns = ['content_id', 'locale', 'title']
        else:
            logger.warning(f"Content_translations tabel heeft slechts {len(df_content_nl_raw.columns)} kolommen")
            df_content_nl = pd.DataFrame(columns=['content_id', 'locale', 'title'])

        if not df_content_nl.empty:
            df_content_nl['locale'] = _normalize_locale(df_content_nl['locale'])
            df_content_nl = df_content_nl[df_content_nl['locale'].str.startswith('nl')].drop_duplicates('content_id')
            logger.info(f"✓ Content translations geladen: {len(df_content_nl)} records (nl locale)")
        else:
            logger.warning("Geen content translations gevonden in content_translations tabel")

        if df_content_nl.empty:
            try:
                df_content_fallback = pd.read_sql("SELECT id, title FROM content", engine)
                if {'id', 'title'}.issubset(df_content_fallback.columns):
                    df_content_nl = df_content_fallback.rename(columns={'id': 'content_id'})[['content_id', 'title']]
                    logger.info(f"✓ Fallback content geladen: {len(df_content_nl)} records")
            except Exception as e:
                logger.warning(f"Kon fallback content niet laden: {e}")

        # Check if users load successfully
        try:
            df_users = pd.read_sql("SELECT id, email, deleted_at FROM users", engine)
            logger.info(f"✓ Users geladen: {len(df_users)} records")
        except Exception as e:
            logger.warning(f"Kon deleted_at niet laden uit users ({e}), val terug op id/email")
            try:
                df_users = pd.read_sql("SELECT id, email FROM users", engine)
                logger.info(f"✓ Users geladen zonder deleted_at: {len(df_users)} records")
            except Exception as e2:
                logger.warning(f"Kon users niet laden: {e2}")
                df_users = pd.DataFrame(columns=['id', 'email', 'deleted_at'])

        for col in ['id', 'email', 'deleted_at']:
            if col not in df_users.columns:
                df_users[col] = pd.NA

        if df_inter.empty:
            logger.warning("Geen interactions gevonden in database")
            return empty_top_artikelen.copy(), pd.DataFrame(), {
                'Totaal views': 0, 'Unieke lezers': 0, 'Unieke artikelen': 0,
                'Lezers met actief account': 0, 'Gefilterde interne accounts': 0,
            }

    except Exception as e:
        logger.warning(f"Kan artikel interacties niet uit database laden: {e}. Retourneer lege dataset.")
        return empty_top_artikelen.copy(), pd.DataFrame(), {
            'Totaal views': 0, 'Unieke lezers': 0, 'Unieke artikelen': 0,
            'Lezers met actief account': 0, 'Gefilterde interne accounts': 0,
        }

    df_inter['user_id'] = pd.to_numeric(df_inter['user_id'], errors='coerce')
    if user_ids is not None:
        uids = set(pd.to_numeric(pd.Series(list(user_ids)), errors='coerce').dropna().astype(int))
        if uids:
            df_candidate = df_inter[df_inter['user_id'].isin(uids)].copy()
            mapped_uids = set()
            try:
                bridge = _laad_participant_user_bridge(DB_URL)
                if not bridge.empty and 'participant_id' in bridge.columns and 'user_id' in bridge.columns:
                    mapped_uids = set(
                        pd.to_numeric(
                            bridge.loc[bridge['participant_id'].isin(uids), 'user_id'],
                            errors='coerce'
                        ).dropna().astype(int)
                    )
            except Exception as e:
                logger.warning(f"Kon participant-user bridge niet gebruiken voor artikelfiltering: {e}")

            if mapped_uids:
                combined_uids = uids | mapped_uids
                df_inter = df_inter[df_inter['user_id'].isin(combined_uids)].copy()
            else:
                df_inter = df_candidate
    if df_inter.empty:
        return empty_top_artikelen, pd.DataFrame(), {
            'Totaal views': 0, 'Unieke lezers': 0, 'Unieke artikelen': 0,
            'Lezers met actief account': 0, 'Gefilterde interne accounts': 0,
        }
    df_inter['interactable_id'] = pd.to_numeric(df_inter['interactable_id'], errors='coerce')
    df_inter['created_at'] = pd.to_datetime(df_inter['created_at'], errors='coerce')

    
    # Load content table to map interactable_id to public_id (content_id)
    try:
        df_content_raw = pd.read_sql("SELECT * FROM content", engine)
        if len(df_content_raw.columns) >= 2:
            # Extract: id and public_id columns
            df_content = df_content_raw.iloc[:, [0, 1]].copy()
            df_content.columns = ['id', 'public_id']
            df_content['id'] = pd.to_numeric(df_content['id'], errors='coerce')
            df_content['public_id'] = pd.to_numeric(df_content['public_id'], errors='coerce')
            logger.info(f"✓ Content geladen: {len(df_content)} records")
        else:
            logger.warning("Content tabel heeft niet genoeg kolommen")
            df_content = pd.DataFrame(columns=['id', 'public_id'])
    except Exception as e:
        logger.warning(f"Kon content niet laden: {e}")
        df_content = pd.DataFrame(columns=['id', 'public_id'])
    
    df_content_nl['content_id'] = pd.to_numeric(df_content_nl['content_id'], errors='coerce')
    df_content_nl = (
        df_content_nl[df_content_nl['locale'].str.startswith('nl')][['content_id', 'title']]
        .drop_duplicates('content_id')
    )

    df_content_nl = df_content_nl.drop_duplicates('content_id')
    
    # Join: interactions -> content_translations. In deze export verwijst
    # interactable_id direct naar content_translations.content_id; content.public_id
    # is niet dezelfde sleutel.
    df_views = (
        df_inter
        .merge(df_users, left_on='user_id', right_on='id', how='left')
        .merge(df_content_nl, left_on='interactable_id', right_on='content_id', how='left') # type: ignore
    )
    
    df_views['title'] = df_views['title'].fillna('Onbekend artikel')

    # Account status bepalen indien deleted_at beschikbaar is in de database
    if 'deleted_at' in df_views.columns:
        df_views['account_status'] = df_views['deleted_at'].isna().map({True: 'Actief account', False: 'Verwijderd account'})
    else:
        df_views['account_status'] = 'Onbekend'

    df_views['email_norm'] = df_views['email'].astype(str).str.strip().str.lower()
    df_views['email_localpart'] = df_views['email_norm'].str.split('@').str[0]

    # Filter interne accounts via de gedeelde definitie (account_filtering)
    internal_mask = interne_email_mask(df_views['email_norm'])
    smarthealth_localparts = set(df_views.loc[internal_mask, 'email_localpart'].dropna())
    df_views = df_views[~df_views['email_localpart'].isin(smarthealth_localparts)].copy()

    if 'user_id' not in df_views.columns:
        df_views['user_id'] = pd.NA

    if df_views.empty:
        return empty_top_artikelen, pd.DataFrame(), {
            'Totaal views': 0, 'Unieke lezers': 0, 'Unieke artikelen': 0,
            'Lezers met actief account': 0, 'Gefilterde interne accounts': int(len(smarthealth_localparts)),
        }
    
    top_artikelen = (
        df_views.groupby(['interactable_id', 'title'], dropna=False)
        .agg(
            Views=('interactable_id', 'count'),
            Unieke_lezers=('user_id', 'nunique'),
            Laatste_view=('created_at', 'max'),
        )
        .reset_index()
        .sort_values(['Unieke_lezers', 'Views'], ascending=False)
        .head(20)
    )

    samenvatting = {
        'Totaal views': int(len(df_views)),
        'Unieke lezers': int(df_views['user_id'].nunique()),
        'Unieke artikelen': int(df_views['interactable_id'].nunique()),
        'Lezers met actief account': (
            int(df_views.loc[df_views['deleted_at'].isna(), 'user_id'].nunique()) 
            if 'deleted_at' in df_views.columns else int(df_views['user_id'].nunique())
        ),
        'Gefilterde interne accounts': int(len(smarthealth_localparts)),
    }
    return top_artikelen, df_views, samenvatting


# ── Scores per opdrachtgever ───────────────────────────────────────────────────
# Vaste absolute afkapwaardes voor het indelen van snapshot-gemiddelden per
# opdrachtgever in categorieën. Per indicator (slug uit users_met_scores):
#   richting   : 'hoger_is_beter' → hogere gemiddelde score is gunstiger
#                'lager_is_beter' → lagere gemiddelde score is gunstiger (bv. BMI, stress)
#   grens_goed : onderste afkapwaarde — hieronder valt de onderste band
#   grens_hoog : bovenste afkapwaarde — vanaf hier valt de bovenste band
# Banden: score < grens_goed | grens_goed ≤ score < grens_hoog | score ≥ grens_hoog
# De waarden sluiten aan op bestaande afleidingen in data_ingestion.py
# (rec_heartrisk_cat: 10/20, rec_ls_stress_cat: 11/16, BMI WHO-grenzen 25/30,
# WAI-ranges 7/8.5) en gangbare DASS-/PSQI-ernstgraden.
# Indicatoren zonder entry vallen terug op de relatieve methode
# (t.o.v. het gewogen totaalgemiddelde).
SCORE_AFKAPWAARDES: dict[str, dict] = {
    # Leefstijlscore (1-10) en dieetscores (0-5) — hoger = beter
    'rec_ls_lifestyle_score':        {'richting': 'hoger_is_beter', 'grens_goed': 6.0, 'grens_hoog': 7.5},
    'rec_ls_score_fruit':            {'richting': 'hoger_is_beter', 'grens_goed': 3.0, 'grens_hoog': 4.0},
    'rec_ls_score_vegetables':       {'richting': 'hoger_is_beter', 'grens_goed': 3.0, 'grens_hoog': 4.0},
    'rec_ls_score_sugar':            {'richting': 'hoger_is_beter', 'grens_goed': 3.0, 'grens_hoog': 4.0},
    'rec_ls_score_saturated_fat':    {'richting': 'hoger_is_beter', 'grens_goed': 3.0, 'grens_hoog': 4.0},
    'rec_ls_score_natrium':          {'richting': 'hoger_is_beter', 'grens_goed': 3.0, 'grens_hoog': 4.0},
    'rec_ls_score_alcohol':          {'richting': 'hoger_is_beter', 'grens_goed': 3.0, 'grens_hoog': 4.0},
    'rec_ls_score_exercise':         {'richting': 'hoger_is_beter', 'grens_goed': 3.0, 'grens_hoog': 4.0},
    # Psycho-sociale scores — hoger = beter (instrumenteigen schalen)
    'rec_wellbeing_score':           {'richting': 'hoger_is_beter', 'grens_goed': 55.0, 'grens_hoog': 75.0},   # WHO-5 stijl, 0-100
    'rec_resilience_score':          {'richting': 'hoger_is_beter', 'grens_goed': 26.0, 'grens_hoog': 33.0},   # CD-RISC-10, 0-40
    'rec_self_efficacy_score':       {'richting': 'hoger_is_beter', 'grens_goed': 25.0, 'grens_hoog': 32.0},   # GSE, 10-40
    # Werkgerelateerde ASR-scores (genormaliseerd 0-10) — hoger = beter;
    # uitzondering: rec_asr_wai_score is de ruwe WAI-totaalscore (7-49),
    # indeling volgt de officiële WAI-klassen (slecht 7-27, matig 28-43, goed 44+)
    'rec_asr_work_ability_score':    {'richting': 'hoger_is_beter', 'grens_goed': 7.0, 'grens_hoog': 8.5},
    'rec_asr_wai_score':             {'richting': 'hoger_is_beter', 'grens_goed': 28.0, 'grens_hoog': 44.0},
    'rec_asr_vitality_score':        {'richting': 'hoger_is_beter', 'grens_goed': 6.0, 'grens_hoog': 7.5},
    'rec_asr_job_satisfaction_score': {'richting': 'hoger_is_beter', 'grens_goed': 6.0, 'grens_hoog': 7.5},
    # Medische en belastingscores — lager = beter
    'rec_med_bmi':                   {'richting': 'lager_is_beter', 'grens_goed': 25.0, 'grens_hoog': 30.0},
    'rec_heartrisk':                 {'richting': 'lager_is_beter', 'grens_goed': 10.0, 'grens_hoog': 20.0},
    # Stresssom: waargenomen bereik 0-28 (korte somscore), niet 0-40 zoals eerst
    # geconfigureerd — daardoor werd de hoogste band nooit bereikt.
    'rec_ls_stress_sum':             {'richting': 'lager_is_beter', 'grens_goed': 10.0, 'grens_hoog': 16.0},
    'rec_ls_sleep_psqi_sum':         {'richting': 'lager_is_beter', 'grens_goed': 7.0, 'grens_hoog': 15.0},
    'rec_dass_stress_score':         {'richting': 'lager_is_beter', 'grens_goed': 15.0, 'grens_hoog': 26.0},
    'rec_dass_anxiety_score':        {'richting': 'lager_is_beter', 'grens_goed': 8.0, 'grens_hoog': 15.0},
    'rec_dass_depression_score':     {'richting': 'lager_is_beter', 'grens_goed': 10.0, 'grens_hoog': 21.0},
    'rec_asr_burn_out_score':        {'richting': 'lager_is_beter', 'grens_goed': 4.0, 'grens_hoog': 6.0},
    'rec_asr_workload_score':        {'richting': 'lager_is_beter', 'grens_goed': 4.0, 'grens_hoog': 6.0},
    'rec_asr_exhaustion_score':      {'richting': 'lager_is_beter', 'grens_goed': 4.0, 'grens_hoog': 6.0},
}

# Label en kleur per band, per richting. Bij 'hoger_is_beter' is een hoge
# score gunstig (Hoog = groen); bij 'lager_is_beter' is een lage score
# gunstig (Laag = groen), aansluitend op RISICO_COLORS-conventies.
AFKAPWAARDE_BANDEN: dict[str, dict] = {
    'hoger_is_beter': {
        'laag':   {'label': 'Laag',  'kleur': '#E74C3C'},  # rood
        'midden': {'label': 'Goed',  'kleur': '#2ECC71'},  # groen
        'hoog':   {'label': 'Hoog',  'kleur': '#1A8A4A'},  # donkergroen
    },
    'lager_is_beter': {
        'laag':   {'label': 'Laag',  'kleur': '#2ECC71'},  # groen
        'midden': {'label': 'Matig', 'kleur': '#E87722'},  # oranje
        'hoog':   {'label': 'Hoog',  'kleur': '#E74C3C'},  # rood
    },
}


def _afkap_categorie(score, config: dict) -> str:
    """Deel één (gemiddelde) score in met vaste absolute afkapwaardes."""
    if pd.isna(score):
        return 'Onbekend'
    banden = AFKAPWAARDE_BANDEN[config['richting']]
    if score >= config['grens_hoog']:
        return banden['hoog']['label']
    if score >= config['grens_goed']:
        return banden['midden']['label']
    return banden['laag']['label']


def _controleer_afkapwaardes(scores, indicator: str) -> str | None:
    """
    Controleer of de geconfigureerde afkapwaardes passen bij het waargenomen
    databereik. Geeft een waarschuwingstekst terug bij een mismatch, anders None.
    Vangt bijvoorbeeld een configuratie met drempels op een 0-40 schaal terwijl
    de data op een 0-10 schaal zit (of andersom).
    """
    cfg = SCORE_AFKAPWAARDES.get(indicator)
    if cfg is None:
        return None
    s = pd.to_numeric(pd.Series(scores), errors='coerce').dropna()
    if s.empty:
        return None
    mn, mx = float(s.min()), float(s.max())
    g1, g2 = float(cfg['grens_goed']), float(cfg['grens_hoog'])
    problemen = []
    if mx < g1:
        problemen.append(
            f"alle scores liggen onder grens_goed ({g1:g}); maximum waargenomen: {mx:.2f}"
        )
    elif mx < g2:
        problemen.append(
            f"maximum ({mx:.2f}) ligt onder grens_hoog ({g2:g}) — de hoogste band wordt nooit bereikt"
        )
    if mn > g2:
        problemen.append(
            f"minimum ({mn:.2f}) ligt boven grens_hoog ({g2:g}) — de onderste band wordt nooit bereikt"
        )
    if problemen:
        return (
            f"Afkapwaardes voor '{indicator}' passen mogelijk niet bij deze data: "
            + "; ".join(problemen)
            + " Controleer de schaal van de scores en pas SCORE_AFKAPWAARDES aan."
        )
    return None


def maak_scores_per_opdrachtgever(base_pad, db_url: str,
                                   indicator: str = 'rec_ls_lifestyle_score',
                                   indicator_label: str = 'Leefstijlscore',
                                   min_n: int = 1,
                                   participant_ids: list | set | pd.Series | None = None,
                                   afkapwaardes: dict | None = None) -> tuple:
    """
    Vergelijkt gemiddelde scores per opdrachtgever (store).
    Alleen stores met minimaal min_n deelnemers worden getoond.

    De gemiddelden worden ingedeeld in categorieën via vaste absolute
    afkapwaardes (zie SCORE_AFKAPWAARDES). Voor indicatoren zonder configuratie
    wordt teruggevallen op een relatieve indeling t.o.v. het gewogen
    totaalgemiddelde.
    """
    from pathlib import Path
    import pandas as pd
    base_pad = Path(base_pad)

    # Scores laden
    df_expanded = load_my_clic_participants_expanded(db_url or DB_URL)
    if df_expanded.empty or 'participant_id' not in df_expanded.columns:
        return _empty_store_plot("Geen data beschikbaar voor deze selectie."), 0, 0

    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        df_expanded = df_expanded[pd.to_numeric(df_expanded['participant_id'], errors='coerce').isin(pids)]
        if df_expanded.empty:
            return _empty_store_plot("Geen data beschikbaar voor deze selectie."), 0, 0

    links = _laad_opdrachtgever_links(db_url or DB_URL)
    if links.empty:
        return _empty_store_plot("Geen opdrachtgeverkoppeling beschikbaar."), 0, 0

    df_expanded['participant_id'] = pd.to_numeric(df_expanded['participant_id'], errors='coerce')
    df_expanded = df_expanded.drop_duplicates(subset='participant_id', keep='first')
    df_expanded = df_expanded.drop(columns=[c for c in ['store_id', 'store_naam', 'store_name'] if c in df_expanded.columns])
    df_expanded = df_expanded.merge(
        links[['participant_id', 'store_id', 'store_name']],
        on='participant_id',
        how='inner',
    )

    # Interne Smart Health testaccounts hier ook uitfilteren: deze loader
    # omzeilt load_users_met_scores en bevat anders extreme testwaarden die
    # gemiddelden (en dus categorieën) vervuilen.
    df_expanded = filter_interne_accounts(
        df_expanded,
        db_url=db_url or DB_URL,
        participant_id_col='participant_id',
        bridge_loader=_laad_participant_user_bridge,
    )

    # Dieetsubscores gelden op de snapshot als 0-5 score. In deze kolommen
    # zitten ook rauwe intake-waarden (gram/mg/per week); die vallen buiten
    # het scoredomein en worden genegeerd i.p.v. het gemiddelde te verstoren.
    if indicator in DIEET_SCORE_COLS:
        _s = pd.to_numeric(df_expanded.get(indicator), errors='coerce')
        _buiten = int((_s.notna() & ((_s < 0) | (_s > 5))).sum())
        if _buiten:
            logging.info(
                f"Snapshot {indicator}: {_buiten} waarde(n) buiten scoredomein [0-5] genegeerd"
            )
            df_expanded[indicator] = _s.where(_s.between(0, 5))

    if indicator in df_expanded.columns:
        df_expanded['score'] = pd.to_numeric(df_expanded[indicator], errors='coerce')
    else:
        df_expanded['score'] = pd.NA
    df_expanded['store_naam'] = df_expanded['store_name']

    # Aggregeer per store
    # n_deelnemers = totaal aantal deelnemers per store (ook zonder score) # type: ignore
    # n_met_score  = aantal deelnemers met een ingevulde score (noemer voor gemiddelde) # type: ignore
    agg = (df_expanded.groupby(['store_id', 'store_naam'])
           .agg(
               gemiddelde=('score', 'mean'), # Gebruik de intern genormaliseerde 'score' kolom
               n_met_score=('score', 'count'),
               n_deelnemers=('participant_id', 'nunique'),
           )
           .reset_index())
    
    # Filter pas na aggregatie en drop pas dan NaNs
    agg = agg.dropna(subset=['gemiddelde'])

    # Filter op deelnemers met score; anders kan één ingevulde score binnen
    # een grote opdrachtgever ten onrechte als representatief gemiddelde tellen.
    if not agg.empty:
        agg['gemiddelde'] = pd.to_numeric(agg['gemiddelde'], errors='coerce').round(2)
        agg = agg[agg['n_met_score'] >= min_n].sort_values('gemiddelde', ascending=True)

    if agg.empty:
        return _empty_store_plot("Geen data beschikbaar voor deze selectie."), 0, 0

    # Gebruik een gewogen totaalgemiddelde zodat grote stores zwaarder meetellen
    # dan kleine stores in de referentielijn.
    totaal_gem = _weighted_average(agg['gemiddelde'], agg['n_met_score'])
    if pd.isna(totaal_gem): totaal_gem = 0

    # Indeling met vaste absolute afkapwaardes; voor indicatoren zonder
    # configuratie een relatieve indeling t.o.v. het gewogen totaalgemiddelde.
    config_afkap = afkapwaardes if afkapwaardes is not None else SCORE_AFKAPWAARDES.get(indicator)
    if config_afkap:
        methode = 'absoluut'
        banden_cfg = AFKAPWAARDE_BANDEN[config_afkap['richting']]
        volgorde = [banden_cfg['laag'], banden_cfg['midden'], banden_cfg['hoog']]
        agg['categorie'] = agg['gemiddelde'].apply(lambda g: _afkap_categorie(g, config_afkap))
        band_uitleg = (
            f"{banden_cfg['laag']['label']} < {config_afkap['grens_goed']:g} · "
            f"{banden_cfg['midden']['label']} {config_afkap['grens_goed']:g}–{config_afkap['grens_hoog']:g} · "
            f"{banden_cfg['hoog']['label']} ≥ {config_afkap['grens_hoog']:g}"
        )
        grenzen = (float(config_afkap['grens_goed']), float(config_afkap['grens_hoog']))
        richting = config_afkap['richting']
    else:
        methode = 'relatief'
        def _categoriseer(score, ref):
            if pd.isna(score) or ref == 0:
                return 'Onbekend'
            ratio = score / ref
            if ratio >= 1.0:
                return 'Goed'
            if ratio >= 0.8:
                return 'Matig'
            return 'Slecht'

        agg['categorie'] = agg['gemiddelde'].apply(lambda g: _categoriseer(g, totaal_gem))
        volgorde = [
            {'label': 'Goed',  'kleur': RISICO_COLORS[0]},
            {'label': 'Matig', 'kleur': RISICO_COLORS[1]},
            {'label': 'Slecht', 'kleur': RISICO_COLORS[2]},
        ]
        band_uitleg = ("Relatief t.o.v. totaalgemiddelde: goed ≥100%, matig 80-100%, "
                       "slecht <80% (geen afkapwaardes geconfigureerd voor deze indicator).")
        grenzen = None
        richting = None

    cat_kleuren = {band['label']: band['kleur'] for band in volgorde}
    cat_kleuren['Onbekend'] = '#CCCCCC'
    kleuren = [cat_kleuren.get(cat, '#CCCCCC') for cat in agg['categorie']]

    # ── Verdeling van individuen over de banden per opdrachtgever ─────────────
    # Methodologisch sterker dan alleen het groepsgemiddelde classificeren:
    # eerst individuele scores indelen, dan het aandeel per band rapporteren.
    # Een gemiddelde in de 'Goed'-band kan bijvoorbeeld nog steeds veel
    # individuen in de risicoband verbergen.
    fig_verdeling = None
    verdeling_tabel = pd.DataFrame()
    if methode == 'absoluut':
        df_scored = df_expanded.dropna(subset=['score']).copy()
        if not df_scored.empty and 'store_naam' in df_scored.columns:
            df_scored['band'] = df_scored['score'].apply(lambda s: _afkap_categorie(s, config_afkap))
            label_volgorde = [band['label'] for band in volgorde]
            kleur_map = {band['label']: band['kleur'] for band in volgorde}
            verdeling = (
                df_scored.groupby(['store_naam', 'band'])
                .size().reset_index(name='n')
            )
            totaal_per_store = verdeling.groupby('store_naam')['n'].transform('sum')
            verdeling['pct'] = (verdeling['n'] / totaal_per_store * 100).round(1)
            verdeling_tabel = (
                verdeling.pivot_table(index='store_naam', columns='band', values='pct', fill_value=0.0)
                .reindex(columns=label_volgorde, fill_value=0.0)
                .reset_index()
            )
            fig_verdeling = go.Figure()
            for lab in label_volgorde:
                sub = verdeling[verdeling['band'] == lab]
                fig_verdeling.add_trace(go.Bar(
                    y=sub['store_naam'],
                    x=sub['pct'],
                    orientation='h',
                    name=lab,
                    marker_color=kleur_map[lab],
                ))
            fig_verdeling.update_layout(
                barmode='stack',
                title=f'{indicator_label} — aandeel individuen per categorie (%)',
                xaxis_title='Percentage deelnemers (%)',
                yaxis_title='',
                height=max(400, len(verdeling_tabel) * 30),
                margin=dict(l=200, r=100),
                legend=dict(
                    orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, title='Categorie',
                ),
            )

    fig = go.Figure(go.Bar(
        x=agg['gemiddelde'],
        y=agg['store_naam'],
        orientation='h',
        marker_color=kleuren,
        text=agg.apply(
            lambda r: f"{r['gemiddelde']:.2f} — {r['categorie']} "
                      f"({r['n_met_score']}/{r['n_deelnemers']} deeln.)",
            axis=1,
        ),
        textposition='outside',
    ))

    fig.add_vline(
        x=totaal_gem, line_dash='dot', line_color='black',
        annotation_text=f'Totaalgemiddelde ({totaal_gem:.2f})',
        annotation_position='top',
    )

    # Toon de absolute afkapwaardes als verticale stippellijnen
    if methode == 'absoluut' and grenzen:
        fig.add_vline(
            x=grenzen[0], line_dash='dash', line_color='#7F8C8D',
            annotation_text=f'Afkapwaarde {grenzen[0]:g}',
            annotation_position='top left',
        )
        fig.add_vline(
            x=grenzen[1], line_dash='dash', line_color='#7F8C8D',
            annotation_text=f'Afkapwaarde {grenzen[1]:g}',
            annotation_position='top right',
        )

    for band in volgorde:
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            marker_color=band['kleur'],
            name=band['label'],
            showlegend=True,
        ))

    titel_suffix = 'met vaste afkapwaardes' if methode == 'absoluut' else 'relatief t.o.v. totaalgemiddelde'
    fig.update_layout(
        title=f'{indicator_label} per opdrachtgever ({titel_suffix})',
        xaxis_title=indicator_label,
        yaxis_title='',
        height=max(400, len(agg) * 30),
        margin=dict(l=200, r=150),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            title='Categorie',
        ),
    )
    cat_counts = agg['categorie'].value_counts().to_dict()
    category_counts = {band['label']: int(cat_counts.get(band['label'], 0)) for band in volgorde}
    afkap_info = {
        'methode': methode,
        'richting': richting,
        'uitleg': band_uitleg,
        'grenzen': grenzen,
        'waarschuwing': _controleer_afkapwaardes(df_expanded.get('score'), indicator),
    }
    return fig, totaal_gem, len(agg), category_counts, afkap_info, fig_verdeling, verdeling_tabel


# ══════════════════════════════════════════════════════════════════════════════
# MEERDERE VRAGENLIJSTEN INVULLERS


# ══════════════════════════════════════════════════════════════════════════════
def maak_meerdere_vragenlijsten_plot(base_pad: Path, participant_ids: list | set | pd.Series | None = None) -> tuple:
    """
    Analyseert wie meerdere vragenlijsten invullen.
    Geeft een plot van de verdeling + een profielanalyse terug.
    """
    base_pad = Path(base_pad)

    known_questionnaire_ids = {1, 2, 3, 4, 5, 6, 7, 8}
    naam_map = {
        1: 'Smart Health Test',
        2: 'Smart Work Test',
        3: 'Resilience',
        4: 'Well-being',
        5: 'Positive Health',
        6: 'Self-efficacy',
        7: 'Negative emotions',
        8: 'Smartphone and stress',
    }
    naam_map = {q_id: naam for q_id, naam in naam_map.items() if q_id in known_questionnaire_ids}
    # Dynamische naam mapping met filter
    try:
        from sqlalchemy import create_engine
        engine = create_engine(DB_URL)
        df_q = pd.read_sql("SELECT id, internal_name, name::text AS name, slug FROM questionnaires", engine)
        for _, row in df_q.iterrows():
            q_id = int(row['id'])
            if q_id not in known_questionnaire_ids:
                continue
            name = row.get('internal_name') or row.get('name') or row.get('slug')
            if name:
                name_low = str(name).lower()
                if not any(x in name_low for x in ['bloeddruk', 'blood_pressure', 'bloodpressure', 'test_']):
                    naam_map[q_id] = str(name)
    except:
        naam_map = {
            1: 'Smart Health Test', 2: 'Smart Work Test', 3: 'Resilience',
            4: 'Well-being', 5: 'Positive Health', 6: 'Self-efficacy',
            7: 'Negative emotions', 8: 'Smartphone and stress'
        }

    df_comp = load_completions(DB_URL)
    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        df_comp = df_comp[pd.to_numeric(df_comp['participant_id'], errors='coerce').isin(pids)].copy()
    df_comp['created_at'] = pd.to_datetime(df_comp['created_at'], errors='coerce')
    df_comp = df_comp[df_comp['questionnaire_id'].isin(naam_map.keys())]

    # Filter completions: gebruik alleen echte vragenlijsten
    df_comp['questionnaire_id'] = pd.to_numeric(df_comp['questionnaire_id'], errors='coerce')
    df_comp = df_comp[df_comp['questionnaire_id'].isin(naam_map.keys())].copy()

    # Aantal unieke vragenlijsten per participant
    per_participant = (
        df_comp.groupby('participant_id')['questionnaire_id']
        .nunique()
        .reset_index(name='n_vragenlijsten')
    )

    # Verdeling
    verdeling = (
        per_participant['n_vragenlijsten']
        .value_counts()
        .sort_index()
        .reset_index()
    )
    verdeling.columns = ['n_vragenlijsten', 'n_deelnemers']
    verdeling['pct'] = (verdeling['n_deelnemers'] / verdeling['n_deelnemers'].sum() * 100).round(1)

    kleuren = [HOOFD_KLEUR if n == 1 else '#2ECC71' for n in verdeling['n_vragenlijsten']]

    fig = go.Figure(go.Bar(
        x=verdeling['n_vragenlijsten'],
        y=verdeling['n_deelnemers'],
        marker_color=kleuren,
        text=verdeling.apply(lambda r: f"{int(r['n_deelnemers']):,} ({r['pct']}%)", axis=1),
        textposition='outside',
    ))
    fig.update_layout(
        title='Aantal ingevulde vragenlijsten per deelnemer',
        xaxis_title='Aantal verschillende vragenlijsten',
        yaxis_title='Aantal deelnemers',
        xaxis=dict(tickmode='linear', dtick=1),
        showlegend=False,
        height=450,
    )

    # Welke combinaties komen het meest voor
    combinaties = (
        df_comp.groupby('participant_id')['questionnaire_id']
        .apply(lambda x: ' + '.join(sorted([naam_map.get(i, str(i)) for i in x.unique()])))
        .reset_index(name='combinatie')
    )
    top_combinaties = (
        combinaties['combinatie']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_combinaties.columns = ['Combinatie', 'Aantal deelnemers']

    samenvatting = {
        'Totaal deelnemers': len(per_participant),
        'Slechts 1 vragenlijst': int((per_participant['n_vragenlijsten'] == 1).sum()),
        '2 of meer vragenlijsten': int((per_participant['n_vragenlijsten'] >= 2).sum()),
        'Alle vragenlijsten (8)': int((per_participant['n_vragenlijsten'] == 8).sum()),
        'Gemiddeld aantal': round(per_participant['n_vragenlijsten'].mean(), 2),
    }

    return fig, top_combinaties, samenvatting, per_participant


# ══════════════════════════════════════════════════════════════════════════════
# PROFIEL VROEGE KOPERS
# ══════════════════════════════════════════════════════════════════════════════
def maak_vroege_kopers_profiel(base_pad: Path, db_url: str,
                               geselecteerde_groepen: list[str] | None = None,
                               include_wellbeing: bool = False,
                               participant_ids: list | set | pd.Series | None = None) -> tuple:
    """
    Analyseert het profiel van vroege kopers: wie kopen er het eerste product?
    Vergelijkt vroege kopers (eerste aankoop <90 dagen na registratie)
    """
    try:
        base_pad = Path(base_pad)
        success_statuses = {'completed', 'paid', 'success', 'shipped', 'delivered', 'voltooid', 'betaald'}

        # Orders laden
        df_orders = None
        if _has_valid_db_url(db_url):
            try:
                engine = create_engine(db_url)
                df_orders = pd.read_sql('SELECT id, user_id, created_at, status FROM orders', engine)
            except Exception:
                df_orders = pd.DataFrame()

        # Aankopen laden via centrale herstelde koppeling.
        df_orders = get_user_purchases(base_pad, db_url)
        if participant_ids is not None:
            pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
            if 'participant_id' in df_orders.columns:
                df_orders = df_orders[pd.to_numeric(df_orders['participant_id'], errors='coerce').isin(pids)]

        for col in ['participant_id', 'user_id', 'status']:
            if col not in df_orders.columns:
                df_orders[col] = pd.NA
        if 'created_at' not in df_orders.columns:
            df_orders['created_at'] = pd.NaT

        if not df_orders.empty:
            df_orders['participant_id'] = pd.to_numeric(df_orders['participant_id'], errors='coerce')
            df_orders['created_at'] = pd.to_datetime(df_orders['created_at'], errors='coerce', utc=True).dt.tz_convert(None)
            df_orders = df_orders[df_orders['status'].astype(str).str.lower().isin(success_statuses)].copy()
            df_orders = df_orders.dropna(subset=['participant_id', 'created_at']).copy()

        eerste_aankoop = (
            df_orders.groupby('participant_id')['created_at']
            .min()
            .reset_index(name='eerste_aankoop')
        )

        # Registratiedatum (Referentie datum bepalen: account creation)
        df_users = load_participants(db_url or DB_URL)[['id', 'created_at']].copy()
        df_users = df_users.rename(columns={'id': 'participant_id', 'created_at': 'registratie_datum'})
        df_users['participant_id'] = pd.to_numeric(df_users['participant_id'], errors='coerce')
        df_users['referentie_datum'] = pd.to_datetime(
            df_users['registratie_datum'], errors='coerce', utc=True
        ).dt.tz_convert(None)

        df_kopers = eerste_aankoop.merge(df_users[['participant_id', 'referentie_datum']], on='participant_id', how='left')
        if 'user_id' in df_orders.columns:
            user_lookup = (
                df_orders[['participant_id', 'user_id']]
                .dropna(subset=['participant_id', 'user_id'])
                .drop_duplicates(subset=['participant_id'])
            )
            df_kopers = df_kopers.merge(user_lookup, on='participant_id', how='left')
        df_kopers['dagen_tot_koop'] = (df_kopers['eerste_aankoop'] - df_kopers['referentie_datum']).dt.days

        # Groepen
        def groep(dagen):
            if pd.isna(dagen): return 'Geen aankoop'
            elif dagen <= 90: return 'Vroege koper (≤90 dagen)'
            elif dagen <= 365: return 'Late koper (91-365 dagen)'
            else: return 'Zeer late koper (>365 dagen)'

        df_kopers['koper_groep'] = df_kopers['dagen_tot_koop'].apply(groep)

        # Scores koppelen
        df_scores_expanded = load_my_clic_participants_expanded(DB_URL)
        df_scores_expanded['participant_id'] = pd.to_numeric(df_scores_expanded['participant_id'], errors='coerce')
        if participant_ids is not None:
            pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
            df_scores_expanded = df_scores_expanded[df_scores_expanded['participant_id'].isin(pids)]
        df_scores_expanded = df_scores_expanded.drop_duplicates(subset='participant_id', keep='first')

        # Alle gebruikers krijgen een groep - select only available columns
        select_cols = ['participant_id']
        for col in ['rec_ls_lifestyle_score', 'rec_med_bmi', 'rec_ls_stress_sum', 'rec_heartrisk',
                     'rec_resilience_score', 'rec_wellbeing_score', 'rec_age_current', 'rec_user_gender']:
            if col in df_scores_expanded.columns:
                select_cols.append(col)
        df_all = df_scores_expanded[select_cols].copy()
        df_all['koper_groep'] = 'Geen aankoop'
        df_all['dagen_tot_koop'] = pd.NA

        participant_lookup = df_kopers.dropna(subset=['participant_id']).set_index('participant_id').to_dict(orient='index')
        user_lookup = df_kopers.dropna(subset=['user_id']).set_index('user_id').to_dict(orient='index')

        for idx, row in df_all.iterrows():
            pid = pd.to_numeric(row.get('participant_id'), errors='coerce')
            uid = pd.to_numeric(row.get('user_id'), errors='coerce')
            if pd.notna(pid) and int(pid) in participant_lookup:
                meta = participant_lookup[int(pid)]
            elif pd.notna(uid) and int(uid) in user_lookup:
                meta = user_lookup[int(uid)]
            else:
                continue
            df_all.at[idx, 'koper_groep'] = meta.get('koper_groep', 'Geen aankoop')
            df_all.at[idx, 'dagen_tot_koop'] = meta.get('dagen_tot_koop', pd.NA)

        indicatoren = {
            'Leefstijlscore': 'rec_ls_lifestyle_score',
            'BMI': 'rec_med_bmi',
            'Stress': 'rec_ls_stress_sum',
            'Heartrisk': 'rec_heartrisk',
            'Veerkracht': 'rec_resilience_score',
        }
        if include_wellbeing:
            indicatoren['Welzijn'] = 'rec_wellbeing_score' # type: ignore

        rijen = []
        groep_volgorde = ['Vroege koper (≤90 dagen)', 'Late koper (91-365 dagen)',
                          'Zeer late koper (>365 dagen)', 'Geen aankoop']
        geselecteerde_groepen = geselecteerde_groepen or groep_volgorde
        geselecteerde_groepen = [groep for groep in groep_volgorde if groep in geselecteerde_groepen]
        for label, kolom in indicatoren.items():
            df_all[kolom] = pd.to_numeric(df_all[kolom], errors='coerce')
            for grp in geselecteerde_groepen:
                sub = df_all[df_all['koper_groep'] == grp][kolom].dropna()
                if len(sub) > 0:
                    rijen.append({
                        'Indicator': label,
                        'Groep': grp,
                        'Gemiddelde': round(sub.mean(), 2),
                        'N': len(sub),
                    })

        df_plot = pd.DataFrame(rijen)

        kleur_map = {
            'Vroege koper (≤90 dagen)':    '#2ECC71',
            'Late koper (91-365 dagen)':   '#E87722',
            'Zeer late koper (>365 dagen)': '#E74C3C',
            'Geen aankoop':                '#95A5A6',
        }

        fig = px.bar(
            df_plot, x='Indicator', y='Gemiddelde',
            color='Groep',
            barmode='group',
            color_discrete_map=kleur_map,
            text='Gemiddelde',
            labels={'Gemiddelde': 'Gemiddelde score', 'Indicator': ''}, # type: ignore
            title=f'Profiel vroege kopers vs late kopers (referentie: account_created)', # Use a default reference
            category_orders={'Groep': geselecteerde_groepen},
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            legend_title_text='',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), # type: ignore
            height=500,
        )

        df_kpi = df_all[df_all['koper_groep'].isin(geselecteerde_groepen)].copy()
        if df_kpi.empty:
            df_kpi = df_all.copy()
        df_dagen = df_kopers[df_kopers['koper_groep'].isin(geselecteerde_groepen)].copy()
        profiel = {
            'Aantal geselecteerde gebruikers': len(df_kpi),
            'Gemiddelde leeftijd': round(pd.to_numeric(df_kpi['rec_age_current'], errors='coerce').mean(), 1),
            '% vrouw': round((pd.to_numeric(df_kpi['rec_user_gender'], errors='coerce') == 0).mean() * 100, 1),
            'Gem. leefstijlscore': round(pd.to_numeric(df_kpi['rec_ls_lifestyle_score'], errors='coerce').mean(), 2),
            'Gem. dagen tot aankoop': round(pd.to_numeric(df_dagen['dagen_tot_koop'], errors='coerce').mean(), 1),
        }

        # Verdeling dagen tot eerste aankoop
        df_kopers_valid = df_kopers[
            df_kopers['dagen_tot_koop'].between(0, 730) &
            df_kopers['koper_groep'].isin(geselecteerde_groepen)
        ].copy()
        if df_kopers_valid.empty and not df_kopers.empty:
            # Fallback to all buyers if specific groups are empty
            df_kopers_valid = df_kopers[df_kopers['dagen_tot_koop'].between(0, 730)].copy() # type: ignore
        elif df_kopers_valid.empty:
            # If still empty, create an empty DataFrame to avoid errors
            df_kopers_valid = pd.DataFrame(columns=['dagen_tot_koop'])
        fig_dagen = px.histogram(
            df_kopers_valid,
            x='dagen_tot_koop',
            nbins=50,
            color_discrete_sequence=[HOOFD_KLEUR],
            labels={'dagen_tot_koop': 'Dagen tot eerste aankoop', 'count': 'Aantal kopers'},
            title='Verdeling: dagen tot eerste aankoop (0-730 dagen)',
        )
        fig_dagen.add_vline(x=90, line_dash='dash', line_color='green',
                            annotation_text='90 dagen grens', annotation_position='top right')
        return fig, fig_dagen, profiel, df_plot
    except Exception as e:
        logger.error(f"Error in maak_vroege_kopers_profiel: {e}")
        # Return empty figures with error message
        empty_df = pd.DataFrame({'Indicator': [], 'Groep': [], 'Gemiddelde': [], 'N': []})
        fig_empty = px.bar(empty_df, x='Indicator', y='Gemiddelde',
                          title='Kopersprofiel: geen data beschikbaar')
        fig_empty.update_layout(height=500) # type: ignore

        fig_dagen_empty = px.histogram(pd.DataFrame(), title='Verdeling: geen data beschikbaar')
        fig_dagen_empty.update_layout(height=350) # type: ignore

        empty_profiel = {
            'Aantal geselecteerde gebruikers': 0,
            'Gemiddelde leeftijd': 0.0,
            '% vrouw': 0.0,
            'Gem. leefstijlscore': 0.0,
            'Gem. dagen tot aankoop': 0.0,
        }
        
        return fig_empty, fig_dagen_empty, empty_profiel, empty_df

def _is_higher_better(slug: str) -> bool:
    """Bepaal of een hogere score ‘beter’ is aan de hand van variabele metadata."""
    if not isinstance(slug, str) or slug == '':
        return True

    for variabele in VARIABELEN_DICT.values():
        if variabele.get('kolom') == slug or variabele.get('label') == slug:
            kleur_richting = variabele.get('kleur_richting')
            if kleur_richting == 'laatste_goed':
                return True
            if kleur_richting == 'eerste_goed':
                return False
            if kleur_richting == 'midden_goed':
                return True
    
    lager_is_beter = ['alcohol', 'smoking', 'stress', 'bmi', 'sugar', 'fat', 'salt', 'blood_pressure', 'diabetes']
    return slug not in lager_is_beter


def maak_risico_migratie_sankey(df_long: pd.DataFrame, slug: str, label_map: dict, lang: str = 'nl') -> go.Figure:
    """Toont de verschuiving tussen risicocategorieën tussen de eerste en laatste meting."""
    if {'Van', 'Na', 'Aantal'}.issubset(df_long.columns):
        matrix = df_long.copy()
    else:
        from analyses import bereken_risico_migratie
        matrix = bereken_risico_migratie(df_long, slug)
    
    if matrix.empty:
        return _empty_store_plot(tr("Onvoldoende deelnemers met herhaalmetingen voor migratie-analyse.", lang))
        
    # Bepaal werkelijk aanwezige categorieën (geen hardcoded filtering)
    present_cats = sorted(set(matrix['Van'].unique()) | set(matrix['Na'].unique()))
    higher_is_better = _is_higher_better(slug)
    display_cats = sorted(present_cats, reverse=higher_is_better)
    ncats = len(display_cats)
    
    if ncats == 0:
        return _empty_store_plot(tr("Onvoldoende data voor visualisatie.", lang))
    
    # Bereken totals per categorie voor eerste en laatste meting
    totals_van = matrix.groupby('Van')['Aantal'].sum().to_dict()
    totals_na = matrix.groupby('Na')['Aantal'].sum().to_dict()

    def label_with_count(i, when='Eerste'):
        base = tr(label_map.get(i, str(i)), lang)
        count = int(totals_van.get(i, 0)) if when == 'Eerste' else int(totals_na.get(i, 0))
        return f"{base} ({tr(when, lang)}) — {count:,}"

    labels_van = [label_with_count(i, 'Eerste') for i in display_cats]
    labels_na = [label_with_count(i, 'Laatste') for i in display_cats]
    all_labels = labels_van + labels_na

    groen, oranje, rood = '#2ECC71', '#E87722', '#E74C3C'

    # Bepaal kleur per categorie op basis van higher_is_better logica
    cat_to_color = {}
    for cat in display_cats:
        if ncats == 1:
            cat_to_color[cat] = groen
        elif ncats == 2:
            if higher_is_better:
                cat_to_color[cat] = groen if cat == max(present_cats) else rood
            else:
                cat_to_color[cat] = groen if cat == min(present_cats) else rood
        elif ncats == 3:
            if higher_is_better:
                if cat == max(present_cats):
                    cat_to_color[cat] = groen
                elif cat == min(present_cats):
                    cat_to_color[cat] = rood
                else:
                    cat_to_color[cat] = oranje
            else:
                if cat == min(present_cats):
                    cat_to_color[cat] = groen
                elif cat == max(present_cats):
                    cat_to_color[cat] = rood
                else:
                    cat_to_color[cat] = oranje
        else:
            # Meer dan 3: verdeel groen/oranje/rood lineair
            if higher_is_better:
                if cat == max(present_cats):
                    cat_to_color[cat] = groen
                elif cat == min(present_cats):
                    cat_to_color[cat] = rood
                else:
                    cat_to_color[cat] = oranje
            else:
                if cat == min(present_cats):
                    cat_to_color[cat] = groen
                elif cat == max(present_cats):
                    cat_to_color[cat] = rood
                else:
                    cat_to_color[cat] = oranje

    cat_colors = [cat_to_color[cat] for cat in display_cats]
    if ncats == 1:
        positions = [0.45]
    else:
        step = 0.8 / (ncats - 1)
        positions = [0.05 + (idx * step) for idx in range(ncats)]

    node_colors = cat_colors + cat_colors
    node_x = [0.02] * ncats + [0.98] * ncats
    node_y = positions + positions

    # Map oude categorieën naar nieuwe indices
    cat_to_idx = {cat: i for i, cat in enumerate(display_cats)}
    
    sources = [cat_to_idx[int(v)] for v in matrix['Van']]
    targets = [cat_to_idx[int(v)] + ncats for v in matrix['Na']]
    values = matrix['Aantal'].tolist()

    # Build link hover text showing count and percentage of source-category
    # Map source totals for percentage calculations
    source_totals = matrix.groupby('Van')['Aantal'].sum().to_dict()
    link_hover = []
    for _, row in matrix.iterrows():
        s = int(row['Van']); t = int(row['Na']); v = int(row['Aantal'])
        total_s = int(source_totals.get(s, 0)) or 1
        pct = v / total_s * 100
        link_hover.append(f"{tr(label_map.get(s, str(s)), lang)} → {tr(label_map.get(t, str(t)), lang)}: {v:,} ({pct:.1f}%)")
    
    link_colors = []
    
    for _, row in matrix.iterrows():
        s, t = int(row['Van']), int(row['Na'])
        if not higher_is_better:
            if t < s: color = 'rgba(46, 204, 113, 0.4)' # Groen (verbetering)
            elif t > s: color = 'rgba(231, 76, 60, 0.4)' # Rood (verslechtering)
            else: color = 'rgba(200, 200, 200, 0.4)'
        else:
            if t > s: color = 'rgba(46, 204, 113, 0.4)'
            elif t < s: color = 'rgba(231, 76, 60, 0.4)'
            else: color = 'rgba(200, 200, 200, 0.4)'
        link_colors.append(color)

    fig = go.Figure(data=[go.Sankey(
        arrangement='freeform',
        node = dict(
            pad = 30,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = all_labels,
            color = node_colors,
            x = node_x,
            y = node_y,
            align = 'center',
            hovertemplate = "%{label}<extra></extra>",
            hoverlabel = dict(font = dict(size=13, color='black', family='Arial, sans-serif'))
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values,
            color = link_colors,
            customdata = link_hover,
            hovertemplate = "%{customdata}<extra></extra>",
            hoverlabel = dict(font = dict(size=12, family='Arial, sans-serif'))
        )
    )])

    fig.update_layout(
        title_text=tr("Risico-migratie: verschuiving tussen eerste en laatste meting", lang),
        height=700,
        margin=dict(l=200, r=200, t=100, b=50),
        font=dict(size=12, family='Arial, sans-serif', color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig


DEFAULT_RISICOMIGRATIE_BINS = {
    'bmi': {
        'bins': [-0.1, 18.5, 25.0, 30.0, 1000.0],
        'labels': ['Ondergewicht (<18.5)', 'Normaal gewicht (18.5-25)', 'Overgewicht (25-30)', 'Obesitas (>30)']
    },
    'fruit': {
        'bins': [-0.1, 0.5, 1.5, 10000.0],
        'labels': ['Weinig (<1/dag)', 'Matig (~1/dag)', 'Veel (≥2/dag)']
    },
    'vegetables': {
        'bins': [-0.1, 150.0, 250.0, 100000.0],
        'labels': ['Laag (<150g)', 'Matig (150-250g)', 'Veel (≥250g)']
    },
    'stress': {
        'bins': [-0.1, 5.0, 14.0, 100.0],
        'labels': ['Weinig stress (<5)', 'Matig stress (5-14)', 'Veel stress (>14)']
    },
    'dass_stress': {
        'bins': [-0.1, 14.0, 18.0, 25.0, 100.0],
        'labels': ['Normaal (0-14)', 'Matig (15-18)', 'Ernstig (19-25)', 'Zeer ernstig (>25)']
    },
    'dass_anxiety': {
        'bins': [-0.1, 7.0, 9.0, 14.0, 100.0],
        'labels': ['Normaal (0-7)', 'Matig (8-9)', 'Ernstig (10-14)', 'Zeer ernstig (>14)']
    },
    'dass_depression': {
        'bins': [-0.1, 9.0, 13.0, 20.0, 100.0],
        'labels': ['Normaal (0-9)', 'Matig (10-13)', 'Ernstig (14-20)', 'Zeer ernstig (>20)']
    },
    'sleep': {
        'bins': [-0.1, 5.0, 10.0, 100.0],
        'labels': ['Goed (PSQI ≤5)', 'Matig (PSQI 6-10)', 'Slecht (PSQI >10)']
    },
    'alcohol': {
        'bins': [-0.1, 0.5, 14.0, 1000.0],
        'labels': ['Geen/Zelden (<1/wk)', 'Matig (1-14/wk)', 'Hoog (>14/wk)']
    },
    'sugar': {
        'bins': [-0.1, 25.0, 50.0, 10000.0],
        'labels': ['Laag (<25g)', 'Matig (25-50g)', 'Hoog (>50g)']
    },
    'fat': {
        'bins': [-0.1, 22.0, 35.0, 10000.0],
        'labels': ['Laag (<22g)', 'Matig (22-35g)', 'Hoog (>35g)']
    },
    'salt': {
        'bins': [-0.1, 2400.0, 3500.0, 100000.0],
        'labels': ['Laag (<2400mg)', 'Matig (2400-3500mg)', 'Hoog (>3500mg)']
    },
    'blood_pressure': {
        'bins': [-0.1, 0.5, 1.5, 10.0],
        'labels': ['Normaal', 'Verhoogd', 'Hoog']
    },
    'steps': {
        'bins': [-0.1, 5000.0, 10000.0, 100000.0],
        'labels': ['Weinig (<5.000)', 'Matig (5.000-10.000)', 'Veel (>10.000)']
    },
    'exercise': {
        'bins': [-0.1, 150.0, 300.0, 10000.0],
        'labels': ['Onvoldoende (<150 min)', 'Voldoende (150-300 min)', 'Actief (>300 min)']
    },
    'wellbeing': {
        'bins': [-0.1, 50.0, 75.0, 100.0],
        'labels': ['Laag welzijn (<50)', 'Matig welzijn (50-75)', 'Hoog welzijn (>75)']
    },
    'resilience': {
        'bins': [-0.1, 20.0, 30.0, 40.0],
        'labels': ['Laag (<20)', 'Gemiddeld (20-30)', 'Hoog (>30)']
    },
    'workload': {
        'bins': [-0.1, 3.0, 7.0, 10.0],
        'labels': ['Lage werkdruk (<3)', 'Normale werkdruk (3-7)', 'Hoge werkdruk (>7)']
    },
    'job_satisfaction': {
        'bins': [-0.1, 5.0, 7.5, 10.0],
        'labels': ['Ontevreden (<5)', 'Voldoende (5-7.5)', 'Tevreden (>7.5)']
    },
    'lifestyle': {
        'bins': [-0.1, 60.0, 80.0, 100.0],
        'labels': ['Matig (<60)', 'Voldoende (60-80)', 'Uitstekend (>80)']
    },
    'heartrisk': {
        'bins': [-0.1, 0.5, 1.5, 10.0],
        'labels': ['Laag risico', 'Matig risico', 'Hoog risico']
    }
}


def bereken_risico_migratie_value_binned(df_long: pd.DataFrame, slug: str, max_unique: int = 6, n_bins: int = 5, custom_bins: dict | None = None):
    """Bereken migratie op basis van `score_value` door waarden te binen wanneer nodig.

    Returns: (matrix_df, label_map) where matrix_df has columns Van(int), Na(int), Aantal
    and label_map maps integer codes to display labels.
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame(), {}

    df_factor = df_long[df_long['slug'] == slug].sort_values(['participant_id', 'completion_created_at'])
    if df_factor.empty:
        return pd.DataFrame(), {}

    ids = df_factor.groupby('participant_id').size()
    multi_ids = ids[ids >= 2].index
    first = df_factor.groupby('participant_id').first().loc[multi_ids]
    last = df_factor.groupby('participant_id').last().loc[multi_ids]

    first_vals = pd.to_numeric(first['score_value'], errors='coerce')
    last_vals = pd.to_numeric(last['score_value'], errors='coerce')
    df_pairs = pd.DataFrame({'Van_raw': first_vals, 'Na_raw': last_vals}).dropna()
    if df_pairs.empty:
        return pd.DataFrame(), {}

    combined = pd.concat([df_pairs['Van_raw'], df_pairs['Na_raw']])
    unique_vals = combined.dropna().unique()

    bins_config = {**DEFAULT_RISICOMIGRATIE_BINS, **(custom_bins or {})}

    if slug in bins_config:
        try:
            cfg = bins_config[slug]
            bins_edges = list(cfg.get('bins'))
            labels = list(cfg.get('labels'))
            if not bins_edges or not labels:
                return pd.DataFrame(), {}
            df_pairs['Van_bin'] = pd.cut(df_pairs['Van_raw'], bins=bins_edges, include_lowest=True)
            df_pairs['Na_bin'] = pd.cut(df_pairs['Na_raw'], bins=bins_edges, include_lowest=True)
            categories = list(pd.Categorical(df_pairs['Van_bin']).categories)
            cat_to_code = {cat: i for i, cat in enumerate(categories)}
            df_pairs['Van'] = df_pairs['Van_bin'].map(cat_to_code)
            df_pairs['Na'] = df_pairs['Na_bin'].map(cat_to_code)
            label_map = {i: labels[i] if i < len(labels) else str(categories[i]) for i in range(len(categories))}
        except Exception:
            return pd.DataFrame(), {}
    else:
        # If few unique integer-like values, use them directly as categories
        is_integer_like = pd.api.types.is_integer_dtype(combined) or all(float(x).is_integer() for x in unique_vals[:min(len(unique_vals), 20)])
        if len(unique_vals) <= max_unique and is_integer_like:
            bins = sorted(int(v) for v in sorted(unique_vals))
            val_to_code = {v: i for i, v in enumerate(bins)}
            label_map = {i: str(v) for v, i in val_to_code.items()}
            df_pairs['Van'] = df_pairs['Van_raw'].astype(int).map(val_to_code)
            df_pairs['Na'] = df_pairs['Na_raw'].astype(int).map(val_to_code)
        else:
            # Use quantile-based bins with formatted labels
            try:
                bins_edges = list(np.unique(np.quantile(combined.dropna(), np.linspace(0, 1, n_bins + 1))))
                if len(bins_edges) <= 1:
                    return pd.DataFrame(), {}
                df_pairs['Van_bin'] = pd.cut(df_pairs['Van_raw'], bins=bins_edges, include_lowest=True)
                df_pairs['Na_bin'] = pd.cut(df_pairs['Na_raw'], bins=bins_edges, include_lowest=True)
                categories = list(pd.Categorical(df_pairs['Van_bin']).categories)
                label_map = {}
                for i, cat in enumerate(categories):
                    label_map[i] = f"{cat.left:.1f} – {cat.right:.1f}"
                cat_to_code = {cat: i for i, cat in enumerate(categories)}
                df_pairs['Van'] = df_pairs['Van_bin'].map(cat_to_code)
                df_pairs['Na'] = df_pairs['Na_bin'].map(cat_to_code)
            except Exception:
                return pd.DataFrame(), {}

    df_pairs = df_pairs.dropna(subset=['Van', 'Na'])
    if df_pairs.empty:
        return pd.DataFrame(), {}

    matrix = df_pairs.groupby(['Van', 'Na']).size().reset_index(name='Aantal')
    matrix['Van'] = matrix['Van'].astype(int)
    matrix['Na'] = matrix['Na'].astype(int)
    return matrix, label_map

def bereken_engagement_journey(
    base_pad: Path,
    db_url: str,
    participant_ids: list | set | pd.Series | None = None,
) -> pd.DataFrame:
    """Bereken aantallen en conversies in de gebruikersreis."""
    try:
        df_p = load_participants(db_url)
        df_c = load_completions(db_url)
        df_o = get_user_purchases(base_pad, db_url)
        df_long = laad_longitudinale_data(base_pad, db_url)
        
        if participant_ids is not None:
            pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
            if not df_p.empty and 'id' in df_p.columns:
                df_p = df_p[pd.to_numeric(df_p['id'], errors='coerce').isin(pids)]
            if not df_c.empty and 'participant_id' in df_c.columns:
                df_c = df_c[pd.to_numeric(df_c['participant_id'], errors='coerce').isin(pids)]
            if not df_o.empty and 'participant_id' in df_o.columns:
                df_o = df_o[pd.to_numeric(df_o['participant_id'], errors='coerce').isin(pids)]
            if not df_long.empty and 'participant_id' in df_long.columns:
                df_long = df_long[pd.to_numeric(df_long['participant_id'], errors='coerce').isin(pids)]

        n_reg = df_p['id'].nunique() if not df_p.empty and 'id' in df_p.columns else 0
        n_comp = df_c['participant_id'].nunique() if not df_c.empty and 'participant_id' in df_c.columns else 0
        geldige_statussen = {'paid', 'completed', 'complete', 'success', 'succeeded'}
        if not df_o.empty and 'participant_id' in df_o.columns:
            status = df_o.get('status', pd.Series('', index=df_o.index)).astype(str).str.lower()
            df_o = df_o[status.isin(geldige_statussen)]
        n_buy = df_o['participant_id'].nunique() if not df_o.empty and 'participant_id' in df_o.columns else 0
        n_repeat_count = int((df_long.groupby('participant_id')['completion_id'].nunique() >= 2).sum()) if not df_long.empty else 0

        journey = pd.DataFrame([
            ('Geregistreerd', n_reg),
            ('Vragenlijst ingevuld', n_comp),
            ('Geldige aankoop gedaan', n_buy),
            ('Herhaalmeting gedaan', n_repeat_count),
        ], columns=['Stap', 'Gebruikers'])
        journey['% van registratie'] = np.where(
            n_reg > 0, journey['Gebruikers'] / n_reg * 100, np.nan,
        ).round(1)
        vorige = journey['Gebruikers'].shift(1)
        journey['Conversie vanaf vorige stap (%)'] = np.where(
            vorige > 0, journey['Gebruikers'] / vorige * 100, np.nan,
        ).round(1)
        journey['Afname vanaf vorige stap'] = (vorige - journey['Gebruikers']).fillna(0).astype(int)
        return journey
    except Exception:
        return pd.DataFrame(columns=[
            'Stap', 'Gebruikers', '% van registratie',
            'Conversie vanaf vorige stap (%)', 'Afname vanaf vorige stap',
        ])


def maak_engagement_funnel_plot(base_pad: Path, db_url: str, lang: str = 'nl', participant_ids: list | set | pd.Series | None = None) -> go.Figure:
    """Visualiseert de funnel van registratie naar herhaalmeting."""
    try:
        journey = bereken_engagement_journey(base_pad, db_url, participant_ids)
        if journey.empty:
            return _empty_store_plot(tr("Geen engagementdata beschikbaar.", lang))
        stages = [tr(value, lang) for value in journey['Stap']]
        values = journey['Gebruikers'].tolist()
        fig = go.Figure(go.Funnel(
            y = stages, x = values,
            textinfo = "value+percent initial",
            marker = {"color": ["#1F6FBF", "#3498DB", "#2ECC71", "#27AE60"]}
        ))
        fig.update_layout(title_text=tr("Engagement Funnel: conversie naar actie", lang), height=450)
        return fig
    except Exception:
        return _empty_store_plot(tr("Fout bij laden van engagement funnel.", lang))

def maak_business_value_plot(df: pd.DataFrame, lang: str = 'nl') -> go.Figure:
    """Toont correlatie tussen WAI en productiviteitsschatting."""
    if df.empty or 'rec_asr_wai_score' not in df.columns:
         return _empty_store_plot(tr("Geen WAI data beschikbaar.", lang))

    df2 = df.dropna(subset=['rec_asr_wai_score']).copy()
    df2['WAI'] = pd.to_numeric(df2['rec_asr_wai_score'], errors='coerce')
    df2 = df2.dropna(subset=['WAI'])
    
    if df2.empty:
        return _empty_store_plot(tr("Geen WAI data beschikbaar.", lang))

    # Wetenschappelijke aanname: 1 punt stijging in WAI correleert met ~3% productiviteitswinst
    df2['Productiviteit (%)'] = (70 + df2['WAI'] * 3).clip(0, 100)
    
    fig = px.scatter(df2, x='WAI', y='Productiviteit (%)', trendline="ols" if HAS_STATSMODELS else None,
                     title=tr("Business Case: Werkvermogen (WAI) vs Geschatte Productiviteit", lang),
                     labels={'WAI': 'Work Ability Index (0-10)', 'Productiviteit (%)': tr('Productiviteit (%)', lang)},
                     color_discrete_sequence=[HOOFD_KLEUR], opacity=0.5)
    fig.update_layout(template='plotly_white')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4 ANALYTISCHE VISUALISATIES VOOR DE VRAGEN
# ══════════════════════════════════════════════════════════════════════════════

def maak_vragenlijst_herhalingen_vs_scores(base_pad: Path, db_url: str, lang: str = 'nl') -> tuple:
    """Backward-compatible wrapper – gebruikt analyse_invulfrequentie_vs_scores."""
    from analyses import analyse_invulfrequentie_vs_scores
    df_plot, samenvatting, meta = analyse_invulfrequentie_vs_scores(
        base_pad, db_url, score_kolom='rec_ls_lifestyle_score'
    )
    if df_plot.empty:
        return _empty_store_plot(tr("Onvoldoende data.", lang)), None

    score_kolom = meta.get('score_kolom', 'rec_ls_lifestyle_score')
    fig = px.box(
        df_plot, x='n_categorie', y=score_kolom, points='outliers',
        title=tr('Invulfrequentie vs Huidige Leefstijlscore', lang),
        labels={
            score_kolom: tr('Leefstijlscore', lang),
            'n_categorie': tr('Aantal verschillende vragenlijsten ingevuld', lang),
        },
        category_orders={'n_categorie': ['1x', '2x', '3x', '4x', '5x', '6+x']},
    )
    fig.update_layout(template='plotly_white')

    samenvatting_dict = {}
    if not samenvatting.empty:
        for _, row in samenvatting.iterrows():
            samenvatting_dict[f"{row['Invulfrequentie']} - gem. score"] = row.get(f'gem_{score_kolom}')
        if 'verschil_1x_vs_6plus' in meta:
            samenvatting_dict['Verschil (6+x vs 1x)'] = meta['verschil_1x_vs_6plus']
    return fig, samenvatting_dict


def maak_invulfrequentie_vs_scores_plot(
    base_pad: Path,
    db_url: str,
    score_kolom: str = 'rec_ls_lifestyle_score',
    score_label: str = 'Leefstijlscore',
    lang: str = 'nl',
) -> tuple:
    """Boxplot + gemiddelde-balken voor scores per invulfrequentie (1x, 2x, ...)."""
    from analyses import analyse_invulfrequentie_vs_scores

    df_plot, samenvatting, meta = analyse_invulfrequentie_vs_scores(base_pad, db_url, score_kolom)
    if df_plot.empty or samenvatting.empty:
        return _empty_store_plot(tr("Onvoldoende data.", lang)), _empty_store_plot(tr("Onvoldoende data.", lang)), pd.DataFrame(), df_plot

    volgorde = ['1x', '2x', '3x', '4x', '5x', '6+x']
    fig_box = px.box(
        df_plot, x='n_categorie', y=score_kolom, points='outliers',
        title=tr('Scoreverdeling per invulfrequentie', lang),
        labels={'n_categorie': tr('Aantal vragenlijsten', lang), score_kolom: score_label},
        category_orders={'n_categorie': volgorde},
    )
    fig_box.update_layout(template='plotly_white')

    plot_agg = samenvatting.copy()
    fig_bar = px.bar(
        plot_agg,
        x='Invulfrequentie',
        y=f'gem_{score_kolom}',
        error_y=f'std_{score_kolom}',
        text='n_deelnemers',
        title=tr('Gemiddelde score per invulfrequentie', lang),
        labels={
            'Invulfrequentie': tr('Invulfrequentie', lang),
            f'gem_{score_kolom}': score_label,
            'n_deelnemers': 'n',
        },
        category_orders={'Invulfrequentie': volgorde},
        color_discrete_sequence=[HOOFD_KLEUR],
    )
    fig_bar.update_traces(
        texttemplate='n=%{text}',
        textposition='outside',
    )
    fig_bar.update_layout(template='plotly_white', showlegend=False)

    display_cols = ['Invulfrequentie', 'n_deelnemers'] + [
        c for c in samenvatting.columns if c.startswith('gem_')
    ]
    return fig_box, fig_bar, samenvatting[display_cols], df_plot


def maak_herhaalde_vragenlijst_scoreverandering_plot(
    base_pad: Path,
    db_url: str,
    questionnaire_id: int,
    score_slug: str,
    lang: str = 'nl',
) -> tuple:
    """
    Visualiseer scoreverandering bij herhaald invullen van dezelfde vragenlijst.

    Returns: (fig_verandering, fig_traject, samenvatting, traject, df_change, meta)
    """
    from analyses import analyse_herhaalde_vragenlijst_scoreverandering

    samenvatting, traject, df_change, meta = analyse_herhaalde_vragenlijst_scoreverandering(
        base_pad, db_url, questionnaire_id, score_slug
    )
    if samenvatting.empty:
        empty = _empty_store_plot(tr("Onvoldoende data voor herhaalanalyse.", lang))
        return empty, empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta

    q_naam = meta.get('questionnaire_naam', f'Vragenlijst {questionnaire_id}')
    score_label = score_slug.replace('rec_', '').replace('_', ' ').title()
    volgorde = ['1x', '2x', '3x', '4x', '5x', '6+x']

    fig_verandering = px.bar(
        samenvatting,
        x='Invulfrequentie',
        y='gem_verandering',
        error_y='std_verandering',
        text='n_deelnemers',
        title=tr(
            'Gemiddelde scoreverandering (laatste − eerste) bij herhaald invullen: {naam}',
            lang,
        ).format(naam=q_naam),
        labels={
            'Invulfrequentie': tr('Aantal keer dezelfde vragenlijst ingevuld', lang),
            'gem_verandering': tr(f'Verandering {score_label}', lang),
            'n_deelnemers': 'n',
        },
        category_orders={'Invulfrequentie': volgorde},
        color_discrete_sequence=[HOOFD_KLEUR],
    )
    fig_verandering.update_traces(texttemplate='n=%{text}', textposition='outside')
    fig_verandering.add_hline(y=0, line_dash='dash', line_color='gray')
    fig_verandering.update_layout(template='plotly_white', showlegend=False)

    fig_traject = _empty_store_plot(tr("Geen trajectdata.", lang))
    if not traject.empty:
        fig_traject = px.line(
            traject,
            x='invul_label',
            y='gem_score',
            markers=True,
            text='n_deelnemers',
            title=tr('Scoreverloop per invulmoment: {naam}', lang).format(naam=q_naam),
            labels={
                'invul_label': tr('Invulmoment', lang),
                'gem_score': score_label,
                'n_deelnemers': 'n',
            },
            color_discrete_sequence=[HOOFD_KLEUR],
        )
        fig_traject.update_traces(
            texttemplate='n=%{text}',
            textposition='top center',
        )
        fig_traject.update_layout(template='plotly_white')

    return fig_verandering, fig_traject, samenvatting, traject, df_change, meta


def maak_account_dataflow_funnel_plot(base_pad: Path, db_url: str, lang: str = 'nl') -> tuple:
    """Funneldiagram van registratie naar scores, incl. naam- en leeftijdstap."""
    from analyses import analyse_account_dataflow_funnel

    funnel_df, meta = analyse_account_dataflow_funnel(base_pad, db_url)
    if funnel_df.empty:
        return _empty_store_plot(tr("Onvoldoende data voor funnel.", lang)), pd.DataFrame(), {}

    funnel_plot_df = funnel_df[funnel_df['stap_key'] != 'naam_parallel'] if 'stap_key' in funnel_df.columns else funnel_df

    funnel_colors = ['#1F6FBF', '#3498DB', '#5DADE2', '#2ECC71', '#27AE60', '#16A085', '#E87722']
    fig = go.Figure(go.Funnel(
        y=[tr(row['stap'], lang) if lang == 'en' else row['stap'] for _, row in funnel_plot_df.iterrows()],
        x=funnel_plot_df['aantal'],
        textinfo='value+percent initial',
        customdata=funnel_plot_df[['pct_afgevallen_sinds_vorige', 'afgevallen_sinds_vorige']].values,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Aantal: %{x}<br>'
            '%{percentInitial} van start<br>'
            'Afgevallen sinds vorige stap: %{customdata[1]} (%{customdata[0]:.1f}%)<extra></extra>'
        ),
        marker={'color': funnel_colors[:len(funnel_plot_df)]},
    ))
    fig.update_layout(
        title=tr('Dataflow: van registratie naar scores', lang),
        height=550,
        template='plotly_white',
    )
    return fig, funnel_df, meta


def _laad_interaction_views(db_url: str | None = None) -> pd.DataFrame:
    """Laad content-views uit interactions (position-based kolommapping)."""
    effective_db_url = db_url or DB_URL
    if not _has_valid_db_url(effective_db_url):
        return pd.DataFrame(columns=['user_id', 'created_at'])

    try:
        engine = create_engine(effective_db_url)
        df_inter_raw = pd.read_sql("SELECT * FROM interactions", engine)
        if df_inter_raw.empty or len(df_inter_raw.columns) < 7:
            return pd.DataFrame(columns=['user_id', 'created_at'])

        df_inter = df_inter_raw.iloc[:, :7].copy()
        df_inter.columns = ['id', 'interactable_type', 'interactable_id', 'user_id', 'type', 'created_at', 'updated_at']
        df_inter = df_inter[
            (df_inter['interactable_type'].astype(str).str.strip() == 'content') &
            (df_inter['type'].astype(str).str.strip() == 'view')
        ].copy()
        df_inter['user_id'] = pd.to_numeric(df_inter['user_id'], errors='coerce')
        df_inter['created_at'] = pd.to_datetime(df_inter['created_at'], errors='coerce')
        df_inter = df_inter.dropna(subset=['user_id'])

        # Interne Smart Health (test)accounts uit de views filteren
        return filter_interne_accounts(df_inter, db_url=effective_db_url)
    except Exception as e:
        logger.warning(f"Kon interactions niet laden: {e}")
        return pd.DataFrame(columns=['user_id', 'created_at'])


def _laad_user_challenges_per_store(db_url: str | None = None) -> pd.DataFrame:
    """Laad challenge-deelname per store (position-based kolommapping)."""
    effective_db_url = db_url or DB_URL
    if not _has_valid_db_url(effective_db_url):
        return pd.DataFrame(columns=['store_id', 'user_ref', 'status', 'created_at'])

    try:
        engine = create_engine(effective_db_url)
        df_raw = pd.read_sql("SELECT * FROM user_challenges", engine)
        if df_raw.empty or len(df_raw.columns) < 5:
            return pd.DataFrame(columns=['store_id', 'user_ref', 'status', 'created_at'])

        df = df_raw.iloc[:, :9].copy()
        df.columns = [
            'id', 'user_ref', 'challenge_id', 'store_id', 'status',
            'weeks_completed', 'challenge_type', 'created_at', 'updated_at',
        ][:len(df.columns)]
        df['store_id'] = pd.to_numeric(df['store_id'], errors='coerce')
        df['user_ref'] = pd.to_numeric(df['user_ref'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df = df.dropna(subset=['store_id'])

        # Interne Smart Health (test)accounts uit challenge-deelname filteren
        # (user_ref is hier de app-user_id).
        return filter_interne_accounts(df, db_url=effective_db_url, user_id_col='user_ref')
    except Exception as e:
        logger.warning(f"Kon user_challenges niet laden: {e}")
        return pd.DataFrame(columns=['store_id', 'user_ref', 'status', 'created_at'])


def _laad_participant_user_bridge(db_url: str | None = None) -> pd.DataFrame:
    """Koppel participant_id aan user_id via my_clic bridge."""
    effective_db_url = db_url or DB_URL
    if not _has_valid_db_url(effective_db_url):
        return pd.DataFrame(columns=['participant_id', 'user_id'])

    try:
        engine = create_engine(effective_db_url)
        bridge = pd.read_sql(
            text("""
                SELECT p.id AS participant_id, m.user_id
                FROM public.participants p
                JOIN smart_health.my_clic_participants m
                    ON m.qe_participant_id = p.public_id
                WHERE m.user_id IS NOT NULL
            """),
            engine,
        )
        bridge['participant_id'] = pd.to_numeric(bridge['participant_id'], errors='coerce')
        bridge['user_id'] = pd.to_numeric(bridge['user_id'], errors='coerce')
        return bridge.dropna(subset=['participant_id', 'user_id']).drop_duplicates('participant_id')
    except Exception as e:
        logger.warning(f"Kon participant-user bridge niet laden: {e}")
        return pd.DataFrame(columns=['participant_id', 'user_id'])


ENGAGEMENT_COMPONENT_DEFAULTS = {
    'vragenlijst': 0.30,
    'herhaalmeting': 0.20,
    'breedte': 0.15,
    'artikelen': 0.15,
    'challenges': 0.10,
    'aankopen': 0.10,
}


def _laad_app_logins(db_url: str | None = None) -> pd.DataFrame:
    """
    Laad login-events uit smart_health.activity_log (benamde kolommen) en
    filter interne Smart Health-testaccounts eruit.
    """
    effective_db_url = db_url or DB_URL
    if not _has_valid_db_url(effective_db_url):
        return pd.DataFrame(columns=['user_id', 'created_at'])

    try:
        engine = create_engine(effective_db_url)
        df_al = pd.read_sql(
            "SELECT causer_id AS user_id, event, created_at "
            "FROM smart_health.activity_log WHERE event = 'users.logged_in'",
            engine,
        )
        if df_al.empty:
            return pd.DataFrame(columns=['user_id', 'created_at'])
        df_al['user_id'] = pd.to_numeric(df_al['user_id'], errors='coerce')
        df_al['created_at'] = pd.to_datetime(df_al['created_at'], errors='coerce')
        df_al = df_al.dropna(subset=['user_id'])

        # Interne Smart Health (test)accounts uit de logins filteren
        return filter_interne_accounts(df_al[['user_id', 'created_at']], db_url=effective_db_url)
    except Exception as e:
        logger.warning(f"Kon activity_log niet laden: {e}")
        return pd.DataFrame(columns=['user_id', 'created_at'])


def bereken_gebruiker_engagement_scores(
    base_pad: Path,
    db_url: str | None = None,
    participant_ids: list | set | pd.Series | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Bereken per deelnemer engagement-componenten (0–100) over koppelbare databronnen.
    """
    base_pad = Path(base_pad)
    effective_db_url = db_url or DB_URL

    links = _laad_opdrachtgever_links(effective_db_url)
    if links.empty:
        return pd.DataFrame()

    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        links = links[pd.to_numeric(links['participant_id'], errors='coerce').isin(pids)]
        if links.empty:
            return pd.DataFrame()

    df_scores = load_my_clic_participants_expanded(effective_db_url)
    if df_scores.empty:
        return pd.DataFrame()

    df = links[['participant_id', 'store_id', 'store_name', 'user_id']].copy()
    df['participant_id'] = pd.to_numeric(df['participant_id'], errors='coerce')
    df = df.drop_duplicates(subset='participant_id', keep='first')

    score_cols = ['rec_ls_lifestyle_score']
    score_cols = [c for c in score_cols if c in df_scores.columns]
    if score_cols:
        df_scores['participant_id'] = pd.to_numeric(df_scores['participant_id'], errors='coerce')
        df = df.merge(
            df_scores[['participant_id'] + score_cols].drop_duplicates('participant_id'),
            on='participant_id',
            how='left',
        )

    if 'user_id' not in df.columns or df['user_id'].isna().all():
        bridge = _laad_participant_user_bridge(effective_db_url)
        if not bridge.empty:
            df = df.drop(columns=['user_id'], errors='ignore').merge(bridge, on='participant_id', how='left')

    known_questionnaire_ids = {1, 2, 3, 4, 5, 6, 7, 8}
    df_comp = load_completions(effective_db_url)
    if not df_comp.empty:
        df_comp = df_comp.copy()
        df_comp['created_at'] = pd.to_datetime(df_comp['created_at'], errors='coerce')
        if start_date is not None:
            df_comp = df_comp[df_comp['created_at'] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df_comp = df_comp[df_comp['created_at'] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
        df_comp['participant_id'] = pd.to_numeric(df_comp['participant_id'], errors='coerce')
        df_comp['questionnaire_id'] = pd.to_numeric(df_comp['questionnaire_id'], errors='coerce')
        df_comp = df_comp[df_comp['questionnaire_id'].isin(known_questionnaire_ids)]
        comp_stats = df_comp.groupby('participant_id').agg(
            n_completions=('completion_id', 'nunique') if 'completion_id' in df_comp.columns else ('questionnaire_id', 'count'),
            n_vragenlijsten=('questionnaire_id', 'nunique'),
        ).reset_index()
        df = df.merge(comp_stats, on='participant_id', how='left')
    else:
        df['n_completions'] = 0
        df['n_vragenlijsten'] = 0

    df['n_completions'] = pd.to_numeric(df.get('n_completions'), errors='coerce').fillna(0)
    df['n_vragenlijsten'] = pd.to_numeric(df.get('n_vragenlijsten'), errors='coerce').fillna(0)

    has_score = pd.to_numeric(df.get('rec_ls_lifestyle_score'), errors='coerce').notna()
    df['score_vragenlijst'] = np.where(has_score, 100.0, np.where(df['n_completions'] > 0, 50.0, 0.0))
    df['score_herhaalmeting'] = np.where(df['n_completions'] >= 2, 100.0, 0.0)
    df['score_breedte'] = (df['n_vragenlijsten'].clip(0, 8) / 8.0 * 100.0).round(1)

    views = _laad_interaction_views(effective_db_url)
    if not views.empty:
        if start_date is not None:
            views = views[views['created_at'] >= pd.Timestamp(start_date)]
        if end_date is not None:
            views = views[views['created_at'] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if not views.empty and 'user_id' in df.columns:
        viewers = set(views['user_id'].dropna().astype(int))
        df['user_id_num'] = pd.to_numeric(df['user_id'], errors='coerce')
        df['score_artikelen'] = np.where(df['user_id_num'].isin(viewers), 100.0, 0.0)
        views_per_user = views.groupby('user_id').size()
        df['n_artikelviews'] = df['user_id_num'].map(views_per_user).fillna(0)
    else:
        df['score_artikelen'] = 0.0
        df['n_artikelviews'] = 0

    purchases = get_user_purchases(base_pad, effective_db_url)
    if not purchases.empty and 'created_at' in purchases.columns:
        purchases = purchases.copy()
        purchases['created_at'] = pd.to_datetime(purchases['created_at'], errors='coerce')
        if start_date is not None:
            purchases = purchases[purchases['created_at'] >= pd.Timestamp(start_date)]
        if end_date is not None:
            purchases = purchases[purchases['created_at'] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if not purchases.empty and 'participant_id' in purchases.columns:
        _pp = purchases.copy()
        _pp['_pid'] = pd.to_numeric(_pp['participant_id'], errors='coerce')
        _status = _pp.get('status', pd.Series('', index=_pp.index)).astype(str).str.lower()
        _pp_betaald = _pp[~_status.isin(['canceled', 'cancelled', 'expired', 'failed', 'payment_pending'])]
        buyers = set(_pp_betaald['_pid'].dropna().astype(int))
        df['score_aankopen'] = np.where(df['participant_id'].isin(buyers), 100.0, 0.0)
        # Herhaalaankopen: >=2 geldige bestellingen.
        _orders_pp = _pp_betaald.groupby('_pid').size()
        df['_f_koper'] = df['participant_id'].isin(buyers)
        df['_n_betaalde_orders'] = df['participant_id'].map(_orders_pp).fillna(0).astype(float)
        df['_f_herhaalkoper'] = df['_f_koper'] & (df['_n_betaalde_orders'] >= 2)
    else:
        df['score_aankopen'] = 0.0
        df['_f_koper'] = False
        df['_n_betaalde_orders'] = 0.0
        df['_f_herhaalkoper'] = False

    # Challenges: store-level deelnemersratio en voltooiingsgraad, toegepast als store-kenmerk
    uc = _laad_user_challenges_per_store(effective_db_url)
    if not uc.empty and 'created_at' in uc.columns:
        uc = uc.copy()
        uc['created_at'] = pd.to_datetime(uc['created_at'], errors='coerce')
        if start_date is not None:
            uc = uc[uc['created_at'] >= pd.Timestamp(start_date)]
        if end_date is not None:
            uc = uc[uc['created_at'] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if not uc.empty:
        _status_lc = uc['status'].astype(str).str.lower()
        challenge_per_store = (
            uc.dropna(subset=['user_ref']).groupby('store_id')['user_ref']
            .nunique()
            .reset_index(name='n_challenge_deelnemers')
        )
        # Voltooiingsgraad: aandeel afgeronde van alle uitgekomen deelnames.
        # Uitgekomen = completed / expired / unsubscribed; signed_up, started en
        # disabled zijn nog lopend of administratief, dus niet meegenomen.
        uc_eind = uc[_status_lc.isin(['completed', 'expired', 'unsubscribed'])]
        term = (
            uc_eind.groupby('store_id')
            .agg(
                n_challenge_terminal=('status', 'size'),
                n_challenge_afgerond=('status',
                                      lambda s: int((s.astype(str).str.lower() == 'completed').sum())),
            )
            .reset_index()
        )
        term['pct_challenges_afgerond'] = np.where(
            term['n_challenge_terminal'] > 0,
            term['n_challenge_afgerond'] / term['n_challenge_terminal'] * 100,
            np.nan,
        )
        store_counts = df.groupby('store_id')['participant_id'].nunique().reset_index(name='n_store_users')
        challenge_per_store = (
            challenge_per_store
            .merge(term, on='store_id', how='outer')
            .merge(store_counts, on='store_id', how='left')
        )
        challenge_per_store['n_challenge_deelnemers'] = (
            pd.to_numeric(challenge_per_store['n_challenge_deelnemers'], errors='coerce')
            .fillna(0)
            .astype(int)
        )
        challenge_per_store['pct_challenge_store'] = (
            challenge_per_store['n_challenge_deelnemers'] / challenge_per_store['n_store_users'].clip(lower=1) * 100
        ).clip(0, 100)
        df = df.merge(
            challenge_per_store[['store_id', 'pct_challenge_store', 'n_challenge_deelnemers',
                                 'pct_challenges_afgerond', 'n_challenge_afgerond']],
            on='store_id',
            how='left',
        )
        df['score_challenges'] = df['pct_challenge_store'].fillna(0.0)
    else:
        for _c in ['pct_challenges_afgerond', 'n_challenge_afgerond']:
            df[_c] = np.nan
        df['score_challenges'] = 0.0

    # App-activiteit: % deelnemers met minstens één login in de laatste 90 dagen,
    # gemeten t.o.v. de meest recente datum in de log zodat historische dumps
    # op dezelfde schaal blijven.
    try:
        logins = _laad_app_logins(effective_db_url)
    except Exception:
        logins = pd.DataFrame(columns=['user_id', 'created_at'])
    if 'user_id' not in df.columns or df['user_id'].isna().all():
        bridge = _laad_participant_user_bridge(effective_db_url)
        if not bridge.empty:
            df = df.drop(columns=['user_id'], errors='ignore').merge(
                bridge[['participant_id', 'user_id']], on='participant_id', how='left')
    if not logins.empty and end_date is not None:
        logins = logins[pd.to_datetime(logins['created_at'], errors='coerce') < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    df['_f_app_actief'] = False
    if not logins.empty and 'user_id' in df.columns:
        _uids = pd.to_numeric(df['user_id'], errors='coerce')
        _ref_datum = pd.to_datetime(logins['created_at'], errors='coerce').max()
        _venster_start = _ref_datum - pd.Timedelta(days=90)
        _recente_gebruikers = set(
            pd.to_numeric(logins.loc[pd.to_datetime(logins['created_at'], errors='coerce') >= _venster_start,
                                     'user_id'], errors='coerce').dropna().astype(int)
        )
        df['_f_app_actief'] = _uids.notna() & _uids.isin(_recente_gebruikers)

    return df


def bereken_engagement_per_opdrachtgever(
    base_pad: Path,
    db_url: str | None = None,
    gewichten: dict | None = None,
    participant_ids: list | set | pd.Series | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Aggregeer engagement naar opdrachtgever-niveau met gewogen samengestelde score.
    """
    gewichten = gewichten or ENGAGEMENT_COMPONENT_DEFAULTS.copy()
    totaal_gewicht = sum(gewichten.values()) or 1.0
    gewichten = {k: v / totaal_gewicht for k, v in gewichten.items()}

    score_args = {'participant_ids': participant_ids}
    if start_date is not None or end_date is not None:
        score_args.update(start_date=start_date, end_date=end_date)
    df_users = bereken_gebruiker_engagement_scores(base_pad, db_url, **score_args)
    if df_users.empty:
        return pd.DataFrame(), 0.0, pd.DataFrame()

    component_map = {
        'vragenlijst': 'score_vragenlijst',
        'herhaalmeting': 'score_herhaalmeting',
        'breedte': 'score_breedte',
        'artikelen': 'score_artikelen',
        'challenges': 'score_challenges',
        'aankopen': 'score_aankopen',
    }

    for key, col in component_map.items():
        if col in df_users.columns:
            df_users[f'w_{key}'] = df_users[col] * gewichten.get(key, 0.0)

    # Fracties per persoon voor interpreteerbare percentages/aantallen in de uitsplitsing
    _nc = pd.to_numeric(df_users.get('n_completions'), errors='coerce').fillna(0)
    _sa = pd.to_numeric(df_users.get('score_artikelen'), errors='coerce').fillna(0)
    _sk = pd.to_numeric(df_users.get('score_aankopen'), errors='coerce').fillna(0)
    df_users['_f_vragenlijst'] = (_nc >= 1).astype(float)
    df_users['_f_herhaalmeting'] = (_nc >= 2).astype(float)
    df_users['_f_artikelen'] = (_sa > 0).astype(float)
    df_users['_f_aankopen'] = (_sk > 0).astype(float)

    weight_cols = [f'w_{k}' for k in component_map if f'w_{k}' in df_users.columns]
    df_users['engagement_score'] = df_users[weight_cols].sum(axis=1).round(1)

    agg_spec = {
        'engagement_score': ('engagement_score', 'mean'),
        'n_deelnemers': ('participant_id', 'nunique'),
    }
    for key, col in component_map.items():
        if col in df_users.columns:
            agg_spec[f'comp_{key}'] = (col, 'mean')

    # Interpreteerbare uitsplitsing: percentage deelnemers + absolute aantallen
    agg_spec['pct_vragenlijst'] = ('_f_vragenlijst', 'mean')
    agg_spec['pct_herhaalmeting'] = ('_f_herhaalmeting', 'mean')
    agg_spec['pct_artikelen'] = ('_f_artikelen', 'mean')
    agg_spec['pct_aankopen'] = ('_f_aankopen', 'mean')
    agg_spec['n_vragenlijst'] = ('_f_vragenlijst', 'sum')
    agg_spec['n_herhaalmeting'] = ('_f_herhaalmeting', 'sum')
    agg_spec['n_artikelen'] = ('_f_artikelen', 'sum')
    agg_spec['n_aankopen'] = ('_f_aankopen', 'sum')
    agg_spec['gem_invullingen'] = ('n_completions', 'mean')
    agg_spec['gem_artikelviews'] = ('n_artikelviews', 'mean')
    agg_spec['gem_aankopen'] = ('_n_betaalde_orders', 'mean')
    if 'n_vragenlijsten' in df_users.columns:
        agg_spec['gem_breedte'] = ('n_vragenlijsten', 'mean')
    if 'score_challenges' in df_users.columns:
        agg_spec['pct_challenges'] = ('score_challenges', 'mean')
    if 'n_challenge_deelnemers' in df_users.columns:
        agg_spec['n_challenges'] = ('n_challenge_deelnemers', 'max')

    # Informatieve maten buiten de compositie om (herhaalaankopen,
    # challenge-voltooiing en recente app-activiteit)
    agg_spec['_r_herhaalkoper'] = ('_f_herhaalkoper', 'mean')
    agg_spec['_r_koper'] = ('_f_aankopen', 'mean')
    agg_spec['_r_app'] = ('_f_app_actief', 'mean')
    agg_spec['n_app_actief'] = ('_f_app_actief', 'sum')
    if 'pct_challenges_afgerond' in df_users.columns:
        agg_spec['pct_challenges_afgerond'] = ('pct_challenges_afgerond', 'mean')
    if 'n_challenge_afgerond' in df_users.columns:
        agg_spec['n_challenges_afgerond'] = ('n_challenge_afgerond', 'max')

    agg = (
        df_users.groupby(['store_id', 'store_name'], as_index=False)
        .agg(**agg_spec)
    )
    agg['engagement_score'] = pd.to_numeric(agg['engagement_score'], errors='coerce').round(1)
    for col in [c for c in agg.columns if c.startswith('comp_')]:
        agg[col] = pd.to_numeric(agg[col], errors='coerce').round(1)
    for col in ['pct_vragenlijst', 'pct_herhaalmeting', 'pct_artikelen', 'pct_aankopen']:
        if col in agg.columns:
            agg[col] = (pd.to_numeric(agg[col], errors='coerce') * 100).clip(0, 100).round(1)
    if 'pct_challenges' in agg.columns:
        agg['pct_challenges'] = pd.to_numeric(agg['pct_challenges'], errors='coerce').clip(0, 100).round(1)
    if 'gem_breedte' in agg.columns:
        agg['gem_breedte'] = pd.to_numeric(agg['gem_breedte'], errors='coerce').round(2)
    for col in ['gem_invullingen', 'gem_artikelviews', 'gem_aankopen']:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors='coerce').fillna(0).round(2)
    for col in ['n_vragenlijst', 'n_herhaalmeting', 'n_artikelen', 'n_aankopen', 'n_challenges']:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors='coerce').fillna(0).astype(int)
    if 'n_app_actief' in agg.columns:
        agg['n_app_actief'] = pd.to_numeric(agg['n_app_actief'], errors='coerce').fillna(0).astype(int)
    if 'n_challenges_afgerond' in agg.columns:
        agg['n_challenges_afgerond'] = pd.to_numeric(agg['n_challenges_afgerond'], errors='coerce').fillna(0).astype(int)
    if {'_r_koper', '_r_herhaalkoper'} <= set(agg.columns):
        _denom = pd.to_numeric(agg['_r_koper'], errors='coerce').clip(lower=0)
        _numer = pd.to_numeric(agg['_r_herhaalkoper'], errors='coerce')
        _waarde = (_numer / _denom.where(_denom > 0)) * 100
        agg['pct_herhaalkopers_van_kopers'] = pd.to_numeric(_waarde, errors='coerce').clip(0, 100).round(1)
        agg.drop(columns=['_r_koper', '_r_herhaalkoper'], inplace=True)
    if '_r_app' in agg.columns:
        agg['pct_app_actief_90d'] = (
            pd.to_numeric(agg['_r_app'], errors='coerce') * 100
        ).clip(0, 100).round(1)
        agg.drop(columns=['_r_app'], inplace=True)
    if 'pct_challenges_afgerond' in agg.columns:
        agg['pct_challenges_afgerond'] = pd.to_numeric(
            agg['pct_challenges_afgerond'], errors='coerce'
        ).clip(0, 100).round(1)

    benchmark = _weighted_average(agg['engagement_score'], agg['n_deelnemers'])
    if pd.isna(benchmark):
        benchmark = 0.0

    koppeling = controleer_gebruikers_koppeling(base_pad, db_url)
    return agg.sort_values('engagement_score', ascending=False), float(benchmark), koppeling


def controleer_gebruikers_koppeling(base_pad: Path, db_url: str | None = None) -> pd.DataFrame:
    """Controleer consistentie van gebruikerskoppeling over databronnen."""
    effective_db_url = db_url or DB_URL
    rows = []

    links = _laad_opdrachtgever_links(effective_db_url)
    n_participants_linked = int(links['participant_id'].nunique()) if not links.empty else 0
    rows.append({'Databron': 'Opdrachtgever-koppeling', 'Beschikbaar': n_participants_linked, 'Koppelbaar': n_participants_linked})

    bridge = _laad_participant_user_bridge(effective_db_url)
    n_bridge = int(bridge['participant_id'].nunique()) if not bridge.empty else 0
    rows.append({'Databron': 'Participant → user_id (my_clic)', 'Beschikbaar': n_bridge, 'Koppelbaar': n_bridge})

    df_scores = load_my_clic_participants_expanded(effective_db_url)
    n_scores = int(df_scores['participant_id'].nunique()) if not df_scores.empty and 'participant_id' in df_scores.columns else 0
    if not links.empty and n_scores:
        score_ids = set(pd.to_numeric(df_scores['participant_id'], errors='coerce').dropna().astype(int))
        link_ids = set(pd.to_numeric(links['participant_id'], errors='coerce').dropna().astype(int))
        koppelbaar = len(score_ids & link_ids)
        rows.append({'Databron': 'Vragenlijstscores', 'Beschikbaar': n_scores, 'Koppelbaar': koppelbaar})
    else:
        rows.append({'Databron': 'Vragenlijstscores', 'Beschikbaar': n_scores, 'Koppelbaar': 0})

    views = _laad_interaction_views(effective_db_url)
    n_views = int(views['user_id'].nunique()) if not views.empty else 0
    if not bridge.empty and n_views:
        bridge_users = set(bridge['user_id'].dropna().astype(int))
        koppelbaar = len(set(views['user_id'].dropna().astype(int)) & bridge_users)
        rows.append({'Databron': 'Artikelviews (interactions)', 'Beschikbaar': n_views, 'Koppelbaar': koppelbaar})
    else:
        rows.append({'Databron': 'Artikelviews (interactions)', 'Beschikbaar': n_views, 'Koppelbaar': 0})

    uc = _laad_user_challenges_per_store(effective_db_url)
    n_challenge_stores = int(uc['store_id'].nunique()) if not uc.empty else 0
    rows.append({
        'Databron': 'Challenges (store-niveau)',
        'Beschikbaar': n_challenge_stores,
        'Koppelbaar': n_challenge_stores,
        'Opmerking': 'Geen betrouwbare user-level koppeling; metric op store-niveau',
    })

    koppeling = pd.DataFrame(rows)
    if 'Opmerking' not in koppeling.columns:
        koppeling['Opmerking'] = ''
    if n_participants_linked:
        _koppel_num = pd.to_numeric(koppeling['Koppelbaar'], errors='coerce')
        _koppel_den = pd.to_numeric(koppeling['Beschikbaar'], errors='coerce').replace(0, np.nan)
        koppeling['Koppelingsgraad (%)'] = (_koppel_num / _koppel_den * 100).round(1)
    return koppeling


def maak_engagement_opdrachtgever_ranking_plot(
    agg: pd.DataFrame,
    benchmark: float,
    min_deelnemers: int = 1,
    toon_klein: bool = True,
    lang: str = 'nl',
) -> go.Figure:
    """Horizontale ranking van engagement scores met benchmarklijn."""
    if agg.empty:
        return _empty_store_plot(tr("Geen engagementdata beschikbaar.", lang))

    df = agg.copy()
    if not toon_klein:
        df = df[df['n_deelnemers'] >= min_deelnemers]
    if df.empty:
        return _empty_store_plot(tr("Geen opdrachtgevers boven de drempel.", lang))

    df = df.sort_values('engagement_score', ascending=True)
    kleuren = ['#2ECC71' if s >= benchmark else '#E74C3C' for s in df['engagement_score']]

    fig = go.Figure(go.Bar(
        x=df['engagement_score'],
        y=df['store_name'],
        orientation='h',
        marker_color=kleuren,
        text=df.apply(
            lambda r: f"{r['engagement_score']:.1f} (n={int(r['n_deelnemers'])})",
            axis=1,
        ),
        textposition='outside',
    ))
    fig.add_vline(
        x=benchmark, line_dash='dash', line_color='black',
        annotation_text=f'Benchmark ({benchmark:.1f})',
        annotation_position='top',
    )
    fig.update_layout(
        title=tr('Engagement score per opdrachtgever', lang),
        xaxis_title=tr('Engagement score (0-100)', lang),
        yaxis_title='',
        height=max(400, len(df) * 28),
        margin=dict(l=200, r=120),
        template='plotly_white',
    )
    return fig


def maak_engagement_breakdown_plot(agg: pd.DataFrame, store_name: str, lang: str = 'nl') -> go.Figure:
    """Gestapelde componenten voor één opdrachtgever."""
    if agg.empty or not store_name:
        return _empty_store_plot(tr("Geen data voor uitsplitsing.", lang))

    row = agg[agg['store_name'] == store_name]
    if row.empty:
        return _empty_store_plot(tr("Opdrachtgever niet gevonden.", lang))

    row = row.iloc[0]
    n_deelnemers = 0
    if 'n_deelnemers' in row.index:
        _n = pd.to_numeric(row['n_deelnemers'], errors='coerce')
        if pd.notna(_n):
            n_deelnemers = int(_n)

    pct_present = [
        c for c in ('pct_vragenlijst', 'pct_herhaalmeting', 'pct_artikelen', 'pct_challenges',
                    'pct_aankopen', 'pct_herhaalkopers_van_kopers', 'pct_challenges_afgerond',
                    'pct_app_actief_90d')
        if c in row.index
    ]

    if pct_present:
        # Nieuwe weergave: percentages deelnemers, aantallen in de tooltip
        items = [
            ('pct_vragenlijst', 'n_vragenlijst', tr('Vragenlijst ingevuld', lang)),
            ('pct_herhaalmeting', 'n_herhaalmeting', tr('Herhaald: ≥2 invullingen', lang)),
            ('pct_artikelen', 'n_artikelen', tr('Artikel gelezen', lang)),
            ('pct_challenges', 'n_challenges', tr('Challenge deelgenomen', lang)),
            ('pct_challenges_afgerond', 'n_challenges_afgerond',
             tr('Challenge afgerond (% van uitgekomen)', lang)),
            ('pct_aankopen', 'n_aankopen', tr('Aankoop gedaan', lang)),
            ('pct_herhaalkopers_van_kopers', None, tr('Herhaalkoper (% van kopers)', lang)),
            ('pct_app_actief_90d', 'n_app_actief', tr('App actief (laatste 90 dagen)', lang)),
        ]
        labels, values, hovers = [], [], []
        for pct_col, n_col, label in items:
            if pct_col not in row.index or pd.isna(row[pct_col]):
                continue
            waarde = float(row[pct_col])
            if n_col is not None and n_col in row.index and pd.notna(row[n_col]) and n_deelnemers > 0:
                hover = f"{float(row[n_col]):.0f}/{n_deelnemers} {tr('deelnemers', lang)} ({waarde:.1f}%)"
            else:
                hover = f"{waarde:.1f}%"
            labels.append(label)
            values.append(waarde)
            hovers.append(hover)

        breedte_noot = ''
        if 'gem_breedte' in row.index and pd.notna(row['gem_breedte']):
            breedte_noot = tr(
                'Gemiddeld {aantal:.1f} verschillende vragenlijsten per deelnemer', lang
            ).format(aantal=float(row['gem_breedte']))

        balk_kleuren = [
            '#1F6FBF', '#3498DB', '#5DADE2', '#85C1E9',
            '#2ECC71', '#27AE60', '#E67E22', '#F1C40F',
        ]
        balk_teksten = []
        for pct_col, n_col, _label in [
            item for item in items
            if item[0] in row.index and not pd.isna(row[item[0]])
        ]:
            waarde = float(row[pct_col])
            if n_col is not None and n_col in row.index and pd.notna(row[n_col]):
                balk_teksten.append(f'{waarde:.1f}% ({int(row[n_col])}/{n_deelnemers})')
            else:
                balk_teksten.append(f'{waarde:.1f}%')

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=balk_kleuren[:len(values)],
            text=balk_teksten,
            textposition='outside',
            customdata=hovers,
            hovertemplate='<b>%{y}</b><br>%{customdata}<extra></extra>',
        ))
        fig.update_layout(
            title=tr('Welke activiteiten doen gebruikers? {store}', lang).format(store=store_name),
            xaxis_title=tr('% van de deelnemers', lang),
            template='plotly_white',
            height=420,
            xaxis_range=[0, 108],
            margin=dict(r=70, t=60, b=55),
        )
        if breedte_noot:
            fig.add_annotation(
                text=breedte_noot, xref='paper', yref='paper',
                x=0, y=-0.24, xanchor='left', showarrow=False,
                font=dict(size=12),
            )
        return fig

    # Fallback: oude weergave op basis van componentscores
    comp_labels = {
        'comp_vragenlijst': tr('Vragenlijst', lang),
        'comp_herhaalmeting': tr('Herhaalmeting', lang),
        'comp_breedte': tr('Breedte (meerdere vragenlijsten)', lang),
        'comp_artikelen': tr('Artikelen', lang),
        'comp_challenges': tr('Challenges (store)', lang),
        'comp_aankopen': tr('Aankopen', lang),
    }
    values = []
    labels = []
    for col, label in comp_labels.items():
        if col in row.index and pd.notna(row[col]):
            values.append(float(row[col]))
            labels.append(label)

    if not values:
        return _empty_store_plot(tr("Geen componentdata.", lang))

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=['#1F6FBF', '#3498DB', '#5DADE2', '#85C1E9', '#2ECC71', '#E87722'][:len(values)],
        text=[f'{v:.1f}' for v in values],
        textposition='outside',
    ))
    fig.update_layout(
        title=tr('Engagement uitsplitsing: {store}', lang).format(store=store_name),
        yaxis_title=tr('Component score (0-100)', lang),
        template='plotly_white',
        height=420,
    )
    return fig


def bereken_engagement_trend(
    base_pad: Path,
    db_url: str | None = None,
    gewichten: dict | None = None,
    min_deelnemers: int = 1,
    participant_ids: list | set | pd.Series | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Maandelijkse engagement trend per opdrachtgever."""
    effective_db_url = db_url or DB_URL
    gewichten = gewichten or ENGAGEMENT_COMPONENT_DEFAULTS.copy()
    totaal_gewicht = sum(gewichten.values()) or 1.0
    gewichten = {k: v / totaal_gewicht for k, v in gewichten.items()}

    links = _laad_opdrachtgever_links(effective_db_url)
    df_comp = load_completions(effective_db_url)
    if links.empty or df_comp.empty:
        return pd.DataFrame()

    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        links = links[pd.to_numeric(links['participant_id'], errors='coerce').isin(pids)]
        df_comp = df_comp[pd.to_numeric(df_comp['participant_id'], errors='coerce').isin(pids)]
        if links.empty or df_comp.empty:
            return pd.DataFrame()

    df_comp = df_comp.copy()
    df_comp['participant_id'] = pd.to_numeric(df_comp['participant_id'], errors='coerce')
    df_comp['created_at'] = pd.to_datetime(df_comp['created_at'], errors='coerce')
    if start_date is not None:
        df_comp = df_comp[df_comp['created_at'] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df_comp = df_comp[df_comp['created_at'] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    df_comp = df_comp.dropna(subset=['participant_id', 'created_at'])
    df_comp = df_comp.merge(
        links[['participant_id', 'store_id', 'store_name']],
        on='participant_id',
        how='inner',
    )
    if df_comp.empty:
        return pd.DataFrame()

    df_comp['jaar_maand'] = df_comp['created_at'].dt.to_period('M').dt.to_timestamp()

    # Cumulatieve actieve deelnemers per store per maand
    maanden = sorted(df_comp['jaar_maand'].dropna().unique())
    trend_rows = []
    for store_id, store_df in df_comp.groupby('store_id'):
        store_name = store_df['store_name'].iloc[0]
        seen = set()
        for maand in maanden:
            nieuw = set(store_df[store_df['jaar_maand'] <= maand]['participant_id'].astype(int))
            seen |= nieuw
            n = len(seen)
            if n < 1:
                continue
            # Engagement-proxy: % deelnemers met ≥1 completion t/m deze maand (100) gewogen met herhaalratio
            comp_counts = store_df[store_df['jaar_maand'] <= maand].groupby('participant_id').size()
            pct_repeat = (comp_counts >= 2).mean() * 100 if len(comp_counts) else 0
            engagement_proxy = round(0.6 * 100 + 0.4 * pct_repeat, 1)
            trend_rows.append({
                'store_id': store_id,
                'store_name': store_name,
                'jaar_maand': maand,
                'engagement_score': engagement_proxy,
                'n_deelnemers': n,
            })

    trend = pd.DataFrame(trend_rows)
    if trend.empty:
        return trend

    trend = trend[trend['n_deelnemers'] >= max(1, min_deelnemers)]
    benchmark_per_month = (
        trend.groupby('jaar_maand')
        .apply(lambda g: _weighted_average(g['engagement_score'], g['n_deelnemers']), include_groups=False)
        .reset_index(name='benchmark')
    )
    trend = trend.merge(benchmark_per_month, on='jaar_maand', how='left')
    return trend


def maak_engagement_trend_plot(
    trend: pd.DataFrame,
    store_name: str | None = None,
    lang: str = 'nl',
) -> go.Figure:
    """Lijngrafiek engagement trend met benchmark."""
    if trend.empty:
        return _empty_store_plot(tr("Onvoldoende data voor trend.", lang))

    df = trend.copy()
    if store_name:
        df_store = df[df['store_name'] == store_name]
    else:
        df_store = (
            df.groupby('jaar_maand', as_index=False)
            .agg(
                engagement_score=('engagement_score', 'mean'),
                n_deelnemers=('n_deelnemers', 'sum'),
                benchmark=('benchmark', 'first'),
            )
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_store['jaar_maand'],
        y=df_store['engagement_score'],
        mode='lines+markers',
        name=store_name or tr('Gemiddelde', lang),
        line=dict(color=HOOFD_KLEUR, width=2),
    ))
    if 'benchmark' in df_store.columns:
        bench = df.groupby('jaar_maand')['benchmark'].first().reset_index()
        fig.add_trace(go.Scatter(
            x=bench['jaar_maand'],
            y=bench['benchmark'],
            mode='lines',
            name=tr('Benchmark (gewogen)', lang),
            line=dict(color='black', dash='dash'),
        ))
    fig.update_layout(
        title=tr('Engagement trend over tijd', lang),
        xaxis_title=tr('Maand', lang),
        yaxis_title=tr('Engagement score', lang),
        template='plotly_white',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def maak_periode_filter_visualisatie(base_pad: Path, db_url: str, lang: str = 'nl') -> go.Figure:
    """
    ✓ VRAAG 2: Kan je een periode selecteren of alleen data tot nu?
    
    Toont cumulatieve groei per jaar + kunnen kiezen welke periode weergegeven.
    """
    df_scores = load_my_clic_participants_expanded(db_url or DB_URL)
    
    if df_scores.empty or 'created_at' not in df_scores.columns:
        return _empty_store_plot(tr("Geen tijddata.", lang))
    
    df_scores['created_at'] = pd.to_datetime(df_scores['created_at'], errors='coerce')
    df_scores['jaar'] = df_scores['created_at'].dt.year
    
    # Cumulatieve groei
    jaren_counts = df_scores.groupby('jaar')['participant_id'].nunique().reset_index()
    jaren_counts['cumulatief'] = jaren_counts['participant_id'].cumsum()
    jaren_counts.columns = ['Jaar', 'Nieuwe deelnemers', 'Totaal cumulatief']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=jaren_counts['Jaar'], y=jaren_counts['Nieuwe deelnemers'],
                         name=tr('Nieuwe deelnemers', lang), marker_color='#3498DB'))
    fig.add_trace(go.Scatter(x=jaren_counts['Jaar'], y=jaren_counts['Totaal cumulatief'],
                             name=tr('Totaal cumulatief', lang), line=dict(color='#E87722', width=3),
                             yaxis='y2', mode='lines+markers'))
    
    fig.update_layout(
        title=tr('Data beschikbaar per periode', lang),
        xaxis_title=tr('Jaar', lang),
        yaxis_title=tr('Nieuwe deelnemers', lang),
        yaxis2=dict(overlaying='y', side='right', title=tr('Totaal cumulatief', lang)),
        barmode='group',
        hovermode='x unified'
    )
    return fig


def maak_leeftijd_vs_scores_plot(df: pd.DataFrame, lang: str = 'nl') -> go.Figure:
    """
    ✓ VRAAG 3: Zou het ook mogelijk zijn op te filteren op leeftijd(categorie)?
    
    Toont leeftijdscategorieën met gemiddelde scores per categorie.
    """
    df = df.copy()
    df['rec_age_current'] = pd.to_numeric(df['rec_age_current'], errors='coerce')
    df['rec_ls_lifestyle_score'] = pd.to_numeric(df['rec_ls_lifestyle_score'], errors='coerce')
    
    # Maak leeftijdscategorieën
    df['leeftijd_cat'] = pd.cut(df['rec_age_current'],
                               bins=[0, 25, 35, 45, 55, 65, 150],
                               labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    df_group = df.dropna(subset=['leeftijd_cat', 'rec_ls_lifestyle_score']).groupby('leeftijd_cat', observed=True).agg(
        gemiddelde_score=('rec_ls_lifestyle_score', 'mean'),
        n=('rec_age_current', 'count'),
        std=('rec_ls_lifestyle_score', 'std')
    ).reset_index()
    
    fig = px.bar(df_group, x='leeftijd_cat', y='gemiddelde_score', error_y='std',
                 title=tr('Leefstijlscore naar leeftijdscategorie', lang),
                 labels={'leeftijd_cat': tr('Leeftijdsgroep', lang), 'gemiddelde_score': tr('Gem. leefstijlscore', lang)},
                 color_discrete_sequence=[HOOFD_KLEUR],
                 custom_data=['n'])
    
    fig.update_traces(customdata=df_group['n'], hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<br>N: %{customdata}')
    return fig


def maak_engagement_metrics_dashboard(base_pad: Path, db_url: str, lang: str = 'nl') -> tuple:
    """
    ✓ VRAAG 4: Zou je ook de engagement op het platform inzichtelijk kunnen maken?
    
    Toont engagement metrics: vragenlijsten, artikelen, challenges per opdrachtgever.
    """
    try:
        df_p = load_participants(db_url or DB_URL)
        df_c = load_completions(db_url or DB_URL)
        df_long = laad_longitudinale_data(base_pad, db_url or DB_URL)
        
        n_reg = len(df_p) if not df_p.empty else 0
        n_unique_completes = df_c['participant_id'].nunique() if not df_c.empty else 0
        pct_vragenlijsten = (n_unique_completes / max(n_reg, 1)) * 100 if n_reg > 0 else 0
        
        # Herhaalmeting engagement (2+ metingen)
        n_repeat = int((df_long.groupby('participant_id')['completion_id'].nunique() >= 2).sum()) if not df_long.empty else 0
        pct_repeat = (n_repeat / max(n_reg, 1)) * 100 if n_reg > 0 else 0
        
        # Metrics per opdrachtgever (als beschikbaar)
        links = _laad_opdrachtgever_links(db_url or DB_URL)
        
        metrics = pd.DataFrame([
            {'Metric': tr('Totaal geregistreerd', lang), 'Waarde': n_reg, 'Percentage': 100},
            {'Metric': tr('Vragenlijst ingevuld', lang), 'Waarde': n_unique_completes, 'Percentage': pct_vragenlijsten},
            {'Metric': tr('Herhaalmetingen gedaan', lang), 'Waarde': n_repeat, 'Percentage': pct_repeat},
        ])
        
        # Visualisatie
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics['Metric'],
            y=metrics['Percentage'],
            text=metrics.apply(lambda r: f"{r['Waarde']:,} ({r['Percentage']:.0f}%)", axis=1),
            textposition='outside',
            marker_color=['#1F6FBF', '#2ECC71', '#E87722'],
            name=tr('Engagement %', lang)
        ))
        
        fig.update_layout(
            title=tr('Engagement Metrics: Participatiegraad per activiteit', lang),
            yaxis_title=tr('Percentage van totaal (%)', lang),
            xaxis_title='',
            showlegend=False,
            height=450
        )
        
        return fig, metrics
    except Exception as e:
        logger.error(f"Error in engagement metrics: {e}")
        return _empty_store_plot(tr("Fout bij laden engagement metrics.", lang)), pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# LOOKER STUDIO DASHBOARD - Analytics & Engagement Visualisaties
# ═══════════════════════════════════════════════════════════════════════════════

# Kleurenpalet voor Looker Studio
LOOKER_KLEUREN = {
    'total_users': '#1B3A6B',        # donkerblauw
    'total_users_light': '#4A90D9',  # lichtblauw
    'new_users': '#E87722',           # oranje
    'new_users_light': '#F4A548',     # lichtoranje
    'views': '#5BA4CF',               # lichtblauw
    'views_light': '#A8D4E6',         # heel lichtblauw
    'views_per_session': '#6B7B8C',   # grijs
    'views_per_session_light': '#A0AEBB',  # lichtgrijs
    'avg_engagement': '#D4A017',      # geel
    'avg_engagement_light': '#F0D060', # lichtgeel
    'returning_visitors': '#2E7D32',  # groen
    'background': '#F8F9FA',
    'text': '#2C3E50',
    'grid': '#E8ECF0',
}

# Mapping van device categorien
DEVICE_CATEGORIES = ['desktop', 'mobile', 'tablet']
TRAFFIC_CHANNELS = ['Organic Search', 'Direct', 'Referral', 'Social', 'Email', 'Paid Search']
PAGE_TITLES = [
    'Homepage', 'Vragenlijst: Leefstijl', 'Vragenlijst: Stress',
    'Vragenlijst: Hartrisico', 'Vragenlijst: Werkvermogen',
    'Challenge: Bewegen', 'Challenge: Voeding', 'Challenge: Slaap',
    'Artikel: Gezond eten', 'Artikel: Stressmanagement',
    'Artikel: Meer bewegen', 'Artikel: Beter slapen',
    'Health Partners', 'Mijn overzicht', 'Resultaten'
]
PAGE_PATHS = [
    '/home', '/questionnaire/lifestyle', '/questionnaire/stress',
    '/questionnaire/heartrisk', '/questionnaire/workability',
    '/challenge/exercise', '/challenge/nutrition', '/challenge/sleep',
    '/article/healthy-eating', '/article/stress-management',
    '/article/exercise-tips', '/article/sleep-better',
    '/health-partners', '/overview', '/results'
]


@st.cache_data(ttl=1800)
def _laad_analytics_data(db_url: str) -> pd.DataFrame:
    """
    Laadt analytics data uit de database en transformeert deze naar een
    gestructureerd formaat voor de Looker Studio dashboard.
    
    Als er geen aparte analytics tabel bestaat, worden de data afgeleid
    uit de completions, participants, stores en partners tabellen.
    """
    try:
        # Probeer eerst een dedicated analytics tabel te laden
        df_analytics = load_table_from_database('page_analytics', db_url)
        if not df_analytics.empty and 'date' in df_analytics.columns:
            df_analytics['date'] = pd.to_datetime(df_analytics['date'], errors='coerce')
            df_analytics = df_analytics.dropna(subset=['date']).copy()
            df_analytics['created_at'] = df_analytics['date']
            df_analytics['year'] = df_analytics['date'].dt.year.astype('Int64')
            df_analytics['jaar'] = df_analytics['year']
            return df_analytics
    except Exception:
        pass
    
    # Fallback: genereer analytics data uit bestaande tabellen
    try:
        df_completions = load_completions(db_url)
        df_participants = load_participants(db_url)
        df_partners = load_table_from_database('partners', db_url)
        df_interactions = load_table_from_database('interactions', db_url)
        
        if df_completions.empty and df_interactions.empty:
            return pd.DataFrame()
        
        # Gebruik interactions als primaire bron (echte page view data)
        if not df_interactions.empty:
            df = df_interactions.copy()
            # Hernoem kolommen voor consistentie
            df = df.rename(columns={
                df.columns[0]: 'id',
                df.columns[1]: 'content_type',
                df.columns[2]: 'content_id',
                df.columns[3]: 'user_id',
                df.columns[4]: 'interaction_type',
                df.columns[5]: 'created_at',
                df.columns[6]: 'updated_at'
            })
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = df.dropna(subset=['created_at'])
            
            # Views = elke interaction is een view
            df['views'] = 1
            
        else:
            # Fallback naar completions
            df = df_completions.copy()
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = df.dropna(subset=['created_at'])
            df['content_type'] = 'questionnaire'
            df['views'] = 1
            df['user_id'] = df['participant_id']
        
        if df.empty:
            return pd.DataFrame()
        
        # Voeg partner info toe via participants -> partners
        if not df_participants.empty and 'id' in df_participants.columns:
            pp = df_participants[['id', 'partner_id']].rename(columns={'id': 'participant_id'})
            if 'participant_id' in df.columns:
                df = df.merge(pp, on='participant_id', how='left')
            elif 'user_id' in df.columns:
                df = df.merge(pp, left_on='user_id', right_on='participant_id', how='left')
        
        if not df_partners.empty and 'id' in df_partners.columns and 'partner_id' in df.columns:
            partner_cols = ['id', 'name', 'code']
            avail = [c for c in partner_cols if c in df_partners.columns]
            df_partners_subset = df_partners[avail].rename(columns={'id': 'partner_id', 'name': 'store_name', 'code': 'store_slug'})
            df = df.merge(df_partners_subset, on='partner_id', how='left')
        
        # Voeg datum-kolommen toe
        df['date'] = df['created_at'].dt.date
        df['week'] = df['created_at'].dt.isocalendar().week.astype(int)
        df['month'] = df['created_at'].dt.month
        df['year'] = df['created_at'].dt.year
        
        # Bepaal of gebruiker nieuw of terugkerend is (gebaseerd op user_id)
        user_col = 'user_id' if 'user_id' in df.columns else 'participant_id'
        df = df.sort_values([user_col, 'created_at'])
        df['is_new_user'] = ~df.duplicated(subset=user_col, keep='first')
        df['is_returning_user'] = ~df['is_new_user']
        
        # Voeg store_slug toe als die niet bestaat
        if 'store_slug' not in df.columns:
            df['store_slug'] = 'onbekend'
        else:
            df['store_slug'] = df['store_slug'].fillna('onbekend')
        
        return df
        
    except Exception as e:
        logger.error(f"Fout bij genereren analytics data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def bereken_kpi_metrics(df: pd.DataFrame, df_vorige: pd.DataFrame = None) -> dict:
    """
    Berekent KPI metrics uit analytics data.
    
    Returns dict met total_users, returning_visitors, views en views_per_user,
    en delta percentages t.o.v. vorige periode.
    """
    if df.empty:
        return {
            'total_users': 0, 'returning_visitors': 0, 'views': 0,
            'views_per_user': 0, 'new_users': 0,
            'delta_total_users': 0, 'delta_returning': 0,
            'delta_views': 0, 'delta_views_per_user': 0,
        }
    
    # Gebruik user_id als die bestaat, anders participant_id
    user_col = 'user_id' if 'user_id' in df.columns else 'participant_id'
    total_users = df[user_col].nunique() if user_col in df.columns else len(df)
    new_users = int(df['is_new_user'].sum()) if 'is_new_user' in df.columns else 0
    returning_visitors = int(df['is_returning_user'].sum()) if 'is_returning_user' in df.columns else 0
    total_views = int(df['views'].sum()) if 'views' in df.columns else len(df)
    views_per_user = total_views / total_users if total_users else 0
    
    # Deltas berekenen
    deltas = {}
    if df_vorige is not None and not df_vorige.empty:
        prev_user_col = 'user_id' if 'user_id' in df_vorige.columns else 'participant_id'
        prev_total = df_vorige[prev_user_col].nunique() if prev_user_col in df_vorige.columns else len(df_vorige)
        prev_returning = int(df_vorige['is_returning_user'].sum()) if 'is_returning_user' in df_vorige.columns else 0
        prev_views = int(df_vorige['views'].sum()) if 'views' in df_vorige.columns else len(df_vorige)
        prev_views_per_user = prev_views / prev_total if prev_total else 0
        
        deltas['delta_total_users'] = ((total_users - prev_total) / max(prev_total, 1)) * 100
        deltas['delta_returning'] = ((returning_visitors - prev_returning) / max(prev_returning, 1)) * 100
        deltas['delta_views'] = ((total_views - prev_views) / max(prev_views, 1)) * 100
        deltas['delta_views_per_user'] = (
            (views_per_user - prev_views_per_user) / max(prev_views_per_user, 0.1)
        ) * 100
    else:
        deltas = {k: 0 for k in [
            'delta_total_users', 'delta_returning', 'delta_views', 'delta_views_per_user'
        ]}
    
    return {
        'total_users': total_users,
        'new_users': new_users,
        'returning_visitors': returning_visitors,
        'views': total_views,
        'views_per_user': round(views_per_user, 1),
        **deltas,
    }


def _get_user_col(df: pd.DataFrame) -> str:
    """Return de user kolom naam (user_id of participant_id)."""
    return 'user_id' if 'user_id' in df.columns else 'participant_id'


def bereken_dagelijkse_trends(df: pd.DataFrame, datum_kolom: str = 'created_at') -> pd.DataFrame:
    """Bereken dagelijkse aggregatie van gebruikers metrics."""
    if df.empty or datum_kolom not in df.columns:
        return pd.DataFrame()
    
    df_temp = df.copy()
    df_temp['dag'] = pd.to_datetime(df_temp[datum_kolom]).dt.date
    user_col = _get_user_col(df_temp)
    
    agg = df_temp.groupby('dag').agg(
        total_users=(user_col, 'nunique'),
        new_users=('is_new_user', 'sum'),
        views=('views', 'sum'),
    ).reset_index()
    
    agg['dag'] = pd.to_datetime(agg['dag'])
    agg = agg.sort_values('dag')
    agg['new_users'] = agg['new_users'].astype(int)
    agg['views_per_user'] = agg['views'] / agg['total_users'].replace(0, np.nan)
    agg['views_per_user'] = agg['views_per_user'].fillna(0).round(1)
    
    return agg


def bereken_wekelijkse_trends(df: pd.DataFrame, datum_kolom: str = 'created_at') -> pd.DataFrame:
    """Bereken wekelijkse aggregatie van gebruikers metrics."""
    if df.empty or datum_kolom not in df.columns:
        return pd.DataFrame()
    
    df_temp = df.copy()
    df_temp['week_start'] = pd.to_datetime(df_temp[datum_kolom]).dt.to_period('W').apply(lambda x: x.start_time)
    user_col = _get_user_col(df_temp)
    
    agg = df_temp.groupby('week_start').agg(
        total_users=(user_col, 'nunique'),
        new_users=('is_new_user', 'sum'),
        views=('views', 'sum'),
    ).reset_index()
    
    agg['week_start'] = pd.to_datetime(agg['week_start'])
    agg = agg.sort_values('week_start')
    agg['new_users'] = agg['new_users'].astype(int)
    agg['views_per_user'] = agg['views'] / agg['total_users'].replace(0, np.nan)
    agg['views_per_user'] = agg['views_per_user'].fillna(0).round(1)
    
    return agg


def bereken_maandelijkse_trends(df: pd.DataFrame, datum_kolom: str = 'created_at') -> pd.DataFrame:
    """Bereken maandelijkse aggregatie van gebruikers metrics."""
    if df.empty or datum_kolom not in df.columns:
        return pd.DataFrame()
    
    df_temp = df.copy()
    df_temp['month_start'] = pd.to_datetime(df_temp[datum_kolom]).dt.to_period('M').apply(lambda x: x.start_time)
    user_col = _get_user_col(df_temp)
    
    agg = df_temp.groupby('month_start').agg(
        total_users=(user_col, 'nunique'),
        new_users=('is_new_user', 'sum'),
        views=('views', 'sum'),
    ).reset_index()
    
    agg['month_start'] = pd.to_datetime(agg['month_start'])
    agg = agg.sort_values('month_start')
    agg['new_users'] = agg['new_users'].astype(int)
    agg['views_per_user'] = agg['views'] / agg['total_users'].replace(0, np.nan)
    agg['views_per_user'] = agg['views_per_user'].fillna(0).round(1)
    
    return agg


def bereken_jaar_trends(df: pd.DataFrame, datum_kolom: str = 'created_at') -> pd.DataFrame:
    """Bereken jaarlijkse aggregatie van gebruikers metrics."""
    if df.empty or datum_kolom not in df.columns:
        return pd.DataFrame()
    
    df_temp = df.copy()
    df_temp['year_start'] = pd.to_datetime(df_temp[datum_kolom]).dt.to_period('Y').apply(lambda x: x.start_time)
    user_col = _get_user_col(df_temp)
    
    agg = df_temp.groupby('year_start').agg(
        total_users=(user_col, 'nunique'),
        new_users=('is_new_user', 'sum'),
        views=('views', 'sum'),
    ).reset_index()
    
    agg['year_start'] = pd.to_datetime(agg['year_start'])
    agg = agg.sort_values('year_start')
    agg['new_users'] = agg['new_users'].astype(int)
    agg['views_per_user'] = agg['views'] / agg['total_users'].replace(0, np.nan)
    agg['views_per_user'] = agg['views_per_user'].fillna(0).round(1)
    
    return agg


def maak_engagement_bar_chart(trends_df: pd.DataFrame, metrics: list = None,
                               aggregatie: str = 'week') -> go.Figure:
    """
    Maakt een mixed chart voor user & engagement performance.
    
    - total_users: bar chart (hoofdmetriek)
    - Andere metrics: lijnen (op secundaire y-as)
    
    Metrics: total_users, new_users, views, views_per_user
    """
    if trends_df.empty:
        return _empty_store_plot("Geen data beschikbaar")
    
    if metrics is None:
        metrics = ['total_users']
    
    # Bepaal de juiste datum kolom
    datum_kolommen = ['dag', 'week_start', 'month_start', 'year_start']
    datum_col = next((c for c in datum_kolommen if c in trends_df.columns), None)
    
    if datum_col is None:
        return _empty_store_plot("Geen data beschikbaar")
    
    # Zorg dat beschikbare metrics in de trends_df staan
    metrics = [m for m in metrics if m in trends_df.columns]
    
    if not metrics:
        return _empty_store_plot("Geen data beschikbaar")
    
    # Kleur mapping
    kleur_map = {
        'total_users': LOOKER_KLEUREN['total_users'],
        'new_users': LOOKER_KLEUREN['new_users'],
        'views': LOOKER_KLEUREN['views'],
        'views_per_user': LOOKER_KLEUREN['views_per_session'],
    }
    
    # Label mapping
    label_map = {
        'total_users': 'Totaal gebruikers',
        'new_users': 'Nieuwe gebruikers',
        'views': 'Weergaven',
        'views_per_user': 'Weergaven per gebruiker',
    }
    
    fig = go.Figure()
    
    # Voeg total_users als bar toe (als beschikbaar)
    if 'total_users' in metrics:
        waarden = trends_df['total_users'].values.astype(float)
        if len(waarden) > 1 and waarden.max() > waarden.min():
            norm = (waarden - waarden.min()) / (waarden.max() - waarden.min())
        else:
            norm = np.ones_like(waarden) * 0.5
        
        # Genereer graduele kleuren voor bars
        bar_kleuren = []
        for n in norm:
            r = int(0x4A + (0x1B - 0x4A) * n)
            g = int(0x90 + (0x3A - 0x90) * n)
            b = int(0xD9 + (0x6B - 0xD9) * n)
            bar_kleuren.append(f'#{max(0,min(255,r)):02X}{max(0,min(255,g)):02X}{max(0,min(255,b)):02X}')
        
        fig.add_trace(go.Bar(
            x=trends_df[datum_col],
            y=trends_df['total_users'],
            marker_color=bar_kleuren,
            text=trends_df['total_users'].round(1),
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate=(
                f"<b>%{{x|%d-%m-%Y}}</b><br>"
                f"Totaal gebruikers: %{{y:,.0f}}<br>"
                f"<extra></extra>"
            ),
            name='Totaal gebruikers',
            yaxis='y1',
        ))
    
    # Voeg andere metrics als lijnen toe (op secundaire y-as)
    line_metrics = [m for m in metrics if m != 'total_users']
    
    for metric in line_metrics:
        label = label_map.get(metric, metric)
        kleur = kleur_map.get(metric, '#666666')
        
        fig.add_trace(go.Scatter(
            x=trends_df[datum_col],
            y=trends_df[metric],
            mode='lines+markers',
            name=label,
            line=dict(color=kleur, width=3),
            marker=dict(size=8),
            hovertemplate=(
                f"<b>%{{x|%d-%m-%Y}}</b><br>"
                f"{label}: %{{y:,.1f}}<br>"
                f"<extra></extra>"
            ),
            yaxis='y2' if line_metrics else 'y1',
        ))
    
    title_text = f"Engagement Metrics per {aggregatie}"
    if len(metrics) == 1:
        title_text = f"{label_map.get(metrics[0], metrics[0])} per {aggregatie}"
    
    # Bepaal y-axes configuratie
    yaxis1_title = 'Totaal gebruikers' if 'total_users' in metrics else (label_map.get(metrics[0], metrics[0]) if metrics else '')
    
    layout_dict = dict(
        title=dict(
            text=title_text,
            font=dict(size=16, color=LOOKER_KLEUREN['text']),
        ),
        xaxis_title='',
        yaxis=dict(
            title=dict(text=yaxis1_title, font=dict(color=LOOKER_KLEUREN.get(metrics[0], LOOKER_KLEUREN['text']))),
            tickfont=dict(color=LOOKER_KLEUREN.get(metrics[0], LOOKER_KLEUREN['text'])),
            gridcolor=LOOKER_KLEUREN['grid'],
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=LOOKER_KLEUREN['text']),
        xaxis=dict(gridcolor=LOOKER_KLEUREN['grid'], showgrid=True),
        height=400,
        margin=dict(l=60, r=60, t=50, b=40),
        bargap=0.3,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )
    
    # Voeg secundaire y-as toe als er lijnen zijn
    if line_metrics:
        line_labels = ', '.join([label_map.get(m, m) for m in line_metrics])
        layout_dict['yaxis2'] = dict(
            title=dict(text=line_labels, font=dict(color=LOOKER_KLEUREN.get(line_metrics[0], '#666666'))),
            tickfont=dict(color=LOOKER_KLEUREN.get(line_metrics[0], '#666666')),
            overlaying='y1',
            side='right',
            gridcolor=LOOKER_KLEUREN['grid'],
        )
    
    fig.update_layout(**layout_dict)
    
    return fig


def bereken_page_performance(df: pd.DataFrame, page_type: str = 'all') -> pd.DataFrame:
    """
    Berekent page performance metrics gegroepeerd per content type.
    
    page_type: 'all', 'content', 'orderable', 'product'
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter op content_type
    df_filtered = df.copy()
    if page_type != 'all' and 'content_type' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['content_type'] == page_type]
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    # Groepeer per content_type
    user_col = _get_user_col(df_filtered)
    group_cols = ['content_type']
    
    agg_dict = {
        'total_users': (user_col, 'nunique'),
        'views': ('views', 'sum'),
    }
    if 'is_new_user' in df_filtered.columns:
        agg_dict['new_users'] = ('is_new_user', 'sum')
    
    agg = df_filtered.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()
    
    # Zorg dat alle kolommen aanwezig zijn
    for col in ['new_users']:
        if col not in agg.columns:
            agg[col] = 0
    
    agg['new_users'] = agg['new_users'].astype(int)
    agg['views_per_user'] = agg['views'] / agg['total_users'].replace(0, np.nan)
    agg['views_per_user'] = agg['views_per_user'].fillna(0).round(1)
    agg = agg.rename(columns={'content_type': 'page_type'})
    
    return agg.sort_values('views', ascending=False)


def bereken_top_health_partners(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Berekent top N met meeste views per partner."""
    if df.empty or 'store_slug' not in df.columns:
        return pd.DataFrame()
    
    user_col = _get_user_col(df)
    
    agg = df.groupby('store_slug', dropna=False).agg(
        total_users=(user_col, 'nunique'),
        views=('views', 'sum'),
    ).reset_index()
    
    if 'is_new_user' in df.columns:
        agg['new_users'] = df.groupby('store_slug', dropna=False)['is_new_user'].sum().values
    
    return agg.sort_values('views', ascending=False).head(top_n)
