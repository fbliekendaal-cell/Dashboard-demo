# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false

import sys
import subprocess
import importlib
import logging
import json
import re
from pathlib import Path
from typing import cast

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE, HTML_DIR as HTML_MAP, FEATURES, CACHE_TTL_SECONDS, setup_logger,
    DB_URL, CODE_DIR, LOOKER_STUDIO_EMBED_URL, DATA_SOURCE,
)
DATA_PATH = BASE

from auth import require_login, logout, is_admin
from data_ingestion import (
    load_completions,
    load_participants,
    load_my_clic_participants_expanded,
    load_participants_with_factor_scores,
    add_app_user_ids_and_addresses,
)
from helpers import (
    filter_by_gender, calculate_kpi_count, get_numeric_clean,
    export_to_csv, normalize_search_text, search_variable_specs
    , _maak_net_label
)

logger = setup_logger(__name__)


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column with NaN values when the column is absent."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype='float64')
    return pd.to_numeric(df[column], errors='coerce')

def load_main_data_safe(db_url: str) -> pd.DataFrame:
    try:
        df = _load_my_clic_participants_cached(db_url)
        if df.empty: 
            st.error("⚠️ Main data is empty"); logger.warning("my_clic_participants (expanded) is leeg"); return pd.DataFrame()
        logger.info(f"✓ Loaded: {len(df)} records"); return df
    except Exception as e:
        st.error(f"❌ Error: {e}"); logger.error(f"Data load error: {e}", exc_info=True); return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_SECONDS['main_data'])
def _load_my_clic_participants_cached(db_url: str) -> pd.DataFrame:
    try:
        # Gebruik de geconsolideerde tabel als primaire bron
        return load_my_clic_participants_expanded(db_url)
    except Exception as e:
        logger.error(f"Error loading participants: {e}", exc_info=True)
        return pd.DataFrame()

st.set_page_config(page_title="Smart Health Dashboard", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")
require_login()
translation_script = CURRENT_DIR / "update_translations.py"
if translation_script.exists():
    try:
        subprocess.run([sys.executable, str(translation_script)], check=False)
    except Exception as exc:
        logger.warning(f"Translations bootstrap script kon niet starten: {exc}")
else:
    logger.info("Translations bootstrap script ontbreekt; veilig overgeslagen.")

from kleuren import RISICO_COLORS, HEARTRISK_LABELS, BMI_LABELS, BMI_COLORS, STRESS_LABELS
from categorieen import categorie_labels
import visualisaties # Import the module first
from variabelen import VARIABELEN_PER_GROEP, VARIABELEN_DICT
from i18n import tr, translate_variable_specs, translate_plotly_figure, translate_matplotlib_figure, translate_dataframe
from analyses import (bereken_datakwaliteit, maak_missende_waarden_plot, bereken_duplicaat_statistieken, bereken_outliers, 
    maak_outlier_boxplot, bereken_t_toetsen, maak_t_toets_plot, maak_t_toets_gemiddelden_plot, maak_correlatiematrix, maak_scatter_correlatie,
    genereer_inzichten, analyse_vragenlijst_herhalingen, analyse_vragenlijst_dropoff, RUWE_WAARDEN, LEEFSTIJL_SCORES,
    T_TOETS_EXTRA_VARIABELEN,
    analyse_invulfrequentie_vs_scores, INVULFREQUENTIE_SCORE_KOLOMMEN,
    haal_vragenlijsten_overzicht, haal_scores_voor_vragenlijst, analyse_herhaalde_vragenlijst_scoreverandering)
# Force reload of the visualisaties module to ensure latest changes are picked up
# Import once after reload to avoid duplicate symbol imports and keep the module API explicit.
importlib.reload(visualisaties)

from visualisaties import (
    laad_data,
    laad_longitudinale_data,
    bereken_verandering,
    maak_geslacht_plot,
    maak_leeftijd_plot,
    maak_heartrisk_plot,
    maak_heartrisk_naar_geslacht_plot,
    maak_bmi_plot,
    maak_stress_plot,
    maak_leefstijl_score_plot,
    maak_bmi_beweging_plot,
    maak_scoreverandering_plot,
    voeg_dieet_score_toe,
    maak_dieet_verdeling_plot,
    maak_dieet_bmi_plot,
    maak_dieet_score_histogram,
    maak_verkenner_plot,
    maak_platform_groei_plot,
    maak_account_activatie_plot,
    maak_vragenlijst_plot,
    maak_kopers_vergelijking_plot,
    maak_artikel_interacties_overzicht,
    maak_scores_per_opdrachtgever,
    maak_store_scoreverbetering_plot,
    maak_gemiddelde_score_over_tijd_plot,
    get_available_stores,
    maak_store_gemiddelde_scores_plot,
    maak_store_gemiddelde_verandering_plot,
    maak_update_verificatie_plot,
    get_available_stores_from_average_scores,
    get_available_stores_with_score_changes,
    maak_meerdere_vragenlijsten_plot,
    maak_vroege_kopers_profiel,
    maak_risico_migratie_sankey,
    bereken_engagement_journey,
    maak_engagement_funnel_plot,
    maak_business_value_plot,
    maak_herhaalde_vragenlijst_scoreverandering_plot,
    maak_account_dataflow_funnel_plot,
    bereken_engagement_per_opdrachtgever,
    build_engagement_summary_stats,
    maak_engagement_opdrachtgever_ranking_plot,
    maak_engagement_breakdown_plot,
    bereken_engagement_trend,
    maak_engagement_trend_plot,
    ENGAGEMENT_COMPONENT_DEFAULTS,
    _laad_analytics_data,
    bereken_kpi_metrics,
    bereken_dagelijkse_trends,
    bereken_wekelijkse_trends,
    bereken_maandelijkse_trends,
    bereken_jaar_trends,
    maak_engagement_bar_chart,
    bereken_page_performance,
    bereken_top_health_partners,
    LOOKER_KLEUREN,
)

# ── Cache helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL_SECONDS['main_data'])
def load_main_data() -> pd.DataFrame:
    df = load_main_data_safe(DB_URL)
    if not df.empty:
        steps = _numeric_column(df, 'rec_ls_exercise_steps_per_day') * 1000
        mins = _numeric_column(df, 'rec_ls_exercise_physical_activity_minutes_total')
        is_active, is_inactive = (steps >= 5000) | (mins >= 250), (steps < 5000) & (mins < 250)
        df['derived_is_inactive'] = np.nan
        df.loc[is_active, 'derived_is_inactive'] = 0
        df.loc[is_inactive, 'derived_is_inactive'] = 1
    return df

ml_module, ML_AVAILABLE, ML_IMPORT_ERROR = None, False, ""
try: 
    print(f"DEBUG: Reloading ML module...")
    ml_module = importlib.reload(__import__('ML'))
    print(f"DEBUG: ML module reloaded. ML_MODEL_VERSION from module: {getattr(ml_module, 'ML_MODEL_VERSION', 'N/A')}")
    ml_module.DATA_PATH = BASE
    ML_AVAILABLE = True
except Exception as e: 
    ML_IMPORT_ERROR = str(e)
    # Ensure ML_AVAILABLE is False if reload fails
    ML_AVAILABLE = False
    logger.warning(f"ML module unavailable: {e}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, recall_score

load_ml_data = load_main_data
train_bp_model = getattr(ml_module, "train_bp_model", None)
train_heartrisk_model = getattr(ml_module, "train_heartrisk_model", None)
train_lifestyle_model = getattr(ml_module, "train_lifestyle_model", None)
train_next_lifestyle_model = getattr(ml_module, "train_next_lifestyle_model", None)
train_dropoff_model = getattr(ml_module, "train_dropoff_model", None)
train_improvement_model = getattr(ml_module, "train_improvement_model", None)
train_purchase_model = getattr(ml_module, "train_purchase_model", None)
plot_feature_importance = getattr(ml_module, "plot_feature_importance", None)
plot_prediction_performance_regression = getattr(ml_module, "plot_prediction_performance_regression", None)
plot_confusion_matrix = getattr(ml_module, "plot_confusion_matrix", None)
plot_model_metrics = getattr(ml_module, "plot_model_metrics", None)
plot_roc_curve = getattr(ml_module, "plot_roc_curve", None)
predict_bp = getattr(ml_module, "predict_bp", lambda m, i: (m.predict(i), m.predict_proba(i)) if m else (None, None))
predict_heartrisk = getattr(ml_module, "predict_heartrisk", lambda m, i: (m.predict(i), m.predict_proba(i)) if m else (None, None))
predict_lifestyle = getattr(ml_module, "predict_lifestyle", lambda m, i: m.predict(i) if m else None)
predict_binary_model = getattr(ml_module, "predict_binary_model", lambda m, i: (m.predict(i), m.predict_proba(i)) if m else (None, None))
plot_local_shap = getattr(ml_module, "plot_local_shap", None)
maak_gebruikers_segmentatie_plot = getattr(ml_module, "maak_gebruikers_segmentatie_plot", None)

LEEFTIJD_CATEGORIEEN = ["<30", "30-40", "40-50", "50-60", "60+"]
PERIODE_PRESETS = {
    "Laatste 3 maanden": "3m",
    "Laatste jaar": "1y",
    "Sinds start": "all",
    "Custom": "custom",
}
DATUM_KOLOMMEN = [
    "latest_completion_at",
    "completion_created_at",
    "completed_at",
    "created_at",
    "updated_at",
    "date",
]
AFDELING_KOLOMMEN = ["department", "afdeling", "team", "business_unit"]
FUNCTIE_KOLOMMEN = ["function", "functie", "job_title", "role"]
MIN_VISUALISATIE_DEELNEMERS = 1
MIN_INTERNATIONALE_DEELNEMERS = 6


def _tab_default_for_path(path: str | None, labels: list[str]) -> str | None:
    """Return the tab label matching a segment of a search path."""
    if not path:
        return None
    targets = [normalize_search_text(segment) for segment in path.split(' → ')[::-1]]
    for target in targets:
        for label in labels:
            normalized_label = normalize_search_text(label)
            if target == normalized_label or target in normalized_label or normalized_label in target:
                return label
    return None


def filter_geslacht(df: pd.DataFrame, geslacht: str) -> pd.DataFrame:
    if df is None or df.empty: return df
    gender_kolom = 'rec_user_gender' if 'rec_user_gender' in df.columns else 'user_gender' if 'user_gender' in df.columns else None
    if not gender_kolom:
        return df
    if geslacht == 'man': return df[_numeric_column(df, gender_kolom) == 1]
    elif geslacht == 'vrouw': return df[_numeric_column(df, gender_kolom) == 0]
    return df


def _gender_participant_sets(df: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """Return participant IDs for the two groups used by comparison mode."""
    if df is None or df.empty or 'participant_id' not in df.columns:
        return {'Man': tuple(), 'Vrouw': tuple()}
    gender = _numeric_column(df, 'rec_user_gender')
    participant_ids = pd.to_numeric(df['participant_id'], errors='coerce')
    return {
        'Man': tuple(participant_ids[gender.eq(1)].dropna().astype(int).unique()),
        'Vrouw': tuple(participant_ids[gender.eq(0)].dropna().astype(int).unique()),
    }


def filter_store(df: pd.DataFrame, store_id: int | None) -> pd.DataFrame:
    if df is None or df.empty or store_id is None:
        return df

    # If the main data contains a matching store_id column with the selected
    # store present, filter directly on that. Otherwise fall back to the
    # canonical participant->store links and filter by participant_id.
    try:
        store_vals = _numeric_column(df, 'store_id').dropna().unique()
        if len(store_vals) and int(store_id) in [int(x) for x in store_vals]:
            return df[_numeric_column(df, 'store_id') == int(store_id)]
    except Exception:
        pass

    # Fallback: use participant->store links to find participants for this store
    try:
        from visualisaties import _laad_opdrachtgever_links
        links = _laad_opdrachtgever_links(DB_URL)
        if links is not None and not links.empty and 'participant_id' in links.columns:
            pids = links.loc[pd.to_numeric(links['store_id'], errors='coerce') == int(store_id), 'participant_id'].dropna().unique()
            if len(pids):
                return df[_numeric_column(df, 'participant_id').isin([int(x) for x in pids])]
    except Exception:
        pass

    # If all else fails, return original df (no filtering)
    return df


def get_store_filter_id() -> int | None:
    return st.session_state.get('global_store_id', None)


def _participant_ids(df: pd.DataFrame | None) -> tuple[int, ...]:
    if df is None or df.empty or 'participant_id' not in df.columns:
        return tuple()
    ids = pd.to_numeric(df['participant_id'], errors='coerce').dropna().astype(int).unique()
    return tuple(sorted(int(x) for x in ids))


def _user_ids(df: pd.DataFrame | None) -> tuple[int, ...]:
    if df is None or df.empty or 'user_id' not in df.columns:
        return tuple()
    ids = pd.to_numeric(df['user_id'], errors='coerce').dropna().astype(int).unique()
    return tuple(sorted(int(x) for x in ids))


def _eerste_bestaande_kolom(df: pd.DataFrame, kandidaten: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    for kolom in kandidaten:
        if kolom in df.columns:
            return kolom
    return None


def _detecteer_datumkolom(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    for kolom in DATUM_KOLOMMEN:
        if kolom in df.columns and pd.to_datetime(df[kolom], errors='coerce').notna().any():
            return kolom
    return None


def _datum_bereik(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str | None]:
    datumkolom = _detecteer_datumkolom(df)
    if not datumkolom:
        return None, None, None
    datums = pd.to_datetime(df[datumkolom], errors='coerce').dropna()
    if datums.empty:
        return None, None, datumkolom
    return datums.min().normalize(), datums.max().normalize(), datumkolom


def _periode_selectie(min_datum: pd.Timestamp | None, max_datum: pd.Timestamp | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if min_datum is None or max_datum is None:
        return None, None

    preset_label = st.session_state.get("global_period_preset", "Sinds start")
    preset = PERIODE_PRESETS.get(preset_label, "all")
    if preset == "3m":
        return max(min_datum, max_datum - pd.DateOffset(months=3)), max_datum
    if preset == "1y":
        return max(min_datum, max_datum - pd.DateOffset(years=1)), max_datum
    if preset == "custom":
        custom = st.session_state.get("global_period_custom")
        if isinstance(custom, tuple) and len(custom) == 2:
            start, einde = pd.to_datetime(custom[0], errors='coerce'), pd.to_datetime(custom[1], errors='coerce')
            if pd.notna(start) and pd.notna(einde):
                return start.normalize(), einde.normalize()
    return min_datum, max_datum


def _filters_actief(geslacht: str) -> bool:
    if geslacht not in {'beide', 'totaal', None}:
        return True
    if st.session_state.get("global_store_id") not in (None, ""):
        return True
    if st.session_state.get("global_period_preset", "Sinds start") != "Sinds start":
        return True
    if st.session_state.get("global_age_categories"):
        return True
    if st.session_state.get("global_department_values"):
        return True
    if st.session_state.get("global_function_values"):
        return True
    return False


def _pct_context_label(geslacht: str) -> str:
    return "van selectie" if _filters_actief(geslacht) else "van totaal"


def _leeftijd_categorieen(df: pd.DataFrame) -> pd.Series:
    leeftijd = _numeric_column(df, 'rec_age_current')
    categorie = pd.cut(
        leeftijd,
        bins=[0, 30, 40, 50, 60, float("inf")],
        labels=LEEFTIJD_CATEGORIEEN,
        right=False,
    )
    return categorie.astype("object").where(categorie.notna(), "Onbekend")


def _filter_op_tekstkolom(df: pd.DataFrame, kolom: str | None, geselecteerd: list[str] | None) -> pd.DataFrame:
    if df is None or df.empty or not kolom or kolom not in df.columns or geselecteerd is None:
        return df
    waarden = df[kolom].fillna("Onbekend").astype(str).str.strip().replace("", "Onbekend")
    return df[waarden.isin(geselecteerd)]


def apply_global_filters(
    df: pd.DataFrame,
    geslacht: str | None = None,
    include_store: bool = True,
    include_period: bool = True,
    include_age: bool = True,
    include_extra: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    gefilterd = df.copy()
    if geslacht is not None:
        gefilterd = filter_geslacht(gefilterd, geslacht)
    if include_store:
        gefilterd = filter_store(gefilterd, get_store_filter_id())

    if include_period:
        min_datum, max_datum, datumkolom = _datum_bereik(gefilterd)
        start, einde = _periode_selectie(min_datum, max_datum)
        if datumkolom and start is not None and einde is not None:
            datums = pd.to_datetime(gefilterd[datumkolom], errors='coerce')
            gefilterd = gefilterd[datums.between(start, einde + pd.Timedelta(days=1), inclusive="left")]

    if include_age and "rec_age_current" in gefilterd.columns:
        gekozen_leeftijden = st.session_state.get("global_age_categories", LEEFTIJD_CATEGORIEEN + ["Onbekend"])
        if gekozen_leeftijden is not None:
            gefilterd = gefilterd[_leeftijd_categorieen(gefilterd).isin(gekozen_leeftijden)]

    if include_extra:
        afdeling_kolom = st.session_state.get("global_department_column")
        functie_kolom = st.session_state.get("global_function_column")
        gefilterd = _filter_op_tekstkolom(
            gefilterd,
            afdeling_kolom,
            st.session_state.get("global_department_values"),
        )
        gefilterd = _filter_op_tekstkolom(
            gefilterd,
            functie_kolom,
            st.session_state.get("global_function_values"),
        )

    return gefilterd


def get_global_filter_signature() -> tuple:
    return (
        st.session_state.get("global_period_preset", "Sinds start"),
        tuple(st.session_state.get("global_period_custom", ()) or ()),
        tuple(st.session_state.get("global_age_categories", LEEFTIJD_CATEGORIEEN + ["Onbekend"]) or ()),
        st.session_state.get("global_store_id"),
        tuple(st.session_state.get("global_department_values", ()) or ()),
        tuple(st.session_state.get("global_function_values", ()) or ()),
    )


def load_main_data_filtered_by_store() -> pd.DataFrame:
    return apply_global_filters(load_main_data(), geslacht=None)


def load_main_data_filtered_by_gender_and_store(geslacht: str) -> pd.DataFrame:
    return apply_global_filters(load_main_data(), geslacht=geslacht)


def _active_participant_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep non-deleted participant rows for data-quality counts."""
    if df is None or df.empty or 'deleted_at' not in df.columns:
        return df
    deleted_values = df['deleted_at'].astype('string').str.strip().str.lower()
    active_mask = df['deleted_at'].isna() | deleted_values.isin({'', 'null', 'none', 'nat'})
    return df[active_mask].copy()


def _count_unique_participants(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    for column in ('participant_id', 'user_id', 'id'):
        if column in df.columns:
            return int(df[column].dropna().nunique())
    return len(df)


def get_available_stores_dashboard() -> pd.DataFrame:
    # Prefer the canonical participant->store company links so the dashboard
    # shows companies (public.stores) instead of individual users with the
    # 'opdrachtgever' role. Fall back to main-data-derived store ids if needed.
    try:
        from visualisaties import _laad_opdrachtgever_links
        links = _laad_opdrachtgever_links(DB_URL)
        if links is not None and not links.empty and 'store_id' in links.columns and 'store_name' in links.columns:
            stores = (
                links[['store_id', 'store_name']]
                .dropna(subset=['store_id'])
                .drop_duplicates(subset=['store_id'])
                .sort_values('store_name')
                .reset_index(drop=True)
            )
            return stores.rename(columns={'store_name': 'store_name'})
    except Exception:
        # Ignore and fall back
        pass

    # Fallback: use main participant data and store metadata (best-effort)
    try:
        df_main = load_main_data()
        if df_main is None or df_main.empty:
            return pd.DataFrame(columns=['store_id', 'store_name'])

        store_ids = _numeric_column(df_main, 'store_id').dropna()
        main_ids = sorted(int(x) for x in store_ids.unique())

        try:
            from data_ingestion import load_stores
            stores_meta = load_stores(DB_URL)
        except Exception:
            stores_meta = pd.DataFrame()

        name_col = None
        for candidate in ('store_name', 'name', 'title', 'display_name'):
            if candidate in stores_meta.columns:
                name_col = candidate
                break

        rows = []
        for sid in main_ids:
            display = None
            if not stores_meta.empty and name_col and 'id' in stores_meta.columns:
                id_values = _numeric_column(stores_meta, 'id')
                match = stores_meta[id_values == int(sid)]
                if not match.empty:
                    display = str(match.iloc[0][name_col])
            if not display:
                display = f"Opdrachtgever {sid}"
            rows.append({'store_id': int(sid), 'store_name': display})

        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning(f"Unable to build available stores for dashboard (fallback): {e}")
        return pd.DataFrame(columns=['store_id', 'store_name'])


def get_store_name(store_id: int | None) -> str | None:
    if store_id is None:
        return None
    stores = get_available_stores_dashboard()
    if stores.empty or 'store_id' not in stores.columns or 'store_name' not in stores.columns:
        return None
    match = stores.loc[stores['store_id'] == int(store_id)]
    if match.empty:
        return None
    return str(match['store_name'].iloc[0])


def _beschikbare_opdrachtgever_indicatoren(df: pd.DataFrame, min_stores: int = 2, min_participants_per_store: int = 2) -> dict[str, str]:
    """
    Filter indicatoren op basis van echte deelnemersdata per opdrachtgever.
    Sorteer op hoeveelheid data (meeste data eerst).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Deelnemersdata met participant_id en scorekolommen
    min_stores : int
        Minimaal aantal opdrachtgevers met data
    min_participants_per_store : int
        Minimaal aantal deelnemers per opdrachtgever
    """

    voorkeurslabels = {
        'rec_ls_lifestyle_score': 'Leefstijlscore',
        'rec_med_bmi': 'BMI',
        'rec_ls_stress_sum': 'Stress',
        'rec_heartrisk': 'Heartrisk score',
        'rec_ls_sleep_psqi_sum': 'Slaap (PSQI)',
        'rec_dass_stress_score': 'DASS stress',
        'rec_dass_anxiety_score': 'DASS angst',
        'rec_dass_depression_score': 'DASS depressie',
        'rec_wellbeing_score': 'Welzijn',
        'rec_resilience_score': 'Veerkracht',
        'rec_self_efficacy_score': 'Zelfeffectiviteit',
        'rec_asr_work_ability_score': 'Werkvermogen (WAI)',
        'rec_asr_burn_out_score': 'Burn-out risico',
        'rec_asr_vitality_score': 'Vitaliteit',
        'rec_asr_job_satisfaction_score': 'Werktevredenheid',
        'rec_asr_workload_score': 'Werkdruk',
        'rec_asr_exhaustion_score': 'Uitputting',
        'rec_ls_score_fruit': 'Fruit (stuks/dag)',
        'rec_ls_score_vegetables': 'Groenten (gram/dag)',
        'rec_ls_score_sugar': 'Suiker (gram/dag)',
        'rec_ls_score_saturated_fat': 'Verzadigd vet (gram/dag)',
        'rec_ls_score_alcohol': 'Alcohol (glazen/week)',
        'rec_ls_score_natrium': 'Natrium (mg/dag)',
        'rec_ls_score_exercise': 'Beweegminuten (min/week)',
    }
    fallback = {label: slug for slug, label in voorkeurslabels.items()}

    try:
        if df is None or df.empty or 'participant_id' not in df.columns:
            return dict(list(fallback.items())[:10])

        df_scores = df.copy()
        df_scores['participant_id'] = pd.to_numeric(df_scores['participant_id'], errors='coerce')
        df_scores = df_scores.dropna(subset=['participant_id'])
        df_scores = df_scores.drop_duplicates(subset='participant_id', keep='first')

        try:
            from visualisaties import _laad_opdrachtgever_links
            links = _laad_opdrachtgever_links(DB_URL)
        except Exception:
            links = pd.DataFrame()

        if not links.empty:
            df_scores = df_scores.drop(columns=[c for c in ['store_id', 'store_name', 'store_naam'] if c in df_scores.columns])
            df_scores = df_scores.merge(
                links[['participant_id', 'store_id', 'store_name']],
                on='participant_id',
                how='inner',
            )

        if df_scores.empty or 'store_id' not in df_scores.columns:
            return dict(list(fallback.items())[:10])
        
        indicator_stats = []
        
        for score_slug, label in voorkeurslabels.items():
            if score_slug not in df_scores.columns:
                continue
                
            df_indicator = df_scores[['participant_id', 'store_id', score_slug]].copy()
            df_indicator['score'] = pd.to_numeric(df_indicator[score_slug], errors='coerce')
            df_indicator = df_indicator.dropna(subset=['store_id', 'score'])
            per_store = df_indicator.groupby('store_id')['participant_id'].nunique()
            n_stores = int((per_store >= min_participants_per_store).sum())
            n_records = int(df_indicator['participant_id'].nunique())
            
            if n_stores >= min_stores:
                    indicator_stats.append({
                        'slug': score_slug,
                        'label': label,
                        'n_records': n_records,
                        'n_stores': n_stores
                    })
        
        # Sorteer op aantal records (meeste data eerst)
        indicator_stats.sort(key=lambda x: x['n_records'], reverse=True)
        
        # Return als dict (label -> slug)
        result = {item['label']: item['slug'] for item in indicator_stats}
        
        if not result:
            logger.warning("Geen opdrachtgever-indicatoren beschikbaar - fallback naar standaard")
            return dict(list(fallback.items())[:10])
        
        logger.info(f"✓ Beschikbare opdrachtgever indicatoren: {len(result)}")
        return result
        
    except Exception as e:
        logger.error(f"Error in _beschikbare_opdrachtgever_indicatoren: {e}")
        return dict(list(fallback.items())[:10])


def _beschikbare_longitudinale_factor_labels(df_data: pd.DataFrame) -> dict[str, str]:
    factor_labels = {
        'age': 'Leeftijd',
        'alcohol': 'Alcohol',
        'blood_pressure': 'Bloeddruk',
        'bmi': 'BMI',
        'competences': 'Competenties',
        'dass_anxiety': 'DASS Angst',
        'dass_depression': 'DASS Depressie',
        'dass_stress': 'DASS Stress',
        'diabetes': 'Diabetes',
        'exercise': 'Bewegen',
        'fat': 'Vet',
        'fruit': 'Fruit',
        'health': 'Gezondheid',
        'heredity': 'Erfelijkheid',
        'job_satisfaction': 'Werktevredenheid',
        'mental_complaints': 'Mentale klachten',
        'menopause_genitourinary': 'Menopauze (genitourinair)',
        'menopause_psychological': 'Menopauze (psychologisch)',
        'menopause_somatic': 'Menopauze (somatisch)',
        'nutrition': 'Voeding',
        'resilience': 'Veerkracht',
        'salt': 'Zout',
        'selfefficacy': 'Zelfeffectiviteit',
        'sleep': 'Slaap',
        'smoking': 'Roken',
        'steps': 'Stappen',
        'stress': 'Stress',
        'sugar': 'Suiker',
        'vegetables': 'Groenten',
        'wellbeing': 'Welzijn',
        'work_life_balance': 'Werk-privébalans',
        'working_attitude': 'Werkhouding',
        'workload': 'Werkdruk',
    } # Removed df_trend and used df_data
    slugs = sorted(df_data['slug'].dropna().astype(str).unique().tolist())
    opties = sorted(
        [(factor_labels.get(slug, _maak_net_label(slug)), slug) for slug in slugs],
        key=lambda x: x[0].lower()
    )
    return {label: slug for label, slug in opties}


OVERZICHTS_KPI_SPECS = {
    'Hoog hart risico': ('rec_heartrisk_cat', 2, 'eq'),
    'Hoog stress': ('rec_ls_stress_cat', 2, 'eq'),
    'Overgewicht': ('rec_med_bmi_cat', 1, 'ge'),
    'Slechte slaap': ('rec_ls_sleep_cat', 1, 'ge'),
    'Niet fysiek actief': ('derived_is_inactive', 1, 'eq'),
    'Roker': ('rec_smoking_answer', 3, 'eq'),
    'Diabetes': ('rec_med_diabetes_cat', 2, 'eq'),
    'Hoge bloeddruk': ('rec_med_blood_pressure_cat', 2, 'eq'),
}


def _bereken_kpi_telling(df: pd.DataFrame, kolom: str, waarde: float, op: str) -> tuple[int, int, float]:
    if df is None or df.empty or kolom not in df.columns: return 0, 0, 0.0
    s = pd.to_numeric(df[kolom], errors='coerce')
    n_geldig = int(s.notna().sum())
    if n_geldig == 0: return 0, 0, 0.0
    count = int(s.ge(waarde).sum() if op == 'ge' else s.eq(waarde).sum())
    return count, n_geldig, count / n_geldig * 100

@st.cache_data
def laad_html(pad: Path) -> str:
    with open(pad, 'r', encoding='utf-8') as f:
        return f.read()

@st.cache_data
def get_longitudinale_data_raw() -> pd.DataFrame:
    return laad_longitudinale_data(BASE, DB_URL)


def get_longitudinale_data() -> pd.DataFrame:
    return apply_global_filters(
        get_longitudinale_data_raw(),
        geslacht=st.session_state.get('global_geslacht', 'beide'),
        include_store=True,
        include_period=True,
        include_age=False,
        include_extra=False,
    )

@st.cache_data
def get_duplicaat_statistieken() -> pd.DataFrame:
     return bereken_duplicaat_statistieken(DB_URL)

@st.cache_data
def get_datakwaliteit(geslacht: str, n_totaal: int | None = None, filter_signature: tuple = ()) -> pd.DataFrame:
    df_actief = _active_participant_rows(load_main_data_filtered_by_gender_and_store(geslacht))
    return bereken_datakwaliteit(df_actief,
                                  n_totaal_override=n_totaal)

@st.cache_data
def get_outliers(geslacht: str, filter_signature: tuple = ()) -> pd.DataFrame:
    return bereken_outliers(load_main_data_filtered_by_gender_and_store(geslacht))

@st.cache_data
def get_t_toetsen(filter_signature: tuple = (), geselecteerde_variabelen: list[str] | None = None) -> pd.DataFrame:
    return bereken_t_toetsen(load_main_data_filtered_by_store(), geselecteerde_variabelen=geselecteerde_variabelen)


def _bereken_store_vs_totaal_t_toets(df: pd.DataFrame, store_id: int | None, kolommen: list[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty or store_id is None:
        return pd.DataFrame()

    df2 = df.copy()
    if 'store_id' not in df2.columns:
        return pd.DataFrame()

    df2['store_id'] = pd.to_numeric(df2['store_id'], errors='coerce')
    store_df = df2[df2['store_id'] == int(store_id)]
    totaal_df = df2
    if store_df.empty or totaal_df.empty:
        return pd.DataFrame()

    kandidaat_kolommen = kolommen or list({**LEEFSTIJL_SCORES, **RUWE_WAARDEN}.values())
    rijen = []
    for kolom in kandidaat_kolommen:
        if kolom not in df2.columns:
            continue
        a = pd.to_numeric(store_df[kolom], errors='coerce').dropna()
        b = pd.to_numeric(totaal_df[kolom], errors='coerce').dropna()
        b = b[~b.index.isin(a.index)]
        if len(a) < 10 or len(b) < 10:
            continue
        toets = stats.ttest_ind(a, b, equal_var=False)
        t_stat = cast(float, toets[0])
        p_val = cast(float, toets[1])
        pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
        cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0
        label = next((k for k, v in {**LEEFSTIJL_SCORES, **RUWE_WAARDEN}.items() if v == kolom), kolom)
        rijen.append({
            'Variabele': label,
            'Gemiddelde store': round(a.mean(), 2),
            'Gemiddelde totaal': round(b.mean(), 2),
            'Verschil': round(a.mean() - b.mean(), 2),
            'N store': len(a),
            'N totaal': len(b),
            't-statistiek': round(t_stat, 3),
            'p-waarde': round(p_val, 4),
            'Significant': 'Ja ✓' if p_val < 0.05 else 'Nee',
            "Cohen's d": round(cohens_d, 3),
        })
    return pd.DataFrame(rijen).sort_values('p-waarde').reset_index(drop=True)

@st.cache_data
def get_platform_groei(participant_ids: tuple[int, ...] = tuple()):
    return maak_platform_groei_plot(BASE, participant_ids=participant_ids or None)

@st.cache_data
def get_account_activatie(participant_ids: tuple[int, ...] = tuple()):
    return maak_account_activatie_plot(BASE, participant_ids=participant_ids or None)

@st.cache_data
def get_artikel_interacties_v2(user_ids: tuple[int, ...] = tuple()):
    return maak_artikel_interacties_overzicht(BASE, user_ids=user_ids or None)

@st.cache_data
def get_longitudinale_stores():
    return get_available_stores(BASE, DB_URL)

@st.cache_data
def get_store_scoreverbetering(score_type: str, maanden_window: int = 3, store_id: int | None = None):
    return maak_store_scoreverbetering_plot(BASE, DB_URL, score_type=score_type, maanden_window=maanden_window, store_id=store_id)

@st.cache_data
def get_store_average_stores():
    return get_available_stores_from_average_scores(BASE)

@st.cache_data
def get_store_average_stores_with_changes(score_slug: str, min_deelnemers: int = 1):
    return get_available_stores_with_score_changes(
        BASE, score_slug=score_slug, min_deelnemers=min_deelnemers
    )

@st.cache_data
def get_store_gemiddelde_scores(score_slug: str, score_label: str, min_deelnemers: int = 1,
                                store_id: int | None = None, richting_label: str = '', participant_ids: tuple[int, ...] = tuple()):
    participant_list = list(participant_ids) if participant_ids else None
    return maak_store_gemiddelde_scores_plot(
        BASE, score_slug=score_slug, score_label=score_label,
        min_deelnemers=min_deelnemers, store_id=store_id, participant_ids=participant_list
    )

@st.cache_data
def get_store_gemiddelde_verandering(score_slug: str, score_label: str, min_deelnemers: int = 1,
                                     store_id: int | None = None, richting_label: str = '', participant_ids: tuple[int, ...] = tuple()):
    participant_list = list(participant_ids) if participant_ids else None
    return maak_store_gemiddelde_verandering_plot(
        BASE, score_slug=score_slug, score_label=score_label,
        min_deelnemers=min_deelnemers, store_id=store_id, participant_ids=participant_list
    )

@st.cache_data
def get_scores_opdrachtgever(indicator: str, label: str, min_n: int, participant_ids: tuple[int, ...] = tuple()):
    participant_list = list(participant_ids) if participant_ids else None
    return maak_scores_per_opdrachtgever(BASE, DB_URL, indicator, label, min_n, participant_ids=participant_list)


@st.cache_data
def get_vragenlijst_data(participant_ids: tuple[int, ...] = tuple()):
    participant_list = list(participant_ids) if participant_ids else None
    return maak_vragenlijst_plot(BASE, DB_URL, participant_ids=participant_list)

@st.cache_data
def get_kopers_data(participant_ids: tuple[int, ...] = tuple()):
    participant_list = list(participant_ids) if participant_ids else None
    return maak_kopers_vergelijking_plot(BASE, DB_URL, participant_ids=participant_list)

@st.cache_data
def get_vragenlijst_gedrag(participant_ids: tuple[int, ...] = tuple()):
    participant_list = list(participant_ids) if participant_ids else None
    return maak_meerdere_vragenlijsten_plot(BASE, participant_ids=participant_list)

@st.cache_data
def get_vroege_kopers_profiel(participant_ids: tuple[int, ...] = tuple()):
    participant_list = list(participant_ids) if participant_ids else None
    return maak_vroege_kopers_profiel(BASE, DB_URL, participant_ids=participant_list)

@st.cache_data
def get_vroege_kopers_profiel_filtered(groepen: tuple[str, ...], include_wellbeing: bool, participant_ids: tuple[int, ...] = tuple()):
    participant_list = list(participant_ids) if participant_ids else None
    return maak_vroege_kopers_profiel(
        BASE, DB_URL, list(groepen), include_wellbeing=include_wellbeing,
        participant_ids=participant_list,
    )

@st.cache_data
def get_vragenlijst_herhalingen(participant_ids: tuple[int, ...] = tuple()):
    return analyse_vragenlijst_herhalingen(BASE, participant_ids=participant_ids or None)

@st.cache_data
def get_vragenlijst_dropoff(participant_ids: tuple[int, ...] = tuple()):
    return analyse_vragenlijst_dropoff(BASE, participant_ids=participant_ids or None)

@st.cache_data
def get_herhaalde_vragenlijst_scoreverandering(questionnaire_id: int, score_slug: str):
    return maak_herhaalde_vragenlijst_scoreverandering_plot(
        BASE, DB_URL, questionnaire_id, score_slug, lang=st.session_state.get('lang', 'nl')
    )

@st.cache_data
def get_vragenlijsten_overzicht():
    return haal_vragenlijsten_overzicht(DB_URL)

@st.cache_data
def get_scores_voor_vragenlijst(questionnaire_id: int):
    return tuple(haal_scores_voor_vragenlijst(DB_URL, questionnaire_id))

@st.cache_data
def get_account_dataflow_funnel(lang: str = 'nl'):
    return maak_account_dataflow_funnel_plot(BASE, DB_URL, lang=lang)

@st.cache_data
def get_engagement_opdrachtgever(
    gewichten_tuple: tuple,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
):
    # cache-versie 2: uitsplitsing nu ook met herhaalkopers, challenge-voltooiing
    # en recente app-activiteit
    gewichten = dict(gewichten_tuple)
    return bereken_engagement_per_opdrachtgever(
        BASE, DB_URL, gewichten, start_date=start_date, end_date=end_date,
    )

@st.cache_data
def get_engagement_trend(
    gewichten_tuple: tuple,
    min_deelnemers: int,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
):
    # cache-versie 2
    gewichten = dict(gewichten_tuple)
    return bereken_engagement_trend(
        BASE, DB_URL, gewichten, min_deelnemers,
        start_date=start_date, end_date=end_date,
    )


def _get_ml_cache_version() -> tuple[int, int]:
    ml_path = Path(__file__).resolve().parent / "ML.py"
    if not ml_path.exists():
        return (0, 0)
    stat = ml_path.stat()
    return (int(stat.st_mtime), int(stat.st_size))


@st.cache_data
def get_ml_models_v2(_cache_version: tuple[int, int], force_retrain_all: bool = False, model_name_to_retrain: str | None = None):
    """Train both ML models and return visualizations."""
    df = load_ml_data()
    res = {}

    # Train lifestyle model
    if train_lifestyle_model:
        try:
            force_ls = force_retrain_all or (model_name_to_retrain == 'lifestyle')
            ls_model, X_test_ls, y_test_ls, features_ls, y_train_ls = train_lifestyle_model(df, force_retrain=force_ls)
            y_pred_ls = predict_lifestyle(ls_model, X_test_ls)
            if y_pred_ls is None or y_test_ls is None:
                raise ValueError("Lifestyle model returned no predictions")
            ls_cv_metrics = getattr(ls_model, "cv_metrics", None) or {}

            # Bepaal of we classificatie of regressie metrics tonen
            is_clf = getattr(ml_module, 'LIFESTYLE_USE_CLASSIFICATION', False)

            ls_metrics = {}
            if is_clf:
                ls_metrics = {
                    'Accuracy': accuracy_score(y_test_ls, y_pred_ls),
                    'Recall Macro': recall_score(y_test_ls, y_pred_ls, average='macro'),
                    'CV Accuracy': float(ls_cv_metrics.get('accuracy_mean', 0.0)),
                    'CV Recall Macro': float(ls_cv_metrics.get('recall_macro_mean', 0.0)),
                }
            else:
                ls_metrics = {
                    'MAE': mean_absolute_error(y_test_ls, y_pred_ls),
                    'RMSE': np.sqrt(mean_squared_error(y_test_ls, y_pred_ls)),
                    'R² Score': r2_score(y_test_ls, y_pred_ls),
                    'CV R² Score': float(ls_cv_metrics.get('r2_mean', 0.0)),
                }
            
            res['lifestyle'] = {
                'model': ls_model,
                'X_test': X_test_ls,
                'y_test': y_test_ls,
                'y_pred': y_pred_ls,
                'features': features_ls,
                'y_train': y_train_ls,
                'metrics': ls_metrics
            }
        except Exception as e:
            res['lifestyle_error'] = str(e)

    # Train blood pressure model
    if train_bp_model:
        try:
            force_bp = force_retrain_all or (model_name_to_retrain == 'bp')
            bp_model, X_test_bp, y_test_bp, features_bp, y_train_bp = train_bp_model(df, force_retrain=force_bp)
            y_pred_bp, y_pred_bp_proba = predict_bp(bp_model, X_test_bp)
            if y_pred_bp is None or y_test_bp is None:
                raise ValueError("Blood pressure model returned no predictions")
            bp_cv_metrics = getattr(bp_model, "cv_metrics", None) or {}

            accuracy_bp = accuracy_score(y_test_bp, y_pred_bp)
            recall_bp = recall_score(y_test_bp, y_pred_bp, average='macro')

            res['bp'] = {
                'model': bp_model,
                'X_test': X_test_bp,
                'y_test': y_test_bp,
                'y_pred': y_pred_bp,
                'y_pred_proba': y_pred_bp_proba,
                'features': features_bp,
                'y_train': y_train_bp,
                'metrics': {
                    'Accuracy': accuracy_bp,
                    'Recall Macro': recall_bp,
                    'CV Accuracy': float(bp_cv_metrics.get('accuracy_mean', 0.0)),
                    'CV Recall Macro': float(bp_cv_metrics.get('recall_macro', 0.0)),
                    'Average Precision': float(bp_cv_metrics.get('holdout_average_precision', 0.0)),
                    'CV Average Precision': float(bp_cv_metrics.get('average_precision_mean', 0.0)),
                    'Brier': float(bp_cv_metrics.get('holdout_brier', 0.0)),
                    'CV Brier': float(bp_cv_metrics.get('brier_mean', 0.0)),
                    'Precision': float(bp_cv_metrics.get('holdout_precision', 0.0)),
                    'Recall': float(bp_cv_metrics.get('holdout_recall', 0.0)),
                    'Decision threshold': float(bp_cv_metrics.get('decision_threshold', 0.5)),
                }
            }
        except Exception as e:
            res['bp_error'] = str(e)

    # Train heart risk model
    if train_heartrisk_model:
        try:
            force_hr = force_retrain_all or (model_name_to_retrain == 'heartrisk')
            heartrisk_model, X_test_hr, y_test_hr, features_hr, y_train_hr = train_heartrisk_model(df, force_retrain=force_hr)
            y_pred_hr, y_pred_hr_proba = predict_heartrisk(heartrisk_model, X_test_hr)
            if y_pred_hr is None or y_test_hr is None:
                raise ValueError("Heart risk model returned no predictions")
            heartrisk_cv_metrics = getattr(heartrisk_model, "cv_metrics", None) or {}

            accuracy_hr = accuracy_score(y_test_hr, y_pred_hr)
            recall_hr = recall_score(y_test_hr, y_pred_hr, average='macro')
            recall_hr_moderate = recall_score(
                (y_test_hr == 1).astype(int), (y_pred_hr == 1).astype(int), zero_division=0,
            )
            recall_hr_high = recall_score(
                (y_test_hr == 2).astype(int), (y_pred_hr == 2).astype(int), zero_division=0,
            )

            res['heartrisk'] = {
                'model': heartrisk_model,
                'X_test': X_test_hr,
                'y_test': y_test_hr,
                'y_pred': y_pred_hr,
                'y_pred_proba': y_pred_hr_proba,
                'features': features_hr,
                'y_train': y_train_hr,
                'metrics': {
                    'Accuracy': accuracy_hr,
                    'Recall macro': recall_hr,
                    'Recall matig risico': recall_hr_moderate,
                    'Recall hoog risico': recall_hr_high,
                    'CV Recall matig risico': float(heartrisk_cv_metrics.get('recall_moderate_mean', 0.0) or 0.0),
                    'CV Recall matig risico std': float(heartrisk_cv_metrics.get('recall_moderate_std', 0.0) or 0.0),
                    'CV Recall hoog risico': float(heartrisk_cv_metrics.get('recall_high_mean', 0.0) or 0.0),
                    'CV Recall hoog risico std': float(heartrisk_cv_metrics.get('recall_high_std', 0.0) or 0.0),
                    'CV Recall macro': float(heartrisk_cv_metrics.get('recall_macro_mean', 0.0) or 0.0),
                    'CV Recall macro std': float(heartrisk_cv_metrics.get('recall_macro_std', 0.0) or 0.0),
                    'Brier hoge risico': float(heartrisk_cv_metrics.get('holdout_brier_high', 0.0) or 0.0),
                }
            }
        except Exception as e:
            res['heartrisk_error'] = str(e)

    # Train next lifestyle score model
    if train_next_lifestyle_model:
        try:
            force_next_lifestyle = force_retrain_all or (model_name_to_retrain == 'next_lifestyle')
            next_model, X_test_next, y_test_next, features_next, y_train_next = train_next_lifestyle_model(
                df,
                force_retrain=force_next_lifestyle,
            )
            y_pred_next = np.clip(next_model.predict(X_test_next), 0.0, 10.0)
            next_cv_metrics = getattr(next_model, "cv_metrics", None) or {}
            res['next_lifestyle'] = {
                'model': next_model,
                'X_test': X_test_next,
                'y_test': y_test_next,
                'y_pred': y_pred_next,
                'features': features_next,
                'y_train': y_train_next,
                'metrics': {
                    'MAE': mean_absolute_error(y_test_next, y_pred_next),
                    'RMSE': np.sqrt(mean_squared_error(y_test_next, y_pred_next)),
                    'R² Score': r2_score(y_test_next, y_pred_next),
                    'CV R² Score': float(next_cv_metrics.get('r2_mean', 0.0)),
                    'CV MAE': float(next_cv_metrics.get('mae_mean', 0.0)),
                    'Baseline MAE': float(next_cv_metrics.get('baseline_mae', 0.0)),
                    'Aantal testmetingen': int(next_cv_metrics.get('n_test', 0)),
                },
            }
        except Exception as e:
            res['next_lifestyle_error'] = str(e)

    # Train drop-off / activation model
    if train_dropoff_model:
        try:
            force_dropoff = force_retrain_all or (model_name_to_retrain == 'dropoff')
            dropoff_model, X_test_dropoff, y_test_dropoff, features_dropoff, y_train_dropoff = train_dropoff_model(df, force_retrain=force_dropoff)
            y_pred_dropoff, y_pred_dropoff_proba = predict_binary_model(dropoff_model, X_test_dropoff)
            if y_pred_dropoff is None or y_test_dropoff is None:
                raise ValueError("Dropoff model returned no predictions")
            dropoff_cv_metrics = getattr(dropoff_model, "cv_metrics", None) or {}
            res['dropoff'] = {
                'model': dropoff_model,
                'X_test': X_test_dropoff,
                'y_test': y_test_dropoff,
                'y_pred': y_pred_dropoff,
                'y_pred_proba': y_pred_dropoff_proba,
                'features': features_dropoff,
                'y_train': y_train_dropoff,
                'metrics': {
                    'Accuracy': accuracy_score(y_test_dropoff, y_pred_dropoff),
                    'Recall Macro': recall_score(y_test_dropoff, y_pred_dropoff, average='macro', zero_division=0),
                    'CV Accuracy': float(dropoff_cv_metrics.get('accuracy_mean', 0.0)),
                    'CV Recall Macro': float(dropoff_cv_metrics.get('recall_macro_mean', 0.0)),
                }
            }
        except Exception as e:
            res['dropoff_error'] = str(e)

    # Train lifestyle improvement model
    if train_improvement_model:
        try:
            force_improvement = force_retrain_all or (model_name_to_retrain == 'improvement')
            improvement_model, X_test_improvement, y_test_improvement, features_improvement, y_train_improvement = train_improvement_model(df, force_retrain=force_improvement)
            y_pred_improvement = improvement_model.predict(X_test_improvement)
            if y_pred_improvement is None or y_test_improvement is None:
                raise ValueError("Improvement model returned no predictions")
            improvement_cv_metrics = getattr(improvement_model, "cv_metrics", None) or {}
            res['improvement'] = {
                'model': improvement_model,
                'X_test': X_test_improvement,
                'y_test': y_test_improvement,
                'y_pred': y_pred_improvement,
                'features': features_improvement,
                'y_train': y_train_improvement,
                'metrics': {
                    'MAE': mean_absolute_error(y_test_improvement, y_pred_improvement),
                    'RMSE': np.sqrt(mean_squared_error(y_test_improvement, y_pred_improvement)),
                    'R² Score': r2_score(y_test_improvement, y_pred_improvement),
                    'CV R² Score': float(improvement_cv_metrics.get('r2_mean', 0.0)),
                    'CV MAE': float(improvement_cv_metrics.get('mae_mean', 0.0)),
                    'Baseline MAE': float(improvement_cv_metrics.get('holdout_baseline_mae', 0.0)),
                }
            }
        except Exception as e:
            res['improvement_error'] = str(e)

    # Train purchase / product propensity model
    if train_purchase_model:
        try:
            force_purchase = force_retrain_all or (model_name_to_retrain == 'purchase')
            purchase_model, X_test_purchase, y_test_purchase, features_purchase, y_train_purchase = train_purchase_model(df, force_retrain=force_purchase)
            y_pred_purchase, y_pred_purchase_proba = predict_binary_model(purchase_model, X_test_purchase)
            if y_pred_purchase is None or y_test_purchase is None:
                raise ValueError("Purchase model returned no predictions")
            purchase_cv_metrics = getattr(purchase_model, "cv_metrics", None) or {}
            res['purchase'] = {
                'model': purchase_model,
                'X_test': X_test_purchase,
                'y_test': y_test_purchase,
                'y_pred': y_pred_purchase,
                'y_pred_proba': y_pred_purchase_proba,
                'features': features_purchase,
                'y_train': y_train_purchase,
                'metrics': {
                    'Accuracy': accuracy_score(y_test_purchase, y_pred_purchase),
                    'Recall Macro': recall_score(y_test_purchase, y_pred_purchase, average='macro', zero_division=0),
                    'CV Accuracy': float(purchase_cv_metrics.get('accuracy_mean', 0.0)),
                    'CV Recall Macro': float(purchase_cv_metrics.get('recall_macro_mean', 0.0)),
                    'CV Average Precision': float(purchase_cv_metrics.get('average_precision_mean', 0.0)),
                    'CV Brier': float(purchase_cv_metrics.get('brier_mean', 0.0)),
                    'Holdout Average Precision': float(purchase_cv_metrics.get('holdout_average_precision', 0.0)),
                    'Holdout Brier': float(purchase_cv_metrics.get('holdout_brier', 0.0)),
                    'Precision': float(purchase_cv_metrics.get('holdout_precision', 0.0)),
                    'Recall': float(purchase_cv_metrics.get('holdout_recall', 0.0)),
                    'Decision threshold': float(purchase_cv_metrics.get('decision_threshold', 0.5)),
                }
            }
        except Exception as e:
            res['purchase_error'] = str(e)
    return res


# ── Helpers ───────────────────────────────────────────────────────────────────
def store_badge(store_id: int | None) -> None:
    if store_id is None:
        return
    lang = st.session_state.get('lang', 'nl')
    store_name = get_store_name(store_id) or str(store_id)
    st.info(tr("🔵 Gefilterd op opdrachtgever: {store}", lang, store=store_name))


def geslacht_badge(geslacht: str) -> None:
    """Toont een info banner als er gefilterd wordt op geslacht."""
    lang = st.session_state.get('lang', 'nl')
    global_geslacht = st.session_state.get('global_geslacht', 'beide')
    if global_geslacht == 'man':
        label = tr('mannen', lang)
        st.info(tr("🔵 Gefilterd op: alleen {label}", lang, label=label))
    elif global_geslacht == 'vrouw':
        label = tr('vrouwen', lang)
        st.info(tr("🔵 Gefilterd op: alleen {label}", lang, label=label))

    store_badge(st.session_state.get('global_store_id'))


def t(text: str, **kwargs) -> str: return tr(text, st.session_state.get('lang', 'nl'), **kwargs)

def _aantal_deelnemers(df: pd.DataFrame | None) -> int:
    if df is None or df.empty:
        return 0
    for kolom in ('participant_id', 'user_id', 'id'):
        if kolom in df.columns:
            return int(pd.to_numeric(df[kolom], errors='coerce').dropna().nunique())
    return len(df)


def _heeft_genoeg_deelnemers(df: pd.DataFrame | None, min_deelnemers: int = MIN_VISUALISATIE_DEELNEMERS) -> bool:
    n = _aantal_deelnemers(df)
    if n < min_deelnemers:
        st.info(t(
            "Visualisatie niet beschikbaar: minimaal {min_n} deelnemers nodig, huidige selectie bevat {n}.",
            min_n=min_deelnemers,
            n=n,
        ))
        return False
    return True


def p(fig, key: str | None = None, participants_df: pd.DataFrame | None = None,
      min_deelnemers: int = MIN_VISUALISATIE_DEELNEMERS, show_reset_scale: bool = True) -> None:
    if participants_df is not None and not _heeft_genoeg_deelnemers(participants_df, min_deelnemers):
        return
    if fig is None:
        st.warning(t("Visualisatie kon niet worden gegenereerd wegens ontbrekende data."))
        return
    try:
        fig_to_render = translate_plotly_figure(fig, st.session_state.get('lang', 'nl'))
        config: dict[str, object] = {
            'displayModeBar': True,
            'responsive': True,
        }
        if show_reset_scale:
            config['modeBarButtonsToAdd'] = ['resetScale2d']
        st.plotly_chart(
            fig_to_render,
            width="stretch",
            key=key,
            config=config,
        )
    except Exception as e:
        logger.warning(f"Plot rendering failed: {e}")
        st.warning(t("De visualisatie kon niet worden getoond. Controleer de data of probeer een andere filter."))


def m(fig, participants_df: pd.DataFrame | None = None,
      min_deelnemers: int = MIN_VISUALISATIE_DEELNEMERS) -> None:
    if participants_df is not None and not _heeft_genoeg_deelnemers(participants_df, min_deelnemers):
        return
    st.pyplot(translate_matplotlib_figure(fig, st.session_state.get('lang', 'nl')))

KAART_PRESETS = {
    'Nederland':   {'center': [52.1,    5.3],     'zoom': 7},
    'Rotterdam':   {'center': [51.9163, 4.4892],  'zoom': 10},
    'Leusden':     {'center': [52.1335, 5.4257],  'zoom': 10},
    'Tegelen':     {'center': [51.3411, 6.1467],  'zoom': 10},
    'Amsterdam':   {'center': [52.3604, 4.9227],  'zoom': 10},
}

def toon_kaart(bestandsnaam: str, hoogte: int = 580) -> None:
    pad = HTML_MAP / bestandsnaam
    if not pad.exists():
        st.warning(t("Kaart niet gevonden: {pad}. Draai eerst kaarten.py.", pad=pad))
        return
    preset = KAART_PRESETS.get(
        st.session_state.get('kaart_preset', 'Nederland'),
        KAART_PRESETS['Nederland']
    )
    zoom_script = f"""
    <script>
    (function() {{
        var c = {preset['center']}, z = {preset['zoom']};
        var t = setInterval(function() {{
            var el = document.querySelector('.folium-map');
            if (!el || !window[el.id]) return;
            clearInterval(t);
            window[el.id].setView(c, z, {{animate: false}});
        }}, 150);
    }})();
    </script>"""
    html = tr(laad_html(pad), st.session_state.get('lang', 'nl'))
    components.html(html + zoom_script, height=hoogte, scrolling=False)


def _modus(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return pd.NA
    mode = s.mode()
    return mode.iloc[0] if not mode.empty else pd.NA


def _dynamische_pc4_kaart(df: pd.DataFrame, kaart_key: str, titel: str, lang: str = 'nl') -> go.Figure | None:
    if df is None or df.empty or 'postal_code' not in df.columns:
        return None

    pc4_pad = CODE_DIR / 'pc4.geojson'
    if not pc4_pad.exists():
        return None
    with open(pc4_pad, 'r', encoding='utf-8') as f:
        geojson_pc4 = json.load(f)

    df_map = df.copy()
    df_map['pc4'] = df_map['postal_code'].astype(str).str.extract(r'(\d{4})')[0]
    df_map = df_map[df_map['pc4'].notna()].copy()
    if df_map.empty:
        return None

    specs = {
        'gebruikersdichtheid_kaart_pc3.html': {
            'type': 'count', 'label': tr('Deelnemers', lang),
            'color_scale': 'YlOrRd',
        },
        'heartrisk_categorie_kaart.html': {
            'type': 'category', 'column': 'rec_heartrisk_cat', 'label': tr('Cardiovasculair risico', lang),
            'labels': HEARTRISK_LABELS, 'colors': RISICO_COLORS,
        },
        'bmi_categorie_kaart.html': {
            'type': 'category', 'column': 'rec_med_bmi_cat', 'label': tr('BMI categorie', lang),
            'labels': BMI_LABELS, 'colors': BMI_COLORS,
        },
        'stress_categorie_kaart.html': {
            'type': 'category', 'column': 'rec_ls_stress_cat', 'label': tr('Stress categorie', lang),
            'labels': STRESS_LABELS, 'colors': RISICO_COLORS,
        },
        'slaap_categorie_kaart.html': {
            'type': 'category', 'column': 'rec_ls_sleep_cat', 'label': tr('Slaapkwaliteit', lang),
            'labels': categorie_labels('sleep'), 'colors': RISICO_COLORS,
        },
        'leeftijd_categorie_kaart.html': {
            'type': 'age', 'label': tr('Leeftijdscategorie', lang),
            'labels': {0: '<30', 1: '30-40', 2: '40-50', 3: '50-60', 4: '60+'},
            'colors': {0: '#1F6FBF', 1: '#2ECC71', 2: '#F4D03F', 3: '#E87722', 4: '#E74C3C'},
        },
        'bmi_mediaan_kaart.html': {
            'type': 'median', 'column': 'rec_med_bmi', 'label': 'BMI', 'color_scale': 'YlOrRd',
        },
        'stress_mediaan_kaart.html': {
            'type': 'median', 'column': 'rec_ls_stress_sum', 'label': tr('Stressscore', lang), 'color_scale': 'YlOrRd',
        },
        'beweging_mediaan_kaart.html': {
            'type': 'median', 'column': 'rec_ls_exercise_steps_per_day', 'label': tr('Stappen per dag', lang), 'color_scale': 'YlGn',
        },
        'beweging_categorie_kaart.html': {
            'type': 'steps_category', 'label': tr('Beweging', lang),
            'labels': {0: 'Weinig (<5.000)', 1: 'Matig (5.000-10.000)', 2: 'Veel (>10.000)'},
            'colors': {0: '#E74C3C', 1: '#E87722', 2: '#2ECC71'},
        },
        'werkvermogen_categorie_kaart.html': {
            'type': 'category', 'column': 'rec_asr_work_ability_cat', 'label': tr('Werkvermogen', lang),
            'labels': {0: 'Laag', 1: 'Matig', 2: 'Goed'},
            'colors': {0: RISICO_COLORS[2], 1: RISICO_COLORS[1], 2: RISICO_COLORS[0]},
        },
    }
    spec = specs.get(kaart_key)
    if spec is None:
        return None

    if spec['type'] == 'count':
        agg = df_map.groupby('pc4').size().reset_index(name='waarde')
        agg['tooltip'] = agg['waarde'].astype(int).astype(str)
        fig = px.choropleth_mapbox(
            agg, geojson=geojson_pc4, locations='pc4', featureidkey='properties.postcode',
            color='waarde', color_continuous_scale=spec['color_scale'],
            mapbox_style='open-street-map', zoom=6, center={'lat': 52.1, 'lon': 5.3},
            labels={'waarde': spec['label']}, hover_data={'pc4': True, 'waarde': True},
        )
    else:
        work = df_map.copy()
        if spec['type'] == 'age':
            age = _numeric_column(work, 'rec_age_current')
            work['_cat'] = pd.cut(age, bins=[0, 30, 40, 50, 60, float('inf')], labels=[0, 1, 2, 3, 4], right=False)
        elif spec['type'] == 'steps_category':
            steps = _numeric_column(work, 'rec_ls_exercise_steps_per_day')
            steps = np.where(steps < 1000, steps * 1000, steps)
            work['_cat'] = pd.cut(steps, bins=[-0.1, 5000, 10000, float('inf')], labels=[0, 1, 2], right=False)
        elif spec['type'] == 'median':
            if spec['column'] not in work.columns:
                return None
            work['_value'] = pd.to_numeric(work[spec['column']], errors='coerce')
            agg = work.dropna(subset=['_value']).groupby('pc4').agg(
                waarde=('_value', 'median'),
                deelnemers=('_value', 'count'),
            ).reset_index()
            if agg.empty:
                return None
            fig = px.choropleth_mapbox(
                agg, geojson=geojson_pc4, locations='pc4', featureidkey='properties.postcode',
                color='waarde', color_continuous_scale=spec['color_scale'],
                mapbox_style='open-street-map', zoom=6, center={'lat': 52.1, 'lon': 5.3},
                labels={'waarde': spec['label']},
                hover_data={'pc4': True, 'waarde': ':.2f', 'deelnemers': True},
            )
            fig.update_layout(title=titel, margin=dict(l=0, r=0, t=45, b=0), height=580)
            return fig
        else:
            if spec['column'] not in work.columns:
                return None
            work['_cat'] = pd.to_numeric(work[spec['column']], errors='coerce').round()

        labels = spec['labels']
        colors = spec['colors']
        valid_codes = set(labels.keys())
        work = work[pd.to_numeric(work['_cat'], errors='coerce').isin(valid_codes)].copy()
        if work.empty:
            return None
        agg = work.groupby('pc4').agg(
            code=('_cat', _modus),
            deelnemers=('_cat', 'count'),
        ).reset_index()
        agg['code'] = pd.to_numeric(agg['code'], errors='coerce').astype(int)
        agg['categorie'] = agg['code'].map(labels)
        color_map = {labels[k]: colors.get(k, '#999999') for k in valid_codes}
        fig = px.choropleth_mapbox(
            agg, geojson=geojson_pc4, locations='pc4', featureidkey='properties.postcode',
            color='categorie', color_discrete_map=color_map,
            mapbox_style='open-street-map', zoom=6, center={'lat': 52.1, 'lon': 5.3},
            labels={'categorie': spec['label']},
            hover_data={'pc4': True, 'categorie': True, 'deelnemers': True},
        )

    fig.update_layout(title=titel, margin=dict(l=0, r=0, t=45, b=0), height=580)
    return fig


def toon_dynamische_kaart(df: pd.DataFrame, bestandsnaam: str, titel: str, lang: str = 'nl') -> bool:
    if bestandsnaam == 'internationale_gebruikers_kaart.html':
        lat_col = 'lat' if 'lat' in df.columns else 'latitude' if 'latitude' in df.columns else None
        lon_col = 'long' if 'long' in df.columns else 'lon' if 'lon' in df.columns else 'longitude' if 'longitude' in df.columns else None
        if lat_col and lon_col:
            df_geo = df.copy()
            df_geo[lat_col] = pd.to_numeric(df_geo[lat_col], errors='coerce')
            df_geo[lon_col] = pd.to_numeric(df_geo[lon_col], errors='coerce')
            df_geo = df_geo.dropna(subset=[lat_col, lon_col])
            if not df_geo.empty:
                fig = px.scatter_mapbox(
                    df_geo, lat=lat_col, lon=lon_col, zoom=2,
                    mapbox_style='open-street-map',
                    hover_data=[c for c in ['participant_id', 'postal_code'] if c in df_geo.columns],
                    title=titel,
                )
                fig.update_layout(margin=dict(l=0, r=0, t=45, b=0), height=580)
                p(fig, participants_df=df)
                return True
        return False

    fig = _dynamische_pc4_kaart(df, bestandsnaam, titel, lang=lang)
    if fig is None:
        return False
    p(fig, participants_df=df)
    return True


def reset_global_filters():
    st.session_state.pop("global_period_preset", None)
    st.session_state.pop("global_period_custom", None)
    st.session_state.pop("global_age_categories", None)
    st.session_state.pop("global_store_select", None)
    st.session_state["global_store_id"] = None
    st.session_state.pop("global_department_values", None)
    st.session_state.pop("global_function_values", None)
    st.session_state.pop("global_gender_radio", None)
    st.session_state["global_geslacht"] = "totaal"


def _opties_voor_tekstfilter(df: pd.DataFrame, kolom: str | None) -> list[str]:
    if df is None or df.empty or not kolom or kolom not in df.columns:
        return []
    waarden = (
        df[kolom]
        .fillna("Onbekend")
        .astype(str)
        .str.strip()
        .replace("", "Onbekend")
    )
    opties = waarden.dropna().unique().tolist()
    return sorted(opties, key=lambda waarde: waarde.lower())


def _multiselect_met_state(label: str, opties: list[str], default: list[str], key: str, **kwargs):
    huidige = st.session_state.get(key)
    if huidige is not None:
        opgeschoond = [optie for optie in huidige if optie in opties]
        if opgeschoond != huidige:
            st.session_state[key] = opgeschoond
        return st.multiselect(label, opties, key=key, **kwargs)
    return st.multiselect(label, opties, default=default, key=key, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# ─ LAZY LOADING & CACHING (Stap 9) ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_SECONDS['main_data'])
def lazy_load_analytics_summary(df: pd.DataFrame) -> dict:
    """
    Compute analytics summary (snel, voor KPI's).
    Cache: 5 minuten.
    """
    if df.empty:
        return {"error": "Geen data"}
    
    lang = 'nl'
    result = {
        "n_deelnemers": len(df),
        "n_met_postcode": df['postal_code'].notna().sum() if 'postal_code' in df.columns else 0,
        "n_hoog_risico": (_bereken_kpi_telling(df, *OVERZICHTS_KPI_SPECS['Hoog hart risico'])[0] 
                         if 'Hoog hart risico' in OVERZICHTS_KPI_SPECS else 0),
        "pct_hoog_risico": (_bereken_kpi_telling(df, *OVERZICHTS_KPI_SPECS['Hoog hart risico'])[2] 
                           if 'Hoog hart risico' in OVERZICHTS_KPI_SPECS else 0),
    }
    return result


@st.cache_data(ttl=CACHE_TTL_SECONDS['statistics'])
def lazy_load_demographic_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load demographic breakdown (geslacht, leeftijd, etc).
    Cache: 30 minuten.
    """
    if df.empty or 'rec_user_gender' not in df.columns:
        return pd.DataFrame()

    gender_dist = df['rec_user_gender'].value_counts(normalize=True) * 100
    return gender_dist.round(1).to_frame(name='percentage')


@st.cache_data(ttl=CACHE_TTL_SECONDS['statistics'])
def lazy_load_risk_breakdown(df: pd.DataFrame) -> dict:
    """
    Load risk indicators (hart risico, BMI, stress).
    Cache: 30 minuten.
    """
    if df.empty:
        return {}
    
    result = {}
    
    # Cardiovasculair risico
    if 'rec_heartrisk_cat' in df.columns:
        hr_dist = pd.to_numeric(df['rec_heartrisk_cat'], errors='coerce').value_counts(normalize=True) * 100
        result['heartrisk'] = hr_dist.round(1).to_dict()
    
    # BMI
    if 'rec_med_bmi_cat' in df.columns:
        bmi_dist = pd.to_numeric(df['rec_med_bmi_cat'], errors='coerce').value_counts(normalize=True) * 100
        result['bmi'] = bmi_dist.round(1).to_dict()
    
    # Stress
    if 'rec_ls_stress_cat' in df.columns:
        stress_dist = pd.to_numeric(df['rec_ls_stress_cat'], errors='coerce').value_counts(normalize=True) * 100
        result['stress'] = stress_dist.round(1).to_dict()
    
    return result


# ─ CONTEXT CARDS (Stap 3) ──────────────────────────────────────────────────────

def maak_context_card(titel: str, waarde: str, context: str = "", trend: str = "", 
                      indicator: str = "ℹ️", kleur: str = "#1f77b4", actionable: str = ""):
    """
    Maak een context card met bedrijfslogica inzicht.
    
    Args:
        titel: Titel van de card
        waarde: Hoofd-gegeven (bv. "42%", "1.234")
        context: Achtergrond info (bv. "vs. 38% vorig kwartaal")
        trend: Trend indicator (bv. "↑ +4%" of "→ Stabiel")
        indicator: Emoji/symbool
        kleur: Hex kleur
        actionable: Aanbeveling/actie (bv. "Zie meer")
    """
    html = f"""
    <div style="background: linear-gradient(135deg, {kleur}22 0%, {kleur}11 100%);
                border-left: 4px solid {kleur};
                padding: 16px;
                border-radius: 8px;
                margin: 8px 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <div style="font-size: 12px; color: #666; font-weight: 500; margin-bottom: 4px;">
                    {indicator} {titel}
                </div>
                <div style="font-size: 32px; font-weight: 700; color: {kleur}; margin-bottom: 8px;">
                    {waarde}
                </div>
                {f'<div style="font-size: 12px; color: #888;">{context}</div>' if context else ''}
            </div>
            <div style="text-align: right; font-size: 14px;">
                {f'<div style="color: {kleur}; font-weight: 600;">{trend}</div>' if trend else ''}
                {f'<div style="font-size: 11px; color: #0066cc; margin-top: 8px;">{actionable}</div>' if actionable else ''}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Smart Health")
    auth_user = st.session_state.get("auth_user")
    if auth_user:
        st.caption(f"Ingelogd als: {auth_user}")
        if st.button("Uitloggen"):
            logout()
            st.rerun()
    st.markdown("---")

    lang = 'nl'
    st.session_state['lang'] = lang

    if st.button(tr("Reset filters", lang), width="stretch"):
        reset_global_filters()
        st.rerun()

    st.markdown("---")
    st.markdown(f"**{tr('Globale filters', lang)}**")
    st.caption(tr("Deze filters gelden voor alle dashboardpagina's.", lang))

    if "global_geslacht" not in st.session_state:
        st.session_state["global_geslacht"] = "totaal"

    st.markdown(f"**{tr('Geslacht filter', lang)}**")
    geslacht_opties = [
        tr("Totaal", lang),
        tr("Vergelijking man vs vrouw", lang),
        tr("Alleen mannen", lang),
        tr("Alleen vrouwen", lang),
    ]
    huidige_geslacht_keuze = st.session_state.get("global_gender_radio")
    if huidige_geslacht_keuze is None or huidige_geslacht_keuze not in geslacht_opties:
        huidige_geslacht_keuze = tr("Totaal", lang)
    geslacht_keuze = st.radio(
        tr("Geslacht filter", lang),
        geslacht_opties,
        label_visibility="collapsed",
        horizontal=True,
        key="global_gender_radio",
        index=geslacht_opties.index(huidige_geslacht_keuze),
    )
    geslacht = {
        tr("Totaal", lang): "totaal",
        tr("Vergelijking man vs vrouw", lang): "beide",
        tr("Alleen mannen", lang): "man",
        tr("Alleen vrouwen", lang): "vrouw",
    }[geslacht_keuze]
    st.session_state['global_geslacht'] = geslacht

    st.markdown(f"**{tr('Opdrachtgever filter', lang)}**")
    available_stores = get_available_stores_dashboard()
    store_options = [(f"🌍 {t('Alle opdrachtgevers')}", None)]
    if not available_stores.empty:
        store_options += [
            (str(row['store_name']), int(row['store_id']))
            for _, row in available_stores.iterrows()
            if pd.notna(row['store_id'])
        ]

    selected_store_option = st.selectbox(
        tr('Opdrachtgever filter', lang),
        store_options,
        format_func=lambda x: x[0] if isinstance(x, tuple) else x,
        key='global_store_select',
    )
    st.session_state['global_store_id'] = selected_store_option[1] if isinstance(selected_store_option, tuple) else None

    df_filter_basis = filter_store(filter_geslacht(load_main_data(), geslacht), get_store_filter_id())
    min_datum, max_datum, datumkolom = _datum_bereik(df_filter_basis)

    st.markdown(f"**{tr('Periode filter', lang)}**")
    periode_labels = list(PERIODE_PRESETS.keys())
    st.selectbox(
        tr("Periode", lang),
        periode_labels,
        index=periode_labels.index(st.session_state.get("global_period_preset", "Sinds start"))
        if st.session_state.get("global_period_preset", "Sinds start") in periode_labels else 2,
        key="global_period_preset",
        label_visibility="collapsed",
    )
    if st.session_state.get("global_period_preset") == "Custom":
        if min_datum is not None and max_datum is not None:
            st.date_input(
                tr("Custom datumselectie", lang),
                value=(min_datum.date(), max_datum.date()),
                min_value=min_datum.date(),
                max_value=max_datum.date(),
                key="global_period_custom",
            )
        else:
            st.info(tr("Geen datumkolom beschikbaar voor deze selectie.", lang))
    elif datumkolom:
        start_datum, eind_datum = _periode_selectie(min_datum, max_datum)
        if start_datum is not None and eind_datum is not None:
            st.caption(f"{start_datum.date()} t/m {eind_datum.date()} ({datumkolom})")
            st.caption(tr("De periodefilter werkt op de beschikbare datumkolom van de gefilterde selectie.", lang))
    else:
        st.caption(tr("Geen datumkolom beschikbaar.", lang))

    with st.expander(tr("Extra filters", lang), expanded=False):
        st.markdown(f"**{tr('Leeftijdscategorie', lang)}**")
        leeftijd_opties = LEEFTIJD_CATEGORIEEN.copy()
        if "rec_age_current" in df_filter_basis.columns and _leeftijd_categorieen(df_filter_basis).eq("Onbekend").any():
            leeftijd_opties.append("Onbekend")
        huidige_leeftijd = st.session_state.get("global_age_categories", leeftijd_opties)
        huidige_leeftijd = [optie for optie in huidige_leeftijd if optie in leeftijd_opties]
        _multiselect_met_state(
            tr("Leeftijdscategorie", lang),
            leeftijd_opties,
            huidige_leeftijd,
            key="global_age_categories",
            label_visibility="collapsed",
        )

        afdeling_kolom = _eerste_bestaande_kolom(df_filter_basis, AFDELING_KOLOMMEN)
        functie_kolom = _eerste_bestaande_kolom(df_filter_basis, FUNCTIE_KOLOMMEN)
        st.session_state["global_department_column"] = afdeling_kolom
        st.session_state["global_function_column"] = functie_kolom

        if afdeling_kolom:
            afdeling_opties = _opties_voor_tekstfilter(df_filter_basis, afdeling_kolom)
            huidige_afdeling = st.session_state.get("global_department_values", afdeling_opties)
            huidige_afdeling = [optie for optie in huidige_afdeling if optie in afdeling_opties]
            _multiselect_met_state(
                tr("Afdeling", lang),
                afdeling_opties,
                huidige_afdeling,
                key="global_department_values",
            )

        if functie_kolom:
            functie_opties = _opties_voor_tekstfilter(df_filter_basis, functie_kolom)
            huidige_functie = st.session_state.get("global_function_values", functie_opties)
            huidige_functie = [optie for optie in huidige_functie if optie in functie_opties]
            _multiselect_met_state(
                tr("Functie", lang),
                functie_opties,
                huidige_functie,
                key="global_function_values",
            )

    df_filter_preview = apply_global_filters(load_main_data(), geslacht=geslacht)
    if 0 < len(df_filter_preview) < 10:
        st.warning(tr("Let op: de gefilterde dataset bevat minder dan 10 rijen. Interpreteer resultaten terughoudend.", lang))
    st.caption(tr("Gefilterde rijen: {n}", lang, n=f"{len(df_filter_preview):,}"))

    st.markdown("---")

    pagina_opties = ["🏠 Overzicht", "📊 Looker Studio", "🗺️ Kaarten", "📊 Variabelen verkenner", "🥗 Dieet & BMI", "📈 Longitudinale analyse", "🏢 Per opdrachtgever", "🛒 Producteffect", "📋 Vragenlijstgedrag", "🛍️ Kopersprofiel", "🔎 Snelle statistieken", "🔍 Datakwaliteit", "📉 Outlier analyse", "⚖️ T-toets man vs vrouw", "🔗 Correlaties", "🤖 ML Model"]
    # De Testpagina (experimentele visualisaties) is alleen zichtbaar voor admins.
    if is_admin():
        pagina_opties.append("🧪 Test")

    visualisatie_paden = [
        {'pad': ['Overzicht', 'Demografisch'], 'pagina': '🏠 Overzicht'},
        {'pad': ['Overzicht', 'Cardiovasculair risico'], 'pagina': '🏠 Overzicht'},
        {'pad': ['Overzicht', 'BMI'], 'pagina': '🏠 Overzicht'},
        {'pad': ['Overzicht', 'Stress'], 'pagina': '🏠 Overzicht'},
        {'pad': ['Overzicht', 'Leefstijl'], 'pagina': '🏠 Overzicht'},
        {'pad': ['Overzicht', 'Platformgroei'], 'pagina': '🏠 Overzicht'},
        {'pad': ['Overzicht', 'Artikelgebruik'], 'pagina': '🏠 Overzicht'},
        {'pad': ['Looker Studio'], 'pagina': '📊 Looker Studio'},
        {'pad': ['Kaarten'], 'pagina': '🗺️ Kaarten'},
        {'pad': ['Variabelen verkenner'], 'pagina': '📊 Variabelen verkenner'},
        {'pad': ['Dieet & BMI', 'Verdeling dieetkwaliteit'], 'pagina': '🥗 Dieet & BMI'},
        {'pad': ['Dieet & BMI', 'Dieet vs BMI'], 'pagina': '🥗 Dieet & BMI'},
        {'pad': ['Dieet & BMI', 'Dieet naar geslacht'], 'pagina': '🥗 Dieet & BMI'},
        {'pad': ['Dieet & BMI', 'Internationale vergelijking'], 'pagina': '🥗 Dieet & BMI'},
        {'pad': ['Longitudinale analyse', 'BMI vs Beweging'], 'pagina': '📈 Longitudinale analyse'},
        {'pad': ['Longitudinale analyse', 'Scoreverandering per factor'], 'pagina': '📈 Longitudinale analyse'},
        {'pad': ['Longitudinale analyse', 'Gemiddelde score over tijd'], 'pagina': '📈 Longitudinale analyse'},
        {'pad': ['Datakwaliteit', 'Grafiek'], 'pagina': '🔍 Datakwaliteit'},
        {'pad': ['Datakwaliteit', 'Tabel'], 'pagina': '🔍 Datakwaliteit'},
        {'pad': ['Datakwaliteit', 'Duplicaat analyse'], 'pagina': '🔍 Datakwaliteit'},
        {'pad': ['Outlier analyse', 'Overzichtstabel'], 'pagina': '📉 Outlier analyse'},
        {'pad': ['Outlier analyse', 'Boxplot per variabele'], 'pagina': '📉 Outlier analyse'},
        {'pad': ['T-toets man vs vrouw', 'Forest plot'], 'pagina': '⚖️ T-toets man vs vrouw'},
        {'pad': ['T-toets man vs vrouw', 'Volledige tabel'], 'pagina': '⚖️ T-toets man vs vrouw'},
        {'pad': ['Correlaties', 'Correlatiematrix'], 'pagina': '🔗 Correlaties'},
        {'pad': ['Correlaties', 'Scatterplot'], 'pagina': '🔗 Correlaties'},
        {'pad': ['Per opdrachtgever', 'Snapshot scores'], 'pagina': '🏢 Per opdrachtgever'},
        {'pad': ['Per opdrachtgever', 'Score verbetering over tijd'], 'pagina': '🏢 Per opdrachtgever'},
        {'pad': ['Vragenlijstgedrag', 'Invulpercentages'], 'pagina': '📋 Vragenlijstgedrag'},
        {'pad': ['Vragenlijstgedrag', 'Populairste combinaties'], 'pagina': '📋 Vragenlijstgedrag'},
        {'pad': ['Vragenlijstgedrag', 'Herhaalinvullers'], 'pagina': '📋 Vragenlijstgedrag'},
        {'pad': ['Vragenlijstgedrag', 'Afhaakgedrag'], 'pagina': '📋 Vragenlijstgedrag'},
        {'pad': ['Kopersprofiel', 'Scores per groep'], 'pagina': '🛍️ Kopersprofiel'},
        {'pad': ['Kopersprofiel', 'Verdeling aankoopdatum'], 'pagina': '🛍️ Kopersprofiel'},
        {'pad': ['Producteffect', 'Meest gekochte producten'], 'pagina': '🛒 Producteffect'},
        {'pad': ['Snelle statistieken', 'Demografisch'], 'pagina': '🔎 Snelle statistieken'},
        {'pad': ['Snelle statistieken', 'Gezondheid'], 'pagina': '🔎 Snelle statistieken'},
        {'pad': ['Snelle statistieken', 'Leefstijl'], 'pagina': '🔎 Snelle statistieken'},
        {'pad': ['Snelle statistieken', 'Psychologisch'], 'pagina': '🔎 Snelle statistieken'},
        {'pad': ['ML Model', 'Gebruikerssegmentatie'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Leefstijlscore voorspelling'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Afhaakrisico'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Voorspelling scoreverbetering'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Product- en aankoopmodel'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Bloeddruk categorie voorspelling'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Cardiovasculair risico voorspelling'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Leefstijlmodel', 'Belangrijkste voorspellers'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Leefstijlmodel', 'Model evaluatie'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Bloeddrukmodel', 'Belangrijkste voorspellers'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Bloeddrukmodel', 'Model evaluatie'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Hartrisicomodel', 'Belangrijkste voorspellers'], 'pagina': '🤖 ML Model'},
        {'pad': ['ML Model', 'Hartrisicomodel', 'Model evaluatie'], 'pagina': '🤖 ML Model'},
        {'pad': ['Test', 'Risicomigratie'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Risicomigratie', 'Transition Matrix'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Engagement & dataflow', 'Herhaalde vragenlijst & scores'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Engagement & dataflow', 'Account dataflow'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Engagement & dataflow', 'Engagement per opdrachtgever'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Engagement & dataflow', 'Engagement per opdrachtgever', 'Ranking & benchmark'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Engagement & dataflow', 'Engagement per opdrachtgever', 'Uitsplitsing'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Engagement & dataflow', 'Engagement per opdrachtgever', 'Trend'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Looker Studio'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Looker Studio', 'Datum filter'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Looker Studio', 'KPI Metrics'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Looker Studio', 'Weekly User & Engagement Performance'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Looker Studio', 'Page Performance Overview', 'Alle content'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Looker Studio', 'Page Performance Overview', 'Content'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Looker Studio', 'Page Performance Overview', 'Orderable'], 'pagina': '🧪 Test'},
        {'pad': ['Test', 'Looker Studio', 'Top Partners by Views'], 'pagina': '🧪 Test'},
    ]

    # ── GLOBALE ZOEKBALK ──────────────────────────────────────────────────────
    st.markdown(f"**{tr('Zoeken', lang)}**")
    search_query = st.text_input(tr("Zoek binnen het dashboard...", lang), key="global_search_input", label_visibility="collapsed", placeholder=tr("bijv. hart, stress, BMI...", lang))

    if search_query:
        from variabelen import VARIABELEN

        zoekterm = normalize_search_text(search_query)
        # Zoek in pagina titels
        page_results = []
        for i, p_name in enumerate(pagina_opties):
            if all(term in normalize_search_text(tr(p_name, lang)) for term in zoekterm.split()):
                page_results.append((p_name, i))
        
        # Zoek in variabelen
        var_results = search_variable_specs(VARIABELEN, search_query)

        # Zoek in visualisaties met een expliciet navigatiepad
        visualization_results = [
            item for item in visualisatie_paden
            if item['pagina'] in pagina_opties
            and all(term in normalize_search_text(' '.join(item['pad'])) for term in zoekterm.split())
        ]

        if page_results or var_results or visualization_results:
            with st.container():
                st.markdown(f"*{tr('Resultaten', lang)}:*")
                for p_name, idx in page_results:
                    if st.button(f"📍 {tr(p_name, lang)}", key=f"glob_res_p_{idx}", width="stretch"):
                        st.session_state['nav_active_index'] = idx
                        st.session_state['nav_selectbox'] = pagina_opties[idx]
                        st.rerun()
                
                for v in var_results[:8]: # Limiteer resultaten voor overzichtelijkheid
                    if st.button(f"📊 {tr(v['label'], lang)}", key=f"glob_res_v_{v['kolom']}", width="stretch"):
                        # Ga naar Variabelen Verkenner
                        explorer_idx = [i for i, p in enumerate(pagina_opties) if "Variabelen verkenner" in p][0]
                        st.session_state['nav_active_index'] = explorer_idx
                        st.session_state['nav_selectbox'] = pagina_opties[explorer_idx]
                        st.session_state['explorer_search_query'] = tr(v['label'], lang)
                        st.rerun()

                for item in visualization_results:
                    pad = ' → '.join(item['pad'])
                    target_index = pagina_opties.index(item['pagina']) if item['pagina'] in pagina_opties else None
                    if target_index is not None and st.button(
                        tr(pad, lang),
                        key=f"glob_res_visualisatie_{normalize_search_text(pad)}",
                        width="stretch",
                    ):
                        st.session_state['nav_active_index'] = target_index
                        st.session_state['nav_selectbox'] = pagina_opties[target_index]
                        st.session_state['pending_navigation_path'] = pad
                        st.rerun()
        else:
            st.caption(tr("Geen resultaten gevonden.", lang))
    
    st.markdown("---")

    st.markdown(f"**{tr('Navigatie', lang)}**")
    st.caption(tr("Kies een dashboardpagina. De filterselecties blijven actief op alle pagina's.", lang))

    # Ruim verouderde navigatiestate op: als de opgeslagen paginawaarde niet
    # meer in de lijst staat (bijv. "🧪 Test" voor een niet-admin, of een
    # hernoemde pagina), resetten we naar het Overzicht om een
    # StreamlitAPIException op de selectbox te voorkomen.
    _nav_state = st.session_state.get('nav_selectbox')
    if _nav_state is not None and _nav_state not in pagina_opties:
        st.session_state.pop('nav_selectbox', None)
        st.session_state['nav_active_index'] = 0

    raw_nav_index = st.session_state.get('nav_active_index', 0)
    try:
        nav_active_index = int(raw_nav_index)
    except (TypeError, ValueError):
        nav_active_index = 0
    if not 0 <= nav_active_index < len(pagina_opties):
        nav_active_index = 0

    pagina = st.selectbox(
        tr("Navigatie", lang),
        pagina_opties,
        index=nav_active_index,
        key='nav_selectbox',
        format_func=lambda x: tr(x, lang),
        label_visibility="collapsed",
    )
    st.session_state['nav_active_index'] = pagina_opties.index(pagina)

    pending_navigation_path = st.session_state.get('pending_navigation_path')
    if pending_navigation_path:
        st.info(f"Gevonden in: {pending_navigation_path}")

    st.markdown("---")
    st.markdown(f"**{tr('Data Management', lang)}**")
    is_synthetic_source = DATA_SOURCE in {"csv", "sqlite"}
    refresh_label = (
        "🔄 Herlaad synthetische demo-data"
        if is_synthetic_source
        else "🔄 Ververs data uit database"
    )
    if st.button(tr(refresh_label, lang)):
        if is_synthetic_source:
            st.cache_data.clear()
            st.success(tr("Synthetische demo-data opnieuw geladen.", lang))
            st.rerun()
        else:
            with st.spinner(tr("Data aan het synchroniseren...", lang)):
                try:
                    ingest_script = Path(__file__).resolve().parent / "data_ingestion.py"
                    result = subprocess.run([sys.executable, str(ingest_script)], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.cache_data.clear()
                        st.success(tr("Data succesvol ververst!", lang))
                        st.rerun()
                    else:
                        st.error(f"Fout bij synchronisatie: {result.stderr}")
                except Exception as e:
                    st.error(f"Systeemfout: {e}")

    st.markdown("---")

    st.markdown(f"**{tr('🔎 Snelle statistieken', lang)}**")
    with st.expander(tr("Filter & bekijk", lang), expanded=False):
        df_alle = load_main_data_filtered_by_store()
        df_sub = pd.DataFrame()
        n_sub = 0
        if df_alle.empty:
            st.info(tr("Geen data beschikbaar", lang))
        else:
            df_stats = df_alle.copy()
            df_stats['leeftijd'] = pd.to_numeric(df_stats.get('rec_age_current'), errors='coerce')
            
            # Handle gender mapping safely
            gender_col = df_stats.get('rec_user_gender')
            if gender_col is not None:
                gender_numeric = pd.to_numeric(gender_col, errors='coerce')
                df_stats['geslacht_label'] = gender_numeric.map({0: 'Vrouw', 1: 'Man'})
            else:
                df_stats['geslacht_label'] = pd.NA

            # Filters
            geslacht_stat = st.selectbox(t("Geslacht"), [t("Beide"), t("Man"), t("Vrouw")], key="stat_geslacht")
            leeftijd_min, leeftijd_max = st.slider(
                t("Leeftijdsrange"), min_value=16, max_value=86,
                value=(16, 86), key="stat_leeftijd"
            )
            df_stats['bmi_val'] = pd.to_numeric(df_stats.get('rec_med_bmi'), errors='coerce')
            bmi_min, bmi_max = st.slider(
                t("BMI range"), min_value=15, max_value=50,
                value=(15, 50), key="stat_bmi"
            )
            stress_filter = st.selectbox(
                t("Stressniveau"), [t("Alle"), t("Laag (0)"), t("Matig (1)"), t("Hoog (2)")],
                key="stat_stress"
            )
            heartrisk_filter = st.selectbox(
                t("Cardiovasculair risico"), [t("Alle"), t("Laag (0)"), t("Matig (1)"), t("Hoog (2)")],
                key="stat_heartrisk"
            )

            # Toepassen
            mask = pd.Series([True] * len(df_stats), index=df_stats.index)
            if geslacht_stat == t("Man"):
                mask &= df_stats['geslacht_label'] == 'Man'
            elif geslacht_stat == t("Vrouw"):
                mask &= df_stats['geslacht_label'] == 'Vrouw'
            mask &= df_stats['leeftijd'].between(leeftijd_min, leeftijd_max)
            mask &= (df_stats['bmi_val'].isna() | df_stats['bmi_val'].between(bmi_min, bmi_max))
            if stress_filter != t("Alle"):
                stress_val = int(stress_filter.split("(")[1].replace(")", ""))
                df_stats['stress_num'] = pd.to_numeric(df_stats.get('rec_ls_stress_cat'), errors='coerce')
                mask &= df_stats['stress_num'].eq(stress_val)
            if heartrisk_filter != t("Alle"):
                hr_val = int(heartrisk_filter.split("(")[1].replace(")", ""))
                df_stats['hr_num'] = pd.to_numeric(df_stats.get('rec_heartrisk_cat'), errors='coerce')
                mask &= df_stats['hr_num'].eq(hr_val)
            df_sub = df_stats[mask]

            n_sub = _count_unique_participants(df_sub)
            n_totaal_stat = _count_unique_participants(df_stats)
        n_totaal_alle = _count_unique_participants(df_alle)
        pct_context = _pct_context_label(geslacht)
        samenvatting = (
            f"**{n_sub:,} deelnemers** ({(n_sub/n_totaal_alle*100 if n_totaal_alle else 0):.1f}% {pct_context})"
            if lang == "nl"
            else f"**{n_sub:,} participants** ({(n_sub/n_totaal_alle*100 if n_totaal_alle else 0):.1f}% of total)"
        )
        st.markdown(samenvatting)
        st.markdown("---")

        # KPIs voor subgroep — zelfde definities als overzichtspagina
        for label, (kolom, waarde, op) in OVERZICHTS_KPI_SPECS.items():
            count, n_geld, pct = _bereken_kpi_telling(df_sub, kolom, waarde, op)
            if n_geld == 0:
                continue
            st.metric(t(label), f"{pct:.1f}%",
                      delta=f"n={int(count)}",
                      delta_color="off")

# ═════════════════════════════════════════════════════════════════════════════
# OVERZICHT
# ═════════════════════════════════════════════════════════════════════════════
if pagina == "📊 Looker Studio":
    st.title("Looker Studio")
    st.caption("Interactief Looker Studio-rapport")

    embed_url = LOOKER_STUDIO_EMBED_URL.strip()
    toegestane_looker_hosts = (
        "https://lookerstudio.google.com/",
        "https://datastudio.google.com/",
    )
    if not embed_url:
        st.info(
            "De Looker Studio-link is nog niet ingesteld. Voeg "
            "LOOKER_STUDIO_EMBED_URL toe aan het .env-bestand en herstart het dashboard."
        )
    elif not embed_url.startswith(toegestane_looker_hosts):
        st.error(
            "Ongeldige Looker Studio-link. Gebruik de HTTPS-embedlink van "
            "lookerstudio.google.com of datastudio.google.com."
        )
    else:
        components.iframe(embed_url, height=900, scrolling=True)

elif pagina == "🏠 Overzicht":
    st.title(tr("Smart Health — Wetenschappersdashboard", lang))
    geslacht_badge(geslacht)
    st.markdown(tr("Interactief overzicht van de PMO data van Smart Health.", lang))

    st.markdown("---")

    df = load_main_data_filtered_by_gender_and_store(geslacht)

    # KPI metrics
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    c1.metric(t("Deelnemers"), f"{_count_unique_participants(df):,}")
    
    # Postcode: defensieve check omdat kolom misschien niet bestaat
    postal_count = df['postal_code'].notna().sum() if 'postal_code' in df.columns else 0
    c2.metric(t("Met postcode"), f"{postal_count:,}")

    hr_count, _, hr_pct = _bereken_kpi_telling(df, *OVERZICHTS_KPI_SPECS['Hoog hart risico'])
    c3.metric(t("Hoog hart risico"), f"{hr_count:,}",
              delta=f"{hr_pct:.1f}% {_pct_context_label(geslacht)}",
              delta_color="off")

    stress_count, _, stress_pct = _bereken_kpi_telling(df, *OVERZICHTS_KPI_SPECS['Hoog stress'])
    c4.metric(t("Hoog stress"), f"{stress_count:,}",
              delta=f"{stress_pct:.1f}% {_pct_context_label(geslacht)}",
              delta_color="off")

    bmi_count, _, bmi_pct = _bereken_kpi_telling(df, *OVERZICHTS_KPI_SPECS['Overgewicht'])
    c5.metric(t("Overgewicht"), f"{bmi_count:,}",
              delta=f"{bmi_pct:.1f}% {_pct_context_label(geslacht)}",
              delta_color="off")

    slaap_count, _, slaap_pct = _bereken_kpi_telling(df, *OVERZICHTS_KPI_SPECS['Slechte slaap'])
    c6.metric(t("Slechte slaap"), f"{slaap_count:,}",
              delta=f"{slaap_pct:.1f}% {_pct_context_label(geslacht)}",
              delta_color="off")

    bew_count, _, bew_pct = _bereken_kpi_telling(df, *OVERZICHTS_KPI_SPECS['Niet fysiek actief'])
    c7.metric(t("Niet fysiek actief"), f"{bew_count:,}",
              delta=f"{bew_pct:.1f}% {_pct_context_label(geslacht)}",
              delta_color="off")

    st.markdown("---")

    # Vaste visualisaties op overzicht in tabs
    overzicht_tab_labels = [
        f"👥 {t('Demografisch')}",
        f"❤️ {t('Cardiovasculair risico')}",
        "⚖️ BMI",
        "😰 Stress" if lang == "nl" else "😰 Stress",
        f"🏃 {t('Leefstijl')}",
        f"📈 {t('Platformgroei')}" if t('Platformgroei') != 'Platformgroei' else ("📈 Platformgroei" if lang == 'nl' else "📈 Platform growth"),
        "📰 Artikelgebruik" if lang == "nl" else "📰 Article usage",
    ]
    tab_demo, tab_cardio, tab_bmi, tab_stress, tab_leefstijl, tab_groei, tab_artikelen = st.tabs(
        overzicht_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, overzicht_tab_labels),
    )

    df_alle = load_main_data_filtered_by_store()

    with tab_demo:
        c1, c2 = st.columns(2)
        with c1: p(maak_geslacht_plot(df, lang=lang), participants_df=df)
        with c2: p(maak_leeftijd_plot(df, geslacht=geslacht, lang=lang), participants_df=df)

    with tab_cardio:
        p(maak_heartrisk_naar_geslacht_plot(df, lang=lang, geslacht=geslacht), participants_df=df)

    with tab_bmi:
        p(maak_bmi_plot(df, geslacht=geslacht, lang=lang), participants_df=df)

    with tab_stress:
        p(maak_stress_plot(df, geslacht=geslacht, lang=lang), participants_df=df)

    with tab_leefstijl:
        p(maak_leefstijl_score_plot(df, geslacht=geslacht, lang=lang), participants_df=df)

    with tab_groei:
        st.caption(t(
            "Aantal nieuwe gebruikersaccounts en ingevulde vragenlijsten per jaar. Blauwe balken = accounts (linker as), oranje lijn = vragenlijsten (rechter as). Deze grafiek gebruikt dezelfde accountdefinitie als accountactivatie: alleen niet-verwijderde accounts."
        ))
        p(get_platform_groei(_participant_ids(df)), participants_df=df)
        
        st.markdown("---")
        st.subheader(t("Data Verificatie"))
        p(maak_update_verificatie_plot(df, lang=lang), participants_df=df)

    with tab_artikelen:
        st.caption(t("Views op contentartikelen op basis van `interactions.csv`, exclusief interne Smart Health accounts en overeenkomende e-mailnamen."))
        try:
            top_artikelen, df_views, samenvatting_artikelen = get_artikel_interacties_v2(_user_ids(df))
            a1, a2, a3 = st.columns(3)
            a1.metric(t("Totaal article views"), f"{samenvatting_artikelen['Totaal views']:,}")
            a2.metric(t("Unieke lezers"), f"{samenvatting_artikelen['Unieke lezers']:,}")
            a3.metric(t("Unieke artikelen"), f"{samenvatting_artikelen['Unieke artikelen']:,}")
            st.subheader(t("Meest gelezen artikelen"))
            st.dataframe(
                translate_dataframe(top_artikelen[['title', 'Views', 'Unieke_lezers', 'Laatste_view']], lang),
                width="stretch",
                hide_index=True
            )
        except Exception as e:
            st.error(t("Fout bij artikelgebruik: {e}", e=e))

# ═════════════════════════════════════════════════════════════════════════════
# KAARTEN
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🗺️ Kaarten":
    st.title(tr("Geografische kaarten", lang))

    kaarten = {
        tr("Gebruikersdichtheid (pc3)", lang): "gebruikersdichtheid_kaart_pc3.html",
        tr("Cardiovasculair risico", lang):    "heartrisk_categorie_kaart.html",
        tr("BMI categorie", lang):             "bmi_categorie_kaart.html",
        tr("Stress categorie", lang):          "stress_categorie_kaart.html",
        tr("Slaapkwaliteit", lang):            "slaap_categorie_kaart.html",
        tr("Beweging (stappen per dag)", lang): "beweging_categorie_kaart.html",
        tr("Leeftijdscategorie", lang):        "leeftijd_categorie_kaart.html",
        tr("Werkvermogen", lang):              "werkvermogen_categorie_kaart.html",
        tr("Gebruikerslocaties (wereld)", lang): "internationale_gebruikers_kaart.html",
    }
    omschrijvingen = {
        tr("Gebruikersdichtheid (pc3)", lang): tr("Aantal PMO deelnemers per pc3 gebied. Grijs = geen deelnemers. Transparantie geeft betrouwbaarheid aan.", lang),
        tr("Cardiovasculair risico", lang):    tr("Meest voorkomende cardiovasculaire risicocategorie per pc3 gebied. Tooltip toont ook hoe dominant deze categorie is binnen het gebied.", lang),
        tr("BMI categorie", lang):             tr("Meest voorkomende BMI categorie per pc3 gebied. Tooltip toont ook hoe dominant deze categorie is binnen het gebied.", lang),
        tr("Stress categorie", lang):          tr("Meest voorkomende stresscategorie per pc3 gebied. Tooltip toont ook hoe dominant deze categorie is binnen het gebied.", lang),
        tr("Slaapkwaliteit", lang):            tr("Meest voorkomende slaapkwaliteitscategorie per pc3 gebied.", lang),
        tr("Beweging (stappen per dag)", lang): tr("Meest voorkomende stappencategorie per pc3 gebied.", lang),
        tr("Leeftijdscategorie", lang):        tr("Meest voorkomende leeftijdscategorie per pc3 gebied.", lang),
        tr("Werkvermogen", lang):              tr("Meest voorkomende werkvermogenscategorie per pc3 gebied.", lang),
        tr("Gebruikerslocaties (wereld)", lang): tr("Alle gebruikers als punt op de kaart op basis van lat/long coördinaten. Blauw = man, roze = vrouw.", lang),
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        keuze = st.selectbox(t("Kies een kaart"), list(kaarten.keys()))
    with col2:
        st.selectbox(t("Regio"), list(KAART_PRESETS.keys()), key='kaart_preset')

    st.caption(omschrijvingen[keuze])
    df_kaart = load_main_data_filtered_by_gender_and_store(geslacht)
    if not toon_dynamische_kaart(df_kaart, kaarten[keuze], keuze, lang=lang):
        pad_stat = HTML_MAP / kaarten[keuze]
        if pad_stat.exists():
            toon_kaart(kaarten[keuze])
        else:
            st.warning(t("Geen kaart beschikbaar voor deze selectie."))

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIES
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Variabelen verkenner":
    st.title(tr("EDA Visualisaties", lang))
    geslacht_badge(geslacht)

    variabelen_specs = translate_variable_specs(list(VARIABELEN_DICT.values()), lang)
    variabelen_dict_t = {v['label']: v for v in variabelen_specs}
    variabelen_per_groep_t = {}
    for v in variabelen_specs:
        variabelen_per_groep_t.setdefault(v['groep'], []).append(v)

    df_alle = load_main_data_filtered_by_store()
    df      = filter_geslacht(df_alle, geslacht)

    # Zoekbalk
    explorer_query = st.session_state.get('explorer_search_query', "")
    zoek = st.text_input(
        tr("🔍 Zoek variabele", lang), 
        value=explorer_query, 
        placeholder=tr("bijv. stress, BMI, slaap, alcohol...", lang)
    )

    if zoek:
        # Zoek in labels en omschrijvingen
        resultaten = search_variable_specs(variabelen_specs, zoek)
        if resultaten:
            st.caption(f"{len(resultaten)} {tr('resultaat/resultaten gevonden.', lang)}")
            labels_gevonden = [v['label'] for v in resultaten]
            var_keuze = st.radio(
                tr("Kies een resultaat", lang),
                labels_gevonden,
                horizontal=True,
            )
            variabele = variabelen_dict_t[var_keuze]
        else:
            st.warning(tr("Geen variabelen gevonden. Probeer een andere zoekterm.", lang))
            variabele = None
    else:
        st.caption(tr("Of blader via groep en variabele:", lang))
        c1, c2 = st.columns(2)
        with c1:
            groep_keuze = st.selectbox(tr("Groep", lang), list(variabelen_per_groep_t.keys()))
        with c2:
            var_labels = [v['label'] for v in variabelen_per_groep_t[groep_keuze]]
            var_keuze  = st.selectbox(tr("Variabele", lang), var_labels)
        variabele = variabelen_dict_t[var_keuze]

    # Wis de globale zoekopdracht na gebruik in deze tab zodat hij niet blijft plakken
    if 'explorer_search_query' in st.session_state:
        del st.session_state['explorer_search_query']

    if variabele:
        st.caption(variabele['omschrijving'])
        df_explorer = apply_global_filters(load_main_data(), geslacht=geslacht)
        fig = maak_verkenner_plot(df_explorer, variabele, geslacht_filter=geslacht, lang=lang)
        p(fig, participants_df=df_explorer)

# ═════════════════════════════════════════════════════════════════════════════
# DIEET & BMI
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🥗 Dieet & BMI":
    st.title(tr("Dieet & BMI", lang))
    geslacht_badge(geslacht)
    st.caption(
        t("Dieetkwaliteit is het gemiddelde van zes voedingsscores: fruit, groenten, suiker, verzadigd vet, alcohol en zout.")
    )
    st.markdown("---")

    df = load_main_data_filtered_by_gender_and_store(geslacht)
    dieet_tab_labels = [t("Verdeling dieetkwaliteit"), t("Dieet vs BMI"), t("Dieet naar geslacht"), t("Internationale vergelijking")]
    tab1, tab2, tab3, tab4 = st.tabs(
        dieet_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, dieet_tab_labels),
    )

    with tab1:
        p(maak_dieet_verdeling_plot(df, geslacht=geslacht, lang=lang), participants_df=df)
    with tab2:
        st.caption(t("Per dieet categorie de BMI verdeling. Een betere dieetkwaliteit zou moeten samenhangen met lagere BMI."))
        p(maak_dieet_bmi_plot(df, geslacht=geslacht, lang=lang), participants_df=df)
    with tab3:
        p(maak_dieet_score_histogram(df, geslacht=geslacht, lang=lang), participants_df=df)

    with tab4:
        st.subheader(t("Internationale vergelijking: Nederland vs Buitenland"))
        st.caption(t("Vergelijk deelnemers in Nederland en daarbuiten. Elke variabele behoudt zijn eigen meeteenheid; BMI wordt apart getoond. Interpreteer kleine groepen voorzichtig."))
        
        df_int = voeg_dieet_score_toe(load_main_data_filtered_by_store())
        df_int = add_app_user_ids_and_addresses(df_int, DB_URL)
        NL_CODES = {'NL', 'NEDERLAND', 'THE NETHERLANDS', 'NETHERLANDS', 'PAYS-BAS', 'NIEDERLANDE'}

        def _bepaal_regio(row):
            country_val = row.get('country')
            if pd.notna(country_val):
                c_str = str(country_val).strip().upper()
                if c_str in NL_CODES:
                    return 'Nederland'
                if c_str not in {'', 'NONE', 'NAN', 'NULL'}:
                    return 'Buitenland'
            pc_val = str(row.get('postal_code') or '').strip()
            if re.match(r'^[1-9]\d{3}(\s*[A-Za-z]{2})?$', pc_val):
                return 'Nederland'
            if pc_val and pc_val.lower() not in {'none', 'nan', ''}:
                return 'Buitenland'
            return pd.NA

        df_int['regio'] = df_int.apply(_bepaal_regio, axis=1)
        df_int = df_int.dropna(subset=['regio']).copy()
        
        metrics_map = {
            'Leefstijlscore': 'rec_ls_lifestyle_score',
            'Bewegingsscore': 'rec_ls_score_exercise',
            'Dieet score': 'dieet_score',
            'Stress': 'rec_ls_stress_sum',
            'Slaapkwaliteit': 'rec_ls_sleep_psqi_sum',
            'Welzijn': 'rec_wellbeing_score',
            'Veerkracht': 'rec_resilience_score',
            'Hart-risico': 'rec_heartrisk',
        }
        comp_data = []
        for label, col in metrics_map.items():
            if col not in df_int.columns:
                continue
            df_metric = df_int[['participant_id', 'regio', col]].copy()
            df_metric[col] = pd.to_numeric(df_metric[col], errors='coerce')
            df_metric = df_metric.dropna(subset=[col]).drop_duplicates(['participant_id', 'regio'])
            grouped = df_metric.groupby('regio').agg(
                Gemiddelde=(col, 'mean'),
                n=('participant_id', 'nunique'),
            ).reset_index()
            for _, row in grouped.iterrows():
                comp_data.append({
                    'Metric': t(label),
                    'Regio': t(row['regio']),
                    'Gemiddelde': float(row['Gemiddelde']),
                    'n': int(row['n']),
                })

        df_comp = pd.DataFrame(comp_data)
        if not df_comp.empty:
            metrics_met_beide_regios = (
                df_comp.groupby('Metric')
                .filter(lambda group: group['Regio'].nunique() >= 2 and (group['n'] > MIN_INTERNATIONALE_DEELNEMERS - 1).all())
                ['Metric']
                .unique()
            )
            df_comp = df_comp[df_comp['Metric'].isin(metrics_met_beide_regios)].copy()
        if df_comp.empty or df_comp['Regio'].nunique() < 2:
            st.info(t("Onvoldoende data voor internationale vergelijking."))
        else:
            kleurkaart = {t('Nederland'): '#E87722', t('Buitenland'): '#1F6FBF'}
            regio_counts = df_comp[['Regio', 'n']].drop_duplicates('Regio').set_index('Regio')['n']
            k1, k2, k3 = st.columns(3)
            k1.metric(t('Metrics vergeleken'), int(df_comp['Metric'].nunique()))
            k2.metric(t('Deelnemers Nederland'), int(regio_counts.get(t('Nederland'), 0)))
            k3.metric(t('Deelnemers buitenland'), int(regio_counts.get(t('Buitenland'), 0)))

            kleine_groepen = regio_counts[regio_counts < MIN_INTERNATIONALE_DEELNEMERS]
            if not kleine_groepen.empty:
                st.warning(t("Minstens één regiogroep heeft minder dan {min_n} deelnemers.", min_n=MIN_INTERNATIONALE_DEELNEMERS))
            else:
                score_fig = px.bar(
                    df_comp,
                    x='Gemiddelde',
                    y='Metric',
                    color='Regio',
                    barmode='group',
                    orientation='h',
                    text='Gemiddelde',
                    color_discrete_map=kleurkaart,
                    labels={'Gemiddelde': t('Gemiddelde waarde'), 'Metric': '', 'Regio': t('Regio')},
                    title=t("Internationale variabelenvergelijking"),
                )
                score_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', cliponaxis=False)
                score_fig.update_layout(
                    height=max(420, 55 * int(df_comp['Metric'].nunique()) + 100),
                    xaxis=dict(autorange=True),
                    legend_title_text='',
                )
                p(score_fig, participants_df=df_int)

            st.markdown(f"**{t('Details per regio')}**")
            pivot_df = (
                df_comp.assign(waarde=df_comp.apply(lambda row: f"{row['Gemiddelde']:.2f} (n={int(row['n'])})", axis=1))
                .pivot(index='Metric', columns='Regio', values='waarde')
                .reset_index()
            )
            st.dataframe(translate_dataframe(pivot_df, lang), width="stretch", hide_index=True)

            df_bmi = df_int[['participant_id', 'regio', 'rec_med_bmi']].copy()
            df_bmi['rec_med_bmi'] = pd.to_numeric(df_bmi['rec_med_bmi'], errors='coerce')
            df_bmi = df_bmi.dropna(subset=['rec_med_bmi']).drop_duplicates(['participant_id', 'regio'])
            bmi_comp = df_bmi.groupby('regio').agg(
                Gemiddelde=('rec_med_bmi', 'mean'),
                n=('participant_id', 'nunique'),
            ).reset_index()
            if bmi_comp['regio'].nunique() >= 2 and (bmi_comp['n'] > MIN_INTERNATIONALE_DEELNEMERS - 1).all():
                bmi_comp['Regio'] = bmi_comp['regio'].apply(t)
                bmi_fig = px.bar(
                    bmi_comp,
                    x='Regio',
                    y='Gemiddelde',
                    color='Regio',
                    text='Gemiddelde',
                    color_discrete_map=kleurkaart,
                    labels={'Gemiddelde': t('Gemiddelde BMI'), 'Regio': ''},
                    title=t("BMI vergelijking: Nederland vs Buitenland"),
                )
                bmi_fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                bmi_fig.update_layout(height=320, showlegend=False)
                p(bmi_fig, participants_df=df_bmi)

# ═════════════════════════════════════════════════════════════════════════════
# LONGITUDINALE ANALYSE
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "📈 Longitudinale analyse":
    st.title(tr("Longitudinale analyse", lang))
    geslacht_badge(geslacht)
    st.caption(tr("Scores over tijd op basis van alle beschikbare herhaalde metingen.", lang))

    with st.spinner(t("Longitudinale data laden...")):
        df_long = get_longitudinale_data() # Already loaded
        df_trend = bereken_verandering(df_long, split_gender=geslacht == 'beide')

    df_long_participants = filter_geslacht(df_long, geslacht)

    longitudinale_tab_labels = [t("BMI vs Beweging"), t("Scoreverandering per factor"), t("Gemiddelde score over tijd")]
    tab1, tab2, tab3 = st.tabs(
        longitudinale_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, longitudinale_tab_labels),
    ) # New tab

    with tab1:
        st.subheader(t("BMI vs Beweging over tijd"))
        m(maak_bmi_beweging_plot(df_long, geslacht=geslacht, lang=lang), participants_df=df_long_participants)
        st.caption(t("BMI = rood, Beweging = blauw."))

    with tab2:
        st.subheader(t("Scoreverandering ten opzichte van eerste meting"))

        df_trend = df_trend[df_trend['slug'] != 'age'].copy()
        factor_opties = _beschikbare_longitudinale_factor_labels(df_trend) # This is correct, still uses df_trend
        if not factor_opties:
            st.info(t("Geen longitudinale factoren beschikbaar met de huidige filters."))
            st.stop()

        col1, col2 = st.columns([3, 1])
        with col1:
            factor_keuze = st.selectbox(t("Factor"), list(factor_opties.keys()))
        with col2:
            toon_alle = st.checkbox(t("Toon alle factoren"), value=False)

        # Geslacht filter op longitudinale data
        df_long_f = df_long.copy()
        if geslacht == 'man':
            df_long_f = df_long_f[df_long_f['user_gender'] == 1]
        elif geslacht == 'vrouw':
            df_long_f = df_long_f[df_long_f['user_gender'] == 0]
        df_trend_f = bereken_verandering(df_long_f, split_gender=geslacht == 'beide')
        df_trend_f = df_trend_f[df_trend_f['slug'] != 'age'].copy()

        if toon_alle:
            m(maak_scoreverandering_plot(df_trend_f, geslacht=geslacht, lang=lang), participants_df=df_long_f)
        else:
            import plotly.graph_objects as go
            slug = factor_opties[factor_keuze]
            df_factor = df_trend_f[df_trend_f['slug'] == slug]

            if df_factor.empty:
                st.warning(t("Onvoldoende data voor deze factor met de huidige filters."))
            else:
                lager_is_beter = ['alcohol', 'smoking', 'stress', 'bmi', 'sugar',
                                   'fat', 'salt', 'blood_pressure', 'diabetes',
                                   'dass_stress', 'dass_anxiety', 'dass_depression',
                                   'menopause_genitourinary', 'menopause_psychological',
                                   'menopause_somatic']
                if slug in lager_is_beter:
                    kleuren = [RISICO_COLORS[0] if x <= 0 else RISICO_COLORS[2] for x in df_factor['mean']]
                else:
                    kleuren = [RISICO_COLORS[0] if x >= 0 else RISICO_COLORS[2] for x in df_factor['mean']]
                fig = go.Figure()
                fig.add_bar(
                    x=df_factor['periode'].astype(str),
                    y=df_factor['mean'].round(3),
                    marker_color=kleuren,
                    text=df_factor['n_participanten'].astype(int).astype(str) + (' deeln.' if lang == 'nl' else ' particip.'),
                    textposition='outside',
                )
                fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)
                fig.update_layout(
                    title=f"Scoreverandering: {factor_keuze}",
                    xaxis_title="Periode na eerste meting",
                    yaxis_title="Gemiddelde verandering t.o.v. eerste meting",
                    showlegend=False,
                )
                p(fig, participants_df=df_long_f[df_long_f['slug'] == slug])
                st.caption(t("Aantal deelnemers per periode staat boven de balken."))

    with tab3:
        st.subheader(t("Gemiddelde score over tijd per factor"))
        st.caption(t("Toont de gemiddelde score van een geselecteerde factor per maand. De stippellijnen tonen het gewogen gemiddelde per jaar, waarbij maanden met meer deelnemers zwaarder meetellen."))

        factor_opties_avg = _beschikbare_longitudinale_factor_labels(df_long)
        col_avg1, col_avg2 = st.columns([3, 1])
        with col_avg1:
            factor_keuze_avg = st.selectbox(t("Kies een factor"), list(factor_opties_avg.keys()), key="long_avg_factor_select")
        
        slug_avg = factor_opties_avg[factor_keuze_avg]
        fig_avg = maak_gemiddelde_score_over_tijd_plot(df_long, slug_avg, lang=lang, geslacht=geslacht)
        p(fig_avg, participants_df=df_long_participants)

# ═════════════════════════════════════════════════════════════════════════════
# DATAKWALITEIT
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🔍 Datakwaliteit":
    st.title(tr("Datakwaliteit", lang))
    geslacht_badge(geslacht)
    st.caption(tr("Overzicht van missende waarden en basisstatistieken per variabele.", lang))

    # Totaal accounts ophalen
    try:
        from data_ingestion import load_participants
        df_alle_accounts = load_participants(DB_URL)
        if df_alle_accounts.empty:
            n_accounts = None
        else:
            actieve_accounts = df_alle_accounts[df_alle_accounts['deleted_at'].isna()]
            n_accounts = _count_unique_participants(actieve_accounts)
    except Exception:
        n_accounts = None

    df_actief_kwaliteit = _active_participant_rows(load_main_data_filtered_by_gender_and_store(geslacht))
    n_actief = _count_unique_participants(df_actief_kwaliteit)

    # Instellingen
    col_set1, col_set2 = st.columns([2, 2])
    with col_set1:
        gebruik_totaal = st.toggle(
            f"{t('Bereken over alle accounts')} ({n_accounts:,})" if n_accounts else t("Bereken over alle accounts"),
            value=False,
            help=t("Uit = missend % over actieve deelnemers. Aan = over alle accounts inclusief nooit-ingevuld.")
        )
    with col_set2:
        min_missend = st.slider(
            t("Minimaal missend % om te tonen"),
            min_value=0, max_value=90, value=0, step=5,
        )

    n_totaal_gebruik = n_accounts if (gebruik_totaal and n_accounts) else None
    df_kwaliteit = get_datakwaliteit(geslacht, n_totaal=n_totaal_gebruik, filter_signature=get_global_filter_signature())
    df_kwaliteit_gefilterd = df_kwaliteit[df_kwaliteit['Missend (%)'] >= min_missend]

    st.markdown("---")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric(t("Totaal accounts"), f"{n_accounts:,}" if n_accounts else "N/A")
    c2.metric(t("Actieve deelnemers"), f"{n_actief:,}")
    c3.metric(t("Variabelen getoond"), len(df_kwaliteit_gefilterd))


    st.markdown("---")

    kwaliteit_tab_labels = [t("Grafiek"), t("Tabel"), t("Duplicaat analyse")]
    tab1, tab2, tab3 = st.tabs(
        kwaliteit_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, kwaliteit_tab_labels),
    )
    with tab1:
        if df_kwaliteit_gefilterd.empty:
            st.info(t("Geen variabelen met ≥{min_missend}% missende waarden.", min_missend=min_missend))
        else:
            p(maak_missende_waarden_plot(df_kwaliteit_gefilterd), participants_df=df_actief_kwaliteit)
    with tab2:
        st.dataframe(translate_dataframe(df_kwaliteit_gefilterd.drop(columns=['Kolom']), lang),
                     width="stretch", hide_index=True)
        csv = df_kwaliteit_gefilterd.drop(columns=['Kolom']).to_csv(index=False).encode('utf-8')
        st.download_button(t("📥 Download als CSV"), csv, "datakwaliteit.csv", "text/csv")
    with tab3:
        st.subheader(t("Duplicaat analyse per tabel"))
        st.caption(t("Overzicht van duplicaten per Parquet bestand. 'Duplicaat rijen' telt rijen die exact hetzelfde zijn over alle kolommen. 'ID duplicaten' telt duplicaten in de 'id' kolom."))
        df_duplicaten = get_duplicaat_statistieken()
        st.dataframe(translate_dataframe(df_duplicaten, lang),
                     width="stretch", hide_index=True)
        csv_duplicaten = df_duplicaten.to_csv(index=False).encode('utf-8')
        st.download_button(t("📥 Download als CSV"), csv_duplicaten, "duplicaten_analyse.csv", "text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# OUTLIER ANALYSE
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "📉 Outlier analyse":
    st.title(tr("Outlier analyse", lang))
    geslacht_badge(geslacht)
    st.caption(
        t("Outliers gedetecteerd via de IQR methode (1.5 × IQR buiten Q1/Q3). Rode lijn = domeingrens.")
    )

    df = load_main_data_filtered_by_gender_and_store(geslacht)
    df_outliers = get_outliers(geslacht, filter_signature=get_global_filter_signature())

    outlier_tab_labels = [t("Overzichtstabel"), t("Boxplot per variabele")]
    tab1, tab2 = st.tabs(
        outlier_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, outlier_tab_labels),
    )

    with tab1:
        st.dataframe(
            translate_dataframe(
            df_outliers[[
                'Variabele', 'N geldig', 'Gemiddelde', 'Mediaan',
                'Min', 'Max', 'IQR grens laag', 'IQR grens hoog',
                'Outliers totaal (n)', 'Outliers (%)', 'Domeingrens',
            ]],
            lang),
            width="stretch", hide_index=True,
        )
        csv = df_outliers.to_csv(index=False).encode('utf-8')
        st.download_button(t("📥 Download als CSV"), csv, "outliers.csv", "text/csv")

    with tab2:
        keuze = st.selectbox(t("Kies een variabele"), list(RUWE_WAARDEN.keys()))
        kolom = RUWE_WAARDEN[keuze]
        p(maak_outlier_boxplot(df, keuze, kolom, geslacht_filter=geslacht), participants_df=df)
        rij = df_outliers[df_outliers['Variabele'] == keuze]
        if not rij.empty:
            r = rij.iloc[0]
            st.info(
                f"**IQR grenzen:** {r['IQR grens laag']:.1f} – {r['IQR grens hoog']:.1f} | "
                f"**Outliers:** {r['Outliers totaal (n)']} ({r['Outliers (%)']:.1f}%) | "
                f"**Max:** {r['Max']:.1f}"
                + (f" | **Domeingrens:** {r['Domeingrens']}" if r['Domeingrens'] else "")
            )

# ═════════════════════════════════════════════════════════════════════════════
# T-TOETS
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "⚖️ T-toets man vs vrouw":
    st.title(tr("T-toets: mannen vs vrouwen", lang))
    st.caption(
        t("Welch t-toets per variabele. Cohen's d = effectgrootte. Blauw = mannen hoger, roze = vrouwen hoger. Doorzichtig = niet significant (p ≥ 0.05).")
    )

    df_toetsen = get_t_toetsen(filter_signature=get_global_filter_signature())
    df_toets_basis = load_main_data_filtered_by_store()

    toets_tab_labels = [t("Forest plot"), t("Volledige tabel")]
    tab1, tab2 = st.tabs(
        toets_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, toets_tab_labels),
    )

    with tab1:
        alle_t_toets_vars = {**LEEFSTIJL_SCORES, **RUWE_WAARDEN, **T_TOETS_EXTRA_VARIABELEN}
        beschikbare_t_toetsen = [label for label, kolom in alle_t_toets_vars.items() if kolom in df_toets_basis.columns]
        gekozen_t_toetsen = st.multiselect(
            t("Variabelen meenemen"),
            beschikbare_t_toetsen,
            default=[
                label for label in [t("Leefstijlscore"), t("BMI"), t("Heartrisk"), t("Stress"), t("Slaap"), t("Leeftijd")]
                if label in beschikbare_t_toetsen
            ] or beschikbare_t_toetsen,
            help=t("Kies welke variabelen in de t-toets worden meegenomen."),
        )
        if gekozen_t_toetsen:
            gekozen_kolommen = [alle_t_toets_vars[label] for label in gekozen_t_toetsen if label in alle_t_toets_vars]
            df_toetsen_toon = get_t_toetsen(
                filter_signature=get_global_filter_signature(),
                geselecteerde_variabelen=gekozen_t_toetsen,
            )
            st.subheader(t("Verschil in effectgrootte"))
            p(maak_t_toets_plot(df_toetsen_toon), participants_df=df_toets_basis)
            st.subheader(t("Gemiddelde scores per geslacht"))
            p(maak_t_toets_gemiddelden_plot(df_toetsen_toon), participants_df=df_toets_basis)
        else:
            st.info(t("Selecteer minimaal één variabele."))
        st.caption(
            t("Verticale stippellijnen bij d = ±0.2 (klein effect) en d = ±0.5 (matig effect). Alleen variabelen met ≥10 deelnemers per geslacht worden meegenomen.")
        )

    with tab2:
        # Filter opties
        col1, col2 = st.columns(2)
        with col1:
            toon_alleen_sig = st.checkbox(t("Toon alleen significante resultaten"), value=False)
        with col2:
            min_effect = st.selectbox(
                t("Minimale effectgrootte"),
                [t("Alle"), t("Klein (d>0.2)"), t("Matig (d>0.5)"), t("Groot (d>0.8)")],
            )

        df_tabel = df_toetsen.copy()
        if toon_alleen_sig:
            df_tabel = df_tabel[df_tabel['Significant'] == 'Ja ✓']
        if min_effect == t("Klein (d>0.2)"):
            df_tabel = df_tabel[df_tabel["Cohen's d"].abs() > 0.2]
        elif min_effect == t("Matig (d>0.5)"):
            df_tabel = df_tabel[df_tabel["Cohen's d"].abs() > 0.5]
        elif min_effect == t("Groot (d>0.8)"):
            df_tabel = df_tabel[df_tabel["Cohen's d"].abs() > 0.8]

        st.dataframe(translate_dataframe(df_tabel, lang), width="stretch", hide_index=True)
        st.caption(t("{n} van {total} variabelen weergegeven.", n=len(df_tabel), total=len(df_toetsen)))
        csv = df_tabel.to_csv(index=False).encode('utf-8')
        st.download_button(t("📥 Download als CSV"), csv, "t_toetsen.csv", "text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# CORRELATIES
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🔗 Correlaties":
    st.title(tr("Correlaties", lang))
    geslacht_badge(geslacht)
    st.caption(
        t("Pearson correlatiecoëfficiënten. Groen = positief verband, rood = negatief verband.")
    )

    df = load_main_data_filtered_by_gender_and_store(geslacht)
    correlatie_tab_labels = [t("Correlatiematrix"), t("Scatterplot")]
    tab1, tab2 = st.tabs(
        correlatie_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, correlatie_tab_labels),
    )

    with tab1:
        alle_corr_vars = {
            'Leefstijlscore': 'rec_ls_lifestyle_score',
            'BMI (waarde)': 'rec_med_bmi',
            'Heartrisk (score)': 'rec_heartrisk',
            'Framingham score': 'rec_framingham_non_invasive',
            'Stress (score)': 'rec_ls_stress_sum',
            'DASS stress': 'rec_dass_stress_score',
            'DASS angst': 'rec_dass_anxiety_score',
            'DASS depressie': 'rec_dass_depression_score',
            'Slaap PSQI': 'rec_ls_sleep_psqi_sum',
            'Veerkracht': 'rec_resilience_score',
            'Welzijn': 'rec_wellbeing_score',
            'Werkvermogen (WAI)': 'rec_asr_wai_score',
            'Werkdruk': 'rec_asr_workload_score',
            'Uitputting': 'rec_asr_exhaustion_score',
            'Burn-out score': 'rec_asr_burn_out_score',
            'Vitaliteit': 'rec_asr_vitality_score',
            'Werktevredenheid': 'rec_asr_job_satisfaction_score',
            'Leeftijd': 'rec_age_current',
            'Stappen per dag': 'rec_ls_exercise_steps_per_day',
            'Beweegminuten': 'rec_ls_exercise_physical_activity_minutes_total',
            **RUWE_WAARDEN,
        }
        unieke_corr_kolommen = set()
        alle_corr_vars = {
            label: kolom
            for label, kolom in alle_corr_vars.items()
            if not (kolom in unieke_corr_kolommen or unieke_corr_kolommen.add(kolom))
        }
        beschikbare_corr = [label for label, kolom in alle_corr_vars.items() if kolom in df.columns]
        gekozen_corr = st.multiselect(
            t("Variabelen meenemen"),
            beschikbare_corr,
            default=[label for label in [t("Leefstijlscore"), t("BMI (waarde)"), t("Heartrisk (score)"), t("Stress (score)")] if label in beschikbare_corr] or beschikbare_corr,
            help=t("Kies welke variabelen in de correlatiematrix worden meegenomen."),
        )
        min_n = st.slider(
            t("Minimaal aantal deelnemers"),
            min_value=1,
            max_value=50,
            value=5,
            help=t("Alleen variabelen met ten minste dit aantal deelnemers worden meegenomen."),
        )
        if gekozen_corr:
            gekozen_kolommen = [alle_corr_vars[label] for label in gekozen_corr if label in alle_corr_vars]
            df_corr_input = df[[c for c in df.columns if c in gekozen_kolommen or c in {'rec_user_gender'}]].copy()
            if geslacht == 'beide':
                col_man, col_vrouw = st.columns(2)
                with col_man:
                    st.markdown(f"**{t('Mannen')}**")
                    p(maak_correlatiematrix(filter_geslacht(df_corr_input, 'man'), min_n=min_n), participants_df=df)
                with col_vrouw:
                    st.markdown(f"**{t('Vrouwen')}**")
                    p(maak_correlatiematrix(filter_geslacht(df_corr_input, 'vrouw'), min_n=min_n), participants_df=df)
            else:
                p(maak_correlatiematrix(df_corr_input, min_n=min_n), participants_df=df)
        else:
            st.info(t("Selecteer minimaal één variabele."))
        st.caption(t("Alleen variabelen met minimaal {min_n} deelnemers worden meegenomen.", min_n=min_n))

    with tab2:
        CONTINUE_VARS = {
            'Leefstijlscore':     'rec_ls_lifestyle_score',
            'BMI (waarde)':       'rec_med_bmi',
            'Heartrisk (score)':  'rec_heartrisk',
            'Framingham score':   'rec_framingham_non_invasive',
            'Stress (score)':     'rec_ls_stress_sum',
            'DASS stress':        'rec_dass_stress_score',
            'DASS angst':         'rec_dass_anxiety_score',
            'DASS depressie':     'rec_dass_depression_score',
            'Slaap PSQI':         'rec_ls_sleep_psqi_sum',
            'Veerkracht':         'rec_resilience_score',
            'Welzijn':            'rec_wellbeing_score',
            'Werkvermogen (WAI)': 'rec_asr_wai_score',
            'Werkdruk':           'rec_asr_workload_score',
            'Uitputting':         'rec_asr_exhaustion_score',
            'Burn-out score':     'rec_asr_burn_out_score',
            'Vitaliteit':         'rec_asr_vitality_score',
            'Werktevredenheid':   'rec_asr_job_satisfaction_score',
            'Leeftijd':           'rec_age_current',
            'Stappen per dag':    'rec_ls_exercise_steps_per_day',
            'Beweegminuten':      'rec_ls_exercise_physical_activity_minutes_total',
            **RUWE_WAARDEN,
        }
        alle_vars   = CONTINUE_VARS
        beschikbaar = {k: v for k, v in alle_vars.items() if v in df.columns}
        labels      = list(beschikbaar.keys())

        col1, col2 = st.columns(2)
        with col1:
            x_keuze = st.selectbox(t("X-as"), labels,
                                   index=labels.index('Stress (score)') if 'Stress (score)' in labels else 0)
        with col2:
            y_keuze = st.selectbox(t("Y-as"), labels,
                                   index=labels.index('BMI (waarde)') if 'BMI (waarde)' in labels else min(1, len(labels)-1))

        fig = maak_scatter_correlatie(df, x_keuze, beschikbaar[x_keuze],
                                       y_keuze, beschikbaar[y_keuze], geslacht_filter=geslacht)
        p(fig, participants_df=df)
        st.caption(
            t("Trendlijn per geslacht via OLS regressie. r = Pearson correlatie, p = significantiewaarde.")
        )

# ═════════════════════════════════════════════════════════════════════════════
# PRODUCTEFFECT# ═════════════════════════════════════════════════════════════════════════════
# PER OPDRACHTGEVER
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🏢 Per opdrachtgever":
    st.title(tr("Scores per opdrachtgever", lang))
    geslacht_badge(geslacht)
    st.caption(t("Vergelijking van gemiddelde scores per opdrachtgever (store)."))

    # ── TAB 1: Snapshot scores per opdrachtgever ───────────────────────────────
    opdrachtgever_tab_labels = [t("📊 Snapshot scores"), t("📈 Score verbetering over tijd")]
    tab1, tab2 = st.tabs(
        opdrachtgever_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, opdrachtgever_tab_labels),
    )

    with tab1:
        INDICATOREN_OPDRACHTGEVER = {
            t(label): kolom
            for label, kolom in _beschikbare_opdrachtgever_indicatoren(load_main_data_filtered_by_store()).items()
        }

        # Fallback: als geen indicatoren beschikbaar zijn (bijv. vanwege store_id mapping issues),
        # gebruik standaard indicatoren
        if not INDICATOREN_OPDRACHTGEVER:
            logger.warning("Geen opdrachtgever indicatoren beschikbaar - fallback naar standaard")
            INDICATOREN_OPDRACHTGEVER = {
                t('Leefstijlscore'): 'rec_ls_lifestyle_score',
                t('BMI'): 'rec_med_bmi',
                t('Stress'): 'rec_ls_stress_sum',
                t('DASS stress'): 'rec_dass_stress_score',
                t('DASS angst'): 'rec_dass_anxiety_score',
                t('Heartrisk score'): 'rec_heartrisk',
                t('Werkvermogen (WAI)'): 'rec_asr_work_ability_score',
                t('Burn-out risico'): 'rec_asr_burn_out_score',
                t('Werktevredenheid'): 'rec_asr_job_satisfaction_score',
                t('Fruit (stuks/dag)'): 'rec_ls_score_fruit',
                t('Groenten (gram/dag)'): 'rec_ls_score_vegetables',
                t('Suiker (gram/dag)'): 'rec_ls_score_sugar',
                t('Verzadigd vet (gram/dag)'): 'rec_ls_score_saturated_fat',
                t('Natrium (mg/dag)'): 'rec_ls_score_natrium',
                t('Beweegminuten (min/week)'): 'rec_ls_score_exercise',
            }

        indicator_keuze = st.selectbox(
            t("Kies een indicator"), list(INDICATOREN_OPDRACHTGEVER.keys())
        )
        min_n = MIN_VISUALISATIE_DEELNEMERS

        # Defensive: check of indicator_keuze geldig is
        if indicator_keuze is None or indicator_keuze not in INDICATOREN_OPDRACHTGEVER:
            st.warning(t("Geen opdrachtgever beschikbaar met deze indicatoren"))
            st.stop()

        indicator_kolom = INDICATOREN_OPDRACHTGEVER[indicator_keuze]

        if geslacht == 'beide':
            st.markdown(f"**{t('Opdrachtgeverresultaten per geslacht')}**")
            gender_sets = _gender_participant_sets(load_main_data_filtered_by_store())
            gender_cols = st.columns(2)
            for gender_label, gender_col in zip(('Man', 'Vrouw'), gender_cols):
                with gender_col:
                    gender_result = get_scores_opdrachtgever(
                        indicator_kolom, indicator_keuze, min_n,
                        participant_ids=gender_sets[gender_label],
                    )
                    p(gender_result[0], key=f"store_snapshot_{gender_label.lower()}")
                    st.caption(f"{gender_label}: {gender_result[2]} {t('opdrachtgevers')}")

        try:
            result = get_scores_opdrachtgever(
                indicator_kolom, indicator_keuze, min_n
            )
            fig, totaal_gem, n_stores = result[0], result[1], result[2]
            cat_counts = result[3] if len(result) > 3 else {}
            afkap_info = result[4] if len(result) > 4 else {}
            fig_verdeling = result[5] if len(result) > 5 else None
            verdeling_tabel = result[6] if len(result) > 6 else None

            waarschuwing = afkap_info.get('waarschuwing') if isinstance(afkap_info, dict) else None
            if waarschuwing:
                st.warning(waarschuwing)

            if indicator_kolom in {'rec_med_bmi', 'rec_med_bmi_cat'}:
                categorieen = [
                    BMI_LABELS[code]
                    for code in sorted(BMI_LABELS)
                    if cat_counts.get(BMI_LABELS[code])
                ]
            else:
                categorieen = [c for c in ('Hoog', 'Goed', 'Matig', 'Slecht', 'Laag') if cat_counts.get(c)]
            cols = st.columns(2 + len(categorieen))
            cols[0].metric(t("Opdrachtgevers getoond"), n_stores)
            cols[1].metric(f"Totaalgemiddelde {indicator_keuze}", f"{totaal_gem:.2f}")
            for i, cat in enumerate(categorieen):
                cols[2 + i].metric(cat, cat_counts.get(cat, 0))

            optie_gem = t("Gemiddelde per opdrachtgever")
            optie_verd = t("Verdeling van individuen (%)")
            if fig_verdeling is not None:
                weergave = st.radio(
                    t("Weergave"),
                    [optie_gem, optie_verd],
                    horizontal=True,
                    key=f"afkap_weergave_{indicator_kolom}",
                )
            else:
                weergave = optie_gem
            p(fig if weergave == optie_gem else fig_verdeling)

            uitleg = afkap_info.get('uitleg') if isinstance(afkap_info, dict) else None
            if not uitleg:
                uitleg = t("Kleuren: groen = goed (≥100% van totaalgemiddelde), oranje = matig (80-100%), rood = slecht (<80%).")
            st.caption(
                uitleg + " " +
                t("Let op: verschillen tussen opdrachtgevers kunnen samenhangen met de samenstelling van de populatie (leeftijd, geslacht) en niet alleen met de effectiviteit van interventies.")
            )

            if fig_verdeling is not None and verdeling_tabel is not None and not verdeling_tabel.empty:
                with st.expander(t("Tabel: aandeel individuen per categorie per opdrachtgever")):
                    st.dataframe(
                        translate_dataframe(verdeling_tabel.set_index('store_naam'), lang),
                        width="stretch",
                    )

            st.markdown("---")
            st.subheader(t("T-toets: opdrachtgever vs totaal"))
            store_id = get_store_filter_id()
            if store_id is None:
                st.info(t("Selecteer eerst een opdrachtgever in de globale filters."))
            else:
                alle_store_vs_total_vars = {
                    **LEEFSTIJL_SCORES,
                    **RUWE_WAARDEN,
                    **T_TOETS_EXTRA_VARIABELEN,
                }
                beschikbare_store_vs_total_vars = [
                    label
                    for label, kolom in alle_store_vs_total_vars.items()
                    if kolom in load_main_data_filtered_by_store().columns
                ]
                gekozen_store_vars = st.multiselect(
                    t("Variabelen meenemen"),
                    beschikbare_store_vs_total_vars,
                    default=(
                        [indicator_keuze]
                        if indicator_keuze in beschikbare_store_vs_total_vars
                        else beschikbare_store_vs_total_vars[:5]
                    ),
                    help=t("Kies welke variabelen je wilt vergelijken tussen de geselecteerde opdrachtgever en het totaal."),
                    key="store_vs_total_vars",
                )
                if gekozen_store_vars:
                    gekozen_store_kolommen = [
                        alle_store_vs_total_vars[label]
                        for label in gekozen_store_vars
                        if label in alle_store_vs_total_vars
                    ]
                    df_store_vs_total = _bereken_store_vs_totaal_t_toets(load_main_data_filtered_by_store(), store_id, gekozen_store_kolommen)
                    if df_store_vs_total.empty:
                        st.info(t("Onvoldoende data voor deze vergelijking."))
                    else:
                        st.dataframe(translate_dataframe(df_store_vs_total, lang), width="stretch", hide_index=True)
                        st.caption(t("Welch t-toets: geselecteerde opdrachtgever vergeleken met het totaal binnen de huidige filters."))
                else:
                    st.info(t("Selecteer minimaal één variabele."))
        except Exception as e:
            st.error(t("Fout: {e}", e=e))


    with tab2:
        st.write(f"**{t('Scoreontwikkeling per opdrachtgever over tijd')}**")
        st.caption(t("Deze twee grafieken tonen per bedrijf het maandelijkse gemiddelde en de maand-op-maand verandering."))

        STORE_SCORE_INDICATOREN = {
            t('Leefstijlscore'): {'slug': 'rec_ls_lifestyle_score'},
            t('BMI'): {'slug': 'rec_med_bmi'},
            t('Heartrisk score'): {'slug': 'rec_heartrisk'},
            t('Stress'): {'slug': 'rec_ls_stress_sum'},
            t('DASS stress'): {'slug': 'rec_dass_stress_score'},
            t('DASS angst'): {'slug': 'rec_dass_anxiety_score'},
            t('DASS depressie'): {'slug': 'rec_dass_depression_score'},
            t('Slaap (PSQI)'): {'slug': 'rec_ls_sleep_psqi_sum'},
            t('Veerkracht'): {'slug': 'rec_resilience_score'},
            t('Welzijn'): {'slug': 'rec_wellbeing_score'},
            t('Zelfeffectiviteit'): {'slug': 'rec_self_efficacy_score'},
            t('Werktevredenheid'): {'slug': 'rec_asr_job_satisfaction_score'},
            t('Werkhouding'): {'slug': 'rec_asr_working_attitude_score'},
            t('Werkdruk'): {'slug': 'rec_asr_workload_score'},
            t('Burn-out score'): {'slug': 'rec_asr_burn_out_score'},
            t('Werkvermogen (WAI)'): {'slug': 'rec_asr_wai_score'},
            t('Beweging (minuten)'): {'slug': 'rec_ls_exercise_physical_activity_minutes_total'},
            t('Beweging (stappen)'): {'slug': 'rec_ls_exercise_steps_per_day'},
            t('Fruit'): {'slug': 'rec_ls_score_fruit'},
            t('Groenten'): {'slug': 'rec_ls_score_vegetables'},
            t('Suiker'): {'slug': 'rec_ls_score_sugar'},
            t('Vet'): {'slug': 'rec_ls_score_saturated_fat'},
            t('Zout'): {'slug': 'rec_ls_score_natrium'},
            t('Alcohol'): {'slug': 'rec_ls_score_alcohol'},
            t('Roken'): {'slug': 'rec_smoking_answer'},
        }

        col1, col3 = st.columns([2, 1])
        with col1:
            score_label_keuze = st.selectbox(
                t("Welke score wil je volgen?"),
                list(STORE_SCORE_INDICATOREN.keys()),
            )
        min_deelnemers = MIN_VISUALISATIE_DEELNEMERS

        score_info = STORE_SCORE_INDICATOREN[score_label_keuze]
        score_slug = score_info['slug']

        with col3:
            st.markdown(f"**{score_label_keuze}**")

        try:
            stores_df = get_store_average_stores_with_changes(score_slug, min_deelnemers)
            stores_list = [(f"🌍 {t('Alle bedrijven')}", None)] + [
                (row['store_name'], row['store_id']) for _, row in stores_df.iterrows()
            ]
        except Exception as e:
            st.warning(t("Niet alle bedrijven konden geladen worden: {e}", e=e))
            stores_list = [(f"🌍 {t('Alle bedrijven')}", None)]

        col4, _ = st.columns([2, 1])
        with col4:
            selected_store_option = st.selectbox(
                t("📊 Bedrijf"),
                stores_list,
                format_func=lambda x: x[0] if isinstance(x, tuple) else x
            )
            selected_store_id = selected_store_option[1] if isinstance(selected_store_option, tuple) else None

        try:
            if geslacht == 'beide':
                st.markdown(f"**{t('Scoreontwikkeling per geslacht')}**")
                gender_sets = _gender_participant_sets(load_main_data_filtered_by_store())
                trend_cols = st.columns(2)
                for gender_label, gender_col in zip(('Man', 'Vrouw'), trend_cols):
                    with gender_col:
                        p(get_store_gemiddelde_scores(
                            score_slug, score_label_keuze, min_deelnemers,
                            store_id=selected_store_id,
                            participant_ids=gender_sets[gender_label],
                        ), key=f"store_trend_{gender_label.lower()}")
                        p(get_store_gemiddelde_verandering(
                            score_slug, score_label_keuze, min_deelnemers,
                            store_id=selected_store_id,
                            participant_ids=gender_sets[gender_label],
                        ), key=f"store_change_{gender_label.lower()}")

            with st.spinner(t("Scoreontwikkeling laden...")):
                fig_scores = get_store_gemiddelde_scores(
                    score_slug=score_slug,
                    score_label=score_label_keuze,
                    min_deelnemers=min_deelnemers,
                    store_id=selected_store_id,
                )
                fig_verandering = get_store_gemiddelde_verandering(
                    score_slug=score_slug,
                    score_label=score_label_keuze,
                    min_deelnemers=min_deelnemers,
                    store_id=selected_store_id,
                )
            p(fig_scores)
            p(fig_verandering)
            st.caption(t("De veranderingsgrafiek toont het verschil ten opzichte van de vorige maand."))
        except Exception as e:
            st.error(t("Fout bij laden van scoreontwikkeling: {e}", e=e))

# ═════════════════════════════════════════════════════════════════════════════
# PRODUCTEFFECT
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🛒 Producteffect":
    st.title(tr("Producteffect op gezondheidsscores", lang))
    st.caption(
        t("Vergelijking van gemiddelde scores tussen gebruikers die een product hebben gekocht en gebruikers die niets hebben gekocht.")
    )

    st.markdown("---")

    try:
        with st.spinner(t("Data laden en koppelen...")):
            df_product = load_main_data_filtered_by_gender_and_store(geslacht)
            fig_kopers, top_producten, n_kopers = get_kopers_data(_participant_ids(df_product))

        if geslacht == 'beide':
            st.markdown(f"**{t('Producteffect per geslacht')}**")
            gender_sets = _gender_participant_sets(df_product)
            product_cols = st.columns(2)
            for gender_label, product_col in zip(('Man', 'Vrouw'), product_cols):
                with product_col:
                    gender_fig, gender_products, gender_n = get_kopers_data(gender_sets[gender_label])
                    st.markdown(f"**{gender_label}**")
                    c_buy, c_no_buy = st.columns(2)
                    c_buy.metric(t("Met aankoop"), f"{gender_n:,}")
                    c_no_buy.metric(t("Zonder aankoop"), f"{max(len(gender_sets[gender_label]) - gender_n, 0):,}")
                    p(gender_fig, participants_df=df_product, key=f"producteffect_gender_{gender_label.lower()}")
                    if not gender_products.empty:
                        st.dataframe(translate_dataframe(gender_products, lang), width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        c1.metric(t("Gebruikers met ≥1 aankoop"), f"{n_kopers:,}")
        c2.metric(t("Gebruikers zonder aankoop"),
                  f"{max(len(df_product) - n_kopers, 0):,}")
        st.markdown("---")
        p(fig_kopers, participants_df=df_product, key="producteffect_kopers")
        st.markdown("---")
        st.subheader(t("Meest gekochte producten"))
        if not top_producten.empty:
            top_fig = px.bar(
                top_producten.head(10),
                x='Product',
                y='Aantal orders',
                color='Aantal orders',
                color_continuous_scale='Blues',
                text='Aantal orders',
            )
            top_fig.update_layout(
                xaxis_title=t('Product'),
                yaxis_title=t('Aantal orders'),
                showlegend=False,
                height=420,
            )
            p(top_fig, participants_df=df_product, key="producteffect_topproducten")
        st.dataframe(translate_dataframe(top_producten, lang), width="stretch", hide_index=True)

    except Exception as e:
        st.error(t("Fout: {e}", e=e))

# ═════════════════════════════════════════════════════════════════════════════
# VRAGENLIJSTGEDRAG
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "📋 Vragenlijstgedrag":
    st.title(tr("Vragenlijstgedrag", lang))
    geslacht_badge(geslacht)
    st.caption(t("Wie vullen meerdere vragenlijsten in, welke combinaties komen het meest voor, wie herhaalt dezelfde vragenlijst en waar lijkt afhaken op te treden?"))

    st.markdown("---")

    try:
        with st.spinner(t("Data laden...")):
            df_vragenlijst_basis = load_main_data_filtered_by_gender_and_store(geslacht)
            actieve_participant_ids = _participant_ids(df_vragenlijst_basis)
            fig_invul, df_vl = get_vragenlijst_data(actieve_participant_ids)
            fig_vl, top_combinaties, samenvatting, per_participant = get_vragenlijst_gedrag(actieve_participant_ids)
            herhalingen, intervallen = get_vragenlijst_herhalingen(actieve_participant_ids)
            df_dropoff, df_dropoff_detail, dropoff_summary = get_vragenlijst_dropoff(actieve_participant_ids)

        if geslacht == 'beide':
            st.markdown(f"**{t('Vragenlijstgedrag per geslacht')}**")
            gender_sets = _gender_participant_sets(df_vragenlijst_basis)
            questionnaire_cols = st.columns(2)
            for gender_label, questionnaire_col in zip(('Man', 'Vrouw'), questionnaire_cols):
                with questionnaire_col:
                    gender_ids = gender_sets[gender_label]
                    gender_invul, _ = get_vragenlijst_data(gender_ids)
                    gender_gedrag, _, gender_summary, _ = get_vragenlijst_gedrag(gender_ids)
                    gender_repeat, gender_intervals = get_vragenlijst_herhalingen(gender_ids)
                    _, _, gender_dropoff_summary = get_vragenlijst_dropoff(gender_ids)
                    st.markdown(f"**{gender_label}**")
                    st.metric(t("Gemiddeld aantal vragenlijsten"), f"{gender_summary['Gemiddeld aantal']:.2f}")
                    st.metric(t("Herhaalinvullers"), f"{gender_repeat['participant_id'].nunique():,}")
                    st.metric(t("Median completion"), f"{gender_dropoff_summary.get('Median completion (%)', 0):.1f}%")
                    p(gender_invul, key=f"vragenlijst_invul_{gender_label.lower()}")
                    p(gender_gedrag, key=f"vragenlijst_gedrag_{gender_label.lower()}")

        # KPI metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(t("Totaal deelnemers"), f"{samenvatting['Totaal deelnemers']:,}")
        c2.metric(t("Slechts 1 vragenlijst"), f"{samenvatting['Slechts 1 vragenlijst']:,}",
                  delta=f"{samenvatting['Slechts 1 vragenlijst']/samenvatting['Totaal deelnemers']*100:.1f}%",
                  delta_color="off")
        c3.metric(t("2 of meer vragenlijsten"), f"{samenvatting['2 of meer vragenlijsten']:,}",
                  delta=f"{samenvatting['2 of meer vragenlijsten']/samenvatting['Totaal deelnemers']*100:.1f}%",
                  delta_color="off")
        c4.metric(t("Alle 8 vragenlijsten"), f"{samenvatting['Alle vragenlijsten (8)']:,}")
        c5.metric(t("Gemiddeld aantal"), f"{samenvatting['Gemiddeld aantal']:.2f}")

        st.markdown("---")

        vragenlijst_tab_labels = [t("Invulpercentages"), t("Populairste combinaties"), t("Herhaalinvullers"), t("Afhaakgedrag")]
        tab1, tab2, tab3, tab4 = st.tabs(
            vragenlijst_tab_labels,
            default=_tab_default_for_path(pending_navigation_path, vragenlijst_tab_labels),
        )

        with tab1:
            st.subheader(t("Deelname per vragenlijst"))
            p(fig_invul, participants_df=per_participant)
            p(fig_vl, participants_df=per_participant)
            st.caption(t("Oranje = slechts 1 vragenlijst ingevuld. Groen = meerdere vragenlijsten."))

        with tab2:
            st.subheader(t("Vragenlijstcombinaties"))
            st.dataframe(
                translate_dataframe(top_combinaties, lang),
                width="stretch",
                hide_index=True,
            )
            st.caption(t("Combinaties zijn gesorteerd op aantal deelnemers dat exact deze set heeft ingevuld."))

        with tab3:
            if herhalingen.empty:
                st.info(t("Geen herhaalde invullingen van dezelfde vragenlijst gevonden."))
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(t("Herhaalinvullers"), f"{herhalingen['participant_id'].nunique():,}")
                c2.metric(t("Herhaalde trajecten"), f"{len(herhalingen):,}")
                gem_dagen = float(intervallen['gem_interval_dagen'].mean()) if not intervallen.empty else 0.0
                c3.metric(t("Gem. dagen ertussen"), f"{gem_dagen:.1f}")
                if not intervallen.empty:
                    df_herhaal_tabel = intervallen.sort_values('dagen_totaal', ascending=False).drop(columns=['participant_id', 'questionnaire_id'], errors='ignore')
                    st.dataframe(
                        translate_dataframe(df_herhaal_tabel, lang),
                        width="stretch",
                        hide_index=True,
                    )
                st.caption(t("Dit laat zien wie dezelfde vragenlijst opnieuw invult en hoeveel dagen er gemiddeld tussen invulmomenten zitten."))

        with tab4:
            if df_dropoff.empty:
                st.info(t("Geen data beschikbaar voor afhaakanalyse."))
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(t("Median completion"), f"{dropoff_summary.get('Median completion (%)', 0):.1f}%")
                c2.metric(t("Afhakers <50%"), f"{dropoff_summary.get('Afhakers <50% (%)', 0):.1f}%")
                corr = dropoff_summary.get('Correlatie lengte vs afhaken')
                c3.metric(t("Lengte vs afhaken"), "-" if corr is None or pd.isna(corr) else f"{corr:.3f}")
                st.dataframe(
                    translate_dataframe(df_dropoff, lang),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(t("Afhaken is hier benaderd via het aandeel beantwoorde score-items per completion. Positieve correlatie betekent: langere vragenlijsten lijken vaker eerder te stoppen."))

    except Exception as e:
        st.error(t("Fout: {e}", e=e))

# ═════════════════════════════════════════════════════════════════════════════
# KOPERSPROFIEL
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🛍️ Kopersprofiel":
    st.title(tr("Kopersprofiel: wie kopen er eerder?", lang))
    st.caption(t(
        "Vergelijking van vroege kopers (eerste aankoop ≤90 dagen na registratie), "
        "late kopers en niet-kopers op gezondheids- en leefstijlscores."
    ))

    st.markdown("---")

    try:
        groep_opties = [
            'Vroege koper (≤90 dagen)',
            'Late koper (91-365 dagen)',
            'Zeer late koper (>365 dagen)',
            'Geen aankoop',
        ]
        gekozen_groepen = st.multiselect(
            t("Toon kopersgroepen"),
            groep_opties,
            default=groep_opties,
        )
        include_wellbeing = st.checkbox(
            t("Welzijn meenemen"),
            value=False,
            help=t("Staat standaard uit omdat hier vaak weinig waarnemingen beschikbaar zijn."),
        )

        if not gekozen_groepen:
            st.info(t("Selecteer minimaal één kopersgroep."))
            st.stop()

        with st.spinner(t("Data laden en koppelen...")):
            df_kopers_basis = load_main_data_filtered_by_gender_and_store(geslacht)
            fig_profiel, fig_dagen, profiel, df_scores_groepen = get_vroege_kopers_profiel_filtered(
                tuple(gekozen_groepen), include_wellbeing, _participant_ids(df_kopers_basis)
            )

        if geslacht == 'beide':
            st.markdown(f"**{t('Kopersprofiel per geslacht')}**")
            gender_sets = _gender_participant_sets(df_kopers_basis)
            profile_cols = st.columns(2)
            for gender_label, profile_col in zip(('Man', 'Vrouw'), profile_cols):
                with profile_col:
                    gender_fig, gender_days, gender_profile, gender_scores = get_vroege_kopers_profiel_filtered(
                        tuple(gekozen_groepen), include_wellbeing, gender_sets[gender_label]
                    )
                    st.markdown(f"**{gender_label}**")
                    st.metric(t("Geselecteerde gebruikers"), f"{gender_profile['Aantal geselecteerde gebruikers']:,}")
                    st.metric(t("Gem. leefstijlscore"), f"{gender_profile['Gem. leefstijlscore']:.2f}")
                    p(gender_fig, participants_df=gender_scores, key=f"kopersprofiel_gender_{gender_label.lower()}")
                    p(gender_days, participants_df=gender_scores, key=f"kopersdagen_gender_{gender_label.lower()}")

        # KPI metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(t("Geselecteerde gebruikers"), f"{profiel['Aantal geselecteerde gebruikers']:,}")
        c2.metric(t("Gem. leeftijd"), f"{profiel['Gemiddelde leeftijd']:.1f} jaar")
        c3.metric(t("% vrouw"), f"{profiel['% vrouw']:.1f}%")
        c4.metric(t("Gem. leefstijlscore"), f"{profiel['Gem. leefstijlscore']:.2f}")
        c5.metric(t("Gem. dagen tot aankoop"), f"{profiel['Gem. dagen tot aankoop']:.0f}")

        st.markdown("---")

        kopers_tab_labels = [t("Scores per groep"), t("Verdeling aankoopdatum")]
        tab1, tab2 = st.tabs(
            kopers_tab_labels,
            default=_tab_default_for_path(pending_navigation_path, kopers_tab_labels),
        )

        with tab1:
            p(fig_profiel, participants_df=df_scores_groepen)
            st.caption(t(
                "Groen = vroege kopers (≤90 dagen), oranje = late kopers (91-365 dagen), "
                "rood = zeer late kopers (>365 dagen), grijs = geen aankoop. "
                "Let op: dit is een descriptieve analyse. Selectiebias is aannemelijk."
            ))
            st.markdown("---")
            st.subheader(t("Ruwe scores per groep"))
            st.dataframe(
                translate_dataframe(df_scores_groepen, lang),
                width="stretch",
                hide_index=True,
            )

        with tab2:
            p(fig_dagen, participants_df=df_scores_groepen)
            st.caption(t(
                "Verdeling van het aantal dagen tussen registratie en eerste aankoop. "
                "Groene lijn = 90 dagen grens (definitie vroege koper). "
                "Alleen kopers met 0-730 dagen worden getoond."
            ))

    except Exception as e:
        st.error(t("Fout: {e}", e=e))
elif pagina == "🔎 Snelle statistieken":
    st.title(tr("Snelle statistieken", lang))
    st.caption(tr("Stel filters in en zie direct de statistieken voor die subgroep.", lang))

    df_alle = load_main_data_filtered_by_store()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown(f"### {t('Filters')}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        geslacht_filter = st.selectbox(
            t("Geslacht"), [t("Beide"), t("Man"), t("Vrouw")], key="qs_geslacht"
        )
    with col2:
        leeftijd_min, leeftijd_max = st.slider(
            t("Leeftijd"), min_value=16, max_value=86, value=(16, 86), key="qs_leeftijd"
        )
    with col3:
        roken_filter = st.selectbox(
            t("Rookstatus"), [t("Alle"), t("Nooit gerookt"), t("Ex-roker"), t("Roker")], key="qs_roken"
        )
    with col4:
        bmi_filter_labels = [BMI_LABELS[code] for code in sorted(BMI_LABELS)]
        bmi_filter = st.selectbox(
            t("BMI categorie"), [t("Alle"), *bmi_filter_labels], key="qs_bmi"
        )

    # ── Filtering toepassen ───────────────────────────────────────────────────
    df = df_alle.copy()

    # Geslacht
    if geslacht_filter == t("Man"):
        df = df[pd.to_numeric(df['rec_user_gender'], errors='coerce') == 1]
    elif geslacht_filter == t("Vrouw"):
        df = df[pd.to_numeric(df['rec_user_gender'], errors='coerce') == 0]

    # Leeftijd
    leeftijd = pd.to_numeric(df['rec_age_current'], errors='coerce')
    df = df[(leeftijd >= leeftijd_min) & (leeftijd <= leeftijd_max)]

    # Roken
    roken_map = {
        t("Nooit gerookt"): [1],
        t("Ex-roker"):      [2],
        t("Roker"):         [3],
    }
    if roken_filter != t("Alle"):
        roken = pd.to_numeric(
            df['rec_smoking_answer'].replace('None', pd.NA), errors='coerce'
        )
        df = df[roken.isin(roken_map[roken_filter])]

    # BMI
    bmi_map = {label: [code] for code, label in BMI_LABELS.items()}
    if bmi_filter != t("Alle"):
        bmi = pd.to_numeric(df['rec_med_bmi_cat'], errors='coerce')
        df = df[bmi.isin(bmi_map[bmi_filter])]

    st.markdown("---")

    # ── Resultaat ─────────────────────────────────────────────────────────────
    n = len(df)
    n_totaal = len(df_alle)
    pct_van_totaal = round(n / n_totaal * 100, 1) if n_totaal > 0 else 0

    if n == 0:
        st.warning(t("Geen deelnemers gevonden met deze filtercombinatie."))
    else:
        st.markdown(t("Resultaten voor {n} deelnemers ({pct}% van totaal)", n=f"{n:,}", pct=f"{pct_van_totaal:.1f}"))

        # Demografisch
        st.markdown(f"#### 👥 {t('Demografisch')}")
        c1, c2, c3, c4, _ = st.columns(5)
        geslacht_s = pd.to_numeric(df['rec_user_gender'], errors='coerce')
        c1.metric("Vrouwen", f"{geslacht_s.eq(0).sum():,}",
                  delta=f"{geslacht_s.eq(0).mean()*100:.1f}%", delta_color="off")
        c2.metric("Mannen", f"{geslacht_s.eq(1).sum():,}",
                  delta=f"{geslacht_s.eq(1).mean()*100:.1f}%", delta_color="off")
        leeftijd_s = pd.to_numeric(df['rec_age_current'], errors='coerce')
        c3.metric("Gemiddelde leeftijd", f"{leeftijd_s.mean():.1f} jaar")
        c4.metric("Met postcode", f"{df['postal_code'].notna().sum():,}",
                  delta=f"{df['postal_code'].notna().mean()*100:.1f}%", delta_color="off")

        st.markdown(f"#### ❤️ {t('Gezondheid')}")
        c1, c2, c3, c4, c5 = st.columns(5)

        hr = pd.to_numeric(df['rec_heartrisk_cat'], errors='coerce')
        c1.metric("Hoog hart risico", f"{hr.eq(2).sum():,}",
                  delta=f"{hr.eq(2).mean()*100:.1f}%", delta_color="off")

        bmi_s = pd.to_numeric(df['rec_med_bmi'], errors='coerce')
        c2.metric("Gem. BMI", f"{bmi_s.mean():.1f}")

        stress = pd.to_numeric(df['rec_ls_stress_cat'], errors='coerce')
        c3.metric("Hoog stress", f"{stress.eq(2).sum():,}",
                  delta=f"{stress.eq(2).mean()*100:.1f}%", delta_color="off")

        slaap = pd.to_numeric(df['rec_ls_sleep_cat'], errors='coerce')
        c4.metric("Slechte slaap", f"{slaap.ge(1).sum():,}",
                  delta=f"{slaap.ge(1).mean()*100:.1f}%", delta_color="off")

        beweging = pd.to_numeric(df['derived_is_inactive'], errors='coerce')
        c5.metric("Niet fysiek actief", f"{beweging.eq(1).sum():,}",
              delta=f"{beweging.eq(1).mean()*100:.1f}%", delta_color="off")

        st.markdown(f"#### 🥗 {t('Leefstijl')}")
        c1, c2, c3, c4, c5 = st.columns(5)

        roken_s = pd.to_numeric(
            df['rec_smoking_answer'].replace('None', pd.NA), errors='coerce'
        )
        c1.metric("Rokers", f"{roken_s.eq(3).sum():,}",
              delta=f"{roken_s.eq(3).mean()*100:.1f}%", delta_color="off")

        alcohol = pd.to_numeric(df['rec_ls_alcohol_total_per_week'], errors='coerce')
        c2.metric("Gem. alcohol/week", f"{alcohol.mean():.1f}")

        groenten = pd.to_numeric(df['rec_ls_vegetables_gram_per_day'], errors='coerce')
        c3.metric("Gem. groenten/dag", f"{groenten.mean():.0f}")

        fruit = pd.to_numeric(df['rec_ls_nutrition_fruit_fruit_per_day'], errors='coerce')
        c4.metric("Gem. fruit/dag", f"{fruit.mean():.1f}")

        ls_score = pd.to_numeric(df['rec_ls_lifestyle_score'], errors='coerce')
        c5.metric("Gem. leefstijlscore", f"{ls_score.mean():.1f}")

        st.markdown(f"#### 🧠 {t('Psychologisch')}")
        c1, c2, c3, _, _ = st.columns(5)

        dass_stress = pd.to_numeric(df['rec_dass_stress_score'], errors='coerce')
        c1.metric("Gem. DASS stress", f"{dass_stress.mean():.1f}")

        veerkracht = pd.to_numeric(df['rec_resilience_score'], errors='coerce')
        c2.metric("Gem. veerkracht", f"{veerkracht.mean():.1f}")

        welzijn = pd.to_numeric(df['rec_wellbeing_score'], errors='coerce')
        c3.metric("Gem. welzijn", f"{welzijn.mean():.1f}")

        st.markdown("---")
        st.caption(
            t(
                "Filters toegepast: geslacht={gender}, leeftijd={age}, roken={smoking}, BMI={bmi}. N={n} ({pct}% van {total} totaal).",
                gender=geslacht_filter,
                age=f"{leeftijd_min}-{leeftijd_max}",
                smoking=roken_filter,
                bmi=bmi_filter,
                n=f"{n:,}",
                pct=f"{pct_van_totaal:.1f}",
                total=f"{n_totaal:,}",
            )
        )

# ═════════════════════════════════════════════════════════════════════════════
# ML MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 ML Model":
    st.title(tr("Machine Learning Modellen", lang))
    st.caption(
        t("Twee voorspellingsmodellen voor leefstijlscore en cardiovasculair risico.")
    )

    if not ML_AVAILABLE:
        st.warning(
            t(
                "De ML-module is niet beschikbaar in deze deployment. "
                "De rest van het dashboard werkt wel. Voeg het ontbrekende `ML.py`-bestand toe "
                "of schakel deze pagina uit voor productie."
            )
        )
        if ML_IMPORT_ERROR:
            st.code(ML_IMPORT_ERROR)
        st.stop()

    st.markdown("---")

    ml_cache_version = _get_ml_cache_version()
    c_refresh, c_version = st.columns([1, 3])
    # Removed global refresh button, now per-model refresh
    # with c_refresh:
    #     if st.button(t("🔄 Herlaad ML modellen")):
    #         get_ml_models_v2.clear()
    #         st.session_state["force_ml_retrain"] = True
    #         st.rerun()
    with c_refresh: # Keep the column for alignment, but no button
        st.caption( # Moved cache version here for better layout
            t(
                "ML cache-versie: {mtime}-{size}",
                mtime=ml_cache_version[0],
                size=ml_cache_version[1],
            )
        )

    # Load ML models
    try:
        with st.spinner(t("ML modellen trainen...")):
            force_ml = st.session_state.get("force_ml_retrain", False) # This is for global refresh, which is removed.
            model_to_retrain_single = st.session_state.get('model_to_retrain_single', None)
            logger.debug(f"Calling get_ml_models_v2 with force_retrain_all={force_ml}, model_name_to_retrain={model_to_retrain_single}")
            ml_data = get_ml_models_v2(ml_cache_version, force_retrain_all=force_ml, model_name_to_retrain=model_to_retrain_single)
            if force_ml:
                st.session_state["force_ml_retrain"] = False
            st.session_state['model_to_retrain_single'] = None # Clear after use
    except Exception as e:
        st.error(t("Fout bij laden van ML modellen: {e}", e=e))
        st.stop()

    # Dynamische tabs op basis van beschikbare ML functies
    ml_tab_names = []
    if maak_gebruikers_segmentatie_plot: ml_tab_names.append(t("👥 Gebruikerssegmentatie"))
    if 'lifestyle' in ml_data: ml_tab_names.append(t("🥗 Leefstijlmodel"))
    if 'next_lifestyle' in ml_data: ml_tab_names.append(t("🔮 Volgende leefstijlscore"))
    if 'dropoff' in ml_data: ml_tab_names.append(t("📋 Afhaakmodel"))
    if 'improvement' in ml_data: ml_tab_names.append(t("📈 Verbeteringsmodel"))
    if 'purchase' in ml_data: ml_tab_names.append(t("🛒 Productmodel"))
    if 'bp' in ml_data: ml_tab_names.append(t("🩸 Bloeddrukmodel"))
    if 'heartrisk' in ml_data: ml_tab_names.append(t("❤️ Hartrisicomodel"))

    # Altijd diagnostics tonen als er modellen ontbreken
    with st.expander(t("🛠️ ML Diagnostics & Systeeminformatie"), expanded=False):
        col_d1, col_d2 = st.columns(2)
        col_d1.write(f"**ML Module:** {'✅ Geladen' if ML_AVAILABLE else '❌ Fout'}")
        if not ML_AVAILABLE: col_d1.error(ML_IMPORT_ERROR)
        
        for key in ('lifestyle_error', 'next_lifestyle_error', 'dropoff_error', 'improvement_error', 'purchase_error', 'bp_error', 'heartrisk_error'):
            if key in ml_data:
                st.error(f"**{key.replace('_', ' ').title()}**: {ml_data[key]}")

    if not ml_tab_names:
        st.info(t("Geen van de ML-modellen kon worden getraind met de huidige dataset."))
        st.stop()

    tabs = st.tabs(
        ml_tab_names,
        default=_tab_default_for_path(pending_navigation_path, ml_tab_names),
    )
    tab_map = dict(zip(ml_tab_names, tabs))

    # ── GEBRUIKERSSEGMENTATIE ──────────────────────────────────────────────────
    if t("👥 Gebruikerssegmentatie") in tab_map:
        with tab_map[t("👥 Gebruikerssegmentatie")]:
            st.subheader(t("Gebruikerssegmentatie (Clustering)"))
            st.write(t("Dit model identificeert groepen gebruikers met vergelijkbare gezondheidsprofielen op basis van leefstijl, BMI en stress."))
            df_ml = load_ml_data()
            p(maak_gebruikers_segmentatie_plot(df_ml, lang=lang), key="ml_seg_plot", participants_df=df_ml)

    # ── LIFESTYLE MODEL ────────────────────────────────────────────────────────
    if t("🥗 Leefstijlmodel") in tab_map:
        with tab_map[t("🥗 Leefstijlmodel")]:
            st.subheader(t("Leefstijlscore Voorspelling"))
            st.write(
                t(
                    "Dit model voorspelt de leefstijlscore op basis van gezondheid, leefstijl, welzijn en demografie."
                )
            )
            if st.button(t("🔄 Herlaad Leefstijlmodel"), key="refresh_lifestyle_model"):
                get_ml_models_v2.clear() # Clear the main cache
                st.session_state['model_to_retrain_single'] = 'lifestyle'
                st.rerun()

            st.markdown("---")

            ls_data = ml_data.get('lifestyle')
            if not ls_data:
                st.info(t("Leefstijlmodel kon niet worden getraind met de huidige data."))
                st.stop()
            metrics_ls = ls_data['metrics']

            if 'Accuracy' in metrics_ls:
                # Classificatie weergave
                col1, col2 = st.columns(2)
                col1.metric(t("Accuracy"), f"{metrics_ls['Accuracy']:.1%}")
                col2.metric(t("Recall Macro"), f"{metrics_ls['Recall Macro']:.1%}")
                st.caption(
                    t("CV (5-fold): Gemiddelde Accuracy {acc:.1%}, Recall {rec:.1%}.", 
                      acc=metrics_ls.get('CV Accuracy', 0), rec=metrics_ls.get('CV Recall Macro', 0))
                )
            else:
                # Regressie weergave
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{metrics_ls['MAE']:.2f}")
                col2.metric("RMSE", f"{metrics_ls['RMSE']:.2f}")
                col3.metric(t("R² Score"), f"{metrics_ls['R² Score']:.2f}")
                st.caption(
                    t("CV (5-fold): Gemiddelde R² Score {r2:.2f}.",
                      r2=metrics_ls.get('CV R² Score', 0))
                )
                if metrics_ls['R² Score'] < 0:
                    st.warning(t("De R² Score is negatief, wat betekent dat het model slechter presteert dan een simpel gemiddelde."))

            st.markdown("---")

            # Feature importance
            st.markdown(f"### {t('Belangrijkste Voorspellers')}")
            fig_importance_ls = plot_feature_importance(
                ls_data['model'],
                ls_data['features'],
                "Leefstijlmodel",
            )
            fig_importance_ls = plot_feature_importance(ls_data['model'], ls_data['features'], "Leefstijlmodel")
            p(fig_importance_ls, key="ml_ls_importance")

            # Performance
            st.markdown(f"### {t('Voorspellingsnauwkeurigheid')}")
            fig_perf_ls = plot_prediction_performance_regression(ls_data['y_test'], ls_data['y_pred'])
            p(fig_perf_ls, key="ml_ls_performance")
            st.caption(t("Hoe dichter de punten langs de rode lijn liggen, hoe beter het model voorspelt."))
            if 'Accuracy' in metrics_ls:
                p(plot_confusion_matrix(ls_data['y_test'], ls_data['y_pred']), key="ml_ls_performance_clf")
            else:
                fig_perf_ls = plot_prediction_performance_regression(ls_data['y_test'], ls_data['y_pred'])
                p(fig_perf_ls, key="ml_ls_performance")
                st.caption(t("Hoe dichter de punten langs de rode lijn liggen, hoe beter het model voorspelt."))

            st.markdown("---")
            st.markdown(f"### {t('Eigen invoer')}")
            lifestyle_defaults = dict(getattr(ml_module, 'LIFESTYLE_INPUT_DEFAULTS', {}))
            c1, c2, c3 = st.columns(3)
            with c1:
                in_age_ls = st.slider(t("Leeftijd"), 18, 85, int(lifestyle_defaults['rec_age_current']), key="ml_ls_age")
                in_gender_ls = st.selectbox(t("Geslacht"), [t("Vrouw"), t("Man")], key="ml_ls_gender")
                in_bmi_ls = st.slider(t("BMI"), 15.0, 45.0, lifestyle_defaults['rec_med_bmi'], 0.1, key="ml_ls_bmi")
                in_stress_ls = st.slider(t("Stress (0-28)"), 0.0, 28.0, lifestyle_defaults['rec_ls_stress_sum'], 0.1, key="ml_ls_stress")
                in_sleep_ls = st.slider(t("Slaap PSQI (0-21)"), 0.0, 21.0, lifestyle_defaults['rec_ls_sleep_psqi_sum'], 0.1, key="ml_ls_sleep")
                in_steps_ls = st.slider(t("Beweging (duizend stappen/dag)"), 0.0, 25.0, lifestyle_defaults['rec_ls_exercise_steps_per_day'], 0.1, key="ml_ls_steps")
                in_activity_minutes_ls = st.slider(t("Beweegminuten per week"), 0.0, 10080.0, lifestyle_defaults['rec_ls_exercise_physical_activity_minutes_total'], 10.0, key="ml_ls_activity_minutes")
            with c2:
                in_fruit_ls = st.slider(t("Fruit per dag"), 0.0, 10.0, lifestyle_defaults['rec_ls_nutrition_fruit_fruit_per_day'], 0.1, key="ml_ls_fruit")
                in_vegetables_ls = st.slider(t("Groenten (gram/dag)"), 0.0, 1000.0, lifestyle_defaults['rec_ls_vegetables_gram_per_day'], 10.0, key="ml_ls_vegetables")
                in_sugar_ls = st.slider(t("Suiker (gram/dag)"), 0.0, 200.0, lifestyle_defaults['rec_ls_nutrition_sugar_per_day'], 1.0, key="ml_ls_sugar")
                in_saturated_fat_ls = st.slider(t("Verzadigd vet (gram/dag)"), 0.0, 100.0, lifestyle_defaults['rec_ls_nutrition_saturated_fat_per_day'], 1.0, key="ml_ls_saturated_fat")
                in_natrium_ls = st.slider(t("Natrium (mg/dag)"), 0.0, 10000.0, lifestyle_defaults['rec_ls_nutrition_natrium_per_day'], 100.0, key="ml_ls_natrium")
                in_alcohol_ls = st.slider(t("Alcohol per week"), 0.0, 60.0, lifestyle_defaults['rec_ls_alcohol_total_per_week'], 1.0, key="ml_ls_alcohol")
            with c3:
                in_smoking_ls = st.selectbox(
                    t("Roken"),
                    [(t("Onbekend"), 0.0), (t("Nooit gerookt"), 1.0), (t("Ex-roker"), 2.0), (t("Roker"), 3.0)],
                    index=1,
                    format_func=lambda option: option[0],
                    key="ml_ls_smoking",
                )[1]
                in_res_ls = st.slider(t("Veerkracht"), 0.0, 10.0, lifestyle_defaults['rec_resilience_score'], 0.1, key="ml_ls_res")
                in_well_ls = st.slider(t("Welzijn"), 0.0, 10.0, lifestyle_defaults['rec_wellbeing_score'], 0.1, key="ml_ls_well")

            ls_input = pd.DataFrame([{
                'rec_age_current': in_age_ls,
                'rec_user_gender': 1 if in_gender_ls == t("Man") else 0,
                'rec_med_bmi': in_bmi_ls,
                'rec_ls_stress_sum': in_stress_ls,
                'rec_ls_sleep_psqi_sum': in_sleep_ls,
                'rec_ls_exercise_steps_per_day': in_steps_ls,
                'rec_ls_exercise_physical_activity_minutes_total': in_activity_minutes_ls,
                'rec_ls_nutrition_fruit_fruit_per_day': in_fruit_ls,
                'rec_ls_vegetables_gram_per_day': in_vegetables_ls,
                'rec_ls_nutrition_sugar_per_day': in_sugar_ls,
                'rec_ls_nutrition_saturated_fat_per_day': in_saturated_fat_ls,
                'rec_ls_nutrition_natrium_per_day': in_natrium_ls,
                'rec_ls_alcohol_total_per_week': in_alcohol_ls,
                'rec_smoking_answer': in_smoking_ls,
                'rec_resilience_score': in_res_ls,
                'rec_wellbeing_score': in_well_ls,
            }])
            
            ls_pred = predict_lifestyle(ls_data['model'], ls_input)[0]
            st.metric(t("Voorspelde leefstijlscore"), f"{ls_pred:.2f}")
            
            # SHAP
            fig_shap_ls = plot_local_shap(ls_data, ls_input, "Leefstijl") if plot_local_shap else None
            if fig_shap_ls:
                st.markdown(f"#### {t('Waarom deze score?')}")
                p(fig_shap_ls, key="ml_ls_local_explanation")

    # ── NEXT LIFESTYLE MODEL ──────────────────────────────────────────────────
    if t("🔮 Volgende leefstijlscore") in tab_map:
        with tab_map[t("🔮 Volgende leefstijlscore")]:
            st.subheader(t("Voorspelling volgende leefstijlscore"))
            st.write(t("Dit model voorspelt de eerstvolgende leefstijlscore op basis van eerdere metingen van dezelfde deelnemer."))
            if st.button(t("🔄 Herlaad toekomstmodel"), key="refresh_next_lifestyle_model"):
                get_ml_models_v2.clear()
                st.session_state['model_to_retrain_single'] = 'next_lifestyle'
                st.rerun()

            data = ml_data['next_lifestyle']
            metrics = data['metrics']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{metrics['MAE']:.2f}")
            c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            c3.metric(t("R² Score"), f"{metrics['R² Score']:.2f}")
            c4.metric(t("CV MAE"), f"{metrics['CV MAE']:.2f}")
            st.caption(t("Baseline MAE (volgende score = huidige score): {mae:.2f}. Het model is pas nuttig als zijn MAE lager is.", mae=metrics['Baseline MAE']))
            st.caption(t("Voor een derde meting gebruikt het model de score op meting 2 en de verandering tussen meting 1 en 2."))

            col_a, col_b = st.columns(2)
            with col_a:
                p(plot_feature_importance(data['model'], data['features'], "Toekomstig leefstijlmodel"), key="ml_next_ls_importance")
            with col_b:
                p(plot_prediction_performance_regression(data['y_test'], data['y_pred']), key="ml_next_ls_performance")

    # ── DROP-OFF MODEL ────────────────────────────────────────────────────────
    if t("📋 Afhaakmodel") in tab_map:
        with tab_map[t("📋 Afhaakmodel")]:
            st.subheader(t("Afhaakrisico / geen scoredata"))
            st.write(t("Dit model voorspelt welke accounts waarschijnlijk geen bruikbare scoredata hebben. Gebruik dit als activatie- en follow-up signaal."))
            data = ml_data.get('dropoff')
            metrics = data['metrics']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t("Accuracy"), f"{metrics['Accuracy']:.1%}")
            c2.metric(t("Recall Macro"), f"{metrics['Recall Macro']:.1%}")
            c3.metric(t("CV Accuracy"), f"{metrics['CV Accuracy']:.1%}")
            c4.metric(t("CV Recall Macro"), f"{metrics['CV Recall Macro']:.1%}")
            st.caption(t("Target: 1 = account zonder leefstijlscore/scoredata, 0 = account met scoredata."))

            col_a, col_b = st.columns(2)
            with col_a:
                p(plot_feature_importance(data['model'], data['features'], "Afhaakmodel"), key="ml_dropoff_importance")
            with col_b:
                p(plot_confusion_matrix(data['y_test'], data['y_pred']), key="ml_dropoff_cm")

    # ── IMPROVEMENT MODEL ─────────────────────────────────────────────────────
    if t("📈 Verbeteringsmodel") in tab_map:
        with tab_map[t("📈 Verbeteringsmodel")]:
            st.subheader(t("Voorspelling scoreverbetering"))
            st.write(t("Dit model voorspelt de scoreverandering naar de volgende leefstijlmeting op basis van huidige en eerdere leefstijlscores."))
            data = ml_data.get('improvement')
            metrics = data['metrics']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{metrics['MAE']:.2f}")
            c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            c3.metric(t("R² Score"), f"{metrics['R² Score']:.2f}")
            c4.metric(t("CV R² Score"), f"{metrics['CV R² Score']:.2f}")
            st.caption(t("Geen-verandering baseline MAE: {baseline:.2f}; CV MAE: {cv_mae:.2f}. Gebruik dit als signaal, niet als causaal bewijs.", baseline=metrics['Baseline MAE'], cv_mae=metrics['CV MAE']))

            col_a, col_b = st.columns(2)
            with col_a:
                p(plot_feature_importance(data['model'], data['features'], "Verbeteringsmodel"), key="ml_improvement_importance")
            with col_b:
                p(plot_prediction_performance_regression(data['y_test'], data['y_pred']), key="ml_improvement_perf")

    # ── PURCHASE / PRODUCT MODEL ──────────────────────────────────────────────
    if t("🛒 Productmodel") in tab_map:
        with tab_map[t("🛒 Productmodel")]:
            st.subheader(t("Product- en aankoopmodel"))
            st.write(t("Dit model rangschikt gebruikers op basis van historische betaalde aankopen. Het ondersteunt propensity-analyse, maar bewijst geen toekomstige aankoop of causaal effect."))
            data = ml_data.get('purchase')
            metrics = data['metrics']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t("Average precision holdout"), f"{metrics['Holdout Average Precision']:.1%}")
            c2.metric(t("Average precision CV"), f"{metrics['CV Average Precision']:.1%}")
            c3.metric(t("Precision"), f"{metrics['Precision']:.1%}")
            c4.metric(t("Recall"), f"{metrics['Recall']:.1%}")
            st.caption(t("Brier-score: CV {cv:.3f}, holdout {holdout:.3f}; beslisdrempel {threshold:.3f}. Target: minstens één betaalde aankoop.", cv=metrics['CV Brier'], holdout=metrics['Holdout Brier'], threshold=metrics['Decision threshold']))

            col_a, col_b = st.columns(2)
            with col_a:
                p(plot_feature_importance(data['model'], data['features'], "Productmodel"), key="ml_purchase_importance")
            with col_b:
                p(plot_confusion_matrix(data['y_test'], data['y_pred']), key="ml_purchase_cm")

    # ── BLOOD PRESSURE MODEL ──────────────────────────────────────────────────
    if t("🩸 Bloeddrukmodel") in tab_map:
        with tab_map[t("🩸 Bloeddrukmodel")]:
            st.subheader(t("Verhoogde bloeddruk"))
            st.write(
                t(
                    "Dit model schat of bloeddruk mogelijk verhoogd is op basis van demografie, BMI en leefstijlfactoren. "
                    "Het onderscheidt geen verhoogde van hoge bloeddruk en is geen medische diagnose."
                )
            )
            if st.button(t("🔄 Herlaad Bloeddrukmodel"), key="refresh_bp_model"):
                get_ml_models_v2.clear() # Clear the main cache
                st.session_state['model_to_retrain_single'] = 'bp'
                st.rerun()

            st.markdown("---")

            # Metrics
            bp_data = ml_data.get('bp')
            if not bp_data:
                st.info(t("Bloeddrukmodel kon niet worden getraind met de huidige data."))
                st.stop()
            metrics_bp = bp_data['metrics']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(t("Average precision holdout"), f"{metrics_bp['Average Precision']:.1%}")
            col2.metric(t("Average precision CV"), f"{metrics_bp['CV Average Precision']:.1%}")
            col3.metric(t("Precision"), f"{metrics_bp['Precision']:.1%}")
            col4.metric(t("Recall"), f"{metrics_bp['Recall']:.1%}")
            st.caption(
                t("Brier-score: CV {cv:.3f}, holdout {holdout:.3f}; drempel {threshold:.3f}.",
                  cv=metrics_bp['CV Brier'], holdout=metrics_bp['Brier'], threshold=metrics_bp['Decision threshold'])
            )
            st.caption(
                t(
                    "Screeningsinstelling: hoge recall; {rate:.1%} van de holdoutdeelnemers kreeg een signaal. "
                    "Dit signaal is geen diagnose.",
                    rate=metrics_bp['Alert rate'],
                )
            )

            st.markdown("---")

            # Feature importance
            st.markdown(f"### {t('Belangrijkste Voorspellers')}")
            fig_importance_bp = plot_feature_importance(
                bp_data['model'],
                bp_data['features'],
                "Bloeddrukmodel",
            )
            p(fig_importance_bp, key="ml_bp_importance")

            # Classificatieresultaten
            st.markdown(f"### {t('Model Evaluatie')}")
            col_ev1, col_ev2, col_ev3 = st.columns(3)
            with col_ev1:
                fig_cm_bp = plot_confusion_matrix(bp_data['y_test'], bp_data['y_pred'], labels=[0, 1])
                p(fig_cm_bp, key="ml_bp_confusion")
            with col_ev2:
                if plot_roc_curve:
                    fig_roc_bp = plot_roc_curve(bp_data['y_test'], bp_data['y_pred_proba'])
                    p(fig_roc_bp, key="ml_bp_roc")
            with col_ev3:
                if hasattr(ml_module, "plot_precision_recall_curve"):
                    fig_pr_bp = ml_module.plot_precision_recall_curve(bp_data['y_test'], bp_data['y_pred_proba'])
                    p(fig_pr_bp, key="ml_bp_pr")

            st.markdown("---")
            st.markdown(f"### {t('Eigen invoer')}")
            c1, c2, c3 = st.columns(3)
            with c1:
                in_age = st.slider("Leeftijd" if lang == "nl" else "Age", 18, 85, 50, key="ml_bp_age")
                in_gender = st.selectbox("Geslacht" if lang == "nl" else "Gender", ["Vrouw", "Man"], key="ml_bp_gender")
                in_bmi = st.slider("BMI", 15.0, 45.0, 26.0, 0.1, key="ml_bp_bmi")
            with c2:
                in_sodium = st.slider("Natrium (mg/dag)", 0, 6000, 2400, 100, key="ml_bp_sodium")
                in_steps = st.slider("Stappen per dag", 0, 20000, 7000, 500, key="ml_bp_steps")
                in_alcohol = st.slider("Alcohol (glazen/week)", 0, 40, 5, 1, key="ml_bp_alcohol")
            with c3:
                in_diabetes = st.selectbox("Diabetes categorie", [0, 1, 2], key="ml_bp_diab")
                in_sleep = st.slider("Slaap PSQI", 0.0, 21.0, 6.0, 0.5, key="ml_bp_sleep")
                in_stress = st.slider("Stress (0-10)", 0.0, 10.0, 3.0, 0.1, key="ml_bp_stress")
                in_resilience = st.slider("Veerkracht", 0.0, 10.0, 6.0, 0.1, key="ml_bp_res")

            bp_input = pd.DataFrame([{
                'rec_age_current': in_age,
                'rec_user_gender': 1 if in_gender in ["Man"] else 0,
                'rec_med_bmi': in_bmi,
                'rec_med_diabetes_cat': in_diabetes,
                'rec_ls_exercise_steps_per_day': in_steps / 1000.0,
                'rec_ls_nutrition_natrium_per_day': in_sodium,
                'rec_ls_alcohol_total_per_week': in_alcohol,
                'rec_ls_sleep_psqi_sum': in_sleep,
                'rec_ls_stress_sum': in_stress,
                'rec_resilience_score': in_resilience
            }])
            
            bp_pred_class, _ = predict_bp(bp_data['model'], bp_input)
            bp_pred = bp_pred_class[0]
            bp_labels = {0: "Normaal", 1: "Verhoogd of hoog"}
            
            st.metric(
                t("Bloeddruksignaal"),
                bp_labels[int(bp_pred)],
            )
            
            # SHAP Plot voor bloeddruk
            fig_shap_bp = plot_local_shap(bp_data, bp_input, "Bloeddruk") if plot_local_shap else None
            if fig_shap_bp:
                st.markdown(f"#### {t('Waarom deze voorspelling?')}")
                p(fig_shap_bp, key="ml_bp_local_explanation")

    # ── HEART RISK MODEL ──────────────────────────────────────────────────────
    if t("❤️ Hartrisicomodel") in tab_map:
        with tab_map[t("❤️ Hartrisicomodel")]:
            st.subheader(t("Cardiovasculair Risico Voorspelling"))
            st.info(
                t(
                    "Opmerking: Dit model voorspelt een *risicoclassificatie* (proxy van Framingham), "
                    "geen medische diagnose. Het analyseert hoe leefstijlfactoren samenhangen met het berekende hartrisico."
                    "Het Framingham-model wordt niet als feature gebruikt om vergelijking mogelijk te maken."
                )
            )
            if st.button(t("🔄 Herlaad Hartrisicomodel"), key="refresh_heartrisk_model"):
                get_ml_models_v2.clear() # Clear the main cache
                st.session_state['model_to_retrain_single'] = 'heartrisk'
                st.rerun()

            st.markdown("---")

            # Metrics
            hr_data = ml_data.get('heartrisk')
            if not hr_data:
                st.info(t("Hartrisicomodel kon niet worden getraind met de huidige data."))
                st.stop()
            metrics_hr = hr_data['metrics']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(t("Accuracy"), f"{metrics_hr['Accuracy']:.3f}")
            col2.metric(t("Recall matig risico"), f"{metrics_hr['Recall matig risico']:.3f}")
            col3.metric(t("Recall hoog risico"), f"{metrics_hr['Recall hoog risico']:.3f}")
            col4.metric(t("Testset grootte"), f"{len(hr_data['y_test']):,}")
            st.caption(
                t(
                    "Recall = aandeel van echte klasse dat je correct terugvindt. Focus: Matig & Hoog. CV (5-fold): Matig {moderate:.3f} ± {moderate_std:.3f}, Hoog {high:.3f} ± {high_std:.3f}.",
                    moderate=metrics_hr.get('CV Recall matig risico', 0),
                    moderate_std=metrics_hr.get('CV Recall matig risico std', 0),
                    high=metrics_hr.get('CV Recall hoog risico', 0),
                    high_std=metrics_hr.get('CV Recall hoog risico std', 0),
                )
            )
            st.caption(
                t("Macro recall (gemiddelde over alle 3 klassen): {value:.3f}.", value=metrics_hr['Recall macro'])
            )
            st.caption(
                t("Brier-score kans op hoog risico (holdout, lager is beter): {value:.3f}.", value=metrics_hr['Brier hoge risico'])
            )

            st.markdown("---")

            # Feature importance
            st.markdown(f"### {t('Belangrijkste Voorspellers')}")
            fig_importance_hr = plot_feature_importance(
                hr_data['model'],
                hr_data['features'],
                "Hartrisicomodel",
            )
            p(fig_importance_hr, key="ml_heartrisk_importance")

            # Model Evaluatie: Confusion matrix & PR Curve
            st.markdown(f"### {t('Model Evaluatie')}")
            col_ev1, col_ev2, col_ev3 = st.columns(3)
            with col_ev1:
                fig_cm = plot_confusion_matrix(hr_data['y_test'], hr_data['y_pred'])
                p(fig_cm, key="ml_heartrisk_confusion")
            with col_ev2:
                if plot_roc_curve:
                    fig_roc = plot_roc_curve(hr_data['y_test'], hr_data['y_pred_proba'])
                    p(fig_roc, key="ml_heartrisk_roc")
            with col_ev3:
                fig_pr = ml_module.plot_precision_recall_curve(hr_data['y_test'], hr_data['y_pred_proba'])
                p(fig_pr, key="ml_heartrisk_pr_curve")


            st.markdown("---")
            st.markdown(f"### {t('Eigen invoer')}")
            c1, c2 = st.columns(2)
            smoking_options = (
                ["Nooit", "Ex-roker", "Roker"]
                if lang == "nl"
                else ["Never", "Former smoker", "Smoker"]
            )
            with c1:
                hr_age = st.slider("Leeftijd" if lang == "nl" else "Age", 18, 80, 55, key="ml_hr_age")
                hr_gender = st.selectbox("Geslacht" if lang == "nl" else "Gender", ["Vrouw", "Man"] if lang == "nl" else ["Woman", "Man"], key="ml_hr_gender")
                hr_bmi = st.slider("BMI", 16.0, 45.0, 27.0, 0.1, key="ml_hr_bmi")
                hr_smoking = st.selectbox("Rookstatus" if lang == "nl" else "Smoking status", smoking_options, key="ml_hr_smoking")
            with c2:
                hr_diabetes = st.selectbox("Diabetes categorie" if lang == "nl" else "Diabetes category", [0, 2], key="ml_hr_diabetes")
                hr_fat = st.slider(t("Verzadigd vet (g/dag)"), 0.0, 80.0, 20.0, 0.5, key="ml_hr_fat")
                hr_sugar = st.slider(t("Suiker (g/dag)"), 0.0, 200.0, 60.0, 1.0, key="ml_hr_sugar")
                hr_sodium = st.slider(t("Natrium (mg/dag)"), 0.0, 6000.0, 2400.0, 50.0, key="ml_hr_sodium")

            heartrisk_input = pd.DataFrame([{
                'rec_age_current': hr_age,
                'rec_user_gender': 1 if hr_gender in ["Man"] else 0,
                'rec_med_bmi': hr_bmi,
                'rec_smoking_cat': smoking_options.index(hr_smoking),
                'rec_med_diabetes_cat': hr_diabetes,
                'rec_ls_nutrition_saturated_fat_per_day': hr_fat,
                'rec_ls_nutrition_sugar_per_day': hr_sugar,
                'rec_ls_nutrition_natrium_per_day': hr_sodium,
            }])
            hr_pred_class, hr_pred_proba = predict_heartrisk(hr_data['model'], heartrisk_input)
            risico_labels = {
                0: "Laag" if lang == "nl" else "Low",
                1: "Matig" if lang == "nl" else "Moderate",
                2: "Hoog" if lang == "nl" else "High",
            }
            model_classes = list(hr_data['model'].named_steps['model'].classes_)
            prob_high = hr_pred_proba[0][model_classes.index(2)] if 2 in model_classes else 0.0
            st.metric(
                t("Voorspeld hartrisico"),
                risico_labels[int(hr_pred_class[0])],
                delta=t("{prob:.1%} kans op hoog risico", prob=prob_high),
                delta_color="normal",
            )

            # SHAP Plot voor hartrisico
            fig_shap_hr = plot_local_shap(hr_data, heartrisk_input, "Hartrisico") if plot_local_shap else None
            if fig_shap_hr:
                st.markdown(f"#### {t('Waarom dit risico?')}")
                p(fig_shap_hr, key="ml_hr_local_explanation")

# ═════════════════════════════════════════════════════════════════════════════
# TESTOMGEVING
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "🧪 Test":
    # Defense-in-depth: de pagina zit niet in de navigatie voor niet-admins,
    # maar controleer het ook hier voor het geval session_state handmatig
    # is aangepast of een oude sessie nog naar deze pagina verwijst.
    if not is_admin():
        st.error("⛔ Deze pagina is alleen beschikbaar voor beheerders.")
        st.stop()
    st.title("🧪 " + tr("Testomgeving voor nieuwe visualisaties", lang))
    st.info(t("Deze pagina bevat experimentele visualisaties gebaseerd op gezondheidswetenschappelijke principes."))

    test_tab_labels = [
        t("Risicomigratie"),
        t("Engagement & dataflow"),
        "Looker Studio",
    ]
    tab_migratie, tab_engagement, tab_looker = st.tabs(
        test_tab_labels,
        default=_tab_default_for_path(pending_navigation_path, test_tab_labels),
    )

    with tab_migratie:
        df_long = get_longitudinale_data()
        df = load_main_data_filtered_by_gender_and_store(geslacht)
        if df is not None and not df.empty and 'participant_id' in df.columns:
            active_pids = set(pd.to_numeric(df['participant_id'], errors='coerce').dropna().astype(int))
            if not df_long.empty and 'participant_id' in df_long.columns:
                df_long = df_long[pd.to_numeric(df_long['participant_id'], errors='coerce').isin(active_pids)]

        st.subheader(t("Transition Matrix: Verschuiving tussen risicogroepen"))
        available_slugs = set(df_long['slug'].dropna().astype(str)) if 'slug' in df_long.columns else set()
        opties = []
        optie_specs = {}
        gebruikte_slugs = set()
        for label, var_spec in VARIABELEN_DICT.items():
            slug = visualisaties._dashboard_score_to_history_slug(var_spec.get('kolom', ''))
            if slug in visualisaties.RISICO_MIGRATIE_EXCLUDED_SLUGS:
                continue
            if slug in available_slugs and slug not in gebruikte_slugs:
                opties.append(label)
                optie_specs[label] = (label, var_spec)
                gebruikte_slugs.add(slug)
        if not opties:
            st.warning(t("Geen variabelen met beschikbare longitudinale data gevonden."))
        else:
            factor_mig = st.selectbox(t("Kies factor voor migratie"), opties, key="test_m_factor")
            _, var_spec = optie_specs[factor_mig]
            kolom = var_spec.get('kolom', '')
            try:
                slug = visualisaties._dashboard_score_to_history_slug(kolom)
            except Exception:
                slug = kolom
            if slug in visualisaties.RISICO_MIGRATIE_EXCLUDED_SLUGS:
                st.info(t("Deze factor is tijdelijk uitgezet in de risicomigratie-visualisatie."))
                st.stop()

            if 'labels' in var_spec and isinstance(var_spec['labels'], dict):
                label_map = {k: v for k, v in var_spec['labels'].items()}
            else:
                label_map = {0: 'Laag', 1: 'Matig', 2: 'Hoog'}

            custom_bins = {
                'fruit': {
                    'bins': [-0.1, 0.5, 1.5, 10000],
                    'labels': ['Weinig (<1/dag)', '~1/dag', 'Veel (>2/dag)']
                },
                'vegetables': {
                    'bins': [-0.1, 150, 250, 100000],
                    'labels': ['Laag (<150g)', 'Matig (150-250g)', 'Veel (>250g)']
                }
            }

            matrix = pd.DataFrame()
            try:
                matrix, label_map_value = visualisaties.bereken_risico_migratie_value_binned(df_long, slug, custom_bins=custom_bins)
                if not matrix.empty:
                    label_map = label_map_value
            except Exception:
                matrix = pd.DataFrame()

            if matrix.empty:
                from analyses import bereken_risico_migratie
                matrix = bereken_risico_migratie(df_long, slug)

            if matrix.empty:
                st.info(tr("Onvoldoende deelnemers met herhaalmetingen voor migratie-analyse.", lang))
            elif int(matrix['Aantal'].sum()) < MIN_VISUALISATIE_DEELNEMERS:
                st.info(t(
                    "Visualisatie niet beschikbaar: minimaal {min_n} deelnemers nodig, huidige selectie bevat {n}.",
                    min_n=MIN_VISUALISATIE_DEELNEMERS,
                    n=int(matrix['Aantal'].sum()),
                ))
            else:
                present_cats = sorted(set(matrix['Van'].unique()) | set(matrix['Na'].unique()))
                matrix_filtered = matrix[matrix['Van'].isin(present_cats) & matrix['Na'].isin(present_cats)]
                totals_van = matrix_filtered.groupby('Van')['Aantal'].sum().to_dict()
                totals_na = matrix_filtered.groupby('Na')['Aantal'].sum().to_dict()
                rows = []
                for i in present_cats:
                    rows.append({
                        tr('Categorie', lang): label_map.get(i, str(i)),
                        tr('Eerste', lang): int(totals_van.get(i, 0)),
                        tr('Laatste', lang): int(totals_na.get(i, 0)),
                    })

                st.write(tr("Totaal per categorie", lang))
                st.table(pd.DataFrame(rows))

                p(
                    maak_risico_migratie_sankey(matrix_filtered, slug, label_map, lang=lang),
                    key="test_mig_sankey",
                    show_reset_scale=False,
                )

    with tab_engagement:
        st.caption(t(
            "Experimentele analyses: invulfrequentie vs scores, account-dataflow en engagement per opdrachtgever."
        ))

        engagement_tab_labels = [
            t("Engagement funnel"),
            t("Herhaalde vragenlijst & scores"),
            t("Account dataflow"),
            t("Engagement per opdrachtgever"),
        ]
        sub_funnel, sub_invul, sub_dataflow, sub_engagement = st.tabs(
            engagement_tab_labels,
            default=_tab_default_for_path(pending_navigation_path, engagement_tab_labels),
        )

        with sub_funnel:
            st.subheader(t("Engagement funnel"))
            st.caption(t(
                "Eerste proefopzet van registratie naar vragenlijst, aankoop en herhaalmeting. "
                "Percentages zijn berekend ten opzichte van het totale aantal geregistreerde deelnemers."
            ))
            try:
                journey_df = bereken_engagement_journey(BASE, DB_URL)
                fig_engagement_funnel = maak_engagement_funnel_plot(BASE, DB_URL, lang=lang)
                p(fig_engagement_funnel, min_deelnemers=0, key="test_engagement_funnel")
                if not journey_df.empty:
                    st.dataframe(
                        translate_dataframe(journey_df, lang),
                        width="stretch",
                        hide_index=True,
                    )
                st.info(t(
                    "De tabel toont per stap het aantal gebruikers, het aandeel van alle registraties "
                    "en de conversie vanaf de vorige stap. De reis is niet tijdgebonden."
                ))
            except Exception as e:
                st.error(t("Fout: {e}", e=e))

        with sub_invul:
            st.subheader(t("Scoreverandering bij herhaald invullen van dezelfde vragenlijst"))
            st.caption(t(
                "Kies één vragenlijst. Per deelnemer wordt gemeten hoe vaak die dezelfde "
                "vragenlijst is ingevuld (1x, 2x, 3x, …). De grafiek toont de gemiddelde "
                "verandering van de gekozen score tussen de eerste en laatste invulling."
            ))

            df_vragenlijsten = get_vragenlijsten_overzicht()
            if df_vragenlijsten.empty:
                st.info(t("Geen vragenlijstdata beschikbaar."))
            else:
                df_vragenlijsten = df_vragenlijsten[df_vragenlijsten['n_completions'] > 0].copy()
                df_vragenlijsten['label'] = df_vragenlijsten.apply(
                    lambda r: f"{r['naam']} (id={int(r['questionnaire_id'])}, "
                              f"{int(r['n_herhaalinvullers'])} herhaalinvullers)",
                    axis=1,
                )
                gekozen_label = st.selectbox(
                    t("Kies vragenlijst"),
                    df_vragenlijsten['label'].tolist(),
                    key="test_herhaal_vragenlijst",
                )
                questionnaire_id = int(
                    df_vragenlijsten.loc[df_vragenlijsten['label'] == gekozen_label, 'questionnaire_id'].iloc[0]
                )

                score_slugs = list(get_scores_voor_vragenlijst(questionnaire_id))
                if not score_slugs:
                    st.info(t("Geen scores gevonden voor deze vragenlijst."))
                else:
                    score_slug = st.selectbox(
                        t("Kies score uit deze vragenlijst"),
                        score_slugs,
                        key="test_herhaal_score",
                    )
                    try:
                        fig_chg, fig_traj, samenvatting, traject, df_change, meta = (
                            get_herhaalde_vragenlijst_scoreverandering(questionnaire_id, score_slug)
                        )
                        if samenvatting.empty:
                            st.info(t("Onvoldoende data voor vergelijking."))
                        else:
                            c1, c2, c3 = st.columns(3)
                            c1.metric(t("Deelnemers"), f"{meta.get('totaal_deelnemers', 0):,}")
                            c2.metric(t("Herhaalinvullers"), f"{meta.get('herhaalinvullers', 0):,}")
                            if len(samenvatting) > 1:
                                diff = samenvatting.iloc[-1]['gem_verandering'] - samenvatting.iloc[0]['gem_verandering']
                                c3.metric(t("Verschil verandering (laatste vs 1x groep)"), f"{diff:+.2f}")

                            n_deelnemers = int(meta.get('totaal_deelnemers', 0))
                            if n_deelnemers >= MIN_VISUALISATIE_DEELNEMERS:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    p(fig_chg, participants_df=df_change, key="test_herhaal_verandering")
                                with col_b:
                                    if not traject.empty:
                                        p(fig_traj, participants_df=df_change, key="test_herhaal_traject")
                            else:
                                st.info(t(
                                    "Visualisatie niet beschikbaar: minimaal {min_n} deelnemers nodig, huidige selectie bevat {n}.",
                                    min_n=MIN_VISUALISATIE_DEELNEMERS,
                                    n=n_deelnemers,
                                ))

                            st.markdown(f"**{t('Gemiddelde verandering per invulfrequentie')}**")
                            st.dataframe(
                                translate_dataframe(samenvatting, lang),
                                width="stretch",
                                hide_index=True,
                            )
                            if not traject.empty:
                                st.markdown(f"**{t('Score per invulmoment (1e, 2e, 3e, …)')}**")
                                st.dataframe(
                                    translate_dataframe(traject, lang),
                                    width="stretch",
                                    hide_index=True,
                                )
                            st.caption(t(
                                "Bij 1x is verandering 0 (geen eerdere meting). Bij 2x of meer: "
                                "verschil tussen laatste en eerste score voor die vragenlijst."
                            ))
                    except Exception as e:
                        st.error(t("Fout: {e}", e=e))

        with sub_dataflow:
            st.subheader(t("Dataflow: waar vallen gebruikers af?"))
            st.caption(t(
                "Funnel van registratie naar scores. Toont hoeveel deelnemers afhaken na leeftijd, "
                "geslacht en het starten van een vragenlijst. Naam wordt apart getoond als parallelle metriek."
            ))
            try:
                fig_funnel, funnel_df, funnel_meta = get_account_dataflow_funnel(lang)
                if funnel_df.empty:
                    st.info(t("Geen data beschikbaar voor dataflow-analyse."))
                else:
                    totaal = funnel_meta.get('totaal_geregistreerd', int(funnel_df['aantal'].iloc[0]))
                    if totaal >= MIN_VISUALISATIE_DEELNEMERS:
                        p(fig_funnel, key="test_account_dataflow_funnel")
                    else:
                        st.info(t(
                            "Visualisatie niet beschikbaar: minimaal {min_n} deelnemers nodig, huidige selectie bevat {n}.",
                            min_n=MIN_VISUALISATIE_DEELNEMERS,
                            n=totaal,
                        ))
                    st.dataframe(
                        translate_dataframe(funnel_df, lang),
                        width="stretch",
                        hide_index=True,
                    )
                    if funnel_meta.get('naam_bron'):
                        st.caption(t("Naam afgeleid uit: {bron}", bron=funnel_meta['naam_bron']))
            except Exception as e:
                st.error(t("Fout: {e}", e=e))

        with sub_engagement:
            st.subheader(tr("Waar besteden gebruikers hun aandacht aan?", lang))
            st.caption(t(
                "Bekijk per opdrachtgever welke activiteiten de meeste gebruikers bereiken: vragenlijsten, "
                "artikelen, challenges en aankopen. De samengestelde engagement-score (0-100) is een samenvatting, "
                "niet de uitleg op zichzelf."
            ))
            st.info(t(
                "Deze databron registreert acties, maar geen betrouwbare tijdsduur per activiteit. "
                "De uitsplitsing hieronder toont daarom het aandeel deelnemers en het aantal deelnemers "
                "dat een activiteit minimaal één keer heeft uitgevoerd."
            ))

            with st.expander(t("Gewichten engagement-componenten"), expanded=False):
                gewichten = {}
                cols = st.columns(3)
                for i, (key, default) in enumerate(ENGAGEMENT_COMPONENT_DEFAULTS.items()):
                    label = t({
                        'vragenlijst': 'Vragenlijst ingevuld',
                        'herhaalmeting': 'Herhaalmeting',
                        'breedte': 'Meerdere vragenlijsten',
                        'artikelen': 'Artikel gelezen',
                        'challenges': 'Challenges (store)',
                        'aankopen': 'Aankoop gedaan',
                    }.get(key, key))
                    with cols[i % 3]:
                        gewichten[key] = st.slider(label, 0.0, 1.0, float(default), 0.05, key=f"test_eng_w_{key}")

            gewichten_tuple = tuple(sorted(gewichten.items()))
            engagement_min, engagement_max, _ = _datum_bereik(load_completions(DB_URL))
            engagement_start, engagement_end = _periode_selectie(engagement_min, engagement_max)

            try:
                with st.spinner(t("Engagementdata berekenen...")):
                    agg, benchmark, koppeling = get_engagement_opdrachtgever(
                        gewichten_tuple, engagement_start, engagement_end,
                    )
                    trend = get_engagement_trend(
                        gewichten_tuple,
                        min_deelnemers=MIN_VISUALISATIE_DEELNEMERS,
                        start_date=engagement_start,
                        end_date=engagement_end,
                    )

                if agg.empty:
                    st.warning(t("Geen engagementdata beschikbaar. Controleer opdrachtgeverkoppelingen."))
                else:
                    summary = build_engagement_summary_stats(agg, benchmark, lang=lang)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(t("Beste opdrachtgever"), summary['beste_opdrachtgever'], delta=f"{summary['top_delta']:+.1f}")
                    c2.metric(t("Benchmark"), f"{summary['benchmark']:.1f}")
                    c3.metric(t("Gemiddeld"), f"{summary['gemiddeld_score']:.1f}")
                    c4.metric(t("Boven benchmark"), f"{summary['boven_benchmark']}/{len(agg):,}")
                    st.info(summary['insight'])

                    with st.expander(t("Datakwaliteit en koppelingen"), expanded=False):
                        kwaliteit = koppeling.copy()
                        if not kwaliteit.empty:
                            kwaliteit['Status'] = kwaliteit.apply(
                                lambda r: (
                                    'Volledig' if r.get('Beschikbaar', 0) == r.get('Koppelbaar', 0) and r.get('Beschikbaar', 0) > 0
                                    else 'Gedeeltelijk' if r.get('Koppelbaar', 0) > 0
                                    else 'Niet beschikbaar'
                                ),
                                axis=1,
                            )
                            st.dataframe(
                                translate_dataframe(kwaliteit, lang),
                                width="stretch",
                                hide_index=True,
                            )
                            st.caption(t(
                                "Gedeeltelijk betekent dat de bron wel bestaat, maar niet alle gebruikers betrouwbaar "
                                "aan een deelnemer kunnen worden gekoppeld."
                            ))

                    with st.expander(t("Hoe lees je deze score?"), expanded=False):
                        st.markdown(
                            "- " + t("De score loopt van 0 tot 100 en is een gewogen samenvatting van activiteit, herhaling, inhoudsgebruik en aankopen.") + "\n"
                            "- " + t("Benchmark = gewogen gemiddelde over alle opdrachtgevers in deze selectie.") + "\n"
                            "- " + t("Een hoger aandeel betekent dat meer deelnemers de activiteit minimaal één keer hebben gedaan.")
                        )

                    with st.expander(t("Datakwaliteit en koppelingen"), expanded=False):
                        kwaliteit = koppeling.copy()
                        if not kwaliteit.empty:
                            kwaliteit['Status'] = kwaliteit.apply(
                                lambda r: (
                                    'Volledig' if r.get('Beschikbaar', 0) == r.get('Koppelbaar', 0) and r.get('Beschikbaar', 0) > 0
                                    else 'Gedeeltelijk' if r.get('Koppelbaar', 0) > 0
                                    else 'Niet beschikbaar'
                                ),
                                axis=1,
                            )
                            st.dataframe(
                                translate_dataframe(kwaliteit, lang),
                                width="stretch",
                                hide_index=True,
                            )
                            st.caption(t(
                                "Gedeeltelijk betekent dat de bron wel bestaat, maar niet alle gebruikers betrouwbaar "
                                "aan een deelnemer kunnen worden gekoppeld."
                            ))

                    engagement_detail_tab_labels = [
                        t("Ranking & benchmark"), t("Uitsplitsing"), t("Trend"),
                    ]
                    eng_rank, eng_breakdown, eng_trend = st.tabs(
                        engagement_detail_tab_labels,
                        default=_tab_default_for_path(pending_navigation_path, engagement_detail_tab_labels),
                    )

                    with eng_rank:
                        fig_rank = maak_engagement_opdrachtgever_ranking_plot(
                            agg, benchmark, MIN_VISUALISATIE_DEELNEMERS, lang=lang,
                        )
                        p(fig_rank, min_deelnemers=0, key="test_eng_rank")
                        display_agg = agg.copy()
                        display_agg['vs_benchmark'] = (display_agg['engagement_score'] - benchmark).round(1)
                        basis_renames = {
                            'store_name': t('Opdrachtgever'),
                            'engagement_score': t('Engagement'),
                            'vs_benchmark': t('vs benchmark'),
                            'n_deelnemers': t('Deelnemers'),
                        }
                        pct_renames = {
                            'pct_vragenlijst': t('Vragenlijst (%)'),
                            'pct_herhaalmeting': t('Herhaal (%)'),
                            'gem_breedte': t('Gem. # vragenlijsten'),
                            'pct_artikelen': t('Artikelen (%)'),
                            'pct_challenges': t('Challenges (%)'),
                            'pct_aankopen': t('Aankopen (%)'),
                            'pct_herhaalkopers_van_kopers': t('Herhaalkopers (% v kopers)'),
                            'pct_challenges_afgerond': t('Challenges afgerond (%)'),
                            'pct_app_actief_90d': t('App actief 90d (%)'),
                            'gem_invullingen': t('Gem. invullingen p.p.'),
                            'gem_artikelviews': t('Gem. artikelviews p.p.'),
                            'gem_aankopen': t('Gem. geldige aankopen p.p.'),
                        }
                        select_cols = ['store_name', 'engagement_score', 'vs_benchmark', 'n_deelnemers']
                        for k, lbl in pct_renames.items():
                            if k in display_agg.columns:
                                basis_renames[k] = lbl
                                select_cols.append(k)
                        st.caption(t("Belangrijkste KPI’s per opdrachtgever; meer details zijn hieronder beschikbaar."))
                        st.dataframe(
                            translate_dataframe(
                                display_agg[select_cols].rename(columns=basis_renames),
                                lang,
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                        with st.expander(t("Meer details per opdrachtgever"), expanded=False):
                            overige_renames = {
                                'gem_invullingen': t('Gem. invullingen p.p.'),
                                'gem_artikelviews': t('Gem. artikelviews p.p.'),
                                'gem_aankopen': t('Gem. geldige aankopen p.p.'),
                            }
                            detail_cols = [c for c in display_agg.columns if c not in {'store_name', 'engagement_score', 'vs_benchmark', 'n_deelnemers'}]
                            st.dataframe(
                                translate_dataframe(
                                    display_agg[['store_name'] + detail_cols].rename(columns={**basis_renames, **overige_renames}),
                                    lang,
                                ),
                                width="stretch",
                                hide_index=True,
                            )

                    with eng_breakdown:
                        st.caption(t(
                            "Dit is de kernweergave: hoe groter het aandeel, hoe meer deelnemers deze activiteit "
                            "hebben uitgevoerd. De balk toont percentage én aantal van het totaal; beweeg over "
                            "een balk voor de exacte toelichting."
                        ))
                        st.caption(t(
                            "Extra maten: Challenge afgerond = aandeel afgeronde van alle uitgekomen challenge-"
                            "runs (expired/unsubscribed tellen als niet afgerond); Herhaalkoper = percentage "
                            "kopers met minstens twee betaalde bestellingen; App actief = minstens één login in de "
                            "laatste 90 dagen."
                        ))
                        store_opties = sorted(agg['store_name'].dropna().unique())
                        gekozen_store = st.selectbox(t("Kies opdrachtgever"), store_opties, key="test_eng_breakdown_store")
                        if gekozen_store:
                            p(maak_engagement_breakdown_plot(agg, gekozen_store, lang=lang), min_deelnemers=0, key="test_eng_breakdown")

                    with eng_trend:
                        store_opties_trend = [t("Alle opdrachtgevers (gemiddelde)")] + sorted(agg['store_name'].dropna().unique())
                        gekozen_trend = st.selectbox(t("Trend voor"), store_opties_trend, key="test_eng_trend_store")
                        store_filter = None if gekozen_trend == t("Alle opdrachtgevers (gemiddelde)") else gekozen_trend
                        if trend.empty:
                            st.info(t("Onvoldoende data voor engagement-trend."))
                        else:
                            p(maak_engagement_trend_plot(trend, store_filter, lang=lang), min_deelnemers=0, key="test_eng_trend")

            except Exception as e:
                st.error(t("Fout: {e}", e=e))

    # ══════════════════════════════════════════════════════════════════════════
    # LOOKER STUDIO TAB - Analytics & Engagement Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    with tab_looker:
        st.title("Looker Studio")
        st.caption(t("Analytics dashboard: gebruikersgedrag, engagement en pagina-performance per opdrachtgever en organisatieonderdeel."))
        st.info(t(
            "Deze analyse gebruikt platformdata. Sessieduur, device, verkeerskanaal, landing page "
            "en gemiddelde engagementtijd zijn niet beschikbaar en worden daarom niet weergegeven."
        ))

        # ── Data laden (eerst data, dan filters) ────────────────────────────
        with st.spinner(t("Analytics data laden...")):
            df_full = _laad_analytics_data(DB_URL)
        
        if df_full.empty:
            st.warning(t("Geen analytics data beschikbaar. Controleer databaseverbinding."))
        else:
            # Bepaal beschikbare datum range uit de data (zorg voor geldige datums)
            df_full['created_at'] = pd.to_datetime(df_full['created_at'], errors='coerce')
            df_dates = df_full[df_full['created_at'].notna()]
            
            if df_dates.empty:
                st.warning(t("Geen geldige datums gevonden in de data."))
            else:
                datum_min = df_dates['created_at'].min()
                datum_max = df_dates['created_at'].max()
                kalender_vandaag = pd.Timestamp.today().normalize()
                
                # ── Datum Filter (gebaseerd op beschikbare data) ─────────────
                st.subheader(t("Datum filter"))
                
                col_d1, col_d2, col_reset = st.columns([1, 3, 1])
                
                # Reset knop
                with col_reset:
                    if st.button(t("🔄 Resetten"), key="looker_reset"):
                        for key in ["looker_datum_preset", "looker_start", "looker_end",
                                    "looker_aggregatie", "looker_agg_idx"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        # Reset ook alle metric checkboxes
                        for key in list(st.session_state.keys()):
                            if key.startswith("kpi_metric_") or key.startswith("looker_metric_"):
                                del st.session_state[key]
                        st.rerun()
                
                with col_d1:
                    datum_preset = st.selectbox(
                        t("Periode"),
                        [
                            "Vandaag", "Gisteren", "Deze week", "Deze maand", "Dit kwartaal", "Dit jaar",
                            "Afgelopen 7 dagen", "Afgelopen 14 dagen", "Afgelopen 28 dagen", "Afgelopen 30 dagen",
                            "Afgelopen week", "Afgelopen maand", "Afgelopen kwartaal", "Afgelopen jaar",
                            "Alle data", "Geavanceerd"
                        ],
                        key="looker_datum_preset"
                    )
                
                with col_d2:
                    if datum_preset == "Geavanceerd":
                        col_start, col_end = st.columns(2)
                        with col_start:
                            start_datum = st.date_input(
                                t("Startdatum"),
                                value=datum_max.date() - pd.Timedelta(days=28),
                                min_value=datum_min.date(),
                                max_value=datum_max.date(),
                                key="looker_start"
                            )
                        with col_end:
                            eind_datum = st.date_input(
                                t("Einddatum"),
                                value=datum_max.date(),
                                min_value=datum_min.date(),
                                max_value=datum_max.date(),
                                key="looker_end"
                            )
                    elif datum_preset == "Alle data":
                        start_datum = datum_min.date()
                        eind_datum = datum_max.date()
                    elif datum_preset in ("Deze week", "Afgelopen week"):
                        vandaag = datum_max
                        start_datum = (vandaag - pd.Timedelta(days=vandaag.weekday() + 1) if vandaag.weekday() != 6 else vandaag).date()
                        eind_datum = (pd.Timestamp(start_datum) + pd.Timedelta(days=6)).date()
                    elif datum_preset in ("Deze maand", "Afgelopen maand"):
                        vandaag = datum_max
                        start_datum = vandaag.replace(day=1).date()
                        eind_datum = (pd.Timestamp(start_datum) + pd.offsets.MonthEnd(0)).date()
                    elif datum_preset in ("Dit kwartaal", "Afgelopen kwartaal"):
                        vandaag = datum_max
                        kwartaal_start_mnd = ((vandaag.month - 1) // 3) * 3 + 1
                        start_datum = vandaag.replace(month=kwartaal_start_mnd, day=1).date()
                        eind_datum = (pd.Timestamp(start_datum) + pd.offsets.MonthEnd(2)).date()
                    elif datum_preset in ("Dit jaar", "Afgelopen jaar"):
                        if datum_preset == "Dit jaar":
                            start_datum = kalender_vandaag.replace(month=1, day=1).date()
                            eind_datum = kalender_vandaag.date()
                        else:
                            vorig_jaar = kalender_vandaag.year - 1
                            start_datum = kalender_vandaag.replace(year=vorig_jaar, month=1, day=1).date()
                            eind_datum = kalender_vandaag.replace(year=vorig_jaar, month=12, day=31).date()
                    elif datum_preset == "Vandaag":
                        start_datum = eind_datum = datum_max.date()
                    elif datum_preset == "Gisteren":
                        start_datum = eind_datum = (datum_max - pd.Timedelta(days=1)).date()
                    elif datum_preset == "Afgelopen 7 dagen":
                        eind_datum = datum_max.date()
                        start_datum = (datum_max - pd.Timedelta(days=7)).date()
                    elif datum_preset == "Afgelopen 14 dagen":
                        eind_datum = datum_max.date()
                        start_datum = (datum_max - pd.Timedelta(days=14)).date()
                    elif datum_preset == "Afgelopen 28 dagen":
                        eind_datum = datum_max.date()
                        start_datum = (datum_max - pd.Timedelta(days=28)).date()
                    elif datum_preset == "Afgelopen 30 dagen":
                        eind_datum = datum_max.date()
                        start_datum = (datum_max - pd.Timedelta(days=30)).date()
                    else:
                        eind_datum = datum_max.date()
                        start_datum = (datum_max - pd.Timedelta(days=7)).date()

                # Filter data op datum
                start_ts = pd.Timestamp(start_datum)
                eind_ts = pd.Timestamp(eind_datum) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                # Vorige periode voor deltas
                periode_lengte = eind_ts - start_ts
                vorige_start = start_ts - periode_lengte
                vorige_eind = start_ts - pd.Timedelta(seconds=1)

                # Filter op datum
                df = df_full[
                    (pd.to_datetime(df_full['created_at'], errors='coerce') >= start_ts) &
                    (pd.to_datetime(df_full['created_at'], errors='coerce') <= eind_ts)
                ].copy()
                
                df_vorige = df_full[
                    (pd.to_datetime(df_full['created_at'], errors='coerce') >= vorige_start) &
                    (pd.to_datetime(df_full['created_at'], errors='coerce') <= vorige_eind)
                ].copy()

                if 'store_slug' in df.columns:
                    store_options = ['Alle opdrachtgevers'] + sorted(
                        df['store_slug'].dropna().astype(str).unique().tolist()
                    )
                    gekozen_store = st.selectbox(
                        t("Opdrachtgever"),
                        store_options,
                        key="looker_store_filter",
                    )
                    if gekozen_store != 'Alle opdrachtgevers':
                        df = df[df['store_slug'].astype(str) == gekozen_store].copy()
                        df_vorige = df_vorige[
                            df_vorige['store_slug'].astype(str) == gekozen_store
                        ].copy()

                if df.empty:
                    st.info(t("Geen data beschikbaar voor de geselecteerde periode."))
                else:
                    # Zet default sortering: neueste eerst
                    if 'created_at' in df.columns:
                        df = df.sort_values('created_at', ascending=False)

                    # ── KPI Overview ───────────────────────────────────────────────
                    st.subheader(t("KPI Metrics"))
                    kpis = bereken_kpi_metrics(df, df_vorige)
                    metrics_beschikbaar = [
                        ('Total Users', kpis['total_users'], kpis['delta_total_users']),
                        ('New Users', kpis['new_users'], kpis.get('delta_new_users', 0)),
                        ('Views', kpis['views'], kpis['delta_views']),
                        ('Returning Visitors', kpis['returning_visitors'], kpis['delta_returning']),
                        ('Views per User', kpis['views_per_user'], kpis['delta_views_per_user']),
                    ]
                    metric_display_cols = st.columns(len(metrics_beschikbaar))
                    for col_idx, (label, waarde, delta) in enumerate(metrics_beschikbaar):
                        with metric_display_cols[col_idx]:
                            st.metric(
                                label=t(label),
                                value=f"{waarde:,}" if isinstance(waarde, int) else f"{waarde}",
                                delta=f"{delta:+.1f}%" if delta != 0 else None,
                                delta_color="normal"
                            )

                    st.markdown("---")

                    # ── Bar Chart: Weekly User & Engagement Performance ────────
                    st.subheader(t("Weekly User & Engagement Performance"))
                    
                    col_agg, col_nav = st.columns([3, 2])
                    
                    with col_agg:
                        aggregatie_opties = ["dag", "week", "maand", "jaar"]
                        huidige_idx = st.session_state.get("looker_agg_idx", 1)
                        aggregatie = st.selectbox(
                            t("Aggregatie"),
                            aggregatie_opties,
                            index=huidige_idx,
                            key="looker_aggregatie"
                        )
                        st.session_state["looker_agg_idx"] = aggregatie_opties.index(aggregatie)
                    
                    with col_nav:
                        col_links, col_huidig, col_rechts = st.columns([1, 3, 1])
                        with col_links:
                            if st.button("◀", key="looker_agg_down"):
                                idx = max(0, st.session_state.get("looker_agg_idx", 1) - 1)
                                st.session_state["looker_agg_idx"] = idx
                                st.rerun()
                        with col_huidig:
                            st.markdown(
                                f"<p style='text-align:center;font-weight:bold;padding-top:8px;'>"
                                f"{aggregatie.title()}</p>",
                                unsafe_allow_html=True
                            )
                        with col_rechts:
                            if st.button("▶", key="looker_agg_up"):
                                idx = min(3, st.session_state.get("looker_agg_idx", 1) + 1)
                                st.session_state["looker_agg_idx"] = idx
                                st.rerun()
                    
                    # Metrics selectie (checkboxes)
                    st.markdown("**Selecteer Metrics:**")
                    metrics_cols = st.columns(4)
                    metrics_opts = {
                        'total_users': 'Total Users',
                        'new_users': 'New Users',
                        'views': 'Views',
                        'views_per_user': 'Views per User',
                    }
                    
                    selected_metrics = []
                    for idx, (metric_key, metric_label) in enumerate(metrics_opts.items()):
                        with metrics_cols[idx]:
                            if st.checkbox(metric_label, value=(metric_key == 'total_users'), key=f"looker_metric_{metric_key}"):
                                selected_metrics.append(metric_key)
                    
                    # Zorg dat minimaal één metric is geselecteerd
                    if not selected_metrics:
                        st.warning(t("Selecteer minimaal één metric"))
                        selected_metrics = ['total_users']

                    # Bereken trends op basis van aggregatie
                    if aggregatie == "dag":
                        trends = bereken_dagelijkse_trends(df)
                    elif aggregatie == "week":
                        trends = bereken_wekelijkse_trends(df)
                    elif aggregatie == "maand":
                        trends = bereken_maandelijkse_trends(df)
                    else:
                        trends = bereken_jaar_trends(df)

                    if not trends.empty:
                        fig_bar = maak_engagement_bar_chart(trends, metrics=selected_metrics, aggregatie=aggregatie)
                        st.plotly_chart(fig_bar, width="stretch", key="looker_bar_chart")
                    else:
                        st.info(t("Geen trenddata beschikbaar voor deze periode."))

                    st.markdown("---")

                    # ── Page Performance Overview ───────────────────────────────
                    st.subheader(t("Page Performance Overview"))
                    
                    performance_tab_labels = [
                        t("Alle content"),
                        t("Content"),
                        t("Orderable"),
                    ]
                    pp_tab1, pp_tab2, pp_tab3 = st.tabs(
                        performance_tab_labels,
                        default=_tab_default_for_path(pending_navigation_path, performance_tab_labels),
                    )

                    page_type_map = [
                        'all',
                        'content',
                        'orderable',
                    ]

                    for idx, (tab_obj, ptype) in enumerate(zip(
                        [pp_tab1, pp_tab2, pp_tab3],
                        page_type_map
                    )):
                        with tab_obj:
                            pp_data = bereken_page_performance(df, page_type=ptype)
                            if not pp_data.empty:
                                pp_display = pp_data.rename(columns={
                                    'page_type': t("Content Type"),
                                    'total_users': t("Total Users"),
                                    'new_users': t("New Users"),
                                    'views': t("Views"),
                                    'views_per_user': t("Views per User"),
                                })
                                st.dataframe(pp_display, width="stretch", height=300)
                            else:
                                st.info(t("Geen data beschikbaar voor dit type content."))

                    # ── Top Partners ──────────────────────────────────
                    st.subheader(t("Top Partners by Views"))
                    hp_data = bereken_top_health_partners(df, top_n=10)
                    if not hp_data.empty:
                        hp_display = hp_data.rename(columns={
                            'store_slug': t("Partner"),
                            'total_users': t("Total Users"),
                            'new_users': t("New Users"),
                            'views': t("Views"),
                        })
                        st.dataframe(hp_display, width="stretch", height=400)
                    else:
                        st.info(t("Geen partner data beschikbaar."))
