import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, inspect
from pathlib import Path
import logging
import sys # Keep sys import as it's used for path manipulation in some contexts
from config import DB_URL, DB_URL_POSTGRES, DB_URL_SMART_HEALTH, CSV_DATA_DIR, DATA_SOURCE
from account_filtering import filter_interne_accounts
from categorieen import leid_categorie_af
# Lokale instellingen
CODE_DIR = Path(__file__).resolve().parent

# All data should be loaded directly from database, not from local parquet files
# This ensures data is always current and no local storage is needed


def _derive_bmi_category(series: pd.Series) -> pd.Series:
    """Derive the five BMI bands used by the dashboard."""
    return leid_categorie_af(series, 'bmi')


def _derive_stress_category(series: pd.Series) -> pd.Series:
    """Derive stress bands from the 0-28 source scale."""
    return leid_categorie_af(series, 'stress')


def _derive_sleep_category(series: pd.Series) -> pd.Series:
    """Derive PSQI bands used by latest_scores: 0-6, 7-14, 15-20."""
    return leid_categorie_af(series, 'sleep')


def _load_table_from_database_always(table_name: str, code_dir: Path) -> pd.DataFrame:
    """Always load from database, ignore local parquet files."""
    return load_table_from_database(table_name, DB_URL)


def load_table_from_database(table_name: str, db_url: str) -> pd.DataFrame:
    """Laadt een tabel direct uit de database, accepteert een URL of een engine."""
    if DATA_SOURCE == 'csv':
        csv_path = CSV_DATA_DIR / f'{table_name}.csv'
        try:
            return pd.read_csv(csv_path)
        except FileNotFoundError:
            logging.warning("CSV-tabel ontbreekt: %s", csv_path)
            return pd.DataFrame()
        except Exception as exc:
            logging.warning("Fout bij laden van CSV-tabel '%s': %s", table_name, exc)
            return pd.DataFrame()

    engine = None
    if isinstance(db_url, str):
        if not db_url:
            print("Fout: DB_URL niet gevonden voor database load.")
            return pd.DataFrame()
        engine = create_engine(db_url) # Use the provided db_url
    else: # Assume it's an engine object
        engine = db_url # Renamed parameter to db_url_or_engine in thought process, but keeping db_url for consistency with existing calls
    
    try:
        return pd.read_sql(text(f'SELECT * FROM "{table_name}"'), engine)
    except Exception as e:
        # Fallback logica: als tabel niet in de huidige db zit, probeer de alternatieve database
        if isinstance(db_url, str) and DB_URL_POSTGRES and DB_URL_SMART_HEALTH:
            alt_url = DB_URL_POSTGRES if db_url == DB_URL_SMART_HEALTH else DB_URL_SMART_HEALTH
            if alt_url and alt_url != db_url: # Ensure alt_url is different from the current db_url
                try:
                    alt_engine = create_engine(alt_url)
                    df_alt = pd.read_sql(text(f'SELECT * FROM "{table_name}"'), alt_engine) # Use alt_engine
                    if not df_alt.empty:
                        logging.info(f"✓ Tabel '{table_name}' gevonden in fallback database.")
                        return df_alt
                except:
                    pass
        logging.warning(f"⚠ Fout bij laden van tabel '{table_name}' uit database: {e}")
        return pd.DataFrame()
def sync_database_to_parquet():
    """
    DEPRECATED: All data is now loaded directly from the database.
    Local parquet files are no longer used or maintained.
    This function is kept for backwards compatibility only.
    """
    logging.warning("sync_database_to_parquet() is deprecated. All data is loaded directly from database.")

def generate_consolidated_scores_in_db(db_url: str) -> tuple[bool, str]: # Changed to accept db_url string
    """
    Genereert de 'users_met_scores' tabel in de database door relevante tabellen te joinen
    en scores te pivoteren. Deze tabel bevat alle geconsolideerde scores en demografische
    gegevens per unieke gebruiker (user_id).
    """
    try:
        logging.info("Starting generate_consolidated_scores_in_db...")
        # 1. Load raw data with detailed logging
        df_fsh = load_table_from_database("factor_score_histories", db_url)
        logging.debug(f"Loaded factor_score_histories: {len(df_fsh)} rows")
        df_qf = load_table_from_database("questionnaire_factors", db_url)
        logging.debug(f"Loaded questionnaire_factors: {len(df_qf)} rows")
        df_participants = load_table_from_database("participants", db_url)
        logging.debug(f"Loaded participants: {len(df_participants)} rows")
        df_store_employees = load_table_from_database("store_employees", db_url)
        logging.debug(f"Loaded store_employees: {len(df_store_employees)} rows")
        df_completions = load_table_from_database("completions", db_url)
        import numpy as np

        if df_participants.empty:
            return False, "De tabel 'participants' is leeg. Kan geen geconsolideerde data genereren."

        # 2. Bereid scores voor (indien aanwezig)
        df_scores_wide = pd.DataFrame(columns=['participant_id'])
        if not df_fsh.empty and not df_qf.empty:
            df_fsh['questionnaire_factor_id'] = pd.to_numeric(df_fsh['questionnaire_factor_id'], errors='coerce')
            df_qf['id'] = pd.to_numeric(df_qf['id'], errors='coerce')
            df_fsh = df_fsh.merge(df_qf[['id', 'slug']], left_on='questionnaire_factor_id', right_on='id', how='left')
            df_fsh = df_fsh.rename(columns={'slug': 'score_slug'}).dropna(subset=['score_slug'])
            
            df_fsh['completion_created_at'] = pd.to_datetime(df_fsh['completion_created_at'], errors='coerce')
            df_fsh = df_fsh.sort_values('completion_created_at', ascending=False)
            df_fsh_latest = df_fsh.drop_duplicates(subset=['participant_id', 'score_slug'])

            if not df_fsh_latest.empty:
                df_scores_wide = df_fsh_latest.pivot(index='participant_id', columns='score_slug', values='score_value').reset_index()
                df_scores_wide.columns.name = None
                df_scores_wide = df_scores_wide.rename(columns=_get_slug_to_dashboard_mapping()) # Use the mapping

        # 3. Start met alle deelnemers
        demo_cols = ['id', 'user_id', 'store_id', 'partner_id', 'created_at', 'deleted_at', 'rec_user_gender', 'rec_age_current', 'postal_code', 'gender']
        available_demo = [c for c in demo_cols if c in df_participants.columns]
        
        df_consolidated = df_participants[available_demo].rename(columns={'id': 'participant_id'})
        
        # NIEUW: Zorg DIRECT dat user_id gevuld is voor de koppeling in stap 5
        if 'user_id' in df_consolidated.columns:
            df_consolidated['user_id'] = df_consolidated['user_id'].fillna(df_consolidated['participant_id'])
        else:
            df_consolidated['user_id'] = df_consolidated['participant_id']
        
        df_consolidated['user_id'] = pd.to_numeric(df_consolidated['user_id'], errors='coerce').astype('Int64')
        logging.debug(f"Starting consolidation with {len(df_consolidated)} participants.")

        # 4. Voeg scores toe aan de deelnemerslijst
        if not df_scores_wide.empty:
            logging.debug(f"Merging scores (wide) with {len(df_scores_wide)} rows.")
            df_consolidated = df_consolidated.merge(df_scores_wide, on='participant_id', how='left')
            print(f"DEBUG: Scores toegevoegd aan {df_consolidated[df_scores_wide.columns[1]].notna().sum()} deelnemers.")

        # 5. Koppel store_id (van participants tabel, gevuld via store_employees)
        # NIEUWE LOGICA: 
        # - Lees store_id rechtstreeks van participants (reeds gevuld)
        # - Fall-back: gebruik store_employees.id = participants.id join als participants.store_id niet aanwezig
        
        if 'store_id' not in df_consolidated.columns:
            df_consolidated['store_id'] = pd.NA
        
        # Check of store_id al gevuld is vanuit participants
        gekoppeld_existing = df_consolidated['store_id'].notna().sum()
        
        if gekoppeld_existing > 0:
            logging.debug(f"Store-linking already present in participants. {gekoppeld_existing} participants have store_id.")
        elif not df_store_employees.empty:
            # Fall-back: Probeer te koppelen via store_employees
            logging.debug("store_id not present in participants, trying store_employees...")
            
            # Voorkom duplicatie van store_id kolom bij merge
            if 'store_id' in df_consolidated.columns:
                df_consolidated = df_consolidated.drop(columns=['store_id'])
            
            # Verbeterde koppeling via user_id of employee_id (fallback naar id)
            link_col = 'user_id' if 'user_id' in df_store_employees.columns else ('employee_id' if 'employee_id' in df_store_employees.columns else 'id')
            
            df_store_link = df_store_employees[[link_col, 'store_id']].dropna().drop_duplicates(subset=link_col, keep='first')
            df_store_link = df_store_link.rename(columns={link_col: 'user_id'})
            
            # Zorg dat types overeenkomen voor de merge
            df_store_link['user_id'] = pd.to_numeric(df_store_link['user_id'], errors='coerce').astype('Int64')
            df_consolidated = df_consolidated.merge(df_store_link, on='user_id', how='left')
            
            gekoppeld = df_consolidated['store_id'].notna().sum()
            logging.debug(f"Store-linking via store_employees.id completed. {gekoppeld} of {len(df_consolidated)} participants have a store_id.")
            logging.debug(f"df_consolidated after store merge: {len(df_consolidated)} rows")
        else: # If df_store_employees is empty
            print("⚠️ WAARSCHUWING: store_employees tabel is leeg. Kan store_id niet koppelen.")
        
        if 'store_id' not in df_consolidated.columns:
            df_consolidated['store_id'] = pd.NA

        # 6. Normaliseer kolomnamen voor het dashboard
        if 'gender' in df_consolidated.columns and 'rec_user_gender' not in df_consolidated.columns:
            df_consolidated = df_consolidated.rename(columns={'gender': 'rec_user_gender'})

        # Zorg dat rec_user_gender 0/1 is (voor visualisaties)
        if 'rec_user_gender' in df_consolidated.columns:
            df_consolidated['rec_user_gender'] = _normalize_gender_series(df_consolidated['rec_user_gender'])
        
        # Dwing aanwezigheid van kritieke kolommen af om KeyErrors te voorkomen
        for col in ['rec_user_gender', 'rec_age_current', 'rec_ls_lifestyle_score', 'rec_heartrisk_cat', 'rec_ls_stress_cat', 'rec_med_bmi_cat', 'rec_ls_sleep_cat', 'postal_code']:
            if col not in df_consolidated.columns:
                df_consolidated[col] = pd.NA

        # Derivatie van categorieën indien de ruwe scores wel aanwezig zijn
        if 'rec_heartrisk_cat' in df_consolidated.columns and df_consolidated['rec_heartrisk_cat'].isna().all():
            if 'rec_heartrisk' in df_consolidated.columns:
                hr = pd.to_numeric(df_consolidated['rec_heartrisk'], errors='coerce')
                df_consolidated['rec_heartrisk_cat'] = np.where(hr >= 20, 2, np.where(hr >= 10, 1, 0))

        if 'rec_med_bmi_cat' in df_consolidated.columns and df_consolidated['rec_med_bmi_cat'].isna().all():
            if 'rec_med_bmi' in df_consolidated.columns:
                df_consolidated['rec_med_bmi_cat'] = _derive_bmi_category(df_consolidated['rec_med_bmi'])

        if 'rec_ls_exercise_steps_cat' not in df_consolidated.columns or df_consolidated['rec_ls_exercise_steps_cat'].isna().all():
            if 'rec_ls_exercise_steps_per_day' in df_consolidated.columns:
                steps = pd.to_numeric(df_consolidated['rec_ls_exercise_steps_per_day'], errors='coerce')
                df_consolidated['rec_ls_exercise_steps_cat'] = np.where(steps >= 10000, 2, np.where(steps >= 5000, 1, 0))

        if 'rec_ls_sleep_psqi_amount_of_hours_slept_text' not in df_consolidated.columns or df_consolidated['rec_ls_sleep_psqi_amount_of_hours_slept_text'].isna().all():
            if 'rec_ls_sleep_psqi_sum' in df_consolidated.columns:
                psqi = pd.to_numeric(df_consolidated['rec_ls_sleep_psqi_sum'], errors='coerce')
                df_consolidated['rec_ls_sleep_psqi_amount_of_hours_slept_text'] = np.select(
                    [psqi <= 5, psqi <= 8, psqi <= 12],
                    ['7-9 uur', '6-7 uur', '9-10 uur'],
                    default='Minder dan 6 uur'
                )

        if 'rec_ls_stress_type_1_worry' not in df_consolidated.columns or df_consolidated['rec_ls_stress_type_1_worry'].isna().all():
            if 'rec_ls_stress_sum' in df_consolidated.columns:
                stress = pd.to_numeric(df_consolidated['rec_ls_stress_sum'], errors='coerce')
                df_consolidated['rec_ls_stress_type_1_worry'] = np.where(stress >= 7.0, 2, np.where(stress >= 4.0, 1, 0))

        if 'rec_ls_stress_type_2_tense' not in df_consolidated.columns or df_consolidated['rec_ls_stress_type_2_tense'].isna().all():
            if 'rec_ls_stress_sum' in df_consolidated.columns:
                stress = pd.to_numeric(df_consolidated['rec_ls_stress_sum'], errors='coerce')
                df_consolidated['rec_ls_stress_type_2_tense'] = np.where(stress >= 6.5, 2, np.where(stress >= 3.5, 1, 0))

        if 'rec_framingham_non_invasive' not in df_consolidated.columns or df_consolidated['rec_framingham_non_invasive'].isna().all():
            if 'rec_heartrisk' in df_consolidated.columns:
                df_consolidated['rec_framingham_non_invasive'] = pd.to_numeric(df_consolidated['rec_heartrisk'], errors='coerce')

        # 7. Voeg laatste invuldatum toe
        if not df_completions.empty:
            df_completions['created_at'] = pd.to_datetime(df_completions['created_at'], errors='coerce')
            # Zorg dat participant_id numeriek is voor de groupby
            df_completions['participant_id'] = pd.to_numeric(df_completions['participant_id'], errors='coerce')
            df_completions = df_completions.dropna(subset=['participant_id'])

            logging.debug(f"Merging latest completion dates from {len(df_completions)} completions.")
            latest_dates = df_completions.groupby('participant_id')['created_at'].max().reset_index()
            df_consolidated = df_consolidated.merge(latest_dates.rename(columns={'created_at': 'latest_completion_at'}), on='participant_id', how='left')
        else:
            df_consolidated['latest_completion_at'] = pd.NaT # Ensure column exists even if empty
        # Dedupliceer op user_id zonder types te forceren (om strings te ondersteunen)
        df_consolidated = df_consolidated.dropna(subset=['user_id']).drop_duplicates(subset='user_id', keep='last')

        # 9. Opslaan naar database
        from data_integration import DataIntegrator
        integrator = DataIntegrator(db_url, backup_before_import=False)
        integrator.upsert_dataframe(df_consolidated, "users_met_scores", primary_keys=['user_id'], create_if_missing=True)

        logging.info(f"✓ 'users_met_scores' table generated with {len(df_consolidated)} records.")
        return True, "users_met_scores generated successfully."

    except Exception as e:
        logging.error(f"⚠ Error generating 'users_met_scores': {e}", exc_info=True)
        return False, str(e)

def _get_slug_to_dashboard_mapping():
    """Geeft de mapping terug van database slugs naar dashboard kolomnamen."""
    return {
        'bmi': 'rec_med_bmi',
        'bmi_cat': 'rec_med_bmi_cat',
        'age': 'rec_age_current',
        'gender': 'rec_user_gender',
        'heartrisk': 'rec_heartrisk',
        'heartrisk_cat': 'rec_heartrisk_cat',
        'stress': 'rec_ls_stress_sum',
        'stress_cat': 'rec_ls_stress_cat',
        'sleep': 'rec_ls_sleep_psqi_sum',
        'sleep_cat': 'rec_ls_sleep_cat',
        'exercise': 'rec_ls_score_exercise',
        'exercise_steps': 'rec_ls_exercise_steps_per_day',
        'fruit': 'rec_ls_score_fruit',
        'vegetables': 'rec_ls_score_vegetables',
        'sugar': 'rec_ls_score_sugar',
        'fat': 'rec_ls_score_saturated_fat',
        'salt': 'rec_ls_score_natrium',
        'alcohol': 'rec_ls_score_alcohol',
        'smoking': 'rec_smoking_answer',
        'dass_stress': 'rec_dass_stress_score',
        'dass_anxiety': 'rec_dass_anxiety_score',
        'dass_depression': 'rec_dass_depression_score',
        'resilience': 'rec_resilience_score',
        'wellbeing': 'rec_wellbeing_score',
        'selfefficacy': 'rec_self_efficacy_score',
        'health': 'rec_health'
    }

def _score_slug_to_dashboard_column(slug) -> str | None:
    """Map oude score-slugs naar dashboardkolommen en behoud huidige rec_/feat_-namen."""
    if pd.isna(slug):
        return None

    slug = str(slug).strip()
    if not slug:
        return None

    if slug.startswith(('rec_', 'feat_')):
        return slug

    return _get_slug_to_dashboard_mapping().get(slug)

def generate_store_averages_in_db(db_url: str) -> tuple[bool, str]: # Changed to accept db_url string
    """
    Genereert de 'store_average_scores' tabel in de database.
    Deze tabel bevat de gemiddelde scores per store per score_slug per maand.
    """
    try:
        logging.info("Starting generate_store_averages_in_db...")
        # 1. Load consolidated scores (which should now be generated by generate_consolidated_scores_in_db) with logging
        df_users_met_scores = load_table_from_database("users_met_scores", db_url)
        logging.debug(f"Loaded users_met_scores: {len(df_users_met_scores)} rows")
        if df_users_met_scores.empty:
            return False, "users_met_scores table is empty, cannot generate store averages."

        # Controleer of de benodigde kolommen bestaan
        for col in ['user_id', 'store_id', 'latest_completion_at']:
            if col not in df_users_met_scores.columns:
                df_users_met_scores[col] = pd.NA

        # Ensure necessary columns are numeric and datetime
        df_users_met_scores['user_id'] = pd.to_numeric(df_users_met_scores['user_id'], errors='coerce')
        df_users_met_scores['store_id'] = pd.to_numeric(df_users_met_scores['store_id'], errors='coerce')
        df_users_met_scores['latest_completion_at'] = pd.to_datetime(df_users_met_scores['latest_completion_at'], errors='coerce')
        logging.debug(f"After type conversion, {df_users_met_scores['store_id'].notna().sum()} usable records with store_id found.")

        df_users_met_scores = df_users_met_scores.dropna(subset=['user_id', 'store_id', 'latest_completion_at'])
        if df_users_met_scores.empty:
            logging.warning("df_users_met_scores is empty after dropping NaN in critical columns.")
            return False, "Geen volledige records (user, store, datum) gevonden voor gemiddelden."

        # Identify score columns (those starting with 'rec_' or 'feat_')
        score_columns = [col for col in df_users_met_scores.columns if col.startswith('rec_') or col.startswith('feat_')]
        
        # Melt the DataFrame to long format for easier aggregation per score_slug
        df_long = df_users_met_scores.melt(
            id_vars=['user_id', 'store_id', 'latest_completion_at'],
            value_vars=score_columns,
            var_name='score_slug',
            value_name='score_value'
        ).copy() # Ensure a copy to avoid SettingWithCopyWarning
        
        df_long['score_value'] = pd.to_numeric(df_long['score_value'], errors='coerce')
        df_long = df_long.dropna(subset=['score_value'])

        # Aggregate by store, score_slug, and month
        df_long['date'] = df_long['latest_completion_at'].dt.to_period('M').dt.to_timestamp()
        
        df_store_averages = df_long.groupby(['store_id', 'score_slug', 'date']).agg(
            average=('score_value', 'mean'),
            participants_count=('user_id', 'nunique')
        ).reset_index()
        logging.debug(f"Generated {len(df_store_averages)} store average records.")

        # Use DataIntegrator for robust UPSERT
        from data_integration import DataIntegrator
        integrator = DataIntegrator(db_url, backup_before_import=False)
        integrator.upsert_dataframe(df_store_averages, "store_average_scores", primary_keys=['store_id', 'score_slug', 'date'], create_if_missing=True)

        logging.info(f"✓ 'store_average_scores' table generated with {len(df_store_averages)} records.")
        return True, "store_average_scores generated successfully."

    except Exception as e:
        logging.error(f"⚠ Error generating 'store_average_scores': {e}", exc_info=True)
        return False, str(e)

# Implement specific load functions using _load_parquet_file or load_table_from_database
def load_factor_score_histories(db_url: str) -> pd.DataFrame:
    return load_table_from_database("factor_score_histories", db_url)

def load_completions(db_url: str) -> pd.DataFrame:
    return load_table_from_database("completions", db_url)

def load_users_met_scores(db_url: str) -> pd.DataFrame:
    return load_table_from_database("users_met_scores", db_url)

def load_participants(db_url: str) -> pd.DataFrame:
    return load_table_from_database("participants", db_url)

def load_my_clic_participants(db_url: str) -> pd.DataFrame:
    return load_table_from_database("my_clic_participants", db_url)


LATEST_SCORE_SLUGS = sorted({
    'rec_age_current',
    'rec_dass_anxiety_score',
    'rec_dass_depression_score',
    'rec_dass_stress_score',
    'rec_health',
    'rec_heartrisk',
    'rec_heartrisk_cat',
    'rec_ls_alcohol_total_per_week',
    'rec_ls_exercise_physical_activity_minutes_total',
    'rec_ls_exercise_steps_per_day',
    'rec_ls_lifestyle_score',
    'rec_ls_nutrition_fruit_fruit_per_day',
    'rec_ls_nutrition_natrium_per_day',
    'rec_ls_nutrition_saturated_fat_per_day',
    'rec_ls_nutrition_sugar_per_day',
    'rec_ls_score_alcohol',
    'rec_ls_score_fruit',
    'rec_ls_score_natrium',
    'rec_ls_score_saturated_fat',
    'rec_ls_score_sugar',
    'rec_ls_score_vegetables',
    'rec_ls_sleep_cat',
    'rec_ls_sleep_psqi_sum',
    'rec_ls_stress_cat',
    'rec_ls_stress_sum',
    'rec_ls_vegetables_gram_per_day',
    'rec_med_bmi',
    'rec_med_bmi_cat',
    'rec_resilience_score',
    'rec_smoking_answer',
    'rec_user_gender',
    'rec_wellbeing_score',
})


def _load_latest_scores_wide(db_url: str) -> pd.DataFrame:
    """Laad geselecteerde latest_scores slugs in wide format op participant_id."""
    try:
        if DATA_SOURCE == 'csv':
            df_latest = load_table_from_database('latest_scores', db_url)
            if df_latest.empty:
                return pd.DataFrame(columns=['participant_id'])
            df_latest = df_latest[df_latest['slug'].astype(str).str.startswith(('rec_', 'feat_'))].copy()
            df_latest['dashboard_col'] = df_latest['slug'].map(_score_slug_to_dashboard_column)
            df_latest = df_latest.dropna(subset=['participant_id', 'dashboard_col'])
            df_wide = df_latest.pivot_table(
                index='participant_id', columns='dashboard_col', values='value', aggfunc='first'
            ).reset_index()
            df_wide.columns.name = None
            return df_wide

        engine = create_engine(db_url) if isinstance(db_url, str) else db_url
        query = text("""
            SELECT participant_id, slug, value
            FROM latest_scores
            WHERE slug LIKE 'rec_%'
               OR slug LIKE 'feat_%'
               OR slug = ANY(:legacy_slugs)
        """)
        df_latest = pd.read_sql(
            query,
            engine,
            params={'legacy_slugs': list(_get_slug_to_dashboard_mapping().keys())},
        )
        if df_latest.empty:
            return pd.DataFrame(columns=['participant_id'])
        df_latest = df_latest.dropna(subset=['participant_id', 'slug'])
        df_latest['dashboard_col'] = df_latest['slug'].map(_score_slug_to_dashboard_column)
        df_latest = df_latest.dropna(subset=['dashboard_col'])
        df_wide = (
        df_latest.pivot_table(
                index='participant_id',
                columns='dashboard_col',
                values='value',
                aggfunc='first'
            )
            .reset_index()
        )
        df_wide.columns.name = None
        return df_wide
    except Exception as e:
        logging.warning(f"Kon latest_scores niet laden: {e}")
        return pd.DataFrame(columns=['participant_id'])


def _merge_latest_scores_into(df: pd.DataFrame, db_url: str) -> pd.DataFrame:
    """Verrijk een participant-level dataframe met geselecteerde latest_scores velden."""
    if df.empty or 'participant_id' not in df.columns:
        return df

    df_latest = _load_latest_scores_wide(db_url)
    if df_latest.empty or 'participant_id' not in df_latest.columns:
        return df

    overlap = [
        col for col in df_latest.columns
        if col != 'participant_id' and col in df.columns
    ]
    rename_map = {col: f"{col}__latest" for col in overlap}
    df_latest = df_latest.rename(columns=rename_map)

    df_merged = df.merge(df_latest, on='participant_id', how='left')
    for col in overlap:
        latest_col = rename_map[col]
        df_merged[col] = df_merged[latest_col].where(df_merged[latest_col].notna(), df_merged[col])
        df_merged = df_merged.drop(columns=[latest_col])

    return df_merged


def _normalize_gender_series(series: pd.Series) -> pd.Series:
    """Normaliseer diverse gender-representaties naar dashboardcodes 0/1."""
    s_num = pd.to_numeric(series, errors='coerce')
    s_txt = series.astype(str).str.strip().str.upper()

    result = pd.Series(pd.NA, index=series.index, dtype='object')
    result = result.where(~s_num.eq(1), 1)
    result = result.where(~s_num.eq(0), 0)
    result = result.where(~s_txt.str.startswith('M'), 1)
    result = result.where(~(s_txt.str.startswith('V') | s_txt.str.startswith('F')), 0)
    return result

def load_my_clic_participants_expanded(db_url: str) -> pd.DataFrame:
    engine = create_engine(db_url)
    inspector = inspect(engine)
    
    # Check if 'users_met_scores' table exists and has data
    if 'users_met_scores' not in inspector.get_table_names() or load_table_from_database("users_met_scores", engine).empty:
        logging.info("Table 'users_met_scores' does not exist or is empty. Attempting to generate it...")
        success, message = generate_consolidated_scores_in_db(engine)
        if not success:
            logging.error(f"Failed to generate 'users_met_scores': {message}")
            df = pd.DataFrame()
            df.attrs["load_error"] = message
            return df
    
    df = load_table_from_database("users_met_scores", db_url)
    if not df.empty:
        # Behoud user_id maar zorg dat participant_id er ook is voor compatibiliteit
        if 'user_id' in df.columns and 'participant_id' not in df.columns:
            df['participant_id'] = df['user_id']
        df = _merge_latest_scores_into(df, db_url)
        if 'rec_user_gender' in df.columns:
            df['rec_user_gender'] = _normalize_gender_series(df['rec_user_gender'])

        # Keep the value from the data source; dashboard views show the stored scale.
    
    if 'rec_ls_lifestyle_score' not in df.columns:
        df['rec_ls_lifestyle_score'] = pd.NA

    return df

def load_store_employees(db_url: str) -> pd.DataFrame:
    return load_table_from_database("store_employees", db_url)

def load_stores(db_url: str) -> pd.DataFrame:
    return load_table_from_database("stores", db_url)

def load_orders(db_url: str) -> pd.DataFrame:
    return load_table_from_database("orders", db_url)

def load_questionnaires(db_url: str) -> pd.DataFrame:
    return load_table_from_database("questionnaires", db_url)

def get_user_purchases(base_pad=None, db_url: str = DB_URL) -> pd.DataFrame:
    """
    Loads purchase data joining orders and products.
    
    Maps app user_id to QE participant_id through smart_health.my_clic_participants,
    so buyer visualisations can compare purchases against questionnaire scores.
    """
    if db_url and not str(db_url).startswith('sqlite'):
        try:
            engine = create_engine(db_url)
            query = text("""
                WITH participant_map AS (
                    SELECT DISTINCT ON (m.user_id)
                        m.user_id,
                        p.id AS participant_id
                    FROM smart_health.my_clic_participants m
                    JOIN public.participants p
                        ON p.public_id = m.qe_participant_id
                    WHERE m.user_id IS NOT NULL
                    ORDER BY m.user_id, p.id DESC
                )
                SELECT
                    pm.participant_id,
                    o.user_id,
                    o.status,
                    o.created_at,
                    pr.name
                FROM public.orders o
                LEFT JOIN public.products pr
                    ON pr.id::text = o.product_id::text
                LEFT JOIN participant_map pm
                    ON pm.user_id = o.user_id
            """)
            df_purchases = pd.read_sql(query, engine)
            if not df_purchases.empty:
                return df_purchases[['participant_id', 'user_id', 'status', 'created_at', 'name']]
        except Exception as e:
            logging.warning(f"Kon aankoopdata niet via participant bridge laden: {e}")


    df_orders = load_table_from_database("orders", db_url)
    df_products = load_table_from_database("products", db_url)

    if df_orders.empty or df_products.empty:
        logging.warning("Orders or products table is empty - no purchase data available")
        return pd.DataFrame(columns=['participant_id', 'user_id', 'status', 'created_at', 'name'])

    # Ensure numeric columns for join
    df_orders['product_id'] = pd.to_numeric(df_orders['product_id'], errors='coerce')
    df_products['id'] = pd.to_numeric(df_products['id'], errors='coerce')

    # Merge orders with products
    df_purchases = df_orders.merge(
        df_products[['id', 'name']],
        left_on='product_id',
        right_on='id',
        how='left',
        suffixes=('_order', '_product')
    )

    if 'participant_id' not in df_purchases.columns or df_purchases['participant_id'].isna().any():
        df_my_clic = load_table_from_database('my_clic_participants', db_url)
        df_participants = load_table_from_database('participants', db_url)
        bridge_columns = {'user_id', 'qe_participant_id'}
        if bridge_columns.issubset(df_my_clic.columns) and {'id', 'public_id'}.issubset(df_participants.columns):
            bridge = df_my_clic[['user_id', 'qe_participant_id']].dropna().drop_duplicates('user_id')
            participants_map = df_participants[['id', 'public_id']].dropna().drop_duplicates('public_id')
            bridge = bridge.merge(
                participants_map,
                left_on='qe_participant_id',
                right_on='public_id',
                how='left',
            )
            bridge = bridge.rename(columns={'id': 'participant_id'})[['user_id', 'participant_id']]
            df_purchases = df_purchases.drop(columns=['participant_id'], errors='ignore').merge(
                bridge,
                on='user_id',
                how='left',
            )

    # Use participant_id from orders table if present (synthetic data stores it directly).
    # Fall back to pd.NA only when the column is genuinely absent.
    for col in ['participant_id', 'status', 'created_at', 'name']:
        if col not in df_purchases.columns:
            df_purchases[col] = pd.NA
        elif col == 'participant_id':
            df_purchases[col] = pd.to_numeric(df_purchases[col], errors='coerce')

    return df_purchases[['participant_id', 'user_id', 'status', 'created_at', 'name']]

def add_app_user_ids_and_addresses(df: pd.DataFrame, db_url: str | None = None) -> pd.DataFrame:
    # TIP: Hier kun je ook de logica toevoegen om 'users_met_scores.parquet' 
    # opnieuw te genereren door de verschillende tabellen hier alvast te joinen.
    # Dat ontlast de dashboard-code aanzienlijk.

    effective_db_url = db_url or DB_URL

    # Load the canonical app addresses table. public.addresses can be a headerless
    # imported copy, while smart_health.addresses has the expected column names.
    try:
        engine = create_engine(effective_db_url)
        df_addresses = pd.read_sql(
            text("""
                SELECT model_id AS app_user_id, lat, long, city, country, postal_code, updated_at
                FROM smart_health.addresses
                WHERE model_type = 'user'
                  AND deleted_at IS NULL
            """),
            engine,
        )
    except Exception as e:
        logging.warning(f"Could not load smart_health.addresses, falling back to addresses table: {e}")
        df_addresses = load_table_from_database("addresses", effective_db_url)

    if df_addresses.empty:
        logging.warning("Addresses table is empty, cannot add address information.")
        return df # Return original df if no address data

    if 'model_type' in df_addresses.columns:
        df_user_addresses = df_addresses[df_addresses['model_type'] == 'user'].copy()
    else:
        df_user_addresses = df_addresses.copy()
    if df_user_addresses.empty:
        logging.warning("No user addresses found in addresses table.")
        return df # Return original df if no user address data

    if 'app_user_id' not in df_user_addresses.columns and 'model_id' in df_user_addresses.columns:
        df_user_addresses = df_user_addresses.rename(columns={'model_id': 'app_user_id'})

    if 'app_user_id' not in df_user_addresses.columns:
        logging.warning("Addresses data is missing app_user_id/model_id. Skipping address merge.")
        return df

    df_user_addresses['app_user_id'] = pd.to_numeric(df_user_addresses['app_user_id'], errors='coerce')

    df = df.copy()
    if 'app_user_id' not in df.columns:
        df['app_user_id'] = pd.NA

    if 'user_id' in df.columns:
        direct_user_id = pd.to_numeric(df['user_id'], errors='coerce')
        address_user_ids = set(df_user_addresses['app_user_id'].dropna().astype(int))
        direct_matches = int(direct_user_id.dropna().astype(int).isin(address_user_ids).sum())
        if direct_matches:
            df['app_user_id'] = direct_user_id

    if df['app_user_id'].isna().all() and ('public_id' in df.columns or 'participant_id' in df.columns):
        try:
            engine = create_engine(effective_db_url)
            is_sqlite = 'sqlite' in str(effective_db_url).lower()
            if 'public_id' in df.columns:
                table_name = "my_clic_participants" if is_sqlite else "smart_health.my_clic_participants"
                try:
                    participant_bridge = pd.read_sql(
                        text(f"""
                            SELECT qe_participant_id AS public_id, user_id AS app_user_id
                            FROM {table_name}
                            WHERE user_id IS NOT NULL
                        """),
                        engine,
                    )
                except Exception:
                    participant_bridge = pd.read_sql(
                        text("""
                            SELECT qe_participant_id AS public_id, user_id AS app_user_id
                            FROM my_clic_participants
                            WHERE user_id IS NOT NULL
                        """),
                        engine,
                    )
                df['public_id'] = df['public_id'].astype(str)
                participant_bridge['public_id'] = participant_bridge['public_id'].astype(str)
                merge_key = 'public_id'
            else:
                p_table = "participants" if is_sqlite else "public.participants"
                m_table = "my_clic_participants" if is_sqlite else "smart_health.my_clic_participants"
                try:
                    participant_bridge = pd.read_sql(
                        text(f"""
                            SELECT p.id AS participant_id, m.user_id AS app_user_id
                            FROM {p_table} p
                            JOIN {m_table} m
                                ON m.qe_participant_id = p.public_id
                            WHERE m.user_id IS NOT NULL
                        """),
                        engine,
                    )
                except Exception:
                    participant_bridge = pd.read_sql(
                        text("""
                            SELECT p.id AS participant_id, m.user_id AS app_user_id
                            FROM participants p
                            JOIN my_clic_participants m
                                ON m.qe_participant_id = p.public_id
                            WHERE m.user_id IS NOT NULL
                        """),
                        engine,
                    )
                df['participant_id'] = pd.to_numeric(df['participant_id'], errors='coerce')
                participant_bridge['participant_id'] = pd.to_numeric(participant_bridge['participant_id'], errors='coerce')
                merge_key = 'participant_id'
            participant_bridge['app_user_id'] = pd.to_numeric(participant_bridge['app_user_id'], errors='coerce')
            df = df.drop(columns=['app_user_id']).merge(
                participant_bridge.drop_duplicates(merge_key),
                on=merge_key,
                how='left',
            )
        except Exception as e:
            logging.warning(f"Could not map participants to app user IDs for addresses: {e}")

    if df['app_user_id'].isna().all():
        logging.warning("Input DataFrame could not be mapped to app user IDs for addresses.")
        return df

    # Select relevant address columns to merge and rename 'model_id' to 'user_id' for merging
    address_cols_to_add = ['app_user_id', 'lat', 'long', 'city', 'country', 'postal_code']
    df_user_addresses_filtered = df_user_addresses[
        [col for col in address_cols_to_add if col in df_user_addresses.columns]
    ]
    df_user_addresses_filtered = df_user_addresses_filtered.drop_duplicates(subset='app_user_id', keep='last')

    # Merge with the input DataFrame using a left merge to keep all users
    df_merged = df.merge(df_user_addresses_filtered, on='app_user_id', how='left', suffixes=('', '_addr'))

    logging.info(f"Added address information to {len(df_merged)} users.")
    return df_merged

def load_participants_with_factor_scores(db_url: str) -> pd.DataFrame:
    """
    Laadt participants met hun latest factor scores uit questionnaire responses.
    Dit bypast de users_met_scores tabel en laadt direct uit factor_score_histories.
    """
    try:
        import numpy as np

        engine = create_engine(db_url) if isinstance(db_url, str) else db_url
        
        # Load factor scores - pandas automatically parses JSON values as numbers
        query = """
        SELECT 
            participant_id,
            questionnaire_factor_id,
            score_value as score_numeric,
            completion_created_at
        FROM factor_score_histories
        WHERE score_value IS NOT NULL
        ORDER BY participant_id, completion_created_at DESC
        """
        
        df_scores = pd.read_sql(text(query), engine)
        
        if df_scores.empty:
            logging.warning("No factor scores found in factor_score_histories")
            return pd.DataFrame()
        
        # Map factor IDs to the current dashboard/raw-value schema.
        factor_map = {
            1: 'rec_resilience_score',
            2: 'rec_wellbeing_score',
            4: 'rec_dass_anxiety_score',
            5: 'rec_dass_depression_score',
            6: 'rec_dass_stress_score',
            7: 'rec_med_bmi',
            8: 'rec_age_current',
            9: 'rec_ls_alcohol_total_per_week',
            12: 'rec_ls_exercise_physical_activity_minutes_total',
            13: 'rec_ls_nutrition_saturated_fat_per_day',
            14: 'rec_ls_nutrition_fruit_fruit_per_day',
            17: 'rec_ls_nutrition_natrium_per_day',
            18: 'rec_ls_sleep_psqi_sum',
            19: 'rec_ls_stress_sum',
            20: 'rec_ls_nutrition_sugar_per_day',
            21: 'rec_ls_vegetables_gram_per_day',
            22: 'rec_smoking_answer',
            23: 'rec_health'
        }
        
        # Robuuste mapping: gebruik slugs uit de database indien beschikbaar
        try:
            df_qf = load_table_from_database('questionnaire_factors', db_url)
            if not df_qf.empty:
                id_to_slug = dict(zip(df_qf['id'], df_qf['slug']))
                df_scores['factor_name'] = (
                    df_scores['questionnaire_factor_id']
                    .map(id_to_slug)
                    .map(_score_slug_to_dashboard_column)
                )
            else:
                df_scores['factor_name'] = df_scores['questionnaire_factor_id'].map(factor_map)
        except:
            df_scores['factor_name'] = df_scores['questionnaire_factor_id'].map(factor_map)

        df_scores = df_scores.dropna(subset=['factor_name'])
        
        if df_scores.empty:
            logging.warning("No recognized factors found in scores")
            return pd.DataFrame()
        
        # Pivot to wide format - take latest value per factor
        df_latest = df_scores.sort_values('completion_created_at', ascending=False).drop_duplicates(
            subset=['participant_id', 'factor_name'], keep='first'
        )
        
        df_wide = df_latest.pivot_table(
            index='participant_id',
            columns='factor_name',
            values='score_numeric',
            aggfunc='first'
        ).reset_index()

        latest_completion = (
            df_latest.groupby('participant_id', as_index=False)['completion_created_at']
            .max()
            .rename(columns={'completion_created_at': 'latest_completion_at'})
        )
        df_wide = df_wide.merge(latest_completion, on='participant_id', how='left')
        
        # Add participant metadata when available.
        df_participants = load_table_from_database('participants', db_url)
        if not df_participants.empty and 'id' in df_participants.columns:
            participant_cols = [
                col for col in [
                    'id', 'user_id', 'created_at', 'deleted_at', 'public_id', 'pmo_id',
                    'postal_code', 'store_id', 'gender', 'rec_user_gender'
                ]
                if col in df_participants.columns
            ]
            df_participants = df_participants[participant_cols].rename(columns={'id': 'participant_id'})
            df_wide = df_wide.merge(df_participants, on='participant_id', how='inner')

        df_wide = _merge_latest_scores_into(df_wide, db_url)

        if 'user_id' not in df_wide.columns:
            df_wide['user_id'] = df_wide['participant_id']
        df_wide['user_id'] = pd.to_numeric(df_wide['user_id'], errors='coerce').fillna(
            pd.to_numeric(df_wide['participant_id'], errors='coerce')
        )

        if 'gender' in df_wide.columns and 'rec_user_gender' not in df_wide.columns:
            df_wide = df_wide.rename(columns={'gender': 'rec_user_gender'})

        if 'rec_user_gender' in df_wide.columns:
            df_wide['rec_user_gender'] = _normalize_gender_series(df_wide['rec_user_gender'])

        # Keep legacy aliases alive for code paths that still expect the old names.
        alias_map = {
            'rec_ls_alcohol_total_per_week': 'rec_ls_score_alcohol',
            'rec_ls_exercise_physical_activity_minutes_total': 'rec_ls_score_exercise',
            'rec_ls_nutrition_fruit_fruit_per_day': 'rec_ls_score_fruit',
            'rec_ls_nutrition_saturated_fat_per_day': 'rec_ls_score_saturated_fat',
            'rec_ls_nutrition_sugar_per_day': 'rec_ls_score_sugar',
            'rec_ls_vegetables_gram_per_day': 'rec_ls_score_vegetables',
            'rec_ls_nutrition_natrium_per_day': 'rec_ls_score_natrium',
            'rec_ls_sleep_psqi_amount_of_hours_slept_text_en': 'rec_ls_sleep_psqi_amount_of_hours_slept_text',
        }
        for source_col, alias_col in alias_map.items():
            if source_col in df_wide.columns and alias_col not in df_wide.columns:
                df_wide[alias_col] = df_wide[source_col]

        if 'store_id' not in df_wide.columns:
            df_wide['store_id'] = pd.NA

        if pd.Series(df_wide['store_id']).isna().all():
            df_store_employees = load_table_from_database('store_employees', db_url)
            if not df_store_employees.empty and {'id', 'store_id'}.issubset(df_store_employees.columns):
                df_store_link = (
                    df_store_employees[['id', 'store_id']]
                    .dropna(subset=['id', 'store_id'])
                    .drop_duplicates(subset='id', keep='first')
                    .rename(columns={'id': 'participant_id'})
                )
                df_wide = df_wide.merge(df_store_link, on='participant_id', how='left', suffixes=('', '_linked'))
                if 'store_id_linked' in df_wide.columns:
                    df_wide['store_id'] = df_wide['store_id'].where(
                        df_wide['store_id'].notna(), df_wide['store_id_linked']
                    )
                    df_wide = df_wide.drop(columns=['store_id_linked'])
        
        # Add derived categories if needed
        if 'rec_med_bmi' in df_wide.columns and (
            'rec_med_bmi_cat' not in df_wide.columns or df_wide['rec_med_bmi_cat'].isna().all()
        ):
            df_wide['rec_med_bmi_cat'] = _derive_bmi_category(df_wide['rec_med_bmi'])
        
        if 'rec_ls_stress_sum' in df_wide.columns and (
            'rec_ls_stress_cat' not in df_wide.columns or df_wide['rec_ls_stress_cat'].isna().all()
        ):
            df_wide['rec_ls_stress_cat'] = _derive_stress_category(df_wide['rec_ls_stress_sum'])

        # Voeg ontbrekende afgeleide kolommen toe voor visualisaties
        if 'rec_heartrisk' in df_wide.columns and ('rec_heartrisk_cat' not in df_wide.columns or df_wide['rec_heartrisk_cat'].isna().all()):
            hr = pd.to_numeric(df_wide['rec_heartrisk'], errors='coerce')
            df_wide['rec_heartrisk_cat'] = np.where(hr >= 20, 2, np.where(hr >= 10, 1, 0))

        if 'rec_ls_sleep_psqi_sum' in df_wide.columns and ('rec_ls_sleep_cat' not in df_wide.columns or df_wide['rec_ls_sleep_cat'].isna().all()):
            df_wide['rec_ls_sleep_cat'] = _derive_sleep_category(df_wide['rec_ls_sleep_psqi_sum'])
            
        if 'rec_age_current' in df_wide.columns and ('rec_med_age_cat' not in df_wide.columns or df_wide['rec_med_age_cat'].isna().all()):
            age = pd.to_numeric(df_wide['rec_age_current'], errors='coerce')
            df_wide['rec_med_age_cat'] = np.select([age >= 55, age >= 40], [2, 1], default=0)
        # Exercise steps category
        if 'rec_ls_exercise_steps_per_day' in df_wide.columns and ('rec_ls_exercise_steps_cat' not in df_wide.columns or df_wide['rec_ls_exercise_steps_cat'].isna().all()):
            steps = pd.to_numeric(df_wide['rec_ls_exercise_steps_per_day'], errors='coerce')
            df_wide['rec_ls_exercise_steps_cat'] = np.select(
                [steps >= 10000, steps >= 5000],
                [2, 1],
                default=0
            )
        # Blood pressure category
        if 'rec_med_blood_pressure' in df_wide.columns and ('rec_med_blood_pressure_cat' not in df_wide.columns or df_wide['rec_med_blood_pressure_cat'].isna().all()):
            bp = pd.to_numeric(df_wide['rec_med_blood_pressure'], errors='coerce')
            df_wide['rec_med_blood_pressure_cat'] = np.select(
                [bp >= 2, bp >= 1], # Illustrative thresholds
                [2, 1],
                default=0
            )

        # Koppel legacy aliases zonder ruwe waarden tot een leefstijlscore te combineren.
        for source_col, alias_col in alias_map.items():
            if source_col in df_wide.columns and alias_col not in df_wide.columns:
                df_wide[alias_col] = df_wide[source_col]

        for col in [
            'rec_user_gender', 'rec_age_current', 'rec_ls_lifestyle_score', 'rec_heartrisk_cat',
            'rec_ls_stress_cat', 'rec_med_bmi_cat', 'rec_ls_sleep_cat', 'postal_code',
            'rec_ls_exercise_steps_per_day'
        ]:
            if col not in df_wide.columns:
                df_wide[col] = pd.NA
        
        logging.info(f"✓ Loaded {len(df_wide)} participants with factor scores")
        return df_wide
        
    except Exception as e:
        logging.error(f"Error loading participants with factor scores: {e}", exc_info=True)
        return pd.DataFrame()



if __name__ == "__main__":
    # 1. Sync naar parquet is uitgeschakeld om database directheid te garanderen.
    # sync_database_to_parquet() 
    
    # 2. Herbouw de geconsolideerde tabellen in de database
    try:
        engine = create_engine(DB_URL)
        logging.info("--- Start Consolidatie in Database ---")
        s1, m1 = generate_consolidated_scores_in_db(engine)
        s2, m2 = generate_store_averages_in_db(engine)
        if s1 and s2:
            logging.info("✓ Consolidatie voltooid.")
        else:
            logging.error(f"⚠ Consolidatie fout: {m1} | {m2}")
    except Exception as e:
        logging.error(f"⚠ Fout tijdens main consolidatie: {e}", exc_info=True)
