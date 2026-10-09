"""
Statistische analysefuncties voor het Smart Health dashboard.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from pathlib import Path
import logging
from collections.abc import Sequence
from typing import Tuple, Union, cast
from helpers import get_numeric_clean, aggregate_by_groups

try:
    import statsmodels.api as sm  # noqa: F401
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

logger = logging.getLogger(__name__)

ParticipantIds = Sequence[int] | set[int] | pd.Series | None
 

from i18n import tr
from config import DB_URL
from data_ingestion import load_users_met_scores, load_my_clic_participants_expanded, load_completions, load_questionnaires, load_participants
CODE_DIR = Path(__file__).resolve().parent
from kleuren import HOOFD_KLEUR, GENDER_COLORS, GENDER_LABELS, RISICO_COLORS

# ── Kolommen voor analyses ────────────────────────────────────────────────────
LEEFSTIJL_SCORES = {
    'Fruit':         'rec_ls_score_fruit',
    'Groenten':      'rec_ls_score_vegetables',
    'Suiker':        'rec_ls_score_sugar',
    'Vet':           'rec_ls_score_saturated_fat',
    'Alcohol':       'rec_ls_score_alcohol',
    'Zout':          'rec_ls_score_natrium',
    'Bewegen':       'rec_ls_score_exercise',
    'Slaap':         'rec_ls_score_sleep',
    'Stress':        'rec_ls_stress_sum',
    'BMI':           'rec_med_bmi_cat',
    'Heartrisk':     'rec_heartrisk_cat',
}

RUWE_WAARDEN = {
    'Fruit (stuks/dag)':        'rec_ls_nutrition_fruit_fruit_per_day',
    'Groenten (gram/dag)':      'rec_ls_vegetables_gram_per_day',
    'Suiker (gram/dag)':        'rec_ls_nutrition_sugar_per_day',
    'Vet (gram/dag)':           'rec_ls_nutrition_saturated_fat_per_day',
    'Natrium (mg/dag)':         'rec_ls_nutrition_natrium_per_day',
    'Alcohol (glazen/week)':    'rec_ls_alcohol_total_per_week',
    'Slaap PSQI':               'rec_ls_sleep_psqi_sum',
    'Stressscore':              'rec_ls_stress_sum',
    'DASS stress':              'rec_dass_stress_score',
    'DASS angst':               'rec_dass_anxiety_score',
    'DASS depressie':           'rec_dass_depression_score',
    'Veerkracht':               'rec_resilience_score',
    'Welzijn':                  'rec_wellbeing_score',
    'Werkvermogen (WAI)':       'rec_asr_wai_score',
    'Leefstijlscore':           'rec_ls_lifestyle_score',
    'Leeftijd':                 'rec_age_current',
}

T_TOETS_EXTRA_VARIABELEN = {
    'Framingham score':         'rec_framingham_non_invasive',
    'Heartrisk (score)':        'rec_heartrisk',
    'BMI (waarde)':             'rec_med_bmi',
    'Werkdruk':                 'rec_asr_workload_score',
    'Uitputting':               'rec_asr_exhaustion_score',
    'Burn-out score':           'rec_asr_burn_out_score',
    'Vitaliteit':               'rec_asr_vitality_score',
    'Werktevredenheid':         'rec_asr_job_satisfaction_score',
    'Stappen per dag':          'rec_ls_exercise_steps_per_day',
    'Beweegminuten':            'rec_ls_exercise_physical_activity_minutes_total',
}

OUTLIER_GRENZEN = {
    'rec_ls_nutrition_natrium_per_day':         8000,
    'rec_ls_nutrition_sugar_per_day':           300,
    'rec_ls_nutrition_saturated_fat_per_day':   150,
    'rec_ls_vegetables_gram_per_day':           800,
    'rec_ls_alcohol_total_per_week':            56,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATAKWALITEIT
# ══════════════════════════════════════════════════════════════════════════════
def bereken_datakwaliteit(df: Union[pd.DataFrame, Path], n_totaal_override: int | None = None) -> pd.DataFrame:
    """
    Berekent missende waarden en basisstatistieken voor ALLE kolommen
    met minimaal 1 ingevulde waarde.
    n_totaal_override: als opgegeven, wordt dit gebruikt als noemer voor missend %.
                       Handig om te vergelijken met totaal aantal accounts.
    """
    if isinstance(df, Path):
        # Laadt van database in plaats van parquet
        try:
            df = load_my_clic_participants_expanded(DB_URL)
        except Exception as exc:
            logger.warning(f"Kon databestand voor datakwaliteit niet laden ({exc})")
            return pd.DataFrame(columns=[
                'Variabele', 'Kolom', 'Ingevuld', 'Missend', 'Missend (%)',
                'Gemiddelde', 'Std', 'Min', 'Max', 'Categorie verdeling'
            ])

    rijen = []
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'Variabele', 'Kolom', 'Ingevuld', 'Missend', 'Missend (%)',
            'Gemiddelde', 'Std', 'Min', 'Max', 'Categorie verdeling'
        ])

    n_totaal = n_totaal_override if n_totaal_override is not None else len(df)
    if n_totaal <= 0:
        n_totaal = len(df)

    # Maak een leesbaar label van de kolomnaam
    def maak_label(kolom: str) -> str:
        # Gebruik bekende labels waar beschikbaar
        voor_labels = {**LEEFSTIJL_SCORES, **RUWE_WAARDEN}
        omgekeerd = {v: k for k, v in voor_labels.items()}
        if kolom in omgekeerd:
            return omgekeerd[kolom]
        # Anders: kolomnaam opschonen
        return (kolom
                .replace('rec_', '').replace('ls_', '').replace('med_', '')
                .replace('asr_', '').replace('_', ' ').strip().capitalize())

    # Voorvoegsels die we overslaan
    skip_prefixes = (
        'rec_the_score_by_age.',               # leeftijdslookup uit geneste JSON
        'show_',                               # binaire adviesmarkeringen
        'rec_ls_nutrition_fat_total_',         # vet per product (al in totaal)
        'rec_ls_nutrition_natrium_',           # natrium per product (al in totaal)
        'rec_ls_nutrition_saturated_fat_',     # vet per product (al in totaal)
        'rec_ls_nutrition_sugar_',             # suiker per product (al in totaal)
        'rec_ls_alcohol_',                     # alcohol per dag (al in totaal per week)
        'rec_ls_exercise_physical_activity_',  # beweging per dag (al in totaal)
        'rec_asr_policy_',                     # beleidsscores
        'rec_coach_',                          # coach aanbevelingen
        'rec_med_hr_',                         # ruwe medische waarden (al verwerkt)
        'rec_med_bmi_user_weight_',            # tussenberekeningen BMI
        'rec_ls_sleep_psqi_component_',        # PSQI subcomponenten (al in totaal)
        'rec_ls_sleep_psqi_amount_',           # slaapuren tekst (al verwerkt)
        'rec_ls_stress_type_',                 # stress subtypen (al in totaal)
        'rec_asr_exhaustion_',                 # uitputting subschalen (al in totaal)
        'rec_asr_cynicism_',                   # cynisme subschalen
        'rec_asr_absorption_',                 # absorptie subschalen
        'rec_asr_dedication_',                 # toewijding subschalen
        'rec_asr_harassment_',                 # intimidatie subschalen
        'rec_asr_coping_',                     # coping subschalen
        'rec_asr_wai_dimension_',              # WAI subschalen (al in totaal)
        'rec_positive_health_',                # positieve gezondheid subschalen
        'rec_digital_detox_smartphone_use_',   # smartphone gebruik details
        'answers_',                            # ruwe antwoordblokken
        'rec_smoking_factor_',                 # rookfactor teksten
        'rec_med_diabetes_factor_',            # diabetes factor teksten
        'rec_med_blood_pressure_factor_',      # bloeddruk factor teksten
        'rec_med_hr_heredity_factor_',         # erfelijkheid factor teksten
        'rec_ls_exercise_steps_per_day_',      # stappen tekst versies
        'rec_ls_nutrition_bread_',             # brood subdetails
        'rec_ls_nutrition_biscuit',            # koekjes subdetails
        'rec_ls_nutrition_candy',              # snoep subdetails
        'rec_ls_nutrition_chocolate',          # chocolade subdetails
        'rec_ls_nutrition_chicken',            # kip subdetails
        'rec_ls_nutrition_custard',            # vla subdetails
        'rec_ls_nutrition_fries',              # friet subdetails
        'rec_ls_nutrition_fruit_juice',        # vruchtensap subdetails
        'rec_ls_nutrition_fruit_yoghurt',      # fruityoghurt subdetails
        'rec_ls_nutrition_gravy',              # jus subdetails
        'rec_ls_nutrition_icecream',           # ijs subdetails
        'rec_ls_nutrition_milk',               # melk subdetails
        'rec_ls_nutrition_muesli',             # muesli subdetails
        'rec_ls_nutrition_pancakes',           # pannenkoek subdetails
        'rec_ls_nutrition_pastry',             # gebak subdetails
        'rec_ls_nutrition_pizza',              # pizza subdetails
        'rec_ls_nutrition_sausage',            # worst subdetails
        'rec_ls_nutrition_snack',              # snack subdetails
        'rec_ls_nutrition_soda',               # frisdrank subdetails
        'rec_ls_nutrition_soup',               # soep subdetails
        'rec_ls_nutrition_yoghurt',            # yoghurt subdetails
        'rec_ls_nutrition_crisps',             # chips subdetails
        'rec_ls_nutrition_fish',               # vis subdetails
        'rec_ls_nutrition_legumes',            # peulvruchten subdetails
        'rec_ls_nutrition_nuts',               # noten subdetails
        'rec_ls_nutrition_dinner_',            # avondeten subdetails
        'rec_ls_nutrition_lettuce',            # sla subdetails
        'rec_ls_nutrition_meat',               # vlees subdetails
        'rec_ls_nutrition_breakfast',          # ontbijt subdetails
        'rec_ls_nutrition_diet_',              # dieet type
        'rec_ls_nutrition_bake',               # bak subdetails
        'rec_ls_nutrition_bread_cereal',       # brood/granen
        'rec_ls_nutrition_bread_spread',       # broodbeleg
        'rec_ls_md_',                          # voedingsrichtlijnen (advies)
        'rec_ls_score_',                       # scores (al als ruwe waarden)
        'rec_asr_manager_',                    # manager subschalen
        'rec_asr_colleagues_',                 # collega subschalen
        'rec_asr_resources_',                  # resources subschalen
        'rec_asr_clarity_',                    # duidelijkheid subschalen
        'rec_asr_culture_',                    # cultuur subschalen (via organizational)
        'rec_asr_burden_',                     # belasting subschalen
        'rec_asr_hectic_',                     # hectiek subschalen
        'rec_asr_feel_better_',                # verbetering subschalen
        'rec_asr_fulfillment_',                # voldoening subschalen
        'rec_asr_labor_',                      # arbeidsmarkt subschalen
        'rec_asr_learning_',                   # leren subschalen
        'rec_asr_mobility_',                   # mobiliteit subschalen
        'rec_asr_neglect_',                    # verwaarlozing subschalen
        'rec_asr_pension_',                    # pensioen subschalen
        'rec_asr_realized_',                   # gerealiseerde mobiliteit
        'rec_asr_retirement_',                 # pensioen subschalen
        'rec_asr_satisfaction_',               # tevredenheid subschalen
        'rec_asr_competence_',                 # competentie subschalen
        'rec_asr_commitment_',                 # betrokkenheid subschalen
        'rec_asr_distraction_',                # afleiding subschalen
        'rec_asr_education_',                  # opleiding subschalen
        'rec_asr_emotion_',                    # emotie subschalen
        'rec_asr_info_',                       # informatie subschalen
        'rec_asr_intervene_',                  # interventie subschalen
        'rec_asr_jobcrafting_',                # jobcrafting subschalen
        'rec_asr_lately_',                     # recent subschalen
        'rec_asr_management_',                 # management subschalen
        'rec_asr_optimistic_',                 # optimisme subschalen
        'rec_asr_personal_',                   # persoonlijk subschalen
        'rec_asr_psychological_',              # psychologisch subschalen
        'rec_asr_relaxation_',                 # ontspanning subschalen
        'rec_asr_resilience_',                 # veerkracht subschalen
        'rec_asr_take_break_',                 # pauze subschalen
        'rec_asr_work_experience_',            # werkervaring subschalen
        'rec_asr_worksituation_',              # werksituatie subschalen
        'rec_asr_working_attitude_',           # werkhouding subschalen
        'rec_asr_job_crafting_',               # jobcrafting subschalen
        'rec_asr_job_security_',               # baanzekerheid subschalen
        'rec_asr_low_',                        # laag subschalen
        'rec_asr_open_',                       # open subschalen
        'rec_asr_guiding_',                    # begeleidend subschalen
        'rec_ls_sleep_type_',                  # slaaptype subschalen
        'rec_ls_fluid_',                       # vocht subdetails
    )

    # Kolommen die we expliciet overslaan (display/tekst duplicaten)
    skip_exact = {
        'id', 'user_id', 'store_id', 'model_id', 'postal_code',
        'participant_id', 'partner_id', 'public_id', 'lat', 'long',
        'city', 'country_code',
        'created_at', 'updated_at', 'deleted_at',
        # Display en tekst duplicaten
        'rec_med_bmi_advised_weight_loss',
        'rec_heartrisk_the_score_lookup',
        'rec_heartrisk_score_used',
        'rec_heartrisk_display',
        'rec_heartrisk_plus',
        'rec_heartrisk_all_factors_optimal',
        'rec_heartrisk_all_factors_optimal_influenceable',
        'rec_heartrisk_all_factors_optimal_influenceable_incl_glucose',
        'rec_lifestyle_all_factors_optimal',
        'rec_the_score_display',
        'rec_the_score_by_age',
        'rec_framingham_non_invasive_cat',
        'rec_med_blood_pressure_text',
        'rec_med_blood_pressure_text_en',
        'rec_med_blood_pressure_display',
        'rec_med_hr_hdl_cholesterol_display',
        'rec_med_hr_total_cholesterol_display',
        'rec_med_hr_glucose_display',
        'rec_med_hr_heredity_text',
        'rec_med_hr_heredity_text_en',
        'rec_smoking_text',
        'rec_smoking_text_en',
        'rec_smoking_answer',           # ruwe rookwaarde (al in cat)
        'rec_ls_alcohol_binge_cat_text',
        'rec_ls_exercise_steps_per_day_text_translated',
        'rec_ls_lifestyle_score_no_penalty',
        'rec_ls_lifestyle_score_sub',
        # Score + categorie duplicaten (bewaar alleen score)
        'rec_asr_work_experience_category',
        'rec_asr_job_satisfaction_category',
        'rec_asr_working_attitude_category',
        'rec_asr_workload_category',
        'rec_asr_worksituation_category',
        'rec_asr_work_ability_category',
        'rec_asr_health_category',
        'rec_asr_minor_mental_complaints_category',
        'rec_asr_personal_competences_category',
        'rec_ls_lifestyle_score_cat',
        'rec_digital_detox_stress_score_cat',
        'rec_digital_detox_smartphone_addiction_risk_cat',
        # Menopauze subschalen
        'menopause_genitourinary_bladder_problems',
        'menopause_genitourinary_sexual_problems',
        'menopause_genitourinary_vagina_dryness',
        'menopause_psychological_anxiety',
        'menopause_psychological_depressive_mood',
        'menopause_psychological_irritability',
        'menopause_psychological_physical_mental_exhaustion',
        'menopause_somatic_heart_discomfort',
        'menopause_somatic_joint_muscular_discomfort',
        'menopause_somatic_sleep_problems',
        'menopause_somatic_sweating',
        # Show salt/sugar (al gefilterd via prefix maar voor zekerheid)
        'show_salt_chocolate_advice',
        'show_sugar_chocolate_advice',
        # Nutrition subdetails die nog over zijn
        'rec_ls_nutrition_cheese_on_bread_per_day',
        'rec_ls_nutrition_cheese_warm_per_day',
        'rec_ls_nutrition_fruit_fruit_per_day',   # al in fruit_cat
        # Exercise subdetails
        'rec_ls_exercise_steps_per_day',          # al in steps_cat
        'rec_ls_exercise_steps_cat',              # bewaar alleen steps_per_day
        # Te weinig data
        'rec_med_cholesterol_hdl_cat', 'rec_med_cholesterol_total_cat',
        'rec_med_glucose_cat', 'rec_the_score',
        # Slechts 1 unieke waarde
        'rec_asr_absent_at_work_score', 'rec_med_health_check_complete',
        # Duplicaten
        'rec_age_round_up', 'rec_ls_exercise_minutes_cat',
        'rec_ls_steps_score', 'rec_ls_exercise_score',
        'rec_ls_nutrition_fruit_cat', 'rec_resilience_cat',
        'rec_self_efficacy_cat', 'rec_wellbeing_cat',
        'rec_dass_anxiety_cat', 'rec_dass_depression_cat', 'rec_dass_stress_cat',
        'rec_med_age_cat', 'rec_ls_nutrition_total_nutrition_cat',
        # ASR leefstijl subschalen (al als aparte variabelen aanwezig)
        'rec_asr_alcohol_score', 'rec_asr_bmi_score', 'rec_asr_movement_score',
        'rec_asr_nutrition_score', 'rec_asr_rest_score', 'rec_asr_sleep_score',
        'rec_asr_smoking_score',
        # ASR subschalen met weinig variatie
        'rec_asr_avoid_problem_score', 'rec_asr_can_working_age_score',
        'rec_asr_complaints_due_work_score', 'rec_asr_worked_when_sick_score',
        'rec_asr_wish_working_age_score', 'rec_asr_absent_at_work_score',
        # Beweging dubbel
        'rec_ls_moderate_exercise_physical_activity_minutes_total',
        'rec_ls_vigorous_exercise_physical_activity_minutes_total',
        'rec_ls_exercise_physical_insufficient_days_total',
        # Slaap dubbel (bewaar psqi_sum als meest complete maat)
        'rec_ls_sleep_sum',
        # Nutrition dubbel
        'rec_ls_nutrition_total_fat_per_day',  # al in saturated_fat_per_day
    }

    # Verwijder _cat duplicaten als de ruwe kolom ook bestaat
    # Bijv: rec_med_blood_pressure_cat weglaten als rec_med_blood_pressure bestaat
    cat_skip = set()
    for kolom in df.columns:
        if kolom.endswith('_cat'):
            ruwe = kolom[:-4]  # verwijder '_cat'
            if ruwe in df.columns:
                cat_skip.add(kolom)
    skip_exact = skip_exact | cat_skip

    for kolom in sorted(df.columns):
        if kolom in skip_exact:
            continue
        if any(kolom.startswith(p) for p in skip_prefixes):
            continue

        # Probeer numeriek te converteren
        s = pd.to_numeric(df[kolom].replace('None', pd.NA), errors='coerce')
        n_geldig = int(s.notna().sum())

        # Alleen kolommen met minstens 1 waarde
        if n_geldig == 0:
            continue

        n_missing = n_totaal - n_geldig

        # Zoek bijbehorende cat kolom voor context
        cat_kolom  = kolom + '_cat'
        cat_info   = None
        if cat_kolom in df.columns:
            s_cat = pd.to_numeric(df[cat_kolom], errors='coerce')
            cat_counts = s_cat.dropna().value_counts().sort_index()
            cat_info = ', '.join([f"{int(str(k))}={int(v)}" for k, v in cat_counts.items()])

        rijen.append({
            'Variabele':        maak_label(kolom),
            'Kolom':            kolom,
            'Ingevuld':         n_geldig,
            'Missend':          n_missing,
            'Missend (%)':      round(n_missing / n_totaal * 100, 1) if n_totaal > 0 else 0.0,
            'Gemiddelde':       round(s.mean(), 2) if n_geldig > 0 else None,
            'Std':              round(s.std(), 2)  if n_geldig > 0 else None,
            'Min':              round(s.min(), 2)  if n_geldig > 0 else None,
            'Max':              round(s.max(), 2)  if n_geldig > 0 else None,
            'Categorie verdeling': cat_info,
        })
    if not rijen:
        return pd.DataFrame(columns=[
            'Variabele', 'Kolom', 'Ingevuld', 'Missend', 'Missend (%)',
            'Gemiddelde', 'Std', 'Min', 'Max', 'Categorie verdeling'
        ])

    return pd.DataFrame(rijen).sort_values('Missend (%)', ascending=False)


def maak_missende_waarden_plot(df_kwaliteit: pd.DataFrame) -> go.Figure:
    """Horizontale bar chart van missende waarden per variabele."""
    df = df_kwaliteit[df_kwaliteit['Missend (%)'] > 0].copy()
    df = df.sort_values('Missend (%)', ascending=True)

    kleuren = [
        RISICO_COLORS[2] if x > 50 else
        RISICO_COLORS[1] if x > 20 else
        RISICO_COLORS[0]
        for x in df['Missend (%)']
    ]

    fig = go.Figure(go.Bar(
        x=df['Missend (%)'],
        y=df['Variabele'],
        orientation='h',
        marker_color=kleuren,
        text=df['Missend (%)'].astype(str) + '%',
        textposition='outside',
    ))
    fig.update_layout(
        title='Percentage missende waarden per variabele',
        xaxis_title='Missend (%)',
        yaxis_title='',
        height=max(400, len(df) * 25),
        margin=dict(l=180),
        showlegend=False,
    )
    fig.add_vline(x=20, line_dash='dash', line_color='white',
                  annotation_text='20% grens', annotation_font_color='white',
                  annotation_position='top right')
    fig.add_vline(x=50, line_dash='dash', line_color='black',
                  annotation_text='50% grens', annotation_font_color='white',
                  annotation_position='top right')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. OUTLIER ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
def bereken_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Berekent outliers via IQR methode per variabele."""
    rijen = []
    for label, kolom in RUWE_WAARDEN.items():
        if kolom not in df.columns:
            continue
        s = pd.to_numeric(df[kolom], errors='coerce').dropna()
        if len(s) < 10:
            continue

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr     = q3 - q1
        grens_laag  = q1 - 1.5 * iqr
        grens_hoog  = q3 + 1.5 * iqr
        n_outlier_laag = (s < grens_laag).sum()
        n_outlier_hoog = (s > grens_hoog).sum()
        n_outlier      = n_outlier_laag + n_outlier_hoog

        # Domeingrens (indien gedefinieerd)
        domein_grens = OUTLIER_GRENZEN.get(kolom)
        n_domein     = int((s > domein_grens).sum()) if domein_grens else None

        rijen.append({
            'Variabele':            label,
            'N geldig':             len(s),
            'Gemiddelde':           round(s.mean(), 1),
            'Mediaan':              round(s.median(), 1),
            'Std':                  round(s.std(), 1),
            'Min':                  round(s.min(), 1),
            'Max':                  round(s.max(), 1),
            'IQR grens laag':       round(grens_laag, 1),
            'IQR grens hoog':       round(grens_hoog, 1),
            'Outliers laag (n)':    int(n_outlier_laag),
            'Outliers hoog (n)':    int(n_outlier_hoog),
            'Outliers totaal (n)':  int(n_outlier),
            'Outliers (%)':         round(n_outlier / len(s) * 100, 1),
            'Domeingrens':          domein_grens,
            'Boven domeingrens (n)': n_domein,
        })
    return pd.DataFrame(rijen).sort_values('Outliers (%)', ascending=False)


def maak_outlier_boxplot(df: pd.DataFrame, variabele_label: str,
                          kolom: str, geslacht_filter: str = 'beide') -> go.Figure:
    """Boxplot voor een specifieke variabele met outlier markering."""
    df2 = df.copy()
    df2['geslacht'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(GENDER_LABELS)
    s = pd.to_numeric(df2[kolom], errors='coerce')

    # Cap extreme outliers voor leesbaarheid
    domein_max = OUTLIER_GRENZEN.get(kolom)
    if domein_max:
        s = s.clip(upper=domein_max * 1.5)

    df2['_waarde'] = s

    split_by_gender = geslacht_filter == 'beide'
    if split_by_gender:
        df2 = df2.dropna(subset=['_waarde', 'geslacht'])
        kleur_map = {v: GENDER_COLORS[v] for v in GENDER_LABELS.values()}
        fig = px.box(
            df2, x='geslacht', y='_waarde',
            color='geslacht', color_discrete_map=kleur_map,
            points='outliers',
            labels={'geslacht': 'Geslacht', '_waarde': variabele_label},
            title=f'Verdeling en outliers: {variabele_label}',
            category_orders={'geslacht': ['Man', 'Vrouw']},
        )
        fig.update_layout(showlegend=False)
    else:
        df2 = df2.dropna(subset=['_waarde'])
        fig = px.box(
            df2, y='_waarde', color_discrete_sequence=[HOOFD_KLEUR],
            labels={'_waarde': variabele_label}, title=f'Verdeling en outliers: {variabele_label}',
        )

    # Domeingrens toevoegen
    if domein_max:
        fig.add_hline(y=domein_max, line_dash='dash', line_color='red',
                      annotation_text=f'Domeingrens ({domein_max})',
                      annotation_position='right')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. T-TOETS MANNEN VS VROUWEN
# ══════════════════════════════════════════════════════════════════════════════
def bereken_t_toetsen(df: pd.DataFrame, geselecteerde_variabelen: list[str] | None = None) -> pd.DataFrame:
    """
    Voert Welch t-toets uit voor alle continue variabelen,
    mannen vs vrouwen. Rapporteert t-statistiek, p-waarde en Cohen's d.
    """
    df2 = df.copy()
    df2['geslacht'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce')
    mannen  = df2[df2['geslacht'] == 1]
    vrouwen = df2[df2['geslacht'] == 0]

    alle_kolommen = {**LEEFSTIJL_SCORES, **RUWE_WAARDEN, **T_TOETS_EXTRA_VARIABELEN}
    rijen = []

    if geselecteerde_variabelen:
        geselecteerde_labels = set(geselecteerde_variabelen)
        alle_kolommen = {label: kolom for label, kolom in alle_kolommen.items() if label in geselecteerde_labels}

    for label, kolom in alle_kolommen.items():
        if kolom not in df.columns:
            continue
        m = pd.to_numeric(mannen[kolom],  errors='coerce').dropna()
        v = pd.to_numeric(vrouwen[kolom], errors='coerce').dropna()

        if len(m) < 10 or len(v) < 10:
            continue

        toets = stats.ttest_ind(m, v, equal_var=False)
        t_stat = cast(float, toets[0])
        p_val = cast(float, toets[1])

        # Cohen's d (effectgrootte)
        pooled_std = np.sqrt((m.std()**2 + v.std()**2) / 2)
        cohens_d   = (m.mean() - v.mean()) / pooled_std if pooled_std > 0 else 0

        # Interpretatie effectgrootte
        abs_d = abs(cohens_d)
        if abs_d < 0.2:     effect = 'Verwaarloosbaar'
        elif abs_d < 0.5:   effect = 'Klein'
        elif abs_d < 0.8:   effect = 'Matig'
        else:               effect = 'Groot'

        rijen.append({
            'Variabele':        label,
            'Gemiddelde man':   round(m.mean(), 2),
            'Gemiddelde vrouw': round(v.mean(), 2),
            'Verschil':         round(m.mean() - v.mean(), 2),
            'N man':            len(m),
            'N vrouw':          len(v),
            't-statistiek':     round(t_stat, 3),
            'p-waarde':         round(p_val, 4),
            'Significant':      'Ja ✓' if p_val < 0.05 else 'Nee',
            "Cohen's d":        round(cohens_d, 3),
            'Effectgrootte':    effect,
        })

    return (pd.DataFrame(rijen)
            .sort_values('p-waarde')
            .reset_index(drop=True))


def maak_t_toets_plot(df_toetsen: pd.DataFrame) -> go.Figure:
    """
    Forest plot van Cohen's d per variabele.
    Geeft visueel inzicht in richting en grootte van het verschil.
    """
    df = df_toetsen.copy()
    df = df.sort_values("Cohen's d")

    kleuren = [
        GENDER_COLORS['Man']   if d > 0 else
        GENDER_COLORS['Vrouw']
        for d in df["Cohen's d"]
    ]
    opaciteit = [
        1.0 if sig == 'Ja ✓' else 0.35
        for sig in df['Significant']
    ]

    fig = go.Figure(go.Bar(
        x=df["Cohen's d"],
        y=df['Variabele'],
        orientation='h',
        marker_color=kleuren,
        marker_opacity=opaciteit,
        text=df['Significant'],
        textposition='outside',
    ))

    fig.add_vline(x=0,    line_color='black', line_width=1)
    fig.add_vline(x=0.2,  line_dash='dot', line_color='gray',
                  annotation_text='Klein', annotation_position='top')
    fig.add_vline(x=-0.2, line_dash='dot', line_color='gray')
    fig.add_vline(x=0.5,  line_dash='dot', line_color='gray',
                  annotation_text='Matig', annotation_position='top')
    fig.add_vline(x=-0.5, line_dash='dot', line_color='gray')

    fig.update_layout(
        title="Cohen's d: verschil mannen vs vrouwen (blauw = mannen hoger, roze = vrouwen hoger)",
        xaxis_title="Cohen's d (effectgrootte)",
        yaxis_title='',
        height=max(400, len(df) * 25),
        margin=dict(l=180),
        showlegend=False,
    )
    return fig


def maak_t_toets_gemiddelden_plot(df_toetsen: pd.DataFrame) -> go.Figure:
    """Vergelijk de gemiddelde scores van mannen en vrouwen per variabele."""
    required = {'Variabele', 'Gemiddelde man', 'Gemiddelde vrouw'}
    if df_toetsen is None or df_toetsen.empty or not required.issubset(df_toetsen.columns):
        return go.Figure()

    df = df_toetsen.sort_values('Verschil', ascending=True).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Man',
        y=df['Variabele'],
        x=df['Gemiddelde man'],
        orientation='h',
        marker_color=GENDER_COLORS['Man'],
        customdata=df[['N man']].to_numpy(),
        hovertemplate='Man: %{x:.2f}<br>N: %{customdata[0]}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Vrouw',
        y=df['Variabele'],
        x=df['Gemiddelde vrouw'],
        orientation='h',
        marker_color=GENDER_COLORS['Vrouw'],
        customdata=df[['N vrouw']].to_numpy(),
        hovertemplate='Vrouw: %{x:.2f}<br>N: %{customdata[0]}<extra></extra>',
    ))
    fig.update_layout(
        barmode='group',
        title='Gemiddelde score per variabele: mannen vs vrouwen',
        xaxis_title='Gemiddelde score',
        yaxis_title='',
        height=max(450, len(df) * 30),
        legend_title_text='Geslacht',
        margin=dict(l=180),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. CORRELATIEMATRIX
# ══════════════════════════════════════════════════════════════════════════════
def maak_correlatiematrix(df: pd.DataFrame,
                           min_n: int = 200, lang: str = 'nl') -> go.Figure:
    """
    Pearson correlatiematrix van leefstijlscores en werkindicatoren.
    Alleen variabelen met voldoende data (min_n) worden meegenomen.
    """
    kandidaat_kolommen = {
        'Leefstijlscore':        'rec_ls_lifestyle_score',
        'BMI (waarde)':          'rec_med_bmi',
        'Heartrisk (score)':     'rec_heartrisk',
        'Framingham score':      'rec_framingham_non_invasive',
        'Stress (score)':        'rec_ls_stress_sum',
        'DASS stress':           'rec_dass_stress_score',
        'DASS angst':            'rec_dass_anxiety_score',
        'DASS depressie':        'rec_dass_depression_score',
        'Slaap PSQI':            'rec_ls_sleep_psqi_sum',
        'Veerkracht':            'rec_resilience_score',
        'Welzijn':               'rec_wellbeing_score',
        'Werkvermogen (WAI)':    'rec_asr_wai_score',
        'Werkdruk':              'rec_asr_workload_score',
        'Uitputting':            'rec_asr_exhaustion_score',
        'Burn-out score':        'rec_asr_burn_out_score',
        'Vitaliteit':            'rec_asr_vitality_score',
        'Werktevredenheid':      'rec_asr_job_satisfaction_score',
        'Leeftijd':              'rec_age_current',
        'Stappen per dag':       'rec_ls_exercise_steps_per_day',
        'Beweegminuten':         'rec_ls_exercise_physical_activity_minutes_total',
        'Fruit (stuks/dag)':     'rec_ls_nutrition_fruit_fruit_per_day',
        'Groenten (gram/dag)':   'rec_ls_vegetables_gram_per_day',
        'Vet (gram/dag)':        'rec_ls_nutrition_saturated_fat_per_day',
        'Alcohol (glazen/week)': 'rec_ls_alcohol_total_per_week',
        'Natrium (mg/dag)':      'rec_ls_nutrition_natrium_per_day',
        'Suiker (gram/dag)':     'rec_ls_nutrition_sugar_per_day',
    }

    kolommen = {}
    for label, kolom in kandidaat_kolommen.items():
        if kolom not in df.columns:
            continue
        s = pd.to_numeric(df[kolom], errors='coerce')
        if s.notna().sum() >= min_n:
            label_bold = f"<b>{tr(label, lang)}</b>"
            kolommen[label_bold] = s

    if len(kolommen) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text=tr('Onvoldoende variabelen met minimaal {min_n} deelnemers.', lang, min_n=min_n),
            showarrow=False,
            font=dict(size=14, color='gray'),
        )
        fig.update_layout(template='plotly_white', height=400)
        return fig

    df_corr = pd.DataFrame(kolommen).corr(method='pearson').dropna(how='all').dropna(how='all', axis=1)

    if df_corr.empty or df_corr.shape[0] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text=tr('Onvoldoende variatie voor correlatiematrix.', lang),
            showarrow=False,
            font=dict(size=14, color='gray'),
        )
        fig.update_layout(template='plotly_white', height=400)
        return fig

    fig = px.imshow(
        df_corr,
        color_continuous_scale='RdYlGn',
        zmin=-1, zmax=1,
        aspect='auto',
        title=tr('Pearson correlatiematrix — leefstijl- en werkindicatoren', lang),
    )
    fig.update_traces(texttemplate='%{z:.2f}', textfont_size=11)
    fig.update_layout(
        height=600,
        coloraxis_colorbar_title='Correlatie',
    )
    return fig


def maak_scatter_correlatie(df: pd.DataFrame,
                             x_label: str, x_kolom: str,
                             y_label: str, y_kolom: str,
                             geslacht_filter: str = 'beide',
                             lang: str = 'nl') -> go.Figure:
    """Scatterplot van twee variabelen met trendlijn en correlatie."""
    if x_kolom not in df.columns or y_kolom not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=tr('Gekozen variabele(n) niet gevonden in data.', lang),
            showarrow=False,
            font=dict(size=14, color='gray'),
        )
        fig.update_layout(template='plotly_white', height=450)
        return fig

    df2 = df.copy()
    df2['_x'] = pd.to_numeric(df2[x_kolom], errors='coerce')
    df2['_y'] = pd.to_numeric(df2[y_kolom], errors='coerce')
    df2 = df2.dropna(subset=['_x', '_y'])

    if len(df2) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text=tr('Onvoldoende datapunten voor deze combinatie van variabelen.', lang),
            showarrow=False,
            font=dict(size=14, color='gray'),
        )
        fig.update_layout(template='plotly_white', height=450)
        return fig

    x_std = float(df2['_x'].std(ddof=0))
    y_std = float(df2['_y'].std(ddof=0))
    if x_std > 0 and y_std > 0:
        try:
            r, p = stats.pearsonr(df2['_x'], df2['_y'])
            title_text = f'{x_label} vs {y_label} (r = {r:.3f}, p = {p:.4f})'
            has_trendline = True
        except Exception:
            title_text = f'{x_label} vs {y_label}'
            has_trendline = False
    else:
        title_text = f'{x_label} vs {y_label} (geen spreiding)'
        has_trendline = False

    if geslacht_filter == 'beide' and 'rec_user_gender' in df2.columns:
        df2['geslacht'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce').map(GENDER_LABELS)
        df2 = df2.dropna(subset=['geslacht'])
        kleur_map = {v: GENDER_COLORS[v] for v in GENDER_LABELS.values()}
        try:
            if has_trendline:
                fig = px.scatter(
                    df2, x='_x', y='_y',
                    color='geslacht', color_discrete_map=kleur_map,
                    opacity=0.5, trendline='ols',
                    labels={'_x': x_label, '_y': y_label, 'geslacht': tr('Geslacht', lang)},
                    title=title_text,
                    category_orders={'geslacht': [GENDER_LABELS.get(1, 'Man'), GENDER_LABELS.get(0, 'Vrouw')]},
                )
            else:
                fig = px.scatter(
                    df2, x='_x', y='_y',
                    color='geslacht', color_discrete_map=kleur_map,
                    opacity=0.5,
                    labels={'_x': x_label, '_y': y_label, 'geslacht': tr('Geslacht', lang)},
                    title=title_text,
                    category_orders={'geslacht': [GENDER_LABELS.get(1, 'Man'), GENDER_LABELS.get(0, 'Vrouw')]},
                )
        except Exception:
            fig = px.scatter(
                df2, x='_x', y='_y',
                color='geslacht', color_discrete_map=kleur_map,
                opacity=0.5,
                labels={'_x': x_label, '_y': y_label, 'geslacht': tr('Geslacht', lang)},
                title=title_text,
            )
        fig.update_layout(legend_title_text=tr('Geslacht', lang))
    else:
        try:
            if has_trendline:
                fig = px.scatter(
                    df2, x='_x', y='_y',
                    opacity=0.5, trendline='ols',
                    color_discrete_sequence=[HOOFD_KLEUR],
                    labels={'_x': x_label, '_y': y_label},
                    title=title_text,
                )
            else:
                fig = px.scatter(
                    df2, x='_x', y='_y',
                    opacity=0.5,
                    color_discrete_sequence=[HOOFD_KLEUR],
                    labels={'_x': x_label, '_y': y_label},
                    title=title_text,
                )
        except Exception:
            fig = px.scatter(
                df2, x='_x', y='_y',
                opacity=0.5,
                color_discrete_sequence=[HOOFD_KLEUR],
                labels={'_x': x_label, '_y': y_label},
                title=title_text,
            )

    fig.update_layout(template='plotly_white', height=500)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. INZICHTEN SAMENVATTING
# ══════════════════════════════════════════════════════════════════════════════
def genereer_inzichten(df: pd.DataFrame,
                        df_kwaliteit: pd.DataFrame,
                        df_outliers: pd.DataFrame,
                        df_toetsen: pd.DataFrame) -> list[dict]:
    """
    Genereert automatisch tekstuele inzichten op basis van de analyses.
    Geeft een lijst van dicts met 'titel', 'tekst' en 'type' (info/warning/success).
    """
    inzichten = []
    n = len(df)

    # Demografisch
    geslacht = pd.to_numeric(df['rec_user_gender'], errors='coerce')
    pct_vrouw = round(geslacht.eq(0).sum() / geslacht.notna().sum() * 100, 1)
    leeftijd  = pd.to_numeric(df['rec_age_current'], errors='coerce')
    inzichten.append({
        'titel': 'Populatie',
        'tekst': (f"Het dashboard bevat {n:,} deelnemers, waarvan {pct_vrouw}% vrouw. "
                  f"De gemiddelde leeftijd is {leeftijd.mean():.1f} jaar "
                  f"(spreiding: {leeftijd.min():.0f}-{leeftijd.max():.0f} jaar)."),
        'type': 'info',
    })

    # Datakwaliteit
    hoog_missing = df_kwaliteit[df_kwaliteit['Missend (%)'] > 50]
    if len(hoog_missing) > 0:
        namen = ', '.join(hoog_missing['Variabele'].tolist()[:5])
        inzichten.append({
            'titel': 'Datakwaliteit: hoge missende waarden',
            'tekst': (f"{len(hoog_missing)} variabelen hebben meer dan 50% missende waarden: "
                      f"{namen}. Dit beperkt de betrouwbaarheid van analyses op deze variabelen."),
            'type': 'warning',
        })

    # Outliers
    hoog_outlier = df_outliers[df_outliers['Outliers (%)'] > 5].head(3)
    for _, row in hoog_outlier.iterrows():
        inzichten.append({
            'titel': f"Outliers: {row['Variabele']}",
            'tekst': (f"{row['Outliers (%)']:.1f}% van de waarden voor {row['Variabele']} "
                      f"valt buiten de IQR grenzen ({row['IQR grens laag']:.1f} - "
                      f"{row['IQR grens hoog']:.1f}). "
                      f"Maximum gemeten waarde: {row['Max']:.1f}. "
                      f"Aanbeveling: controleer of dit meetfouten zijn of extreme maar valide waarden."),
            'type': 'warning',
        })

    # T-toets significante verschillen
    sig = df_toetsen[df_toetsen['Significant'] == 'Ja ✓']
    groot_effect = sig[sig['Effectgrootte'].isin(['Matig', 'Groot'])]
    for _, row in groot_effect.head(5).iterrows():
        richting = 'mannen' if row['Verschil'] > 0 else 'vrouwen'
        inzichten.append({
            'titel': f"Geslachtsverschil: {row['Variabele']}",
            'tekst': (f"Er is een statistisch significant verschil tussen mannen en vrouwen "
                      f"voor {row['Variabele']} (p = {row['p-waarde']:.4f}, "
                      f"Cohen's d = {row["Cohen's d"]:.2f}, effectgrootte: {row['Effectgrootte']}). "
                      f"{richting.capitalize()} scoren gemiddeld hoger "
                      f"({row['Gemiddelde man']:.1f} vs {row['Gemiddelde vrouw']:.1f})."),
            'type': 'info',
        })

    return inzichten

def bereken_risico_migratie(df_long: pd.DataFrame, slug: str, participant_ids: ParticipantIds = None) -> pd.DataFrame:
    """
    Berekent de verschuiving tussen risicocategorieën tussen de eerste en laatste meting.
    Nodig voor de 'Transition Matrix' (Sankey).
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()
        
    df_factor = df_long[df_long['slug'] == slug].copy()
    if participant_ids is not None:
        pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
        df_factor = df_factor[pd.to_numeric(df_factor['participant_id'], errors='coerce').isin(pids)]

    df_factor = df_factor.sort_values(['participant_id', 'completion_created_at'])
    
    if df_factor.empty:
        return pd.DataFrame()
        
    # Alleen diegenen met minstens 2 metingen
    ids = df_factor.groupby('participant_id').size()
    multi_measure_ids = ids[ids >= 2].index
    
    # Pak eerste en laatste meting per participant
    first = df_factor.groupby('participant_id').first().loc[multi_measure_ids]
    last = df_factor.groupby('participant_id').last().loc[multi_measure_ids]
    
    migratie = pd.DataFrame({
        'Van': pd.to_numeric(first['score_category_value'], errors='coerce'),
        'Na': pd.to_numeric(last['score_category_value'], errors='coerce')
    }).dropna()
    
    if migratie.empty:
        return pd.DataFrame()
        
    return migratie.groupby(['Van', 'Na'], observed=False).size().reset_index(name='Aantal')
# ──────────────────────────────────────────────────────────────────────────────
# FIX #1: ONTBREKENDE FUNCTIE - bereken_duplicaat_statistieken
# ──────────────────────────────────────────────────────────────────────────────

def bereken_duplicaat_statistieken(base_path: Path | str) -> pd.DataFrame:
    """
    Berekent duplicaatstatistieken voor alle Parquet bestanden in een directory.
    
    Parameters:
    -----------
    base_path : Path | str
        Directory met parquet bestanden of een DB-url/string.
    
    Returns:
    --------
    pd.DataFrame
        Statistieken per tabel
    """
    if isinstance(base_path, str) and base_path.startswith(("postgresql://", "postgresql+")):
        from sqlalchemy import create_engine, inspect, text

        engine = create_engine(base_path)
        inspector = inspect(engine)
        rijen = []

        for table_name in sorted(inspector.get_table_names(schema='public')):
            quoted = '"' + table_name.replace('"', '""') + '"'
            try:
                total_rows = int(pd.read_sql(text(f'SELECT COUNT(*) AS n FROM public.{quoted}'), engine)['n'].iloc[0])
                columns = [col['name'] for col in inspector.get_columns(table_name, schema='public')]

                id_duplicates = 0
                if 'id' in columns and total_rows:
                    id_duplicates = int(pd.read_sql(
                        text(f'SELECT COUNT(*) - COUNT(DISTINCT id) AS n FROM public.{quoted}'),
                        engine,
                    )['n'].iloc[0])

                duplicate_rows = None
                unique_rows = None
                pct_duplicates = None
                if total_rows <= 200_000:
                    unique_rows = int(pd.read_sql(
                        text(f'SELECT COUNT(*) AS n FROM (SELECT DISTINCT to_jsonb(t) AS row_json FROM public.{quoted} t) d'),
                        engine,
                    )['n'].iloc[0])
                    duplicate_rows = total_rows - unique_rows
                    pct_duplicates = round((duplicate_rows / total_rows * 100), 2) if total_rows else 0.0

                rijen.append({
                    'Tabelnaam': table_name,
                    'Totaal rijen': total_rows,
                    'Duplicaat rijen': duplicate_rows,
                    'Unieke rijen': unique_rows,
                    'Duplicaten (%)': pct_duplicates,
                    'ID duplicaten': id_duplicates,
                })
            except Exception as e:
                logger.warning(f"Error processing database table {table_name}: {e}")
                rijen.append({
                    'Tabelnaam': table_name,
                    'Totaal rijen': None,
                    'Duplicaat rijen': None,
                    'Unieke rijen': None,
                    'Duplicaten (%)': None,
                    'ID duplicaten': None,
                    'Fout': str(e),
                })

        df_out = pd.DataFrame(rijen)
        if df_out.empty:
            return pd.DataFrame(columns=[
                'Tabelnaam', 'Totaal rijen', 'Duplicaat rijen', 'Unieke rijen',
                'Duplicaten (%)', 'ID duplicaten', 'Fout'
            ])
        sort_col = 'ID duplicaten' if df_out['Duplicaten (%)'].isna().all() else 'Duplicaten (%)'
        return df_out.sort_values(sort_col, ascending=False, na_position='last')

    if isinstance(base_path, str):
        from config import CODE_DIR
        base_path = CODE_DIR
    else:
        base_path = Path(base_path)

    rijen = []
    
    for file_path in sorted(base_path.glob("*.parquet")):
        tabelnaam = file_path.stem
        
        try:
            df = pd.read_parquet(file_path)
            total_rows = len(df)
            
            if total_rows == 0:
                rijen.append({
                    'Tabelnaam': tabelnaam,
                    'Totaal rijen': 0,
                    'Duplicaat rijen': 0,
                    'Unieke rijen': 0,
                    'Duplicaten (%)': 0.0,
                    'ID duplicaten': 0,
                })
                continue
            
            # Volledige rij duplicaten
            duplicate_rows = df.astype(str).duplicated().sum()
            unique_rows = total_rows - duplicate_rows
            pct_duplicates = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0
            
            # ID duplicaten
            id_duplicates = 0
            if 'id' in df.columns:
                id_duplicates = df['id'].duplicated().sum()
            
            rijen.append({
                'Tabelnaam': tabelnaam,
                'Totaal rijen': total_rows,
                'Duplicaat rijen': duplicate_rows,
                'Unieke rijen': unique_rows,
                'Duplicaten (%)': round(pct_duplicates, 2),
                'ID duplicaten': id_duplicates,
            })
            
            logger.info(f"✓ {tabelnaam}: {total_rows} rows, {pct_duplicates:.1f}% duplicates")
            
        except Exception as e:
            logger.warning(f"Error processing {tabelnaam}: {e}")
            rijen.append({
                'Tabelnaam': tabelnaam,
                'Fout': str(e),
            })
    
    df_out = pd.DataFrame(rijen)
    
    if df_out.empty:
        return pd.DataFrame(columns=[
            'Tabelnaam', 'Totaal rijen', 'Duplicaat rijen', 'Unieke rijen',
            'Duplicaten (%)', 'ID duplicaten'
        ])
    
    return df_out.sort_values('Duplicaten (%)', ascending=False, na_position='last')


# ──────────────────────────────────────────────────────────────────────────────
# FIX #2: HELPER FUNCTIONS - Consolideer repetitieve code
# ──────────────────────────────────────────────────────────────────────────────

def _get_numeric_series(series: pd.Series) -> pd.Series:
    """Helper: Convert to numeric and drop NA."""
    return pd.to_numeric(series, errors='coerce').dropna()


def _calculate_effect_interpretation(cohens_d: float) -> str:
    """Helper: Interpret Cohen's d magnitude."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return 'Verwaarloosbaar'
    elif abs_d < 0.5:
        return 'Klein'
    elif abs_d < 0.8:
        return 'Matig'
    else:
        return 'Groot'


# ──────────────────────────────────────────────────────────────────────────────
# FIX #3: VERBETERDE bereken_outliers met logging
# ──────────────────────────────────────────────────────────────────────────────

def bereken_outliers_improved(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berekent outliers met beter logging en defensive checks.
    Gebruik dit in plaats van de huidge bereken_outliers().
    """
    from analyses import RUWE_WAARDEN, OUTLIER_GRENZEN
    
    rijen = []
    
    for label, kolom in RUWE_WAARDEN.items():
        if kolom not in df.columns:
            logger.debug(f"Column {kolom} not found")
            continue
        
        s = _get_numeric_series(df[kolom])
        
        if len(s) < 10:
            logger.debug(f"Too few values for {label}: {len(s)}")
            continue
        
        # IQR berekening
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        
        grens_laag = q1 - 1.5 * iqr
        grens_hoog = q3 + 1.5 * iqr
        
        n_outlier_laag = (s < grens_laag).sum()
        n_outlier_hoog = (s > grens_hoog).sum()
        n_outlier = n_outlier_laag + n_outlier_hoog
        
        # Domein check
        domein_grens = OUTLIER_GRENZEN.get(kolom)
        n_domein = int((s > domein_grens).sum()) if domein_grens else None
        
        rijen.append({
            'Variabele': label,
            'N geldig': len(s),
            'Gemiddelde': round(s.mean(), 1),
            'Mediaan': round(s.median(), 1),
            'Std': round(s.std(), 1),
            'Min': round(s.min(), 1),
            'Max': round(s.max(), 1),
            'IQR grens laag': round(grens_laag, 1),
            'IQR grens hoog': round(grens_hoog, 1),
            'Outliers laag (n)': int(n_outlier_laag),
            'Outliers hoog (n)': int(n_outlier_hoog),
            'Outliers totaal (n)': int(n_outlier),
            'Outliers (%)': round(n_outlier / len(s) * 100, 1),
            'Domeingrens': domein_grens,
            'Boven domeingrens (n)': n_domein,
        })
    
    return pd.DataFrame(rijen).sort_values('Outliers (%)', ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# FIX #4: VERBETERDE bereken_t_toetsen met defensive checks
# ──────────────────────────────────────────────────────────────────────────────

def bereken_t_toetsen_improved(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verbeterde t-toets met defensive checks en logging.
    Gebruik dit in plaats van bereken_t_toetsen().
    """
    from scipy import stats
    from analyses import LEEFSTIJL_SCORES, RUWE_WAARDEN
    from kleuren import GENDER_LABELS
    
    df2 = df.copy()
    df2['geslacht'] = pd.to_numeric(df2['rec_user_gender'], errors='coerce')
    
    mannen = df2[df2['geslacht'] == 1]
    vrouwen = df2[df2['geslacht'] == 0]
    
    logger.info(f"T-tests: comparing {len(mannen)} men vs {len(vrouwen)} women")
    
    alle_kolommen = {**LEEFSTIJL_SCORES, **RUWE_WAARDEN}
    rijen = []
    
    for label, kolom in alle_kolommen.items():
        if kolom not in df.columns:
            continue
        
        m = _get_numeric_series(mannen[kolom])
        v = _get_numeric_series(vrouwen[kolom])
        
        if len(m) < 10 or len(v) < 10:
            logger.debug(f"Too few samples for {label}: men={len(m)}, women={len(v)}")
            continue
        
        try:
            toets = stats.ttest_ind(m, v, equal_var=False)
            t_stat = cast(float, toets[0])
            p_val = cast(float, toets[1])
            
            # Cohen's d
            pooled_std = np.sqrt((m.std()**2 + v.std()**2) / 2)
            cohens_d = (m.mean() - v.mean()) / pooled_std if pooled_std > 0 else 0
            
            effect = _calculate_effect_interpretation(cohens_d)
            
            rijen.append({
                'Variabele': label,
                'Gemiddelde man': round(m.mean(), 2),
                'Gemiddelde vrouw': round(v.mean(), 2),
                'Verschil': round(m.mean() - v.mean(), 2),
                'N man': len(m),
                'N vrouw': len(v),
                't-statistiek': round(t_stat, 3),
                'p-waarde': round(p_val, 4),
                'Significant': 'Ja ✓' if p_val < 0.05 else 'Nee',
                "Cohen's d": round(cohens_d, 3),
                'Effectgrootte': effect,
            })
        
        except Exception as e:
            logger.warning(f"T-test failed for {label}: {e}")
            continue
    
    return pd.DataFrame(rijen).sort_values('p-waarde').reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# FIX #5: NEW FUNCTION - Vragenlijst herhalingen analyseren
# ──────────────────────────────────────────────────────────────────────────────

def analyse_vragenlijst_herhalingen(base_pad: Path, db_url: str | None = None,
                                    participant_ids: ParticipantIds = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyseert wie dezelfde vragenlijst meerdere keren invullen.
    Toont gehashte participant IDs en echte vragenlijstnamen uit database.
    
    Parameters:
    -----------
    base_pad : Path
        Directory met data
    db_url : str, optional
        Database URL voor vragenlijstnamen
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (herhalingen, intervallen_stats)
    """
    import hashlib
    
    try:
        try:
            df_comp = load_completions(DB_URL)
            if participant_ids is not None:
                pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
                df_comp = df_comp[pd.to_numeric(df_comp['participant_id'], errors='coerce').isin(pids)].copy()
            df_comp['created_at'] = pd.to_datetime(df_comp['created_at'], errors='coerce')
        except Exception as e:
            logger.error(f"Failed to load completions: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Haal vragenlijstnamen uit database of gebruik fallback
        vraag_namen = {}
        effective_db_url = db_url or DB_URL
        if effective_db_url:
            try:
                from sqlalchemy import create_engine
                engine = create_engine(effective_db_url)
                df_questionnaires = pd.read_sql("SELECT id, title FROM questionnaires", engine)
                vraag_namen = dict(zip(df_questionnaires['id'], df_questionnaires['title']))
            except Exception as e:
                logger.warning(f"Kon vragenlijstnamen niet uit database laden: {e}. Gebruik fallback.")
        
        # Fallback hardcoded namen als database query faalt
        if not vraag_namen:
            vraag_namen = {
                1: 'Smart Health Test', 2: 'Smart Work Test', 3: 'Resilience',
                4: 'Well-being', 5: 'Positive Health', 6: 'Self-efficacy',
                7: 'Negative emotions', 8: 'Smartphone and stress'
            }
        
        herhalingen = (
            df_comp.groupby(['participant_id', 'questionnaire_id'])
            .size()
            .reset_index(name='keren_ingevuld')
        )
        herhalingen = herhalingen[herhalingen['keren_ingevuld'] > 1].copy()
        
        # Hash participant IDs voor privacy
        herhalingen['participant_id_hashed'] = herhalingen['participant_id'].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12]  # Eerste 12 chars van hash
        )
        
        if herhalingen.empty:
            logger.info("No repeated questionnaire completions found")
            return herhalingen, pd.DataFrame()
        
        # Voeg intervallen toe
        intervals = []
        for _, row in herhalingen.iterrows():
            participant_id = row['participant_id']
            participant_id_hashed = row['participant_id_hashed']
            questionnaire_id = row['questionnaire_id']
            
            dates = df_comp[
                (df_comp['participant_id'] == participant_id) &
                (df_comp['questionnaire_id'] == questionnaire_id)
            ]['created_at'].sort_values()
            
            if len(dates) > 1:
                first_date = dates.iloc[0]
                last_date = dates.iloc[-1]
                dagen_total = (last_date - first_date).days
                
                # Intervallen tussen invullingen
                intervallen = []
                for i in range(1, len(dates)):
                    intervallen.append((dates.iloc[i] - dates.iloc[i-1]).days)
                
                intervals.append({
                    'participant_id_hashed': participant_id_hashed,
                    'Vragenlijst': vraag_namen.get(int(questionnaire_id), f"Vragenlijst {questionnaire_id}"),
                    'questionnaire_id': questionnaire_id,
                    'keren_ingevuld': row['keren_ingevuld'],
                    'eerste_datum': first_date,
                    'laatste_datum': last_date,
                    'dagen_totaal': dagen_total,
                    'gem_interval_dagen': round(np.mean(intervallen), 1) if intervallen else None,
                    'min_interval_dagen': min(intervallen) if intervallen else None,
                    'max_interval_dagen': max(intervallen) if intervallen else None,
                })
        
        intervallen_stats = pd.DataFrame(intervals)
        
        logger.info(f"Found {len(herhalingen)} repeat completions")
        
        return herhalingen, intervallen_stats
    
    except Exception as e:
        logger.error(f"Error in analyse_vragenlijst_herhalingen: {e}")
        return pd.DataFrame(), pd.DataFrame()


def analyse_vragenlijst_dropoff(base_pad: Path,
                                participant_ids: ParticipantIds = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Benadert afhaken in vragenlijsten via het aantal beantwoorde score-items per completion.
    Omdat er lokaal geen expliciete completion-status beschikbaar is, gebruiken we het
    aantal unieke slugs als proxy voor hoe ver iemand in een vragenlijst kwam.
    Fallback: retourneert lege data als parquet files niet beschikbaar zijn.
    """
    try:
        try:
            df_scores = pd.read_parquet(base_pad / 'completion_scores.parquet')
            logger.info(f"analyse_vragenlijst_dropoff: df_scores (completion_scores) rows: {len(df_scores)}")
            df_comp = pd.read_parquet(base_pad / 'completions.parquet')
            logger.info(f"analyse_vragenlijst_dropoff: df_comp (completions) rows: {len(df_comp)}")
            df_questions = pd.read_parquet(base_pad / 'questionnaires.parquet')
        except Exception as e:
            logger.error(f"Failed to load questionnaire dropoff data from parquet: {e}")
            # Try loading from database instead
            try:
                from data_ingestion import load_completions, load_questionnaires, load_table_from_database
                df_comp = load_completions(DB_URL)
                df_scores_raw = load_table_from_database('factor_score_histories', DB_URL)
                df_factors = load_table_from_database('questionnaire_factors', DB_URL)
                df_questions = load_questionnaires(DB_URL)

                if not df_scores_raw.empty and not df_factors.empty:
                    # Koppel slugs aan de histories zodat de item-telling per vragenlijst werkt
                    df_scores = df_scores_raw.merge(
                        df_factors[['id', 'slug']], 
                        left_on='questionnaire_factor_id', 
                        right_on='id', 
                        how='left'
                    )
                else:
                    df_scores = df_scores_raw

                if df_comp.empty or df_scores.empty:
                    logger.info("Questionnaire dropoff data not available in database")
                    return pd.DataFrame(), pd.DataFrame(), {}
            except Exception as e2:
                logger.error(f"Also failed to load from database: {e2}")
                return pd.DataFrame(), pd.DataFrame(), {}

        if df_scores.empty or df_comp.empty:
            return pd.DataFrame(), pd.DataFrame(), {}
        if participant_ids is not None:
            pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
            df_comp = df_comp[pd.to_numeric(df_comp['participant_id'], errors='coerce').isin(pids)].copy()
        logger.info(f"analyse_vragenlijst_dropoff: Dataframes loaded, proceeding with analysis.")

        df_scores['completion_id'] = pd.to_numeric(df_scores['completion_id'], errors='coerce')
        df_comp['id'] = pd.to_numeric(df_comp['id'], errors='coerce')
        df_comp['questionnaire_id'] = pd.to_numeric(df_comp['questionnaire_id'], errors='coerce')

        beantwoorde_items = (
            df_scores.dropna(subset=['completion_id', 'slug'])
            .groupby('completion_id')['slug']
            .nunique()
            .reset_index(name='beantwoorde_items')
        )

        df = df_comp[['id', 'participant_id', 'questionnaire_id', 'created_at']].merge(
            beantwoorde_items,
            left_on='id',
            right_on='completion_id',
            how='left',
        )
        logger.info(f"analyse_vragenlijst_dropoff: df after merge with beantwoorde_items rows: {len(df)}")
        df['beantwoorde_items'] = pd.to_numeric(df['beantwoorde_items'], errors='coerce').fillna(0)

        vraag_namen = {}
        if not df_questions.empty and 'id' in df_questions.columns:
            # Ensure 'internal_name' and 'slug' are treated as strings for mapping
            df_questions['internal_name'] = df_questions['internal_name'].astype(str)
            df_questions['slug'] = df_questions['slug'].astype(str)
            for _, row in df_questions[['id', 'slug', 'internal_name']].drop_duplicates('id').iterrows():
                label = row.get('internal_name') or row.get('slug') or f"Vragenlijst {row['id']}"
                vraag_namen[pd.to_numeric(row['id'], errors='coerce')] = str(label)

        # Gebruik de 95e percentiel-score als benadering voor een "volledige" vragenlijst.
        verwacht = (
            df.groupby('questionnaire_id')['beantwoorde_items']
            .quantile(0.95)
            .round()
            .clip(lower=1)
            .reset_index(name='geschat_totaal_items')
        )
        logger.info(f"analyse_vragenlijst_dropoff: verwacht (estimated total items) rows: {len(verwacht)}")

        df = df.merge(verwacht, on='questionnaire_id', how='left')
        df['geschat_totaal_items'] = pd.to_numeric(df['geschat_totaal_items'], errors='coerce').fillna(1)
        df['completion_pct'] = (df['beantwoorde_items'] / df['geschat_totaal_items']).clip(0, 1)
        df['half_afgehaakt'] = df['completion_pct'] < 0.5
        df['bijna_af'] = df['completion_pct'] >= 0.8
        df['vragenlijst'] = df['questionnaire_id'].map(vraag_namen).fillna(df['questionnaire_id'].map(lambda x: f"Vragenlijst {int(x)}" if pd.notna(x) else "Onbekend"))

        per_vragenlijst = (
            df.groupby(['questionnaire_id', 'vragenlijst'], dropna=False)
            .agg(
                completions=('id', 'count'),
                deelnemers=('participant_id', 'nunique'),
                geschat_totaal_items=('geschat_totaal_items', 'max'),
                gem_beantwoorde_items=('beantwoorde_items', 'mean'),
                median_completion_pct=('completion_pct', 'median'),
                dropoff_half_pct=('half_afgehaakt', 'mean'),
                bijna_af_pct=('bijna_af', 'mean'),
            )
            .reset_index()
        )

        logger.info(f"analyse_vragenlijst_dropoff: per_vragenlijst rows: {len(per_vragenlijst)}")
        per_vragenlijst['gem_beantwoorde_items'] = per_vragenlijst['gem_beantwoorde_items'].round(1)
        per_vragenlijst['median_completion_pct'] = (per_vragenlijst['median_completion_pct'] * 100).round(1)
        per_vragenlijst['dropoff_half_pct'] = (per_vragenlijst['dropoff_half_pct'] * 100).round(1)
        per_vragenlijst['bijna_af_pct'] = (per_vragenlijst['bijna_af_pct'] * 100).round(1)
        per_vragenlijst = per_vragenlijst.sort_values(['dropoff_half_pct', 'completions'], ascending=[False, False])

        samenvatting = {
            'Totaal completions': int(len(df)),
            'Median completion (%)': round(float(df['completion_pct'].median() * 100), 1),
            'Afhakers <50% (%)': round(float(df['half_afgehaakt'].mean() * 100), 1),
            'Bijna af >=80% (%)': round(float(df['bijna_af'].mean() * 100), 1),
            'Correlatie lengte vs afhaken': round(
                float(per_vragenlijst['geschat_totaal_items'].corr(per_vragenlijst['dropoff_half_pct'])),
                3,
            ) if len(per_vragenlijst) >= 2 else None,
        }

        return per_vragenlijst, df, samenvatting
    
    except Exception as e:
        logger.error(f"Error in analyse_vragenlijst_dropoff: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}


# ──────────────────────────────────────────────────────────────────────────────
# FIX #6: NEW FUNCTION - Voert duplicate detector uit
# ──────────────────────────────────────────────────────────────────────────────

def detect_profile_duplicates(df: pd.DataFrame, similarity_threshold: float = 0.9) -> pd.DataFrame:
    """
    Detecteert potentieel duplicaat profielen op basis van vergelijkbare waarden.
    
    Parameters:
    -----------
    df : pd.DataFrame
        User scores dataframe
    similarity_threshold : float
        Gelijkenis drempel (0-1)
    
    Returns:
    --------
    pd.DataFrame
        Potentiële duplicaten
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Selecteer numerieke kolommen
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 3:
        logger.warning("Too few numeric columns for duplicate detection")
        return pd.DataFrame()
    
    # Normaliseer
    df_norm = df[numeric_cols].fillna(0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_norm)
    
    # Bereken similariteit
    similarity = cosine_similarity(df_scaled)
    
    # Vind hoge similariteiten
    duplicates = []
    for i in range(len(similarity)):
        for j in range(i+1, len(similarity)):
            if similarity[i, j] > similarity_threshold:
                duplicates.append({
                    'user_id_1': df.iloc[i]['user_id'] if 'user_id' in df.columns else i,
                    'user_id_2': df.iloc[j]['user_id'] if 'user_id' in df.columns else j,
                    'similarity': round(similarity[i, j], 3),
                })
    
    return pd.DataFrame(duplicates).sort_values('similarity', ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# INVULFREQUENTIE VS SCORES & ACCOUNT DATAFLOW
# ══════════════════════════════════════════════════════════════════════════════

INVULFREQUENTIE_SCORE_KOLOMMEN = {
    'Leefstijlscore': 'rec_ls_lifestyle_score',
    'BMI': 'rec_med_bmi',
    'Heartrisk': 'rec_heartrisk',
    'Stress': 'rec_ls_stress_sum',
    'Welzijn': 'rec_wellbeing_score',
    'Veerkracht': 'rec_resilience_score',
}


def analyse_invulfrequentie_vs_scores(
    base_pad: Path,
    db_url: str | None = None,
    score_kolom: str = 'rec_ls_lifestyle_score',
    participant_ids: ParticipantIds = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Vergelijk gemiddelde scores per aantal ingevulde vragenlijsten (1x, 2x, 3x, ...).

    Gebruikt het aantal unieke vragenlijsten per deelnemer (niet het totaal aantal
    completion-events), consistent met de overige vragenlijstgedrag-analyses.
    """
    from data_ingestion import load_completions, load_my_clic_participants_expanded

    try:
        effective_db_url = db_url or DB_URL
        if not effective_db_url:
            return pd.DataFrame(), pd.DataFrame(), {}
        df_comp = load_completions(effective_db_url)
        df_scores = load_my_clic_participants_expanded(effective_db_url)

        if participant_ids is not None:
            pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
            if not df_comp.empty and 'participant_id' in df_comp.columns:
                df_comp = df_comp[pd.to_numeric(df_comp['participant_id'], errors='coerce').isin(pids)]
            if not df_scores.empty and 'participant_id' in df_scores.columns:
                df_scores = df_scores[pd.to_numeric(df_scores['participant_id'], errors='coerce').isin(pids)]

        if df_comp.empty or df_scores.empty:
            return pd.DataFrame(), pd.DataFrame(), {}


        known_questionnaire_ids = {1, 2, 3, 4, 5, 6, 7, 8}
        df_comp = df_comp.copy()
        df_comp['questionnaire_id'] = pd.to_numeric(df_comp['questionnaire_id'], errors='coerce')
        df_comp = df_comp[df_comp['questionnaire_id'].isin(known_questionnaire_ids)]

        n_invullingen = (
            df_comp.groupby('participant_id')['questionnaire_id']
            .nunique()
            .reset_index(name='n_vragenlijsten')
        )
        n_invullingen['participant_id'] = pd.to_numeric(n_invullingen['participant_id'], errors='coerce')

        score_cols = [c for c in INVULFREQUENTIE_SCORE_KOLOMMEN.values() if c in df_scores.columns]
        if score_kolom not in score_cols and score_kolom in df_scores.columns:
            score_cols = [score_kolom] + score_cols
        elif score_kolom in score_cols:
            score_cols = [score_kolom] + [c for c in score_cols if c != score_kolom]
        if not score_cols:
            return pd.DataFrame(), pd.DataFrame(), {}

        df_scores = df_scores.copy()
        df_scores['participant_id'] = pd.to_numeric(df_scores['participant_id'], errors='coerce')
        df_plot = n_invullingen.merge(
            df_scores[['participant_id'] + score_cols],
            on='participant_id',
            how='inner',
        )
        if df_plot.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        for col in score_cols:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

        df_plot = df_plot.dropna(subset=[score_kolom])
        if df_plot.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        df_plot['n_categorie'] = df_plot['n_vragenlijsten'].apply(
            lambda n: f'{int(n)}x' if n <= 5 else '6+x'
        )
        volgorde = ['1x', '2x', '3x', '4x', '5x', '6+x']
        df_plot['n_categorie'] = pd.Categorical(df_plot['n_categorie'], categories=volgorde, ordered=True)

        agg_rows = []
        for cat in volgorde:
            subset = df_plot[df_plot['n_categorie'] == cat]
            if subset.empty:
                continue
            row = {
                'Invulfrequentie': cat,
                'n_deelnemers': int(subset['participant_id'].nunique()),
            }
            for col in score_cols:
                row[f'gem_{col}'] = round(subset[col].mean(), 2)
                row[f'std_{col}'] = round(subset[col].std(), 2) if len(subset) > 1 else 0.0
            agg_rows.append(row)

        samenvatting = pd.DataFrame(agg_rows)
        if samenvatting.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        meta = {
            'score_kolom': score_kolom,
            'totaal_deelnemers': int(df_plot['participant_id'].nunique()),
            'score_cols': score_cols,
        }
        if '1x' in samenvatting['Invulfrequentie'].values and '6+x' in samenvatting['Invulfrequentie'].values:
            gem_1x = samenvatting.loc[samenvatting['Invulfrequentie'] == '1x', f'gem_{score_kolom}'].iloc[0]
            gem_6p = samenvatting.loc[samenvatting['Invulfrequentie'] == '6+x', f'gem_{score_kolom}'].iloc[0]
            meta['verschil_1x_vs_6plus'] = round(gem_6p - gem_1x, 2)

        return df_plot, samenvatting, meta
    except Exception as e:
        logger.error(f"Error in analyse_invulfrequentie_vs_scores: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}


def haal_vragenlijsten_overzicht(db_url: str | None = None) -> pd.DataFrame:
    """Laad vragenlijst-id's met namen en aantal completions."""
    from sqlalchemy import create_engine, text
    from data_ingestion import load_completions

    effective_db_url = db_url or __import__('config').DB_URL
    if not effective_db_url:
        return pd.DataFrame(columns=['questionnaire_id', 'naam', 'n_completions', 'n_herhaalinvullers'])

    try:
        engine = create_engine(effective_db_url)
        df_q = pd.read_sql(
            text("SELECT id AS questionnaire_id, internal_name AS naam FROM questionnaires ORDER BY id"),
            engine,
        )
        df_comp = load_completions(effective_db_url)
        if df_comp.empty or df_q.empty:
            return pd.DataFrame(columns=['questionnaire_id', 'naam', 'n_completions', 'n_herhaalinvullers'])

        df_comp['questionnaire_id'] = pd.to_numeric(df_comp['questionnaire_id'], errors='coerce')
        stats = df_comp.groupby('questionnaire_id').agg(
            n_completions=('id', 'count'),
            n_deelnemers=('participant_id', 'nunique'),
        ).reset_index()
        herhaal = (
            df_comp.groupby(['participant_id', 'questionnaire_id'])
            .size()
            .reset_index(name='n')
        )
        herhaal_stats = (
            herhaal[herhaal['n'] > 1]
            .groupby('questionnaire_id')
            .agg(n_herhaalinvullers=('participant_id', 'nunique'))
            .reset_index()
        )
        out = df_q.merge(stats, on='questionnaire_id', how='left').merge(
            herhaal_stats, on='questionnaire_id', how='left'
        )
        out['n_herhaalinvullers'] = out['n_herhaalinvullers'].fillna(0).astype(int)
        return out.fillna(0)
    except Exception as e:
        logger.error(f"Error in haal_vragenlijsten_overzicht: {e}")
        return pd.DataFrame(columns=['questionnaire_id', 'naam', 'n_completions', 'n_herhaalinvullers'])


def haal_scores_voor_vragenlijst(db_url: str, questionnaire_id: int) -> list[str]:
    """Geef beschikbare rec_-scores voor een specifieke vragenlijst."""
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(db_url)
        slugs = pd.read_sql(
            text("""
                SELECT cs.slug, COUNT(*) AS n
                FROM completion_scores cs
                JOIN completions c ON c.id = cs.completion_id
                WHERE c.questionnaire_id = :qid
                GROUP BY cs.slug
                ORDER BY n DESC
            """),
            engine,
            params={'qid': int(questionnaire_id)},
        )
        if slugs.empty:
            return []
        mask = (
            slugs['slug'].astype(str).str.startswith('rec_')
            & ~slugs['slug'].astype(str).str.endswith('_cat')
            & ~slugs['slug'].astype(str).str.startswith('answers_')
        )
        return slugs.loc[mask, 'slug'].tolist()
    except Exception as e:
        logger.error(f"Error in haal_scores_voor_vragenlijst: {e}")
        return []


def analyse_herhaalde_vragenlijst_scoreverandering(
    base_pad: Path,
    db_url: str | None = None,
    questionnaire_id: int = 2,
    score_slug: str = 'rec_ls_lifestyle_score',
    participant_ids: list | set | pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Analyseer scoreverandering bij herhaald invullen van dezelfde vragenlijst.

    Per deelnemer worden completions chronologisch genummerd (1e, 2e, 3e, ...).
    Per totaal aantal invullingen (1x, 2x, ...) wordt de gemiddelde verandering
    van eerste naar laatste score berekend. Daarnaast een traject per invulmoment.
    """
    from sqlalchemy import create_engine, text

    effective_db_url = db_url or __import__('config').DB_URL
    if not effective_db_url:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    try:
        engine = create_engine(effective_db_url)
        df = pd.read_sql(
            text("""
                SELECT
                    c.participant_id,
                    c.id AS completion_id,
                    c.created_at AS completion_at,
                    cs.slug,
                    cs.value AS score
                FROM completions c
                JOIN completion_scores cs ON cs.completion_id = c.id
                WHERE c.questionnaire_id = :qid
                  AND cs.slug = :slug
            """),
            engine,
            params={'qid': int(questionnaire_id), 'slug': score_slug},
        )
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        if participant_ids is not None:
            pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
            df = df[pd.to_numeric(df['participant_id'], errors='coerce').isin(pids)]
            if df.empty:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


        df['participant_id'] = pd.to_numeric(df['participant_id'], errors='coerce')
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df['completion_at'] = pd.to_datetime(df['completion_at'], errors='coerce')
        df = df.dropna(subset=['participant_id', 'score', 'completion_at'])

        # Eén score per completion (gemiddelde bij duplicaten)
        df = (
            df.groupby(['participant_id', 'completion_id', 'completion_at'], as_index=False)
            .agg(score=('score', 'mean'))
        )

        timeline_rows = []
        change_rows = []
        for participant_id, grp in df.groupby('participant_id'):
            grp = grp.sort_values('completion_at').reset_index(drop=True)
            n_totaal = len(grp)
            for i, (_, row) in enumerate(grp.iterrows(), start=1):
                timeline_rows.append({
                    'participant_id': participant_id,
                    'invul_nr': i,
                    'n_totaal': n_totaal,
                    'score': row['score'],
                    'completion_at': row['completion_at'],
                })
            eerste = grp.iloc[0]['score']
            laatste = grp.iloc[-1]['score']
            change_rows.append({
                'participant_id': participant_id,
                'n_totaal': n_totaal,
                'eerste_score': eerste,
                'laatste_score': laatste,
                'verandering': laatste - eerste,
            })

        df_timeline = pd.DataFrame(timeline_rows)
        df_change = pd.DataFrame(change_rows)
        if df_change.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        df_change['n_categorie'] = df_change['n_totaal'].apply(
            lambda n: f'{int(n)}x' if n <= 5 else '6+x'
        )
        volgorde = ['1x', '2x', '3x', '4x', '5x', '6+x']
        df_change['n_categorie'] = pd.Categorical(df_change['n_categorie'], categories=volgorde, ordered=True)

        samenvatting_rows = []
        for cat in volgorde:
            subset = df_change[df_change['n_categorie'] == cat]
            if subset.empty:
                continue
            samenvatting_rows.append({
                'Invulfrequentie': cat,
                'n_deelnemers': int(subset['participant_id'].nunique()),
                'gem_eerste_score': round(subset['eerste_score'].mean(), 2),
                'gem_laatste_score': round(subset['laatste_score'].mean(), 2),
                'gem_verandering': round(subset['verandering'].mean(), 2),
                'std_verandering': round(subset['verandering'].std(), 2) if len(subset) > 1 else 0.0,
            })
        samenvatting = pd.DataFrame(samenvatting_rows)

        traject_rows = []
        max_invul = int(df_timeline['invul_nr'].max()) if not df_timeline.empty else 0
        for invul_nr in range(1, min(max_invul, 10) + 1):
            subset = df_timeline[df_timeline['n_totaal'] >= invul_nr]
            scores_at_nr = subset[subset['invul_nr'] == invul_nr]
            if scores_at_nr.empty:
                continue
            traject_rows.append({
                'invul_nr': invul_nr,
                'invul_label': f'{invul_nr}e invulling',
                'gem_score': round(scores_at_nr['score'].mean(), 2),
                'n_deelnemers': int(scores_at_nr['participant_id'].nunique()),
            })
        traject = pd.DataFrame(traject_rows)

        q_info = haal_vragenlijsten_overzicht(effective_db_url)
        q_naam = ''
        if not q_info.empty:
            match = q_info[q_info['questionnaire_id'] == int(questionnaire_id)]
            if not match.empty:
                q_naam = str(match.iloc[0]['naam'])

        meta = {
            'questionnaire_id': int(questionnaire_id),
            'questionnaire_naam': q_naam,
            'score_slug': score_slug,
            'totaal_deelnemers': int(df_change['participant_id'].nunique()),
            'herhaalinvullers': int((df_change['n_totaal'] > 1).sum()),
        }
        return samenvatting, traject, df_change, meta
    except Exception as e:
        logger.error(f"Error in analyse_herhaalde_vragenlijst_scoreverandering: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


def analyse_account_dataflow_funnel(base_pad: Path, db_url: str | None = None, participant_ids: ParticipantIds = None) -> tuple[pd.DataFrame, dict]:
    """
    Bouw een account-/vragenlijstfunnel op basis van beschikbare databronnen.

    Stappen: geregistreerd → naam → leeftijd → geslacht → vragenlijst gestart → scores.
    Naam komt uit users.name of smart_health.addresses (first_name/last_name).
    """
    from sqlalchemy import create_engine, text
    from data_ingestion import load_completions, load_my_clic_participants_expanded

    try:
        effective_db_url = db_url or __import__('config').DB_URL
        if not effective_db_url:
            return pd.DataFrame(), {}

        engine = create_engine(effective_db_url)
        is_sqlite = 'sqlite' in str(effective_db_url).lower()
        
        if is_sqlite:
            bridge_query = """
                SELECT
                    p.id AS participant_id,
                    m.user_id,
                    NULL AS user_name,
                    NULL AS first_name,
                    NULL AS last_name
                FROM participants p
                LEFT JOIN my_clic_participants m
                    ON m.qe_participant_id = p.public_id
            """
        else:
            bridge_query = """
                SELECT
                    p.id AS participant_id,
                    m.user_id,
                    u.name AS user_name,
                    a.first_name,
                    a.last_name
                FROM public.participants p
                LEFT JOIN smart_health.my_clic_participants m
                    ON m.qe_participant_id = p.public_id
                LEFT JOIN public.users u
                    ON u.id = m.user_id
                LEFT JOIN smart_health.addresses a
                    ON a.model_type = 'user'
                   AND a.model_id = m.user_id
                   AND a.deleted_at IS NULL
            """
        try:
            bridge = pd.read_sql(text(bridge_query), engine)
        except Exception:
            # Fallback simple table query
            bridge = pd.read_sql(
                text("""
                    SELECT p.id AS participant_id, m.user_id, NULL AS user_name, NULL AS first_name, NULL AS last_name
                    FROM participants p
                    LEFT JOIN my_clic_participants m ON m.qe_participant_id = p.public_id
                """),
                engine,
            )

        if participant_ids is not None:
            pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
            bridge = bridge[pd.to_numeric(bridge['participant_id'], errors='coerce').isin(pids)]

        bridge['participant_id'] = pd.to_numeric(bridge['participant_id'], errors='coerce')
        bridge = bridge.drop_duplicates(subset=['participant_id'], keep='first')

        df_scores = load_my_clic_participants_expanded(effective_db_url)
        if not df_scores.empty:
            df_scores['participant_id'] = pd.to_numeric(df_scores['participant_id'], errors='coerce')
            score_cols = ['rec_age_current', 'rec_user_gender', 'rec_ls_lifestyle_score']
            score_cols = [c for c in score_cols if c in df_scores.columns]
            df = bridge.merge(
                df_scores[['participant_id'] + score_cols].drop_duplicates('participant_id'),
                on='participant_id',
                how='left',
            )
        else:
            df = bridge.copy()
            for col in ['rec_age_current', 'rec_user_gender', 'rec_ls_lifestyle_score']:
                df[col] = pd.NA

        df_comp = load_completions(effective_db_url)
        comp_ids = set()
        if not df_comp.empty:
            comp_ids = set(pd.to_numeric(df_comp['participant_id'], errors='coerce').dropna().astype(int))

        def _has_name(row) -> bool:
            for val in (row.get('user_name'), row.get('first_name'), row.get('last_name')):
                if pd.notna(val) and str(val).strip():
                    return True
            return False

        df['has_name'] = df.apply(_has_name, axis=1)
        df['has_leeftijd'] = pd.to_numeric(df['rec_age_current'], errors='coerce').notna() if 'rec_age_current' in df.columns else False
        df['has_geslacht'] = pd.to_numeric(df['rec_user_gender'], errors='coerce').notna() if 'rec_user_gender' in df.columns else False
        df['has_gestart'] = df['participant_id'].isin(comp_ids)
        df['has_scores'] = pd.to_numeric(df['rec_ls_lifestyle_score'], errors='coerce').notna() if 'rec_ls_lifestyle_score' in df.columns else False

        # Herhaalde invulling / andere vragenlijst, op basis van completions.
        herhaal_ids = set()
        andere_vl_ids = set()
        if not df_comp.empty:
            df_comp_num = df_comp.copy()
            df_comp_num['participant_id'] = pd.to_numeric(df_comp_num['participant_id'], errors='coerce')
            df_comp_num['questionnaire_id'] = pd.to_numeric(df_comp_num['questionnaire_id'], errors='coerce')
            df_comp_num = df_comp_num.dropna(subset=['participant_id', 'questionnaire_id'])

            if participant_ids is not None:
                pids = set(pd.to_numeric(pd.Series(list(participant_ids)), errors='coerce').dropna().astype(int))
                df_comp_num = df_comp_num[df_comp_num['participant_id'].isin(pids)]

            # Dezelfde vragenlijst >1x ingevuld (zelfde patroon als haal_vragenlijsten_overzicht).
            herhaal = (
                df_comp_num.groupby(['participant_id', 'questionnaire_id'])
                .size()
                .reset_index(name='n')
            )
            herhaal_ids = set(herhaal.loc[herhaal['n'] > 1, 'participant_id'].astype(int))

            # Meer dan 1 unieke vragenlijst ingevuld.
            n_unieke_vl = df_comp_num.groupby('participant_id')['questionnaire_id'].nunique()
            andere_vl_ids = set(n_unieke_vl[n_unieke_vl > 1].index.astype(int))

        df['has_herhaald'] = df['participant_id'].isin(herhaal_ids)
        df['has_andere_vl'] = df['participant_id'].isin(andere_vl_ids)

        # Sequentiele funnel in productvolgorde (monotoon dalend) voor basisfunnel,
        # gevolgd door twee parallelle vervolgmetrieken binnen de groep met scores.
        df['step_leeftijd'] = df['has_leeftijd']
        df['step_geslacht'] = df['step_leeftijd'] & df['has_geslacht']
        df['step_gestart'] = df['step_geslacht'] & df['has_gestart']
        df['step_scores'] = df['step_gestart'] & df['has_scores']
        df['step_herhaald'] = df['step_scores'] & df['has_herhaald']
        df['step_andere_vl'] = df['step_scores'] & df['has_andere_vl']

        stage_flags = {
            'geregistreerd': pd.Series(True, index=df.index),
            'leeftijd': df['step_leeftijd'],
            'geslacht': df['step_geslacht'],
            'vragenlijst_gestart': df['step_gestart'],
            'scores_beschikbaar': df['step_scores'],
            'andere_vl_parallel': df['step_andere_vl'],
            'herhaald_parallel': df['step_herhaald'],
        }

        stage_labels = {
            'geregistreerd': 'Account / deelnemer geregistreerd',
            'leeftijd': 'Leeftijd ingevuld',
            'geslacht': 'Geslacht ingevuld',
            'vragenlijst_gestart': 'Vragenlijst gestart',
            'scores_beschikbaar': 'Scores beschikbaar',
            'andere_vl_parallel': 'Andere vragenlijst ingevuld',
            'herhaald_parallel': 'Vragenlijst opnieuw ingevuld',
        }

        n_start = int(stage_flags['geregistreerd'].sum())
        n_scores_totaal = int(stage_flags['scores_beschikbaar'].sum())
        rows = []
        prev_n = n_start
        for key, label in stage_labels.items():
            n = int(stage_flags[key].sum())
            if key in ('herhaald_parallel', 'andere_vl_parallel'):
                ref_n = n_scores_totaal
            else:
                ref_n = prev_n

            dropoff = ref_n - n if key != 'geregistreerd' else 0
            rows.append({
                'stap': label,
                'stap_key': key,
                'aantal': n,
                'pct_van_start': round(n / max(n_start, 1) * 100, 1),
                'afgevallen_sinds_vorige': max(dropoff, 0),
                'pct_afgevallen_sinds_vorige': round(max(dropoff, 0) / max(ref_n, 1) * 100, 1) if key != 'geregistreerd' else 0.0,
            })
            if key not in ('herhaald_parallel', 'andere_vl_parallel'):
                prev_n = n

        funnel_df = pd.DataFrame(rows)

        # Naam is een parallelle profielstap (niet sequentieel in de hoofdfunnel)
        n_naam_totaal = int(df['has_name'].sum())
        n_naam_na_leeftijd = int((df['step_leeftijd'] & df['has_name']).sum())
        funnel_df = pd.concat([
            funnel_df,
            pd.DataFrame([{
                'stap': 'Naam ingevuld (parallelle metriek)',
                'stap_key': 'naam_parallel',
                'aantal': n_naam_totaal,
                'pct_van_start': round(n_naam_totaal / max(n_start, 1) * 100, 1),
                'afgevallen_sinds_vorige': max(int(df['step_leeftijd'].sum()) - n_naam_na_leeftijd, 0),
                'pct_afgevallen_sinds_vorige': round(
                    max(int(df['step_leeftijd'].sum()) - n_naam_na_leeftijd, 0)
                    / max(int(df['step_leeftijd'].sum()), 1) * 100, 1
                ),
            }]),
        ], ignore_index=True)

        n_herhaald_binnen_scores = int(stage_flags['herhaald_parallel'].sum())
        n_andere_vl_binnen_scores = int(stage_flags['andere_vl_parallel'].sum())

        meta = {
            'totaal_geregistreerd': n_start,
            'naam_bron': 'users.name + smart_health.addresses',
            'naam_totaal': n_naam_totaal,
            'naam_na_leeftijd': n_naam_na_leeftijd,
            'koppeling_bridge': int(df['user_id'].notna().sum()),
            'scores_totaal': n_scores_totaal,
            'herhaald_totaal': n_herhaald_binnen_scores,
            'andere_vl_totaal': n_andere_vl_binnen_scores,
        }
        return funnel_df, meta
    except Exception as e:
        logger.error(f"Error in analyse_account_dataflow_funnel: {e}")
        return pd.DataFrame(), {}
