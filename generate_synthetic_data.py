"""
Synthetische Data Generator voor Smart Health Dashboard.
Genereert geanonimiseerde, realistische testdata in SQLite database `synthetic_health.db`.
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "synthetic_health.db"

# Seed vastleggen voor formuleerbare reproduceerbare data
DEFAULT_SEED = 42

# The original script was hardwired to 450 participants. Make this
# configurable so we can produce a controlled demo dataset.
DEFAULT_NUM_USERS = 300

np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)

# Nederlandse steden en 2-cijferige prefixen voor correcte 4-cijferige postcodes
NL_CITIES = [
    {"city": "Amsterdam", "pc_prefix": "10", "lat": 52.3676, "long": 4.9041},
    {"city": "Utrecht", "pc_prefix": "35", "lat": 52.0907, "long": 5.1214},
    {"city": "Rotterdam", "pc_prefix": "30", "lat": 51.9244, "long": 4.4777},
    {"city": "Den Haag", "pc_prefix": "25", "lat": 52.0705, "long": 4.3007},
    {"city": "Eindhoven", "pc_prefix": "56", "lat": 51.4416, "long": 5.4697},
    {"city": "Groningen", "pc_prefix": "97", "lat": 53.2194, "long": 6.5665},
    {"city": "Tilburg", "pc_prefix": "50", "lat": 51.5606, "long": 5.0919},
    {"city": "Almere", "pc_prefix": "13", "lat": 52.3508, "long": 5.2647},
    {"city": "Breda", "pc_prefix": "48", "lat": 51.5896, "long": 4.7760},
    {"city": "Nijmegen", "pc_prefix": "65", "lat": 51.8449, "long": 5.8494},
    {"city": "Venlo", "pc_prefix": "59", "lat": 51.3700, "long": 6.1724},
    {"city": "Amersfoort", "pc_prefix": "38", "lat": 52.1561, "long": 5.3878},
    {"city": "Leusden", "pc_prefix": "38", "lat": 52.1319, "long": 5.4294},
    {"city": "Roermond", "pc_prefix": "60", "lat": 51.1942, "long": 5.9875},
]

# Internationale steden voor representatieve internationale vergelijking (~15%)
FOREIGN_CITIES = [
    {"city": "Antwerpen", "country": "België", "postal_code": "2000", "lat": 51.2194, "long": 4.4025},
    {"city": "Brussel", "country": "België", "postal_code": "1000", "lat": 50.8503, "long": 4.3517},
    {"city": "Gent", "country": "België", "postal_code": "9000", "lat": 51.0543, "long": 3.7174},
    {"city": "Düsseldorf", "country": "Duitsland", "postal_code": "40213", "lat": 51.2277, "long": 6.7735},
    {"city": "Köln", "country": "Duitsland", "postal_code": "50667", "lat": 50.9375, "long": 6.9603},
    {"city": "London", "country": "Verenigd Koninkrijk", "postal_code": "SW1A 1AA", "lat": 51.5074, "long": -0.1278},
]

STORES_INFO = [
    {"id": 1, "name": "Hoofdkantoor Utrecht", "city": "Utrecht", "lat": 52.0907, "long": 5.1214},
    {"id": 2, "name": "Distributiecentrum Venlo", "city": "Venlo", "lat": 51.3700, "long": 6.1724},
    {"id": 3, "name": "Filiaal Amsterdam Centrum", "city": "Amsterdam", "lat": 52.3676, "long": 4.9041},
    {"id": 4, "name": "Logistiek Rotterdam", "city": "Rotterdam", "lat": 51.9244, "long": 4.4777},
    {"id": 5, "name": "Innovation Lab Eindhoven", "city": "Eindhoven", "lat": 51.4416, "long": 5.4697},
    {"id": 6, "name": "Service Center Amersfoort", "city": "Amersfoort", "lat": 52.1561, "long": 5.3878},
]

FACTOR_SLUGS = [
    (1, "bmi", "BMI"),
    (2, "heartrisk", "Cardiovasculair Risico"),
    (3, "stress", "Stress Niveau"),
    (4, "sleep", "Slaapkwaliteit PSQI"),
    (5, "exercise", "Lichaamsbeweging"),
    (6, "fruit", "Fruit Consumptie"),
    (7, "vegetables", "Groente Consumptie"),
    (8, "sugar", "Suiker Inname"),
    (9, "fat", "Verzadigd Vet Inname"),
    (10, "salt", "Zout Inname"),
    (11, "alcohol", "Alcohol Consumptie"),
    (12, "smoking", "Rookgedrag"),
    (13, "resilience", "Veerkracht / Resilience"),
    (14, "wellbeing", "Welzijn Score"),
    (15, "selfefficacy", "Zelfeffectiviteit"),
    (16, "dass_stress", "DASS Stress"),
    (17, "dass_anxiety", "DASS Angst"),
    (18, "dass_depression", "DASS Depressie"),
    (19, "job_satisfaction", "Werktevredenheid"),
    (20, "workload", "Werkdruk"),
    (21, "work_life_balance", "Werk-Privé Balans"),
    (22, "burnout_risk", "Burn-out Risico"),
    (23, "vitality", "Vitaliteit"),
]

QUESTIONNAIRES = [
    {"id": 1, "slug": "lifestyle", "internal_name": "Leefstijl", "title": "Leefstijl vragenlijst", "name": "Leefstijl vragenlijst"},
    {"id": 2, "slug": "medical", "internal_name": "Medisch profiel", "title": "Medisch profiel", "name": "Medisch profiel"},
    {"id": 3, "slug": "stress_sleep", "internal_name": "Stress en slaap", "title": "Stress en slaap", "name": "Stress en slaap"},
    {"id": 4, "slug": "mental_health", "internal_name": "Mentale gezondheid", "title": "Mentale gezondheid", "name": "Mentale gezondheid"},
    {"id": 5, "slug": "work_ability", "internal_name": "Werkvermogen", "title": "Werkvermogen", "name": "Werkvermogen"},
    {"id": 6, "slug": "wellbeing", "internal_name": "Welzijn", "title": "Welzijn", "name": "Welzijn"},
    {"id": 7, "slug": "nutrition", "internal_name": "Voeding", "title": "Voeding", "name": "Voeding"},
    {"id": 8, "slug": "engagement", "internal_name": "Engagement", "title": "Engagement", "name": "Engagement"},
]

PRODUCTS = [
    {"id": 101, "name": "Leefstijl Coaching Pakket", "price": 149.0},
    {"id": 102, "name": "Slaapverbetering Training", "price": 79.0},
    {"id": 103, "name": "Voedingsadvies op Maat", "price": 99.0},
    {"id": 104, "name": "Stress & Veerkracht Workshop", "price": 129.0},
    {"id": 105, "name": "Hartgezondheid PMO Check", "price": 199.0},
]

FACTOR_TO_DASHBOARD_SLUG = {
    "bmi": "rec_med_bmi",
    "heartrisk": "rec_heartrisk",
    "stress": "rec_ls_stress_sum",
    "sleep": "rec_ls_sleep_psqi_sum",
    "exercise": "rec_ls_score_exercise",
    "fruit": "rec_ls_score_fruit",
    "vegetables": "rec_ls_score_vegetables",
    "sugar": "rec_ls_score_sugar",
    "fat": "rec_ls_score_saturated_fat",
    "salt": "rec_ls_score_natrium",
    "alcohol": "rec_ls_score_alcohol",
    "smoking": "rec_smoking_answer",
    "resilience": "rec_resilience_score",
    "wellbeing": "rec_wellbeing_score",
    "selfefficacy": "rec_self_efficacy_score",
    "dass_stress": "rec_dass_stress_score",
    "dass_anxiety": "rec_dass_anxiety_score",
    "dass_depression": "rec_dass_depression_score",
    "job_satisfaction": "rec_asr_job_satisfaction_score",
    "workload": "rec_asr_workload_score",
    "work_life_balance": "rec_asr_work_ability_score",
    "burnout_risk": "rec_asr_burn_out_score",
    "vitality": "rec_asr_vitality_score",
}


def generate_all(num_users: int = DEFAULT_NUM_USERS, seed: int = DEFAULT_SEED):
    """Regenerate the full SQLite demo database.

    Parameters
    ----------
    num_users:
        Number of synthetic participants to generate.
    seed:
        Fixed random seed for deterministic synthetic output.
    """
    print(f"🔄 Genereer synthetische gezondheids- en werkplekdata voor {num_users} deelnemers...")
    np.random.seed(seed)
    random.seed(seed)

    conn = sqlite3.connect(DB_PATH)
    
    num_users = int(num_users)
    
    # 1. STORES
    df_stores = pd.DataFrame([
        {
            "id": s["id"],
            "name": s["name"],
            "city": s["city"],
            "is_active": 1,
            "created_at": "2019-01-01 00:00:00",
            "deleted_at": None,
        }
        for s in STORES_INFO
    ])
    df_stores.to_sql("stores", conn, if_exists="replace", index=False)

    # 2. QUESTIONNAIRE FACTORS
    df_qf = pd.DataFrame([
        {"id": f[0], "slug": f[1], "name": f[2]} for f in FACTOR_SLUGS
    ])
    df_qf.to_sql("questionnaire_factors", conn, if_exists="replace", index=False)

    # 3. USERS, PARTICIPANTS, MY_CLIC_PARTICIPANTS & ADDRESSES
    users_met_scores_data = []
    user_accounts_data = []
    participants_data = []
    my_clic_data = []
    addresses_data = []
    store_emp_data = []
    completions_data = []
    histories_data = []
    completion_scores_data = []
    latest_score_rows = {}
    content_data = []
    content_translation_data = []
    interactions_data = []

    start_date = datetime(2019, 1, 1)

    completion_id_counter = 1
    history_id_counter = 1
    completion_score_id_counter = 1

    for u_id in range(1, num_users + 1):
        p_id = u_id
        store = random.choice(STORES_INFO)
        store_id = store["id"]

        is_foreign = random.random() < 0.15
        if is_foreign:
            city_info = random.choice(FOREIGN_CITIES)
            country = city_info["country"]
            pc = city_info["postal_code"]
        else:
            city_info = random.choice(NL_CITIES)
            country = "Nederland"
            pc = f"{city_info['pc_prefix']}{random.randint(10, 99)} {random.choice(['AB', 'CD', 'EF', 'GH', 'JK', 'LM', 'NP', 'RS', 'TV', 'XZ'])}"

        gender = random.choice([0, 1])  # 0=female, 1=male
        age = int(np.clip(np.random.normal(41, 11), 20, 67))

        user_created = start_date + timedelta(days=random.randint(0, 1800))
        user_created_str = user_created.strftime("%Y-%m-%d %H:%M:%S")
        public_id = f"PUB-{100000 + u_id}"

        # Biological & Assessment base metrics
        bmi = round(float(np.clip(np.random.normal(25.5, 4.2), 17.5, 39.0)), 1)
        bmi_cat = 2 if bmi >= 30 else (1 if bmi >= 25 else 0)

        heartrisk = round(float(np.clip(np.random.exponential(6.0), 1.0, 32.0)), 1)
        heartrisk_cat = 2 if heartrisk >= 20 else (1 if heartrisk >= 10 else 0)

        stress_sum = round(float(np.clip(np.random.normal(3.8, 2.1), 0.0, 10.0)), 1)
        stress_cat = 2 if stress_sum >= 6.5 else (1 if stress_sum >= 3.5 else 0)

        psqi = round(float(np.clip(np.random.normal(6.5, 3.2), 0.0, 18.0)), 1)
        sleep_cat = 2 if psqi >= 10 else (1 if psqi >= 5 else 0)

        lifestyle_score = round(float(np.clip(8.5 - 0.25 * stress_sum - 0.1 * (bmi - 22 if bmi > 22 else 0), 2.0, 9.8)), 1)

        wai = round(float(np.clip(9.5 - 0.4 * stress_sum + np.random.normal(0, 0.5), 1.0, 10.0)), 1)
        burnout = round(float(np.clip(1.5 + 0.6 * stress_sum + np.random.normal(0, 0.4), 0.0, 10.0)), 1)
        vitality = round(float(np.clip(lifestyle_score * 0.9 + np.random.normal(0, 0.5), 1.0, 10.0)), 1)
        job_sat = round(float(np.clip(7.2 + np.random.normal(0, 1.2), 2.0, 10.0)), 1)
        workload = round(float(np.clip(5.0 + 0.4 * stress_sum + np.random.normal(0, 1.0), 1.0, 10.0)), 1)
        exhaustion = round(float(np.clip(burnout * 0.95, 0.0, 10.0)), 1)

        fruit = round(float(np.clip(np.random.normal(1.8, 0.8), 0.0, 5.0)), 1)
        vegetables = round(float(np.clip(np.random.normal(220, 80), 30.0, 500.0)), 0)
        sugar = round(float(np.clip(np.random.normal(45, 20), 5.0, 140.0)), 0)
        fat = round(float(np.clip(np.random.normal(28, 10), 5.0, 75.0)), 0)
        natrium = round(float(np.clip(np.random.normal(2400, 600), 600.0, 5000.0)), 0)
        alcohol = round(float(np.clip(np.random.exponential(3.5), 0.0, 24.0)), 1)
        steps = int(np.clip(np.random.normal(7800, 2500), 1500, 18000))
        activity_min = int(np.clip(np.random.normal(180, 70), 20, 500))
        smoking = 1 if random.random() < 0.14 else 0

        resilience = round(float(np.clip(np.random.normal(6.8, 1.5), 1.0, 10.0)), 1)
        wellbeing = round(float(np.clip(np.random.normal(7.1, 1.4), 1.0, 10.0)), 1)
        self_efficacy = round(float(np.clip(np.random.normal(7.3, 1.3), 1.0, 10.0)), 1)
        dass_stress = round(float(np.clip(stress_sum * 3.5, 0.0, 42.0)), 1)
        dass_anxiety = round(float(np.clip(np.random.exponential(4.0), 0.0, 36.0)), 1)
        dass_depression = round(float(np.clip(np.random.exponential(4.5), 0.0, 42.0)), 1)

        # Additional rec_* dimensions used by the dashboard/ML surface.
        # These are score-like normalized synthetic ratings so charts and model
        # selectors can look them up through the same names as the real object.
        work_ability_score = round(float(np.clip(wai + np.random.normal(0, 0.55), 0.0, 10.0)), 1)
        work_ability_cat = 2 if work_ability_score >= 7 else (1 if work_ability_score >= 4 else 0)
        working_attitude_score = round(float(np.clip(job_sat + np.random.normal(0, 0.7), 0.0, 10.0)), 1)
        working_attitude_cat = 2 if working_attitude_score >= 7 else (1 if working_attitude_score >= 4 else 0)
        personal_competences_score = round(float(np.clip(6.3 + (self_efficacy - 5.5) * 0.35 + np.random.normal(0, 0.5), 0.0, 10.0)), 1)
        personal_competences_cat = 2 if personal_competences_score >= 7 else (1 if personal_competences_score >= 4 else 0)
        minor_mental_complaints_score = round(float(np.clip(max(0.0, stress_sum * 1.05 + burnout * 0.2), 0.0, 10.0)), 1)
        minor_mental_complaints_cat = 2 if minor_mental_complaints_score >= 7 else (1 if minor_mental_complaints_score >= 4 else 0)
        diabetes_cat = 2 if bmi >= 30 else (1 if bmi >= 25 else 0)
        blood_pressure_cat = 2 if heartrisk >= 15 else (1 if heartrisk >= 8 else 0)

        # Generate multiple historical survey completions per participant over time.
        # A repeated questionnaire per participant keeps repeat-change analyses non-empty.
        num_completions = random.choices([3, 4, 5, 6], weights=[0.25, 0.35, 0.25, 0.15])[0]
        primary_questionnaire_id = random.choice([q["id"] for q in QUESTIONNAIRES])
        comp_date = user_created

        for c_idx in range(num_completions):
            comp_date_str = comp_date.strftime("%Y-%m-%d %H:%M:%S")
            c_id = completion_id_counter
            completion_id_counter += 1
            questionnaire_id = (
                primary_questionnaire_id
                if c_idx in {0, num_completions - 1}
                else random.choice([q["id"] for q in QUESTIONNAIRES])
            )

            completions_data.append({
                "id": c_id,
                "participant_id": p_id,
                "questionnaire_id": questionnaire_id,
                "created_at": comp_date_str,
                "updated_at": comp_date_str,
                "status": "completed",
            })

            # Time progression simulation (scores improve slightly over assessments)
            trend_factor = c_idx * 0.15
            c_bmi = round(max(17.0, bmi - trend_factor * 0.4), 1)
            c_stress = round(max(0.0, stress_sum - trend_factor * 0.5), 1)
            c_lifestyle = round(min(10.0, lifestyle_score + trend_factor * 0.4), 1)
            c_wellbeing = round(min(10.0, wellbeing + trend_factor * 0.3), 1)

            factor_values = {
                1: c_bmi,
                2: heartrisk,
                3: c_stress,
                4: psqi,
                5: round(min(5.0, 2.5 + trend_factor), 1),
                6: round(min(5.0, fruit + trend_factor * 0.2), 1),
                7: vegetables,
                8: sugar,
                9: fat,
                10: natrium,
                11: alcohol,
                12: smoking,
                13: resilience,
                14: c_wellbeing,
                15: self_efficacy,
                16: dass_stress,
                17: dass_anxiety,
                18: dass_depression,
                19: job_sat,
                20: workload,
                21: wai,
                22: burnout,
                23: vitality,
            }

            for f_id, f_val in factor_values.items():
                factor_slug = next(slug for factor_id, slug, _name in FACTOR_SLUGS if factor_id == f_id)
                histories_data.append({
                    "id": history_id_counter,
                    "participant_id": p_id,
                    "questionnaire_factor_id": f_id,
                    "completion_id": c_id,
                    "score_value": f_val,
                    "score_category_value": None,
                    "completion_created_at": comp_date_str,
                    "created_at": comp_date_str,
                    "updated_at": comp_date_str,
                })
                completion_scores_data.append({
                    "id": completion_score_id_counter,
                    "completion_id": c_id,
                    "participant_id": p_id,
                    "questionnaire_id": questionnaire_id,
                    "slug": factor_slug,
                    "value": f_val,
                    "score_value": f_val,
                    "created_at": comp_date_str,
                    "updated_at": comp_date_str,
                })
                completion_score_id_counter += 1
                dashboard_slug = FACTOR_TO_DASHBOARD_SLUG.get(factor_slug)
                if dashboard_slug:
                    completion_scores_data.append({
                        "id": completion_score_id_counter,
                        "completion_id": c_id,
                        "participant_id": p_id,
                        "questionnaire_id": questionnaire_id,
                        "slug": dashboard_slug,
                        "value": f_val,
                        "score_value": f_val,
                        "created_at": comp_date_str,
                        "updated_at": comp_date_str,
                    })
                    completion_score_id_counter += 1
                history_id_counter += 1

            comp_date += timedelta(days=random.randint(90, 240))

        latest_comp_str = comp_date_str

        # Participant row
        participants_data.append({
            "id": p_id,
            "public_id": public_id,
            "user_id": u_id,
            "store_id": store_id,
            "partner_id": 1,
            "created_at": user_created_str,
            "deleted_at": None,
            "rec_user_gender": gender,
            "rec_age_current": age,
            "postal_code": pc,
            "gender": "Man" if gender == 1 else "Vrouw",
        })

        # User Consolidated Score row (users_met_scores)
        users_met_scores_data.append({
            "user_id": u_id,
            "participant_id": p_id,
            "store_id": store_id,
            "partner_id": 1,
            "created_at": user_created_str,
            "latest_completion_at": latest_comp_str,
            "rec_user_gender": gender,
            "rec_age_current": age,
            "postal_code": pc,
            "rec_med_bmi": bmi,
            "rec_med_bmi_cat": bmi_cat,
            "rec_heartrisk": heartrisk,
            "rec_heartrisk_cat": heartrisk_cat,
            "rec_ls_lifestyle_score": lifestyle_score,
            "rec_ls_stress_sum": stress_sum,
            "rec_ls_stress_cat": stress_cat,
            "rec_ls_sleep_psqi_sum": psqi,
            "rec_ls_sleep_cat": sleep_cat,
            "rec_ls_score_fruit": round(min(5.0, 1.0 + fruit), 1),
            "rec_ls_score_vegetables": round(min(5.0, 1.0 + vegetables / 100), 1),
            "rec_ls_score_sugar": round(min(5.0, 5.0 - sugar / 30), 1),
            "rec_ls_score_saturated_fat": round(min(5.0, 5.0 - fat / 20), 1),
            "rec_ls_score_natrium": round(min(5.0, 5.0 - natrium / 1000), 1),
            "rec_ls_score_alcohol": round(min(5.0, 5.0 - alcohol / 5), 1),
            "rec_ls_score_exercise": round(min(5.0, steps / 2500), 1),
            "rec_ls_exercise_steps_per_day": steps,
            "rec_ls_exercise_physical_activity_minutes_total": activity_min,
            "rec_ls_alcohol_total_per_week": alcohol,
            "rec_ls_vegetables_gram_per_day": vegetables,
            "rec_ls_nutrition_fruit_fruit_per_day": fruit,
            "rec_ls_nutrition_sugar_per_day": sugar,
            "rec_ls_nutrition_saturated_fat_per_day": fat,
            "rec_ls_nutrition_natrium_per_day": natrium,
            "rec_smoking_answer": smoking,
            "rec_dass_stress_score": dass_stress,
            "rec_dass_anxiety_score": dass_anxiety,
            "rec_dass_depression_score": dass_depression,
            "rec_resilience_score": resilience,
            "rec_wellbeing_score": wellbeing,
            "rec_self_efficacy_score": self_efficacy,
            "rec_health": round((lifestyle_score + wellbeing) / 2, 1),
            "rec_asr_wai_score": wai,
            "rec_asr_work_ability_score": work_ability_score,
            "rec_asr_work_ability_cat": work_ability_cat,
            "rec_asr_burn_out_score": burnout,
            "rec_asr_vitality_score": vitality,
            "rec_asr_job_satisfaction_score": job_sat,
            "rec_asr_working_attitude_score": working_attitude_score,
            "rec_asr_working_attitude_category": working_attitude_cat,
            "rec_asr_personal_competences_score": personal_competences_score,
            "rec_asr_personal_competences_category": personal_competences_cat,
            "rec_asr_minor_mental_complaints_score": minor_mental_complaints_score,
            "rec_asr_minor_mental_complaints_category": minor_mental_complaints_cat,
            "rec_asr_workload_score": workload,
            "rec_asr_exhaustion_score": exhaustion,
            "rec_med_diabetes_cat": diabetes_cat,
            "rec_med_blood_pressure_cat": blood_pressure_cat,
            "rec_framingham_non_invasive": heartrisk,
            "rec_ls_exercise_steps_cat": 2 if steps >= 10000 else (1 if steps >= 5000 else 0),
            "rec_ls_sleep_psqi_amount_of_hours_slept_text": "7-9 uur" if 7 <= 8.5 - 0.2 * psqi <= 9 else ("6-7 uur" if psqi <= 8 else ("9-10 uur" if psqi <= 11 else "Minder dan 6 uur")),
            "rec_ls_stress_type_1_worry": 2 if stress_sum >= 6.5 else (1 if stress_sum >= 3.5 else 0),
            "rec_ls_stress_type_2_tense": 2 if stress_sum >= 6.0 else (1 if stress_sum >= 3.0 else 0),
        })

        user_accounts_data.append({
            "id": u_id,
            "email": f"demo{u_id}@example.org",
            "deleted_at": None,
        })

        my_clic_data.append({
            "id": u_id,
            "user_id": u_id,
            "qe_participant_id": public_id,
            "public_id": public_id,
            "created_at": user_created_str,
        })

        addresses_data.append({
            "id": u_id,
            "model_type": "user",
            "model_id": u_id,
            "app_user_id": u_id,
            "lat": city_info["lat"] + np.random.normal(0, 0.03),
            "long": city_info["long"] + np.random.normal(0, 0.04),
            "city": city_info["city"],
            "country": country,
            "postal_code": pc,
            "created_at": user_created_str,
            "updated_at": user_created_str,
            "deleted_at": None,
        })

        store_emp_data.append({
            "id": u_id,
            "user_id": u_id,
            "store_id": store_id,
            "created_at": user_created_str,
            "deleted_at": None,
            "archived_at": None,
        })

    # Synthetic content layer that the article/engagement visualisations can read
    content_ids = [101, 102, 103, 104, 105]
    content_titles = [
        "Gezond bewegen",
        "Stress in balans",
        "Voeding en leefstijl",
        "Slaapverbetering",
        "Werkdruk begrijpen",
    ]
    for idx, content_id in enumerate(content_ids):
        content_data.append({
            "id": content_id,
            "public_id": f"pub-content-{content_id}",
            "title": content_titles[idx],
        })
        content_translation_data.append({
            "content_id": content_id,
            "locale": "nl_NL",
            "title": content_titles[idx],
        })
    # A small, deterministic set of content views across the demo users
    for idx, user_id in enumerate(range(1, min(num_users, 120) + 1)):
        content_id = content_ids[(idx % len(content_ids))]
        view_time = (datetime(2024, 1, 1) + timedelta(days=idx % 365)).strftime("%Y-%m-%d %H:%M:%S")
        interactions_data.append({
            "id": len(interactions_data) + 1,
            "interactable_type": "content",
            "interactable_id": content_id,
            "user_id": user_id,
            "type": "view",
            "created_at": view_time,
            "updated_at": view_time,
        })

    for row in users_met_scores_data:
        pid = row["participant_id"]
        for slug, value in row.items():
            if slug.startswith(("rec_", "feat_")):
                latest_score_rows[(pid, slug)] = {
                    "participant_id": pid,
                    "slug": slug,
                    "value": value,
                    "created_at": row["latest_completion_at"],
                    "updated_at": row["latest_completion_at"],
                }

    for history_row in histories_data:
        factor_slug = next(
            slug for factor_id, slug, _name in FACTOR_SLUGS
            if factor_id == history_row["questionnaire_factor_id"]
        )
        latest_score_rows[(history_row["participant_id"], factor_slug)] = {
            "participant_id": history_row["participant_id"],
            "slug": factor_slug,
            "value": history_row["score_value"],
            "created_at": history_row["completion_created_at"],
            "updated_at": history_row["completion_created_at"],
        }

    for s in STORES_INFO:
        addresses_data.append({
            "id": 100000 + s["id"],
            "model_type": "store",
            "model_id": s["id"],
            "app_user_id": None,
            "lat": s["lat"],
            "long": s["long"],
            "city": s["city"],
            "country": "Nederland",
            "postal_code": f"{s['id']:04d} ZZ",
            "created_at": "2019-01-01 00:00:00",
            "updated_at": "2019-01-01 00:00:00",
            "deleted_at": None,
        })

    # Generate synthetic user_challenges table
    challenges_data = []
    for c_u_id in range(1, min(num_users, 150) + 1):
        challenges_data.append({
            "id": c_u_id,
            "user_ref": c_u_id,
            "challenge_id": random.choice([1, 2, 3, 4]),
            "store_id": random.choice([1, 2, 3, 4, 5, 6]),
            "status": random.choice(["completed", "active", "in_progress"]),
            "weeks_completed": random.randint(1, 6),
            "challenge_type": "lifestyle",
            "created_at": "2024-01-01 00:00:00",
            "updated_at": "2024-02-01 00:00:00",
        })

    # Save to SQLite tables
    pd.DataFrame(QUESTIONNAIRES).to_sql("questionnaires", conn, if_exists="replace", index=False)
    pd.DataFrame(users_met_scores_data).to_sql("users_met_scores", conn, if_exists="replace", index=False)
    pd.DataFrame(user_accounts_data).to_sql("users", conn, if_exists="replace", index=False)
    pd.DataFrame(participants_data).to_sql("participants", conn, if_exists="replace", index=False)
    pd.DataFrame(my_clic_data).to_sql("my_clic_participants", conn, if_exists="replace", index=False)
    pd.DataFrame(addresses_data).to_sql("addresses", conn, if_exists="replace", index=False)
    pd.DataFrame(store_emp_data).to_sql("store_employees", conn, if_exists="replace", index=False)
    pd.DataFrame(completions_data).to_sql("completions", conn, if_exists="replace", index=False)
    pd.DataFrame(histories_data).to_sql("factor_score_histories", conn, if_exists="replace", index=False)
    pd.DataFrame(completion_scores_data).to_sql("completion_scores", conn, if_exists="replace", index=False)
    pd.DataFrame(latest_score_rows.values()).to_sql("latest_scores", conn, if_exists="replace", index=False)
    pd.DataFrame(content_data).to_sql("content", conn, if_exists="replace", index=False)
    pd.DataFrame(content_translation_data).to_sql("content_translations", conn, if_exists="replace", index=False)
    pd.DataFrame(interactions_data).to_sql("interactions", conn, if_exists="replace", index=False)
    pd.DataFrame(challenges_data).to_sql("user_challenges", conn, if_exists="replace", index=False)

    # 4. PRODUCTS & ORDERS
    df_products = pd.DataFrame(PRODUCTS)
    df_products.to_sql("products", conn, if_exists="replace", index=False)

    orders_data = []
    order_id = 1
    for u_id in range(1, num_users + 1):
        if random.random() < 0.35:  # 35% of users purchased something
            prod = random.choice(PRODUCTS)
            orders_data.append({
                "id": order_id,
                "user_id": u_id,
                "product_id": prod["id"],
                "status": "completed",
                "created_at": (datetime(2022, 1, 1) + timedelta(days=random.randint(0, 1000))).strftime("%Y-%m-%d %H:%M:%S"),
            })
            order_id += 1
    pd.DataFrame(orders_data).to_sql("orders", conn, if_exists="replace", index=False)

    # 5. STORE AVERAGE SCORES
    store_avg_data = []
    months = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
    score_slugs = ["rec_med_bmi", "rec_heartrisk", "rec_ls_stress_sum", "rec_ls_lifestyle_score", "rec_wellbeing_score", "rec_resilience_score"]
    
    for s in STORES_INFO:
        s_id = s["id"]
        for m in months:
            m_str = m.strftime("%Y-%m-%d")
            for slug in score_slugs:
                avg_val = round(float(np.random.normal(6.5 if "score" in slug or "wellbeing" in slug else 12.0, 1.5)), 2)
                store_avg_data.append({
                    "store_id": s_id,
                    "score_slug": slug,
                    "date": m_str,
                    "average": avg_val,
                    "participants_count": random.randint(25, 80),
                })
    pd.DataFrame(store_avg_data).to_sql("store_average_scores", conn, if_exists="replace", index=False)

    conn.close()
    print(f"✅ Synthetische database aangemaakt: {DB_PATH}")


if __name__ == "__main__":
    generate_all()
