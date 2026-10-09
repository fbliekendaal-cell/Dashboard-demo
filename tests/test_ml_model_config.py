import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ML import (
    LIFESTYLE_FEATURES,
    LIFESTYLE_INPUT_DEFAULTS,
    IMPROVEMENT_FEATURES,
    NEXT_LIFESTYLE_FEATURES,
    BP_FEATURES,
    BP_SCREENING_THRESHOLD,
    _build_next_lifestyle_training_data,
    _attach_cv_metrics,
    _extract_feature_importances,
    _find_best_binary_f1_threshold,
    _load_purchase_target,
    _make_bp_classifier,
    _make_heartrisk_classifier,
    _make_lifestyle_regressor,
    HEARTRISK_FEATURES,
    _prepare_bp_target,
    _prepare_classification_data,
    _prepare_regression_data,
    predict_binary_model,
    predict_heartrisk,
    _resolve_ensemble_config,
    _resolve_regression_config,
    train_heartrisk_model,
    train_bp_model,
    benchmark_model_by_group,
)


def test_lifestyle_input_defaults_match_the_model_features():
    assert set(LIFESTYLE_INPUT_DEFAULTS) == set(LIFESTYLE_FEATURES)
    assert all(isinstance(value, float) for value in LIFESTYLE_INPUT_DEFAULTS.values())


def test_lifestyle_training_excludes_invalid_target_values_and_keeps_zero():
    feature = "rec_age_current"
    df = pd.DataFrame({
        feature: list(range(6)),
        "rec_med_bmi": [20.0] * 6,
        "rec_ls_stress_sum": [5.0] * 6,
        "rec_ls_lifestyle_score": [0.0, 4.0, 6.0, 8.0, 10.0, 2200.0],
    })

    _, _, y_train, y_test, _ = _prepare_regression_data(
        df,
        "rec_ls_lifestyle_score",
        [feature, "rec_med_bmi", "rec_ls_stress_sum"],
        min_rows=5,
    )

    target_values = pd.concat([y_train, y_test])
    assert target_values.between(0.0, 10.0).all()
    assert 0.0 in target_values.values
    assert 2200.0 not in target_values.values


def test_classification_split_keeps_participants_separate():
    df = pd.DataFrame({
        'participant_id': [participant for participant in range(40) for _ in range(3)],
        'feature_a': range(120),
        'feature_b': [value % 7 for value in range(120)],
        'feature_c': [value % 5 for value in range(120)],
        'target': [participant % 3 for participant in range(40) for _ in range(3)],
    })

    X_train, X_test, _, _, _ = _prepare_classification_data(
        df,
        'target',
        ['feature_a', 'feature_b', 'feature_c'],
    )

    train_participants = set(df.loc[X_train.index, 'participant_id'])
    test_participants = set(df.loc[X_test.index, 'participant_id'])
    assert train_participants.isdisjoint(test_participants)


def test_heartrisk_logistic_model_exposes_feature_importances():
    X = pd.DataFrame({
        'rec_age_current': range(30),
        'rec_user_gender': [value % 2 for value in range(30)],
        'rec_smoking_cat': [value % 3 for value in range(30)],
    })
    y = pd.Series([value % 3 for value in range(30)])

    model = _make_heartrisk_classifier(list(X.columns)).fit(X, y)
    importances = _extract_feature_importances(model)
    probabilities = model.predict_proba(X)

    assert model.named_steps['model'].class_weight == 'balanced'
    assert model.named_steps['model'].C == 0.3
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert importances is not None
    assert importances.shape == (X.shape[1],)
    assert np.isfinite(importances).all()
    assert (importances >= 0).all()


def test_classification_cv_metrics_include_foldwise_class_recalls():
    X = pd.DataFrame({
        'rec_age_current': range(90),
        'rec_user_gender': [value % 2 for value in range(90)],
        'rec_smoking_cat': [value % 3 for value in range(90)],
    })
    y = pd.Series([value % 3 for value in range(90)])
    model = _make_heartrisk_classifier(list(X.columns))

    _attach_cv_metrics(model, X, y)

    assert 'recall_moderate_mean' in model.cv_metrics
    assert 'recall_moderate_std' in model.cv_metrics
    assert 'recall_high_mean' in model.cv_metrics
    assert 'recall_high_std' in model.cv_metrics
    assert 'holdout_recall_moderate' not in model.cv_metrics


def test_heartrisk_model_excludes_blood_pressure_and_uses_categorical_smoking():
    assert 'rec_med_blood_pressure_cat' not in HEARTRISK_FEATURES
    assert 'rec_smoking_cat' in HEARTRISK_FEATURES
    assert 'rec_smoking_answer' not in HEARTRISK_FEATURES

    row_count = 120
    target = np.arange(row_count) % 3
    df = pd.DataFrame({
        'participant_id': np.arange(row_count),
        'rec_heartrisk_cat': target,
        'rec_age_current': 30 + target * 15 + np.arange(row_count) % 5,
        'rec_user_gender': np.arange(row_count) % 2,
        'rec_med_bmi': 21 + target * 2 + np.arange(row_count) % 3,
        'rec_smoking_cat': np.arange(row_count) % 3,
        'rec_med_diabetes_cat': np.arange(row_count) % 2 * 2,
        'rec_ls_nutrition_saturated_fat_per_day': np.arange(row_count) % 40,
        'rec_ls_nutrition_sugar_per_day': np.arange(row_count) % 60,
        'rec_ls_nutrition_natrium_per_day': np.arange(row_count) % 2000,
        'rec_med_blood_pressure_cat': np.arange(row_count) % 3,
    })

    model, X_test, _, features, _ = train_heartrisk_model(df)
    predictions, probabilities = predict_heartrisk(model, X_test)

    assert features == HEARTRISK_FEATURES
    assert 'rec_med_blood_pressure_cat' not in X_test.columns
    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert not hasattr(model, 'best_thresholds')


def test_bp_target_merges_elevated_and_high_categories():
    prepared = _prepare_bp_target(pd.DataFrame({'rec_med_blood_pressure_cat': [0, 1, 2, None, 3]}))

    assert prepared['bp_elevated_target'].iloc[:3].tolist() == [0.0, 1.0, 1.0]
    assert pd.isna(prepared['bp_elevated_target'].iloc[3])
    assert pd.isna(prepared['bp_elevated_target'].iloc[4])


def test_bp_classifier_and_cv_metrics_are_binary():
    X = pd.DataFrame({'age': range(90), 'bmi': [20.0 + value / 3 for value in range(90)], 'activity': [value % 7 for value in range(90)]})
    y = pd.Series([value % 2 for value in range(90)])
    model = _make_bp_classifier(y)

    _attach_cv_metrics(model, X, y)
    fitted = _make_bp_classifier(y).fit(X, y)

    assert fitted.classes_.tolist() == [0, 1]
    assert fitted.named_steps['model'].C == 0.03
    assert fitted.predict_proba(X).shape == (90, 2)
    assert 'average_precision_mean' in model.cv_metrics
    assert 'positive_recall_mean' in model.cv_metrics
    assert 'recall_moderate_mean' not in model.cv_metrics


def test_bp_model_uses_fixed_high_recall_screening_threshold():
    row_count = 160
    target = np.arange(row_count) % 5 == 0
    data = {
        'participant_id': np.arange(row_count),
        'rec_med_blood_pressure_cat': target.astype(int),
    }
    for feature_index, feature in enumerate(BP_FEATURES):
        data[feature] = (np.arange(row_count) + feature_index) % 17

    model, X_test, y_test, _, _ = train_bp_model(pd.DataFrame(data))
    predictions, _ = predict_binary_model(model, X_test)

    assert model.decision_threshold == BP_SCREENING_THRESHOLD == 0.01
    assert model.cv_metrics['decision_threshold'] == BP_SCREENING_THRESHOLD
    assert model.cv_metrics['threshold_policy'] == 'fixed_high_recall_screening'
    assert model.cv_metrics['holdout_alert_rate'] == pytest.approx(float(predictions.mean()))
    assert len(predictions) == len(y_test)


def test_purchase_target_counts_only_paid_orders(monkeypatch):
    from data_ingestion import get_user_purchases

    purchases = pd.DataFrame({
        'participant_id': [1, 2, 3, 4, 5],
        'status': ['paid', 'expired', 'payment_pending', 'canceled', 'refunded'],
    })
    monkeypatch.setattr('data_ingestion.get_user_purchases', lambda db_url=None: purchases)
    participants = pd.DataFrame({'participant_id': [1, 2, 3, 4, 5]})

    labeled = _load_purchase_target(participants)

    assert labeled['has_purchase'].tolist() == [1, 0, 0, 0, 0]


def test_binary_f1_threshold_uses_validation_probabilities():
    y_validation = pd.Series([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.35, 0.4, 0.8])

    result = _find_best_binary_f1_threshold(y_validation, probabilities)

    assert 0.0 <= result['threshold'] <= 1.0
    assert result['f1'] > 0.0


def test_binary_prediction_applies_model_decision_threshold():
    class ThresholdModel:
        classes_ = np.array([0, 1])
        feature_names_in_ = np.array(['feature'])
        decision_threshold = 0.3

        def predict_proba(self, input_df):
            return np.array([[0.8, 0.2], [0.6, 0.4]])

        def predict(self, input_df):
            return np.zeros(len(input_df), dtype=int)

    predictions, probabilities = predict_binary_model(
        ThresholdModel(),
        pd.DataFrame({'feature': [1, 2]}),
    )

    assert predictions.tolist() == [0, 1]
    assert probabilities.shape == (2, 2)


def test_lifestyle_regressor_uses_gradient_boosting_and_produces_feature_importances():
    X = pd.DataFrame({
        'age': range(30),
        'bmi': [20.0 + value / 3 for value in range(30)],
        'activity': [value % 7 for value in range(30)],
    })
    y = pd.Series([float(value % 10) for value in range(30)])

    model = _make_lifestyle_regressor().fit(X, y)
    importances = _extract_feature_importances(model)

    assert model.named_steps['model'].__class__.__name__ == 'GradientBoostingRegressor'
    assert importances is not None
    assert importances.shape == (X.shape[1],)


def test_next_lifestyle_training_uses_previous_improvement_for_third_measurement():
    snapshots = pd.DataFrame({
        'participant_id': [1, 1, 1, 2, 2],
        'completion_created_at': pd.to_datetime([
            '2026-01-01', '2026-02-01', '2026-03-01', '2026-01-15', '2026-02-15',
        ]),
        'lifestyle_score': [4.0, 5.0, 7.0, 6.0, 6.5],
    })

    training_data = _build_next_lifestyle_training_data(snapshots)
    third_measurement_row = training_data.loc[
        (training_data['participant_id'] == 1) & (training_data['current_lifestyle_score'] == 5.0)
    ].iloc[0]

    assert set(NEXT_LIFESTYLE_FEATURES).issubset(training_data.columns)
    assert third_measurement_row['next_lifestyle_score'] == 7.0
    assert third_measurement_row['previous_lifestyle_score'] == 4.0
    assert third_measurement_row['previous_lifestyle_improvement'] == 1.0
    assert third_measurement_row['days_since_previous_measurement'] == 31.0
    assert third_measurement_row['next_lifestyle_improvement'] == 2.0
    assert set(IMPROVEMENT_FEATURES) == set(NEXT_LIFESTYLE_FEATURES)


def test_resolve_ensemble_config_uses_more_conservative_settings_for_imbalanced_classification():
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 2])

    config = _resolve_ensemble_config("classification", y)

    assert config["n_estimators"] >= 300
    assert config["min_samples_leaf"] >= 2
    assert config["class_weight"] in {"balanced", "balanced_subsample"}
    assert config["max_features"] in {"sqrt", "log2"}


def test_resolve_regression_config_uses_stable_ensemble_settings_for_continuous_targets():
    config = _resolve_regression_config(default_n_estimators=240)

    assert config["n_estimators"] >= 240
    assert config["learning_rate"] <= 0.1
    assert config["max_depth"] >= 4
    assert config["l2_regularization"] >= 0.01


def test_benchmark_model_by_group_returns_group_metrics():
    df = pd.DataFrame({
        "rec_user_gender": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        "rec_age_current": [31, 33, 35, 37, 44, 46, 48, 50, 32, 34, 42, 45],
        "rec_med_bmi": [22.1, 23.4, 24.0, 25.1, 28.2, 29.7, 31.0, 30.8, 22.9, 24.3, 27.5, 29.1],
        "rec_ls_sleep_psqi_sum": [5, 6, 7, 6, 8, 9, 10, 9, 5, 6, 8, 9],
        "rec_med_blood_pressure_cat": [0, 1, 0, 1, 1, 2, 1, 2, 0, 1, 1, 2],
    })

    benchmark = benchmark_model_by_group(
        df,
        "rec_med_blood_pressure_cat",
        ["rec_age_current", "rec_med_bmi", "rec_ls_sleep_psqi_sum"],
        group_cols=["rec_user_gender"],
    )

    assert "rec_user_gender" in benchmark
    assert benchmark["rec_user_gender"]
    assert all("accuracy_mean" in item for item in benchmark["rec_user_gender"])
    assert all("recall_macro_mean" in item for item in benchmark["rec_user_gender"])
