# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportUnboundVariable=false

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Callable, cast

import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from i18n import tr
from kleuren import HOOFD_KLEUR

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer
    from sklearn.cluster import KMeans
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        VotingClassifier,
        VotingRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, precision_recall_curve, recall_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

logger = logging.getLogger(__name__)

# Defensieve import voor ML bibliotheken
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.cluster import KMeans
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        VotingClassifier,
        VotingRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, precision_recall_curve, recall_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
    HAS_SKLEARN = True
except Exception as e:
    HAS_SKLEARN = False
    SKLEARN_IMPORT_ERROR = str(e)
    logger.warning(f"Scikit-learn niet beschikbaar. ML functionaliteit beperkt: {e}")
    KMeans = cast(Any, None)
    ColumnTransformer = cast(Any, None)
    ExtraTreesClassifier = cast(Any, None)
    ExtraTreesRegressor = cast(Any, None)
    GradientBoostingRegressor = cast(Any, None)
    HistGradientBoostingClassifier = cast(Any, None)
    HistGradientBoostingRegressor = cast(Any, None)
    RandomForestClassifier = cast(Any, None)
    RandomForestRegressor = cast(Any, None)
    VotingClassifier = cast(Any, None)
    VotingRegressor = cast(Any, None)
    SimpleImputer = cast(Any, None)
    LogisticRegression = cast(Any, None)
    confusion_matrix = cast(Any, None)
    precision_recall_curve = cast(Any, None)
    recall_score = cast(Any, None)
    cross_validate = cast(Any, None)
    train_test_split = cast(Any, None)
    KFold = cast(Any, None)
    GroupKFold = cast(Any, None)
    GroupShuffleSplit = cast(Any, None)
    Pipeline = cast(Any, None)
    StandardScaler = cast(Any, None)
    OneHotEncoder = cast(Any, None)
    label_binarize = cast(Any, None)
else:
    SKLEARN_IMPORT_ERROR = ""


def _safe_recall_score(*args: Any, **kwargs: Any) -> Any:
    if recall_score is None:
        raise RuntimeError("Scikit-learn is niet beschikbaar.")
    return recall_score(*args, **kwargs)


def _split_ml_data(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    test_size: float = 0.25,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    use_group_split = (
        groups is not None
        and groups.notna().all()
        and groups.duplicated().any()
    )
    if not use_group_split:
        stratify_values = y if stratify and y.value_counts().min() >= 2 else None
        return cast(
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify_values,
            ),
        )

    splitter = GroupShuffleSplit(n_splits=25 if stratify else 1, test_size=test_size, random_state=42)
    best_split = None
    best_score = None
    for train_indices, test_indices in splitter.split(X, y, groups):
        train_labels = set(y.iloc[train_indices])
        test_labels = set(y.iloc[test_indices])
        score = (
            len(train_labels & test_labels),
            len(train_labels | test_labels),
            -abs(len(test_indices) / len(y) - test_size),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_split = (train_indices, test_indices)

    if best_split is None:
        raise ValueError("Kon geen train/test-splitsing maken.")

    train_indices, test_indices = best_split
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


def _groups_for_rows(df: pd.DataFrame, row_index: pd.Index) -> pd.Series | None:
    if 'participant_id' not in df.columns:
        return None
    return df.loc[row_index, 'participant_id']


def _as_float(value: Any) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _ensure_model_metadata(model: Any) -> Any:
    if not hasattr(model, 'cv_metrics'):
        model.cv_metrics = {}
    return model


BP_FEATURES = [
    'rec_age_current',
    'rec_user_gender',
    'rec_med_bmi',
    'rec_med_diabetes_cat',
    'rec_ls_exercise_steps_per_day',
    'rec_ls_nutrition_natrium_per_day',
    'rec_ls_alcohol_total_per_week',
    'rec_ls_sleep_psqi_sum',
    'rec_ls_stress_sum',
    'rec_resilience_score',
]
BP_TARGET = 'bp_elevated_target'
BP_SCREENING_THRESHOLD = 0.01

HEARTRISK_FEATURES = [
    'rec_age_current',
    'rec_user_gender',
    'rec_med_bmi',
    'rec_smoking_cat',
    'rec_med_diabetes_cat',
    'rec_ls_nutrition_saturated_fat_per_day',
    'rec_ls_nutrition_sugar_per_day',
    'rec_ls_nutrition_natrium_per_day',
]

HEARTRISK_CATEGORICAL_FEATURES = {
    'rec_user_gender',
    'rec_smoking_cat',
    'rec_med_diabetes_cat',
}

ML_MODEL_VERSION = "v9-bp-high-recall-screening-2026-10-09"
LIFESTYLE_USE_CLASSIFICATION = False
LIFESTYLE_SCORE_RANGE = (0.0, 10.0)

ACTIVATION_FEATURES = [
    'rec_age_current',
    'rec_user_gender',
    'store_id',
    'has_postal_code',
]

LIFESTYLE_FEATURES = [
    'rec_age_current',
    'rec_user_gender',
    'rec_med_bmi',
    'rec_ls_stress_sum',
    'rec_ls_sleep_psqi_sum',
    'rec_ls_exercise_steps_per_day',
    'rec_ls_exercise_physical_activity_minutes_total',
    'rec_ls_nutrition_fruit_fruit_per_day',
    'rec_ls_vegetables_gram_per_day',
    'rec_ls_nutrition_sugar_per_day',
    'rec_ls_nutrition_saturated_fat_per_day',
    'rec_ls_nutrition_natrium_per_day',
    'rec_ls_alcohol_total_per_week',
    'rec_smoking_answer',
    'rec_resilience_score',
    'rec_wellbeing_score',
]

# Defaults for interactive predictions. Keep these keys aligned with the training features.
LIFESTYLE_INPUT_DEFAULTS = {
    'rec_age_current': 50.0,
    'rec_user_gender': 0.0,
    'rec_med_bmi': 25.0,
    'rec_ls_stress_sum': 8.0,
    'rec_ls_sleep_psqi_sum': 6.0,
    'rec_ls_exercise_steps_per_day': 7.0,
    'rec_ls_exercise_physical_activity_minutes_total': 570.0,
    'rec_ls_nutrition_fruit_fruit_per_day': 2.0,
    'rec_ls_vegetables_gram_per_day': 250.0,
    'rec_ls_nutrition_sugar_per_day': 50.0,
    'rec_ls_nutrition_saturated_fat_per_day': 25.0,
    'rec_ls_nutrition_natrium_per_day': 2000.0,
    'rec_ls_alcohol_total_per_week': 3.0,
    'rec_smoking_answer': 1.0,
    'rec_resilience_score': 6.0,
    'rec_wellbeing_score': 7.0,
}

NEXT_LIFESTYLE_FEATURES = [
    'current_lifestyle_score',
    'previous_lifestyle_score',
    'previous_lifestyle_improvement',
    'days_since_previous_measurement',
    'measurement_number',
]

IMPROVEMENT_FEATURES = NEXT_LIFESTYLE_FEATURES

PURCHASE_FEATURES = [
    'rec_age_current',
    'rec_user_gender',
    'rec_med_bmi',
    'rec_ls_lifestyle_score',
    'rec_heartrisk_cat',
    'rec_ls_stress_sum',
    'rec_ls_sleep_psqi_sum',
    'rec_resilience_score',
    'rec_wellbeing_score',
    'store_id',
]

FEATURE_LABELS = {
    'rec_age_current': 'Leeftijd',
    'rec_user_gender': 'Geslacht',
    'rec_med_bmi': 'BMI',
    'rec_med_diabetes_cat': 'Diabetes',
    'rec_ls_exercise_steps_per_day': 'Stappen per dag',
    'rec_ls_exercise_physical_activity_minutes_total': 'Beweegminuten per week',
    'rec_ls_vegetables_gram_per_day': 'Groenten',
    'rec_ls_nutrition_natrium_per_day': 'Natrium',
    'rec_ls_alcohol_total_per_week': 'Alcohol',
    'rec_ls_sleep_psqi_sum': 'Slaap PSQI',
    'rec_resilience_score': 'Veerkracht',
    'rec_smoking_answer': 'Roken',
    'rec_med_blood_pressure_cat': 'Bloeddruk',
    'rec_ls_nutrition_saturated_fat_per_day': 'Verzadigd vet',
    'rec_ls_nutrition_sugar_per_day': 'Suiker',
    'rec_ls_lifestyle_score': 'Leefstijlscore',
    'rec_ls_stress_sum': 'Stress',
    'rec_wellbeing_score': 'Welzijn',
    'rec_smoking_cat': 'Roken',
    'store_id': 'Opdrachtgever',
    'has_postal_code': 'Postcode aanwezig',
    'improvement_lifestyle_score': 'Verbetering leefstijlscore',
    'next_lifestyle_score': 'Volgende leefstijlscore',
    'previous_lifestyle_improvement': 'Vorige scoreverandering',
    'has_purchase': 'Heeft aankoop',
    'dropoff_risk': 'Geen scoredata',
}


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14))
    fig.update_layout(template='plotly_white', height=420)
    return fig


def _prepare_classification_data(df: pd.DataFrame, target: str, features: list[str], min_rows: int = 50):
    if not HAS_SKLEARN:
        raise RuntimeError("Scikit-learn is niet beschikbaar.")
    if df is None or df.empty:
        raise ValueError("Geen data beschikbaar voor het ML model.")
    available = [c for c in features if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Doelkolom ontbreekt: {target}")
    if len(available) < 3:
        raise ValueError("Te weinig bruikbare features voor het ML model.")

    df_ml = df[available + [target]].apply(pd.to_numeric, errors='coerce')
    if 'participant_id' in df.columns:
        df_ml['participant_id'] = df['participant_id']
    df_ml = df_ml.dropna(subset=[target])
    df_ml = df_ml[df_ml[target].isin([0, 1, 2])]
    if len(df_ml) < min_rows:
        raise ValueError("Te weinig rijen met geldige targetwaarden voor ML.")

    X = df_ml[available]
    y = df_ml[target].astype(int)
    groups = df_ml['participant_id'] if 'participant_id' in df_ml.columns else None
    X_train, X_test, y_train, y_test = _split_ml_data(X, y, groups=groups, stratify=True)
    return X_train, X_test, y_train, y_test, available


def _resolve_ensemble_config(model_type: str, y: pd.Series | None = None, default_n_estimators: int = 350) -> dict[str, Any]:
    """Tune ensemble settings to be more robust for class imbalance without overfitting."""
    n_estimators = max(300, int(default_n_estimators))
    min_samples_leaf = 2
    max_features = 'sqrt'
    class_weight = 'balanced_subsample'

    if y is not None and not y.empty:
        y_values = pd.to_numeric(pd.Series(y), errors='coerce').dropna()
        if not y_values.empty and y_values.nunique() > 1:
            counts = y_values.value_counts(normalize=False)
            majority = float(counts.max())
            minority = float(counts.min())
            imbalance_ratio = majority / max(minority, 1.0)
            if imbalance_ratio > 2.5:
                n_estimators = max(n_estimators, 420)
                min_samples_leaf = 3
                class_weight = 'balanced'
                max_features = 'log2'
            elif imbalance_ratio > 1.5:
                n_estimators = max(n_estimators, 360)
                min_samples_leaf = 2
                class_weight = 'balanced'
                max_features = 'sqrt'

    if model_type == 'regression':
        return {
            'n_estimators': max(250, n_estimators),
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'max_features': max_features,
        }

    return {
        'n_estimators': n_estimators,
        'min_samples_leaf': min_samples_leaf,
        'class_weight': class_weight,
        'max_features': max_features,
    }


def _make_classifier(n_estimators: int = 350, y: pd.Series | None = None) -> Any:
    config = _resolve_ensemble_config('classification', y=y, default_n_estimators=n_estimators)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(
                    n_estimators=config['n_estimators'],
                    random_state=42,
                    class_weight=config['class_weight'],
                    min_samples_leaf=config['min_samples_leaf'],
                    max_features=config['max_features'],
                    n_jobs=-1,
                )),
                ('et', ExtraTreesClassifier(
                    n_estimators=max(220, config['n_estimators']),
                    random_state=42,
                    class_weight=config['class_weight'],
                    min_samples_leaf=config['min_samples_leaf'],
                    max_features=config['max_features'],
                    n_jobs=-1,
                )),
                ('hgb', HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    l2_regularization=0.05,
                )),
            ],
            voting='soft',
        )),
    ])


def _make_heartrisk_classifier(features: list[str]) -> Any:
    categorical_features = [
        feature for feature in features if feature in HEARTRISK_CATEGORICAL_FEATURES
    ]
    numeric_features = [
        feature for feature in features if feature not in categorical_features
    ]
    transformers = []
    if numeric_features:
        transformers.append((
            'numeric',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            'categorical',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]),
            categorical_features,
        ))
    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(
            max_iter=2000,
            C=0.3,
            class_weight='balanced',
            random_state=42,
        )),
    ])


def _make_bp_classifier(y: pd.Series | None = None) -> Any:
    del y
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=1500, C=0.03, class_weight=None, random_state=42)),
    ])


def benchmark_model_by_group(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    group_cols: list[str] | None = None,
    min_samples_per_group: int = 4,
) -> dict[str, list[dict[str, Hashable | float | int]]]:
    """Return simple accuracy and macro-recall metrics per segment for model improvement analysis."""
    if df is None or df.empty:
        return {}

    cols = [c for c in features if c in df.columns]
    if not cols or target not in df.columns:
        return {}

    group_cols = group_cols or []
    valid_groups = [col for col in group_cols if col in df.columns]
    if not valid_groups:
        return {"all": [{"accuracy_mean": 0.0, "recall_macro_mean": 0.0, "n": int(len(df))}]}

    results: dict[str, list[dict[str, Hashable | float | int]]] = {col: [] for col in valid_groups}
    for group_name, group_df in df.groupby(valid_groups, dropna=False):
        group_values = group_name if isinstance(group_name, tuple) else (group_name,)
        for col, value in zip(valid_groups, group_values):
            subset = group_df[cols + [target]].copy()
            subset = subset.apply(pd.to_numeric, errors='coerce').dropna(subset=[target])
            subset = subset[subset[target].isin([0, 1, 2])]
            if len(subset) < max(min_samples_per_group, 3):
                continue

            X = subset[cols]
            y = subset[target].astype(int)
            if y.nunique() < 2:
                continue

            if len(subset) >= 8 and y.value_counts().min() >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=42,
                    stratify=y,
                )
                model = _make_classifier(y=y_train)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                target_y, target_pred = y_test, y_pred
            else:
                model = _make_classifier(y=y)
                model.fit(X, y)
                y_pred = model.predict(X)
                target_y, target_pred = y, y_pred

            accuracy = float((target_y == target_pred).mean())
            recall_macro = float(recall_score(target_y, target_pred, average='macro', zero_division=0))
            results[col].append({
                'group_value': value,
                'accuracy_mean': accuracy,
                'recall_macro_mean': recall_macro,
                'n': int(len(subset)),
            })

    return {key: value for key, value in results.items() if value}


def _resolve_regression_config(default_n_estimators: int = 350) -> dict[str, Any]:
    """Prefer more stable, slightly more conservative tree ensembles for continuous targets."""
    n_estimators = max(240, int(default_n_estimators))
    learning_rate = 0.05
    max_depth = 6
    l2_regularization = 0.05

    if n_estimators >= 400:
        learning_rate = 0.04
        max_depth = 7
        l2_regularization = 0.08
    elif n_estimators >= 300:
        learning_rate = 0.05
        max_depth = 6
        l2_regularization = 0.05

    return {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'l2_regularization': l2_regularization,
    }


def _make_regressor(n_estimators: int = 350) -> Any:
    config = _resolve_regression_config(default_n_estimators=n_estimators)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(
                    n_estimators=config['n_estimators'],
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                    max_features='sqrt',
                )),
                ('et', ExtraTreesRegressor(
                    n_estimators=max(220, config['n_estimators']),
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                    max_features='sqrt',
                )),
                ('gbr', GradientBoostingRegressor(
                    n_estimators=max(150, config['n_estimators'] // 2),
                    learning_rate=config['learning_rate'],
                    random_state=42,
                )),
                ('hgr', HistGradientBoostingRegressor(
                    learning_rate=config['learning_rate'],
                    max_depth=config['max_depth'],
                    random_state=42,
                    l2_regularization=config['l2_regularization'],
                )),
            ],
        )),
    ])


def _make_lifestyle_regressor(n_estimators: int = 320) -> Any:
    config = _resolve_regression_config(default_n_estimators=n_estimators)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingRegressor(
            n_estimators=max(150, config['n_estimators'] // 2),
            learning_rate=config['learning_rate'],
            random_state=42,
        )),
    ])


def _apply_custom_thresholds(proba, thresholds, classes):
    """Helper om drempelwaarden toe te passen op kans-voorspellingen."""
    n_samples = proba.shape[0]
    preds = np.zeros(n_samples, dtype=int)
    
    # Haal drempels op (fallback naar 0.33 voor 3 klassen)
    t_mod = thresholds.get('moderate_threshold', 0.33)
    t_high = thresholds.get('high_threshold', 0.33)
    
    model_classes = list(classes)
    idx_mod = model_classes.index(1) if 1 in model_classes else -1
    idx_high = model_classes.index(2) if 2 in model_classes else -1
    
    for i in range(n_samples):
        p_mod = proba[i, idx_mod] if idx_mod != -1 else -1.0
        p_high = proba[i, idx_high] if idx_high != -1 else -1.0
        
        if p_high >= t_high:
            preds[i] = 2
        elif p_mod >= t_mod:
            preds[i] = 1
        else:
            preds[i] = 0
    return preds


def _prepare_regression_data(df: pd.DataFrame, target: str, features: list[str], min_rows: int = 80):
    if not HAS_SKLEARN:
        raise RuntimeError("Scikit-learn is niet beschikbaar.")
    if df is None or df.empty:
        raise ValueError("Geen data beschikbaar voor het ML model.")
    available = [c for c in features if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Doelkolom ontbreekt: {target}")
    if len(available) < 3:
        raise ValueError("Te weinig bruikbare features voor het ML model.")

    df_ml = df[available + [target]].apply(pd.to_numeric, errors='coerce').dropna(subset=[target])
    if 'participant_id' in df.columns:
        df_ml['participant_id'] = df.loc[df_ml.index, 'participant_id']
    if target == 'rec_ls_lifestyle_score':
        lower, upper = LIFESTYLE_SCORE_RANGE
        df_ml = df_ml.loc[df_ml[target].between(lower, upper)]
    if len(df_ml) < min_rows:
        raise ValueError("Te weinig rijen voor een betrouwbare regressie.")

    X = df_ml[available]
    y = df_ml[target].astype(float)
    groups = df_ml['participant_id'] if 'participant_id' in df_ml.columns else None
    X_train, X_test, y_train, y_test = _split_ml_data(X, y, groups=groups)
    return X_train, X_test, y_train, y_test, available


def _prepare_binary_data(df: pd.DataFrame, target: str, features: list[str], min_rows: int = 80):
    if not HAS_SKLEARN:
        raise RuntimeError("Scikit-learn is niet beschikbaar.")
    if df is None or df.empty:
        raise ValueError("Geen data beschikbaar voor het ML model.")
    available = [c for c in features if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Doelkolom ontbreekt: {target}")
    if len(available) < 3:
        raise ValueError("Te weinig bruikbare features voor het ML model.")

    df_ml = df[available + [target]].apply(pd.to_numeric, errors='coerce').dropna(subset=[target])
    if 'participant_id' in df.columns:
        df_ml['participant_id'] = df.loc[df_ml.index, 'participant_id']
    df_ml = df_ml[df_ml[target].isin([0, 1])]
    if len(df_ml) < min_rows or df_ml[target].nunique() < 2:
        raise ValueError("Te weinig rijen of targetvariatie voor classificatie.")

    X = df_ml[available]
    y = df_ml[target].astype(int)
    groups = df_ml['participant_id'] if 'participant_id' in df_ml.columns else None
    X_train, X_test, y_train, y_test = _split_ml_data(X, y, groups=groups, stratify=True)
    return X_train, X_test, y_train, y_test, available


def _prepare_bp_target(df: pd.DataFrame) -> pd.DataFrame:
    if 'rec_med_blood_pressure_cat' not in df.columns:
        raise ValueError("Doelkolom ontbreekt: rec_med_blood_pressure_cat")
    result = df.copy()
    categories = pd.to_numeric(result['rec_med_blood_pressure_cat'], errors='coerce')
    result[BP_TARGET] = categories.map({0: 0, 1: 1, 2: 1})
    return result


def _with_basic_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if 'postal_code' in result.columns:
        result['has_postal_code'] = result['postal_code'].notna().astype(int)
    elif 'has_postal_code' not in result.columns:
        result['has_postal_code'] = 0
    return result


def _normalize_for_model_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if s.notna().sum() == 0:
        return s

    low = float(s.quantile(0.05))
    high = float(s.quantile(0.95))
    if pd.isna(low) or pd.isna(high) or high <= low:
        max_value = s.max()
        if pd.isna(max_value) or max_value <= 0:
            return s
        return 1.0 + 9.0 * (s / max_value).clip(0, 1)

    scaled = (s - low) / (high - low)
    return 1.0 + 9.0 * scaled.clip(0, 1)


def _load_lifestyle_improvement_target(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from config import DB_URL
        from data_ingestion import load_factor_score_histories, load_table_from_database

        hist = load_factor_score_histories(DB_URL)
        if hist.empty:
            raise ValueError("factor_score_histories is leeg.")

        hist = hist.copy()
        if 'slug' not in hist.columns:
            qf = load_table_from_database('questionnaire_factors', DB_URL)
            if qf.empty or 'id' not in qf.columns or 'slug' not in qf.columns:
                raise ValueError("Kan score-slugs niet koppelen.")
            hist['questionnaire_factor_id'] = pd.to_numeric(hist['questionnaire_factor_id'], errors='coerce')
            qf['id'] = pd.to_numeric(qf['id'], errors='coerce')
            hist = hist.merge(qf[['id', 'slug']], left_on='questionnaire_factor_id', right_on='id', how='left')

        hist['participant_id'] = pd.to_numeric(hist['participant_id'], errors='coerce')
        hist['score_value'] = pd.to_numeric(hist['score_value'], errors='coerce')
        hist['completion_created_at'] = pd.to_datetime(hist['completion_created_at'], errors='coerce')
        hist = hist.dropna(subset=['participant_id', 'score_value', 'completion_created_at'])

        slug_candidates = {'rec_ls_lifestyle_score', 'ls_lifestyle_score', 'lifestyle_score', 'lifestyle'}
        hist_lifestyle = hist[hist['slug'].astype(str).isin(slug_candidates)].copy()
        if hist_lifestyle.empty:
            factor_map = {
                9: 'rec_ls_alcohol_total_per_week',
                12: 'rec_ls_exercise_physical_activity_minutes_total',
                13: 'rec_ls_nutrition_saturated_fat_per_day',
                14: 'rec_ls_nutrition_fruit_fruit_per_day',
                17: 'rec_ls_nutrition_natrium_per_day',
                18: 'rec_ls_sleep_psqi_sum',
                19: 'rec_ls_stress_sum',
                20: 'rec_ls_nutrition_sugar_per_day',
                21: 'rec_ls_vegetables_gram_per_day',
            }
            hist_components = hist.copy()
            hist_components['factor_name'] = pd.to_numeric(
                hist_components['questionnaire_factor_id'],
                errors='coerce',
            ).map(factor_map)
            hist_components = hist_components.dropna(subset=['factor_name'])
            if hist_components.empty:
                raise ValueError("Geen historische leefstijlscore of leefstijlcomponenten gevonden.")

            wide = hist_components.pivot_table(
                index=['participant_id', 'completion_created_at'],
                columns='factor_name',
                values='score_value',
                aggfunc='mean',
            ).reset_index()
            component_cols = [c for c in wide.columns if c in factor_map.values()]
            if not component_cols:
                raise ValueError("Geen historische leefstijlcomponenten gevonden.")

            normalized = wide[component_cols].apply(_normalize_for_model_score)
            wide['score_value'] = normalized.mean(axis=1)
            hist_lifestyle = wide[['participant_id', 'completion_created_at', 'score_value']].dropna(subset=['score_value'])

        hist = hist_lifestyle.copy()
        hist['score_value'] = _normalize_for_model_score(hist['score_value'])
        hist = hist.sort_values(['participant_id', 'completion_created_at'])

        grouped = hist.groupby('participant_id')
        first = grouped.first()[['score_value']].rename(columns={'score_value': 'first_lifestyle_score'})
        last = grouped.last()[['score_value']].rename(columns={'score_value': 'last_lifestyle_score'})
        counts = grouped.size().rename('n_lifestyle_measurements')
        target = first.join(last).join(counts).reset_index()
        target = target[target['n_lifestyle_measurements'] >= 2].copy()
        target['improvement_lifestyle_score'] = target['last_lifestyle_score'] - target['first_lifestyle_score']

        result = df.copy()
        if 'participant_id' in result.columns:
            result['participant_id'] = pd.to_numeric(result['participant_id'], errors='coerce')
            result = result.merge(
                target[['participant_id', 'first_lifestyle_score', 'improvement_lifestyle_score', 'n_lifestyle_measurements']],
                on='participant_id',
                how='left',
            )
        else:
            result['first_lifestyle_score'] = pd.NA
            result['improvement_lifestyle_score'] = pd.NA
            result['n_lifestyle_measurements'] = pd.NA
        return result
    except Exception as exc:
        logger.warning(f"Kon verbeterdoel niet maken: {exc}")
        result = df.copy()
        result['improvement_lifestyle_score'] = pd.NA
        result['n_lifestyle_measurements'] = pd.NA
        return result


def _build_next_lifestyle_training_data(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Turn ordered lifestyle measurements into one row per next-score prediction."""
    required = {'participant_id', 'completion_created_at', 'lifestyle_score'}
    if not required.issubset(snapshots.columns):
        missing = ', '.join(sorted(required - set(snapshots.columns)))
        raise ValueError(f"Historische leefstijlmetingen missen kolommen: {missing}")

    result = snapshots[['participant_id', 'completion_created_at', 'lifestyle_score']].copy()
    result['participant_id'] = pd.to_numeric(result['participant_id'], errors='coerce')
    result['completion_created_at'] = pd.to_datetime(result['completion_created_at'], errors='coerce')
    result['lifestyle_score'] = pd.to_numeric(result['lifestyle_score'], errors='coerce')
    lower, upper = LIFESTYLE_SCORE_RANGE
    result = result.dropna().loc[lambda frame: frame['lifestyle_score'].between(lower, upper)]
    result = result.drop_duplicates(['participant_id', 'completion_created_at'], keep='last')
    result = result.sort_values(['participant_id', 'completion_created_at']).reset_index(drop=True)

    grouped = result.groupby('participant_id', sort=False)
    result['current_lifestyle_score'] = result['lifestyle_score']
    result['previous_lifestyle_score'] = grouped['lifestyle_score'].shift(1)
    result['previous_lifestyle_improvement'] = (
        result['current_lifestyle_score'] - result['previous_lifestyle_score']
    )
    result['days_since_previous_measurement'] = (
        result['completion_created_at'] - grouped['completion_created_at'].shift(1)
    ).dt.total_seconds() / 86400.0
    result['measurement_number'] = grouped.cumcount() + 1
    result['next_lifestyle_score'] = grouped['lifestyle_score'].shift(-1)
    result['next_lifestyle_improvement'] = result['next_lifestyle_score'] - result['current_lifestyle_score']
    return result.dropna(subset=['next_lifestyle_score']).reset_index(drop=True)


def _load_next_lifestyle_training_data() -> pd.DataFrame:
    from config import DB_URL
    from data_ingestion import load_factor_score_histories, load_table_from_database

    hist = load_factor_score_histories(DB_URL).copy()
    if hist.empty:
        raise ValueError("factor_score_histories is leeg.")
    if 'slug' not in hist.columns:
        factors = load_table_from_database('questionnaire_factors', DB_URL)
        if factors.empty or not {'id', 'slug'}.issubset(factors.columns):
            raise ValueError("Kan historische score-slugs niet koppelen.")
        hist = hist.merge(
            factors[['id', 'slug']],
            left_on='questionnaire_factor_id',
            right_on='id',
            how='left',
        )

    snapshots = hist.loc[
        hist['slug'].astype(str).isin({'rec_ls_lifestyle_score', 'ls_lifestyle_score', 'lifestyle_score', 'lifestyle'}),
        ['participant_id', 'completion_created_at', 'score_value'],
    ].rename(columns={'score_value': 'lifestyle_score'})
    if snapshots.empty:
        factor_map = {
            9: 'rec_ls_alcohol_total_per_week',
            12: 'rec_ls_exercise_physical_activity_minutes_total',
            13: 'rec_ls_nutrition_saturated_fat_per_day',
            14: 'rec_ls_nutrition_fruit_fruit_per_day',
            17: 'rec_ls_nutrition_natrium_per_day',
            18: 'rec_ls_sleep_psqi_sum',
            19: 'rec_ls_stress_sum',
            20: 'rec_ls_nutrition_sugar_per_day',
            21: 'rec_ls_vegetables_gram_per_day',
        }
        components = hist.copy()
        components['factor_name'] = pd.to_numeric(
            components['questionnaire_factor_id'], errors='coerce'
        ).map(factor_map)
        components = components.dropna(subset=['factor_name'])
        if components.empty:
            raise ValueError("Geen historische, samengestelde leefstijlscores gevonden.")
        wide = components.pivot_table(
            index=['participant_id', 'completion_created_at'],
            columns='factor_name',
            values='score_value',
            aggfunc='mean',
        ).reset_index()
        component_cols = [column for column in wide.columns if column in factor_map.values()]
        if not component_cols:
            raise ValueError("Geen historische leefstijlcomponenten gevonden.")
        snapshots = wide[['participant_id', 'completion_created_at']].copy()
        snapshots['lifestyle_score'] = wide[component_cols].apply(_normalize_for_model_score).mean(axis=1)
    return _build_next_lifestyle_training_data(snapshots)


def _prepare_grouped_regression_data(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    group_col: str,
    min_rows: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    if group_col not in df.columns:
        raise ValueError(f"Groepskolom ontbreekt: {group_col}")
    available = [feature for feature in features if feature in df.columns]
    if len(available) < 2 or target not in df.columns:
        raise ValueError("Te weinig bruikbare features voor het toekomstige leefstijlmodel.")

    model_df = df[available + [target, group_col]].copy()
    model_df[available + [target]] = model_df[available + [target]].apply(pd.to_numeric, errors='coerce')
    model_df[group_col] = pd.to_numeric(model_df[group_col], errors='coerce')
    model_df = model_df.dropna(subset=[target, group_col])
    if len(model_df) < min_rows or model_df[group_col].nunique() < 2:
        raise ValueError("Te weinig historische vervolgmetingen voor een betrouwbaar toekomstmodel.")

    X = model_df[available]
    y = model_df[target].astype(float)
    groups = model_df[group_col]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], groups.iloc[train_idx], available


def _attach_grouped_regression_cv_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> None:
    n_splits = min(5, int(groups.nunique()))
    if n_splits < 2:
        model.cv_metrics = {}
        return
    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=GroupKFold(n_splits=n_splits),
        scoring={'r2': 'r2', 'neg_mean_absolute_error': 'neg_mean_absolute_error'},
        error_score='raise',
    )
    model.cv_metrics = {
        'r2_mean': float(scores['test_r2'].mean()),
        'r2_std': float(scores['test_r2'].std()),
        'mae_mean': float(-scores['test_neg_mean_absolute_error'].mean()),
        'mae_std': float(scores['test_neg_mean_absolute_error'].std()),
    }


def train_next_lifestyle_model(df: pd.DataFrame, force_retrain: bool = False):
    del df, force_retrain
    history_df = _load_next_lifestyle_training_data()
    X_train, X_test, y_train, y_test, train_groups, features = _prepare_grouped_regression_data(
        history_df,
        'next_lifestyle_score',
        NEXT_LIFESTYLE_FEATURES,
        group_col='participant_id',
    )
    model = _make_regressor(n_estimators=280)
    model.fit(X_train, y_train)
    _attach_grouped_regression_cv_metrics(model, X_train, y_train, train_groups)

    y_pred = np.clip(model.predict(X_test), *LIFESTYLE_SCORE_RANGE)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    baseline_pred = X_test['current_lifestyle_score']
    model.cv_metrics = {
        **(getattr(model, 'cv_metrics', {}) or {}),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred)),
        'baseline_mae': float(mean_absolute_error(y_test, baseline_pred)),
        'n_test': int(len(y_test)),
        'n_train_participants': int(train_groups.nunique()),
    }
    return model, X_test, y_test, features, y_train


def _load_purchase_target(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from config import DB_URL
        from data_ingestion import get_user_purchases

        purchases = get_user_purchases(db_url=DB_URL)
        result = df.copy()
        result['has_purchase'] = 0
        if purchases.empty:
            return result

        purchases = purchases.copy()
        if 'status' not in purchases.columns:
            raise ValueError("Aankoopdata mist de orderstatus; betaalde aankopen zijn niet vast te stellen.")
        status = purchases['status'].astype(str).str.strip().str.lower()
        purchases = purchases.loc[status.eq('paid')]

        if 'participant_id' in result.columns and 'participant_id' in purchases.columns:
            buyers = pd.to_numeric(purchases['participant_id'], errors='coerce').dropna().unique()
            result['has_purchase'] = pd.to_numeric(result['participant_id'], errors='coerce').isin(buyers).astype(int)
        elif 'user_id' in result.columns and 'user_id' in purchases.columns:
            buyers = pd.to_numeric(purchases['user_id'], errors='coerce').dropna().unique()
            result['has_purchase'] = pd.to_numeric(result['user_id'], errors='coerce').isin(buyers).astype(int)
        return result
    except Exception as exc:
        logger.warning(f"Kon aankoopdoel niet maken: {exc}")
        result = df.copy()
        result['has_purchase'] = 0
        return result


def _attach_cv_metrics(model: Any, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None) -> None:
    if y.nunique(dropna=True) <= 5:
        min_class = int(y.value_counts().min()) if not y.empty else 0
        if min_class < 3:
            model.cv_metrics = {}
            return

    y_numeric = pd.to_numeric(y, errors='coerce')
    if groups is not None and groups.notna().all() and groups.duplicated().any():
        use_group_cv = True
        n_splits = min(5, int(groups.nunique()))
        if n_splits < 2:
            model.cv_metrics = {}
            return
        cv_splitter = GroupKFold(n_splits=n_splits)
        cv_groups = groups
    else:
        use_group_cv = False
        cv_groups = None
        cv_splitter = min(5, int(y.value_counts().min()) if not y.empty else 5)
        cv_splitter = max(2, cv_splitter)

    def run_cross_validate(scoring: dict[str, Any]) -> dict[str, np.ndarray]:
        kwargs: dict[str, Any] = {'cv': cv_splitter, 'scoring': scoring, 'error_score': 'raise'}
        if cv_groups is not None:
            kwargs['groups'] = cv_groups
        return cross_validate(model, X, y, **kwargs)

    if y_numeric.nunique(dropna=True) <= 8 and y_numeric.dropna().between(0, 2).all():
        scoring: dict[str, Any] = {
            'accuracy': 'accuracy',
            'recall_macro': 'recall_macro',
        }
        is_binary_target = y_numeric.nunique(dropna=True) == 2 and y_numeric.dropna().isin([0, 1]).all()
        if is_binary_target:
            scoring.update({
                'average_precision': 'average_precision',
                'roc_auc': 'roc_auc',
                'neg_brier_score': 'neg_brier_score',
            })
        for class_label in (1, 2):
            if class_label not in y.unique():
                continue

            def class_recall_scorer(
                estimator: Any,
                X_validation: pd.DataFrame,
                y_validation: pd.Series,
                label: int = class_label,
            ) -> float:
                return float(_safe_recall_score(
                    y_validation,
                    estimator.predict(X_validation),
                    labels=[label],
                    average='macro',
                    zero_division=0,
                ))

            recall_key = 'recall_positive' if is_binary_target else f'recall_class_{class_label}'
            scoring[recall_key] = class_recall_scorer

        scores = run_cross_validate(scoring)
        model.cv_metrics = {
            'accuracy_mean': float(scores['test_accuracy'].mean()),
            'recall_macro': float(scores['test_recall_macro'].mean()),
            'recall_macro_mean': float(scores['test_recall_macro'].mean()),
            'recall_macro_std': float(scores['test_recall_macro'].std()),
        }
        if is_binary_target:
            model.cv_metrics.update({
                'average_precision_mean': float(scores['test_average_precision'].mean()),
                'average_precision_std': float(scores['test_average_precision'].std()),
                'roc_auc_mean': float(scores['test_roc_auc'].mean()),
                'brier_mean': float(-scores['test_neg_brier_score'].mean()),
                'positive_recall_mean': float(scores['test_recall_positive'].mean()),
                'positive_recall_std': float(scores['test_recall_positive'].std()),
            })
        elif 'test_recall_class_1' in scores:
            model.cv_metrics.update({
                'recall_moderate_mean': float(scores['test_recall_class_1'].mean()),
                'recall_moderate_std': float(scores['test_recall_class_1'].std()),
            })
        if 'test_recall_class_2' in scores:
            model.cv_metrics.update({
                'recall_high_mean': float(scores['test_recall_class_2'].mean()),
                'recall_high_std': float(scores['test_recall_class_2'].std()),
            })
        return

    if not use_group_cv:
        cv_splitter = KFold(n_splits=min(5, max(2, int(len(y) / 20))), shuffle=True, random_state=42)
    scores = run_cross_validate({'r2': 'r2', 'neg_mean_absolute_error': 'neg_mean_absolute_error'})
    model.cv_metrics = {
        'r2_mean': float(scores['test_r2'].mean()),
        'r2_std': float(scores['test_r2'].std()),
        'mae_mean': float(-scores['test_neg_mean_absolute_error'].mean()),
        'mae_std': float(scores['test_neg_mean_absolute_error'].std()),
    }


def train_bp_model(df: pd.DataFrame, force_retrain: bool = False):
    df_model = _prepare_bp_target(df)
    X_train, X_test, y_train, y_test, features = _prepare_binary_data(
        df_model,
        BP_TARGET,
        BP_FEATURES,
        min_rows=50,
    )
    train_groups = _groups_for_rows(df_model, X_train.index)
    model = _make_bp_classifier(y_train).fit(X_train, y_train)
    model.decision_threshold = BP_SCREENING_THRESHOLD
    _attach_cv_metrics(model, X_train, y_train, groups=train_groups)

    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score
    positive_index = list(model.classes_).index(1)
    y_probability = model.predict_proba(X_test)[:, positive_index]
    y_prediction = (y_probability >= model.decision_threshold).astype(int)
    model.cv_metrics.update({
        'decision_threshold': float(model.decision_threshold),
        'threshold_policy': 'fixed_high_recall_screening',
        'holdout_average_precision': float(average_precision_score(y_test, y_probability)),
        'holdout_brier': float(brier_score_loss(y_test, y_probability)),
        'holdout_precision': float(precision_score(y_test, y_prediction, zero_division=0)),
        'holdout_recall': float(recall_score(y_test, y_prediction, zero_division=0)),
        'holdout_f1': float(f1_score(y_test, y_prediction, zero_division=0)),
        'holdout_alert_rate': float(np.mean(y_prediction)),
    })

    return model, X_test, y_test, features, y_train


def train_lifestyle_model(df: pd.DataFrame, force_retrain: bool = False):
    X_train, X_test, y_train, y_test, features = _prepare_regression_data(
        df,
        'rec_ls_lifestyle_score',
        LIFESTYLE_FEATURES,
        min_rows=80,
    )
    model = _make_lifestyle_regressor(n_estimators=320)
    model.fit(X_train, y_train)
    _attach_cv_metrics(model, X_train, y_train, groups=_groups_for_rows(df, X_train.index))

    y_pred = np.clip(model.predict(X_test), *LIFESTYLE_SCORE_RANGE)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    model.cv_metrics = {
        **(getattr(model, 'cv_metrics', {}) or {}),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred)),
        'n_test': int(len(y_test)),
    }
    return model, X_test, y_test, features, y_train


def train_dropoff_model(df: pd.DataFrame, force_retrain: bool = False):
    df_model = _with_basic_ml_features(df)
    score_col = df_model.get('rec_ls_lifestyle_score')
    if score_col is None:
        df_model['dropoff_risk'] = 0
    else:
        df_model['dropoff_risk'] = pd.to_numeric(score_col, errors='coerce').isna().astype(int)
    X_train, X_test, y_train, y_test, features = _prepare_binary_data(
        df_model,
        'dropoff_risk',
        ACTIVATION_FEATURES,
        min_rows=120,
    )
    model = _make_classifier(n_estimators=240, y=y_train)
    model.fit(X_train, y_train)
    _attach_cv_metrics(model, X_train, y_train, groups=_groups_for_rows(df_model, X_train.index))
    return model, X_test, y_test, features, y_train


def train_improvement_model(df: pd.DataFrame, force_retrain: bool = False):
    del df, force_retrain
    history_df = _load_next_lifestyle_training_data()
    X_train, X_test, y_train, y_test, train_groups, features = _prepare_grouped_regression_data(
        history_df,
        'next_lifestyle_improvement',
        IMPROVEMENT_FEATURES,
        group_col='participant_id',
    )
    model = _make_regressor(n_estimators=280)
    model.fit(X_train, y_train)
    _attach_grouped_regression_cv_metrics(model, X_train, y_train, train_groups)
    from sklearn.metrics import mean_absolute_error, r2_score
    model.cv_metrics.update({
        'holdout_mae': float(mean_absolute_error(y_test, model.predict(X_test))),
        'holdout_baseline_mae': float(mean_absolute_error(y_test, np.zeros(len(y_test)))),
        'holdout_r2': float(r2_score(y_test, model.predict(X_test))),
    })

    # Bewaar naast de omvang ook de richting van de scoreverandering.
    direction_y = pd.Series(
        np.select(
            [y_train < -0.25, y_train > 0.25],
            [0, 2],
            default=1,
        ),
        index=y_train.index,
        dtype=int,
    )
    if direction_y.nunique() >= 2:
        direction_model = _make_classifier(n_estimators=280, y=direction_y)
        direction_model.fit(X_train, direction_y)
        _attach_cv_metrics(direction_model, X_train, direction_y, groups=train_groups)
        model.direction_model = direction_model
        model.direction_labels = {
            0: 'Verslechtering',
            1: 'Stabiel',
            2: 'Verbetering',
        }
    else:
        model.direction_model = None
        model.direction_labels = {}
    return model, X_test, y_test, features, y_train


def train_purchase_model(df: pd.DataFrame, force_retrain: bool = False):
    df_model = _load_purchase_target(df)
    X_train, X_test, y_train, y_test, features = _prepare_binary_data(
        df_model,
        'has_purchase',
        PURCHASE_FEATURES,
        min_rows=120,
    )
    train_groups = _groups_for_rows(df_model, X_train.index)
    X_fit, X_validation, y_fit, y_validation = _split_ml_data(
        X_train,
        y_train,
        groups=train_groups,
        test_size=0.2,
        stratify=True,
    )
    validation_model = _make_classifier(n_estimators=240, y=y_fit)
    validation_model.fit(X_fit, y_fit)
    threshold_metrics = _find_best_binary_f1_threshold(
        y_validation,
        validation_model.predict_proba(X_validation)[:, list(validation_model.classes_).index(1)],
    )

    model = _make_classifier(n_estimators=240, y=y_train)
    model.fit(X_train, y_train)
    model.decision_threshold = threshold_metrics['threshold']
    _attach_cv_metrics(model, X_train, y_train, groups=train_groups)

    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score
    y_probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    y_prediction = (y_probability >= model.decision_threshold).astype(int)
    model.cv_metrics.update({
        'decision_threshold': float(model.decision_threshold),
        'validation_f1': float(threshold_metrics['f1']),
        'holdout_average_precision': float(average_precision_score(y_test, y_probability)),
        'holdout_brier': float(brier_score_loss(y_test, y_probability)),
        'holdout_precision': float(precision_score(y_test, y_prediction, zero_division=0)),
        'holdout_recall': float(recall_score(y_test, y_prediction, zero_division=0)),
        'holdout_f1': float(f1_score(y_test, y_prediction, zero_division=0)),
    })
    return model, X_test, y_test, features, y_train


def _align_input_features(model, input_df: pd.DataFrame) -> pd.DataFrame:
    """Align input columns to the trained feature order to avoid sklearn feature-name mismatches."""
    if input_df is None:
        raise ValueError("Geen invoerdata beschikbaar voor leefstijlvoorspelling.")

    candidate_features = None
    if hasattr(model, 'feature_names_in_'):
        candidate_features = list(model.feature_names_in_)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', model), 'feature_names_in_'):
        candidate_features = list(model.named_steps['model'].feature_names_in_)

    if candidate_features:
        aligned = input_df.copy()
        aligned = aligned.reindex(columns=candidate_features, fill_value=np.nan)
        return aligned
    return input_df


def predict_lifestyle(model, input_df: pd.DataFrame):
    aligned_df = _align_input_features(model, input_df)
    return np.clip(model.predict(aligned_df), *LIFESTYLE_SCORE_RANGE)


def predict_binary_model(model, input_df: pd.DataFrame):
    aligned_df = _align_input_features(model, input_df)
    proba = model.predict_proba(aligned_df)
    if hasattr(model, 'decision_threshold') and 1 in model.classes_:
        positive_index = list(model.classes_).index(1)
        predictions = (proba[:, positive_index] >= model.decision_threshold).astype(int)
        return predictions, proba
    return model.predict(aligned_df), proba


def _find_best_binary_f1_threshold(y_true: pd.Series, positive_probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, positive_probability)
    if len(thresholds) == 0:
        return {'threshold': 0.5, 'f1': 0.0}
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.argmax(f1_values))
    return {
        'threshold': float(thresholds[best_index]),
        'f1': float(f1_values[best_index]),
    }


def train_heartrisk_model(df: pd.DataFrame, force_retrain: bool = False):
    del force_retrain
    X_train, X_test, y_train, y_test, features = _prepare_classification_data(
        df,
        'rec_heartrisk_cat',
        HEARTRISK_FEATURES,
    )
    train_groups = _groups_for_rows(df, X_train.index)
    model = _make_heartrisk_classifier(features)
    model.fit(X_train, y_train)
    _attach_cv_metrics(model, X_train, y_train, groups=train_groups)

    y_pred_proba_test = model.predict_proba(X_test)
    y_pred_hr_test = model.predict(X_test)
    class_labels = sorted(y_test.unique())
    rec_moderate = _safe_recall_score(
        (y_test == 1).astype(int), (y_pred_hr_test == 1).astype(int), zero_division=0
    ) if 1 in class_labels else 0.0
    rec_high = _safe_recall_score(
        (y_test == 2).astype(int), (y_pred_hr_test == 2).astype(int), zero_division=0
    ) if 2 in class_labels else 0.0

    model = _ensure_model_metadata(model)

    from sklearn.metrics import brier_score_loss
    high_class_index = list(model.classes_).index(2) if 2 in model.classes_ else None
    model.cv_metrics.update({
        'holdout_recall_moderate': float(rec_moderate),
        'holdout_recall_high': float(rec_high),
        'holdout_brier_high': float(brier_score_loss(
            (y_test == 2).astype(int), y_pred_proba_test[:, high_class_index]
        )) if high_class_index is not None else 0.0,
    })
    return model, X_test, y_test, features, y_train


def predict_bp(model, input_df: pd.DataFrame):
    if hasattr(model, 'decision_threshold'):
        return predict_binary_model(model, input_df)
    aligned_df = _align_input_features(model, input_df)
    proba = model.predict_proba(aligned_df)
    if hasattr(model, 'best_thresholds'):
        preds = _apply_custom_thresholds(proba, model.best_thresholds, model.classes_)
        return preds, proba
    return model.predict(aligned_df), proba


def predict_heartrisk(model, input_df: pd.DataFrame):
    aligned_df = _align_input_features(model, input_df)
    proba = model.predict_proba(aligned_df)
    if hasattr(model, 'best_thresholds'):
        preds = _apply_custom_thresholds(proba, model.best_thresholds, model.classes_)
        return preds, proba
    return model.predict(aligned_df), proba


def _extract_feature_importances(model) -> np.ndarray | None:
    """Return tree importances or mean absolute coefficients for the fitted model."""
    estimator = model.named_steps.get('model') if hasattr(model, 'named_steps') else model

    direct = getattr(estimator, 'feature_importances_', None)
    if direct is not None:
        return np.asarray(direct, dtype=float)

    coefficients = getattr(estimator, 'coef_', None)
    if coefficients is not None:
        coefficient_array = np.asarray(coefficients, dtype=float)
        transformed_importances = (
            np.abs(coefficient_array)
            .reshape(-1, coefficient_array.shape[-1])
            .mean(axis=0)
        )
        preprocessor = (
            model.named_steps.get('preprocess')
            if hasattr(model, 'named_steps')
            else None
        )
        input_features = list(getattr(model, 'feature_names_in_', []))
        if (
            preprocessor is None
            or len(input_features) == len(transformed_importances)
            or not hasattr(preprocessor, 'get_feature_names_out')
        ):
            return transformed_importances

        transformed_features = [
            name.split('__', 1)[-1]
            for name in preprocessor.get_feature_names_out()
        ]
        aggregated = []
        for feature in input_features:
            indices = [
                index
                for index, transformed_feature in enumerate(transformed_features)
                if transformed_feature == feature
                or transformed_feature.startswith(f'{feature}_')
            ]
            if not indices:
                return None
            aggregated.append(float(transformed_importances[indices].mean()))
        return np.asarray(aggregated)

    estimators = getattr(estimator, 'estimators_', None) or getattr(estimator, 'named_estimators_', None)
    if isinstance(estimators, dict):
        estimators = list(estimators.values())

    if estimators:
        importances = []
        for est in estimators:
            imp = getattr(est, 'feature_importances_', None)
            if imp is not None:
                importances.append(np.asarray(imp, dtype=float))
        if importances:
            return np.mean(np.vstack(importances), axis=0)

    return None


def plot_feature_importance(model, features: list[str], titel: str = "Model") -> go.Figure:
    importances = _extract_feature_importances(model)
    if importances is None or len(importances) != len(features):
        return _empty_fig("Geen feature importance beschikbaar voor dit model.")
    labels = [FEATURE_LABELS.get(f, f) for f in features]
    df_imp = pd.DataFrame({'Feature': labels, 'Belangrijkheid': importances}).sort_values('Belangrijkheid')
    fig = px.bar(
        df_imp,
        x='Belangrijkheid',
        y='Feature',
        orientation='h',
        title=f'Belangrijkste voorspellers - {titel}',
        color_discrete_sequence=[HOOFD_KLEUR],
    )
    fig.update_layout(template='plotly_white', height=480, yaxis_title='')
    return fig


def plot_confusion_matrix(y_true, y_pred, labels: list[int] | None = None) -> go.Figure:
    labels = labels or [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = {0: 'Laag/Normaal', 1: 'Matig/Verhoogd', 2: 'Hoog'}
    if labels == [0, 1]:
        label_names[0] = 'Normaal'
        label_names[1] = 'Verhoogd/hoog'
    display_labels = [label_names.get(label, str(label)) for label in labels]
    fig = px.imshow(
        cm,
        x=display_labels,
        y=display_labels,
        text_auto=True,
        color_continuous_scale='Blues',
        labels={'x': 'Voorspeld', 'y': 'Werkelijk', 'color': 'Aantal'},
        title='Confusion matrix',
    )
    fig.update_layout(template='plotly_white', height=420)
    return fig


def plot_prediction_performance_regression(y_true, y_pred) -> go.Figure:
    """Visualiseert regressievoorspellingen tegen de werkelijke waarden."""
    y_true = pd.to_numeric(pd.Series(y_true), errors='coerce').reset_index(drop=True)
    y_pred = pd.to_numeric(pd.Series(y_pred), errors='coerce').reset_index(drop=True)

    if y_true.empty or y_pred.empty or len(y_true) != len(y_pred):
        return _empty_fig("Geen bruikbare regressie-data voor de voorspellinggrafiek.")

    aligned = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    if aligned.empty:
        return _empty_fig("Geen bruikbare regressie-data voor de voorspellinggrafiek.")

    fig = px.scatter(
        aligned,
        x='y_true',
        y='y_pred',
        opacity=0.7,
        title='Werkelijke vs. voorspelde waarden',
        labels={'y_true': 'Werkelijke waarde', 'y_pred': 'Voorspelde waarde'},
        color_discrete_sequence=[HOOFD_KLEUR],
    )
    min_val = min(aligned['y_true'].min(), aligned['y_pred'].min())
    max_val = max(aligned['y_true'].max(), aligned['y_pred'].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Ideale lijn',
        line=dict(color='red', dash='dash'),
    ))
    fig.update_layout(template='plotly_white', height=420, showlegend=True)
    return fig


def plot_precision_recall_curve(y_true, y_pred_proba) -> go.Figure:
    fig = go.Figure()
    observed_classes = sorted(pd.Series(y_true).dropna().unique())
    if len(observed_classes) == 2:
        positive_class = 1 if 1 in observed_classes else observed_classes[-1]
        y_bin = (np.asarray(y_true) == positive_class).astype(int)
        probability = np.asarray(y_pred_proba)
        positive_index = observed_classes.index(positive_class)
        precision, recall, _ = precision_recall_curve(y_bin, probability[:, positive_index])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Verhoogd/hoog'))
        fig.update_layout(
            template='plotly_white',
            title='Precision-recall curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=420,
        )
        return fig

    classes = [0, 1, 2]
    y_bin = np.asarray(label_binarize(y_true, classes=classes), dtype=int)
    labels = {0: 'Laag', 1: 'Matig', 2: 'Hoog'}
    for idx, cls in enumerate(classes):
        if y_bin[:, idx].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], np.asarray(y_pred_proba)[:, idx])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=labels[cls]))
    fig.update_layout(
        template='plotly_white',
        title='Precision-recall curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=420,
    )
    return fig


def plot_local_shap(model_data: dict, input_df: pd.DataFrame, titel: str = "Model") -> go.Figure:
    model = model_data.get('model') if isinstance(model_data, dict) else None
    features = model_data.get('features', list(input_df.columns)) if isinstance(model_data, dict) else list(input_df.columns)
    importances = _extract_feature_importances(model)
    if importances is None or len(importances) != len(features):
        return _empty_fig("Geen lokale uitleg beschikbaar.")

    row = input_df.reindex(columns=features)
    baseline = pd.DataFrame(model_data.get('X_test', row)).reindex(columns=features)
    baseline_mean = baseline.apply(pd.to_numeric, errors='coerce').mean()
    values = row.iloc[0].apply(pd.to_numeric, errors='coerce')
    signed = (values - baseline_mean).fillna(0) * importances
    df_local = pd.DataFrame({
        'Feature': [FEATURE_LABELS.get(f, f) for f in features],
        'Invloed': signed.values,
    }).sort_values('Invloed')
    fig = px.bar(df_local, x='Invloed', y='Feature', orientation='h', title=f'Lokale uitleg - {titel}')
    fig.update_layout(template='plotly_white', height=420, yaxis_title='')
    return fig


def maak_gebruikers_segmentatie_plot(df: pd.DataFrame, lang: str = 'nl') -> go.Figure:
    if not HAS_SKLEARN:
        return _empty_fig(tr("Scikit-learn is niet beschikbaar.", lang))
    features = [
        'rec_ls_lifestyle_score',
        'rec_med_bmi',
        'rec_ls_stress_sum',
        'rec_wellbeing_score',
        'rec_resilience_score',
        'rec_ls_sleep_psqi_sum',
        'rec_dass_stress_score',
        'rec_dass_anxiety_score',
        'rec_dass_depression_score',
        'rec_self_efficacy_score',
        'rec_asr_wai_score',
    ]
    available = [c for c in features if c in df.columns and df[c].notna().any()]
    if len(available) < 3:
        return _empty_fig(tr("Onvoldoende data voor gebruikerssegmentatie.", lang))
    df_seg = df[available].apply(pd.to_numeric, errors='coerce')
    df_seg = df_seg.dropna(how='all')
    if len(df_seg) < 10:
        return _empty_fig(tr("Te weinig datapunten voor gebruikerssegmentatie.", lang))

    # Gebruik beschikbare profielinformatie per gebruiker; zo verdwijnen gebruikers
    # met slechts enkele ontbrekende vragen niet volledig uit de visualisatie.
    df_seg = df_seg.fillna(df_seg.median()).dropna(axis=1, how='all')
    if len(df_seg.columns) < 3:
        return _empty_fig(tr("Onvoldoende data voor gebruikerssegmentatie.", lang))

    plot_df = df_seg.copy()
    x_col = 'rec_ls_lifestyle_score' if 'rec_ls_lifestyle_score' in plot_df.columns else available[0]
    y_col = 'rec_med_bmi' if 'rec_med_bmi' in plot_df.columns else available[1]
    size_col = 'rec_ls_stress_sum' if 'rec_ls_stress_sum' in plot_df.columns else None
    if x_col == 'rec_ls_lifestyle_score':
        # De leefstijlscore hoort voor visualisaties op een 0-10-schaal te staan.
        plot_df[x_col] = plot_df[x_col].clip(lower=0, upper=10)

    # Cluster op de twee zichtbare assen, zodat de kleuren direct interpreteerbaar zijn.
    cluster_input = plot_df[[x_col, y_col]]
    scaled = StandardScaler().fit_transform(cluster_input)
    n_clusters = min(3, len(plot_df))
    clusters = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit_predict(scaled)
    plot_df['Segment'] = [f"Profielgroep {c + 1}" for c in clusters]
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color='Segment',
        size=size_col,
        opacity=0.65,
        title=tr('Gebruikerssegmenten op basis van gezondheidsprofiel', lang),
        labels={x_col: tr(FEATURE_LABELS.get(x_col, x_col), lang), y_col: tr(FEATURE_LABELS.get(y_col, y_col), lang)},
        hover_data={
            column: ':.1f'
            for column in available
            if column not in {x_col, y_col}
        },
    )
    fig.update_layout(
        template='plotly_white',
        height=520,
        legend_title_text=tr('Kleur = vergelijkbare profielgroep', lang),
    )
    return fig

def maak_feature_importance_plot(df: pd.DataFrame, lang: str = 'nl') -> go.Figure:
    """
    Visualiseert welke factoren de leefstijlscore het meest beïnvloeden
    met behulp van een Random Forest Regressor.
    """
    if not HAS_SKLEARN:
        fig = go.Figure()
        fig.add_annotation(
            text=tr("Machine Learning module (scikit-learn) is niet beschikbaar in deze omgeving.", lang),
            showarrow=False, font=dict(size=14)
        )
        fig.update_layout(template='plotly_white', height=400)
        return fig

    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text=tr("Geen data beschikbaar.", lang), showarrow=False)
        return fig

    # Mapping van interne namen naar leesbare labels voor de grafiek
    feature_labels = {
        'rec_age_current': tr('Leeftijd', lang),
        'rec_ls_score_fruit': tr('Fruit', lang),
        'rec_ls_score_vegetables': tr('Groenten', lang),
        'rec_ls_score_sugar': tr('Suiker', lang),
        'rec_ls_score_saturated_fat': tr('Verzadigd vet', lang),
        'rec_ls_score_alcohol': tr('Alcohol', lang),
        'rec_ls_score_natrium': tr('Zout', lang),
        'rec_ls_stress_sum': tr('Stress', lang),
        'rec_ls_sleep_psqi_sum': tr('Slaap', lang),
    }
    
    target = 'rec_ls_lifestyle_score'
    
    # Alleen de kolommen gebruiken die daadwerkelijk in de dataframe zitten
    available_features = [c for c in feature_labels.keys() if c in df.columns]
    
    if target not in df.columns or len(available_features) < 3:
        fig = go.Figure()
        fig.add_annotation(text=tr("Onvoldoende data voor machine learning analyse.", lang), showarrow=False)
        return fig

    # Data voorbereiden: numeriek maken en rijen met missende waarden verwijderen
    df_ml = df[available_features + [target]].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_ml) < 50:
        fig = go.Figure()
        fig.add_annotation(text=tr("Te weinig datapunten voor een betrouwbare analyse.", lang), showarrow=False)
        return fig

    X = df_ml[available_features]
    y = df_ml[target]

    # Model trainen
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Importance data verzamelen en sorteren
    importance_df = pd.DataFrame({
        'Feature': [feature_labels[c] for c in available_features],
        'Belangrijkheid': model.feature_importances_
    }).sort_values('Belangrijkheid', ascending=True)

    fig = px.bar(
        importance_df, 
        x='Belangrijkheid', 
        y='Feature',
        orientation='h',
        title=tr('Impact van factoren op Leefstijlscore', lang),
        labels={'Belangrijkheid': tr('Relatieve impact', lang), 'Feature': ''},
        color_discrete_sequence=[HOOFD_KLEUR]
    )
    
    fig.update_layout(
        template='plotly_white',
        height=500,
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=False)
    )
    
    return fig