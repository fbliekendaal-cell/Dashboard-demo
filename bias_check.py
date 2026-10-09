import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple

# Importeer de ML-module
import ML

def _get_model_predictions(model: Pipeline, X: pd.DataFrame, y_true: pd.Series, model_type: str) -> np.ndarray:
    """Helper om voorspellingen te krijgen, rekening houdend met modeltype en thresholds."""
    if model_type == "classification":
        if hasattr(model, 'decision_threshold') and hasattr(ML, 'predict_binary_model'):
            preds, _ = ML.predict_binary_model(model, X)
        elif hasattr(model, 'best_thresholds') and hasattr(ML, 'predict_bp'):
            preds, _ = ML.predict_bp(model, X)
        else: # Algemene classificatie (bijv. lifestyle classificatie)
            preds = model.predict(X)
    else: # Regression
        preds = model.predict(X)
    return preds

def perform_bias_check_classification(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_attribute: str,
    sensitive_groups: Dict[Any, str],
    model_name: str,
) -> pd.DataFrame:
    """Voert een bias-check uit voor een classificatiemodel."""
    results = []
    
    # Zorg dat de gevoelige attribuutkolom numeriek is en in X_test staat
    if sensitive_attribute not in X_test.columns:
        print(f"Waarschuwing: Gevoelige attribuut '{sensitive_attribute}' niet gevonden in X_test voor {model_name}.")
        return pd.DataFrame()
    
    X_test_with_sensitive = X_test.copy()
    X_test_with_sensitive[sensitive_attribute] = pd.to_numeric(X_test_with_sensitive[sensitive_attribute], errors='coerce')

    # Haal alle unieke klassen op uit de testset
    all_classes = sorted(y_test.unique())

    for group_value, group_label in sensitive_groups.items():
        group_mask = X_test_with_sensitive[sensitive_attribute] == group_value
        X_group = X_test[group_mask]
        y_group = y_test[group_mask]

        if len(y_group) == 0:
            print(f"Geen data voor groep '{group_label}' in {model_name}.")
            continue

        preds_group = _get_model_predictions(model, X_group, y_group, "classification")

        row_data = {
            "Model": model_name,
            "Attribuut": sensitive_attribute,
            "Groep": group_label,
            "N": len(y_group),
            "Accuracy": accuracy_score(y_group, preds_group),
            "Recall (macro)": recall_score(y_group, preds_group, average="macro", zero_division=0),
            "Precision (macro)": precision_score(y_group, preds_group, average="macro", zero_division=0),
            "F1-score (macro)": f1_score(y_group, preds_group, average="macro", zero_division=0),
        }
        
        # Recall en Precision per klasse
        for cls in all_classes:
            # Maak binaire labels voor de huidige klasse
            y_true_binary = (y_group == cls).astype(int)
            y_pred_binary = (preds_group == cls).astype(int)
            
            recall_cls = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            precision_cls = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            
            row_data[f"Recall (klasse {cls})"] = recall_cls
            row_data[f"Precision (klasse {cls})"] = precision_cls

        results.append(row_data)

    return pd.DataFrame(results).round(3)

def perform_bias_check_regression(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_attribute: str,
    sensitive_groups: Dict[Any, str],
    model_name: str,
) -> pd.DataFrame:
    """Voert een bias-check uit voor een regressiemodel."""
    results = []

    if sensitive_attribute not in X_test.columns:
        print(f"Waarschuwing: Gevoelige attribuut '{sensitive_attribute}' niet gevonden in X_test voor {model_name}.")
        return pd.DataFrame()
    
    X_test_with_sensitive = X_test.copy()
    X_test_with_sensitive[sensitive_attribute] = pd.to_numeric(X_test_with_sensitive[sensitive_attribute], errors='coerce')

    for group_value, group_label in sensitive_groups.items():
        group_mask = X_test_with_sensitive[sensitive_attribute] == group_value
        X_group = X_test[group_mask]
        y_group = y_test[group_mask]

        if len(y_group) == 0:
            print(f"Geen data voor groep '{group_label}' in {model_name}.")
            continue

        preds_group = _get_model_predictions(model, X_group, y_group, "regression")

        results.append(
            {
                "Model": model_name,
                "Attribuut": sensitive_attribute,
                "Groep": group_label,
                "N": len(y_group),
                "MAE": mean_absolute_error(y_group, preds_group),
                "RMSE": np.sqrt(mean_squared_error(y_group, preds_group)),
                "R2-score": r2_score(y_group, preds_group),
            }
        )

    return pd.DataFrame(results).round(3)

def main():
    print("--- Starten van Bias Check ---")

    # Laad de volledige dataset
    df_raw = ML.load_data()
    print(f"Totaal aantal records geladen: {len(df_raw)}")

    # Definieer gevoelige attributen en hun groepen
    sensitive_attributes = {
        "rec_user_gender": {0: "Vrouw", 1: "Man"},
        "rec_age_current": {
            (0, 30): "Jong (<30)",
            (30, 50): "Middel (30-50)",
            (50, 100): "Oud (>50)",
        },
    }

    all_bias_results = []

    # --- Leefstijlmodel ---
    print("\n--- Analyseren Leefstijlmodel ---")
    try:
        lifestyle_model, X_test_ls, y_test_ls, _, _ = ML.train_lifestyle_model(df_raw)
        
        # Voeg gevoelige attributen toe aan X_test_ls voor filtering
        X_test_ls_full = X_test_ls.copy()
        for attr in sensitive_attributes.keys():
            if attr in df_raw.columns:
                X_test_ls_full[attr] = pd.to_numeric(df_raw.loc[X_test_ls.index, attr], errors='coerce')
            else:
                print(f"Waarschuwing: '{attr}' niet gevonden in ruwe data voor Leefstijlmodel.")

        if ML.LIFESTYLE_USE_CLASSIFICATION:
            print("Leefstijlmodel is een classificatiemodel.")
            for attr, groups in sensitive_attributes.items():
                if attr == "rec_age_current":
                    # Discretiseer leeftijd voor de bias check
                    age_bins = [g[0] for g in groups.keys()] + [groups[list(groups.keys())[-1]][-1]]
                    age_labels = list(groups.values())
                    X_test_ls_full['age_group'] = pd.cut(X_test_ls_full[attr], bins=age_bins, labels=age_labels, right=False)
                    age_groups_map = {label: label for label in age_labels} # Map labels to themselves
                    results = perform_bias_check_classification(
                        lifestyle_model, X_test_ls_full, y_test_ls, 'age_group', age_groups_map, "Leefstijl"
                    )
                else:
                    results = perform_bias_check_classification(
                        lifestyle_model, X_test_ls_full, y_test_ls, attr, groups, "Leefstijl"
                    )
                if not results.empty:
                    all_bias_results.append(results)
        else:
            print("Leefstijlmodel is een regressiemodel.")
            for attr, groups in sensitive_attributes.items():
                if attr == "rec_age_current":
                    age_bins = [g[0] for g in groups.keys()] + [groups[list(groups.keys())[-1]][-1]]
                    age_labels = list(groups.values())
                    X_test_ls_full['age_group'] = pd.cut(X_test_ls_full[attr], bins=age_bins, labels=age_labels, right=False)
                    age_groups_map = {label: label for label in age_labels}
                    results = perform_bias_check_regression(
                        lifestyle_model, X_test_ls_full, y_test_ls, 'age_group', age_groups_map, "Leefstijl"
                    )
                else:
                    results = perform_bias_check_regression(
                        lifestyle_model, X_test_ls_full, y_test_ls, attr, groups, "Leefstijl"
                    )
                if not results.empty:
                    all_bias_results.append(results)
    except Exception as e:
        print(f"Fout bij analyse Leefstijlmodel: {e}")

    # --- Bloeddrukmodel ---
    print("\n--- Analyseren Bloeddrukmodel ---")
    try:
        bp_model, X_test_bp, y_test_bp, _, _ = ML.train_bp_model(df_raw)
        
        X_test_bp_full = X_test_bp.copy()
        for attr in sensitive_attributes.keys():
            if attr in df_raw.columns:
                X_test_bp_full[attr] = pd.to_numeric(df_raw.loc[X_test_bp.index, attr], errors='coerce')
            else:
                print(f"Waarschuwing: '{attr}' niet gevonden in ruwe data voor Bloeddrukmodel.")

        for attr, groups in sensitive_attributes.items():
            if attr == "rec_age_current":
                age_bins = [g[0] for g in groups.keys()] + [groups[list(groups.keys())[-1]][-1]]
                age_labels = list(groups.values())
                X_test_bp_full['age_group'] = pd.cut(X_test_bp_full[attr], bins=age_bins, labels=age_labels, right=False)
                age_groups_map = {label: label for label in age_labels}
                results = perform_bias_check_classification(
                    bp_model, X_test_bp_full, y_test_bp, 'age_group', age_groups_map, "Bloeddruk"
                )
            else:
                results = perform_bias_check_classification(
                    bp_model, X_test_bp_full, y_test_bp, attr, groups, "Bloeddruk"
                )
            if not results.empty:
                all_bias_results.append(results)
    except Exception as e:
        print(f"Fout bij analyse Bloeddrukmodel: {e}")

    # --- Hartrisicomodel ---
    print("\n--- Analyseren Hartrisicomodel ---")
    try:
        heartrisk_model, X_test_hr, y_test_hr, _, _ = ML.train_heartrisk_model(df_raw)
        
        X_test_hr_full = X_test_hr.copy()
        for attr in sensitive_attributes.keys():
            if attr in df_raw.columns:
                X_test_hr_full[attr] = pd.to_numeric(df_raw.loc[X_test_hr.index, attr], errors='coerce')
            else:
                print(f"Waarschuwing: '{attr}' niet gevonden in ruwe data voor Hartrisicomodel.")

        for attr, groups in sensitive_attributes.items():
            if attr == "rec_age_current":
                age_bins = [g[0] for g in groups.keys()] + [groups[list(groups.keys())[-1]][-1]]
                age_labels = list(groups.values())
                X_test_hr_full['age_group'] = pd.cut(X_test_hr_full[attr], bins=age_bins, labels=age_labels, right=False)
                age_groups_map = {label: label for label in age_labels}
                results = perform_bias_check_classification(
                    heartrisk_model, X_test_hr_full, y_test_hr, 'age_group', age_groups_map, "Hartrisico"
                )
            else:
                results = perform_bias_check_classification(
                    heartrisk_model, X_test_hr_full, y_test_hr, attr, groups, "Hartrisico"
                )
            if not results.empty:
                all_bias_results.append(results)
    except Exception as e:
        print(f"Fout bij analyse Hartrisicomodel: {e}")

    # --- Samenvatting resultaten ---
    if all_bias_results:
        final_results_df = pd.concat(all_bias_results, ignore_index=True)
        print("\n--- Samenvatting Bias Check Resultaten ---")
        print(final_results_df.to_string())
        
        # Optioneel: opslaan naar CSV
        output_path = ML.CODE_DIR / "bias_check_results.csv"
        final_results_df.to_csv(output_path, index=False)
        print(f"\nGedetailleerde resultaten opgeslagen in: {output_path}")
    else:
        print("\nGeen bias check resultaten gegenereerd.")

    print("\n--- Bias Check Voltooid ---")

if __name__ == "__main__":
    main()