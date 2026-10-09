"""Centrale categorie-definities voor brondata en dashboardweergave."""

import numpy as np
import pandas as pd


CATEGORIEEN = {
    'bmi': {
        'kolom': 'rec_med_bmi',
        'categorie_kolom': 'rec_med_bmi_cat',
        'grenzen': [16.0, 18.5, 25.0, 30.0],
        'codes': [-2, -1, 0, 1, 2],
        'labels': {
            -2: 'Ernstig ondergewicht (<16)',
            -1: 'Ondergewicht (16-18.5)',
             0: 'Normaal gewicht (18.5-25)',
             1: 'Overgewicht (25-30)',
             2: 'Obesitas (>=30)',
        },
    },
    'heartrisk': {
        'kolom': 'rec_heartrisk',
        'categorie_kolom': 'rec_heartrisk_cat',
        'grenzen': [10.0, 20.0],
        'codes': [0, 1, 2],
        'labels': {0: 'Laag (<10)', 1: 'Matig (10-20)', 2: 'Hoog (>=20)'},
    },
    'stress': {
        'kolom': 'rec_ls_stress_sum',
        'categorie_kolom': 'rec_ls_stress_cat',
        'grenzen': [11.0, 16.0],
        'codes': [0, 1, 2],
        'labels': {0: 'Laag (<11)', 1: 'Matig (11-16)', 2: 'Hoog (>=16)'},
    },
    'sleep': {
        'kolom': 'rec_ls_sleep_psqi_sum',
        'categorie_kolom': 'rec_ls_sleep_cat',
        'grenzen': [7.0, 15.0],
        'codes': [0, 1, 2],
        'labels': {0: 'Goed (0-6)', 1: 'Matig (7-14)', 2: 'Slecht (>=15)'},
    },
}


def categorie_labels(naam: str) -> dict:
    """Return a copy so callers cannot mutate the shared definitions."""
    return CATEGORIEEN[naam]['labels'].copy()


def leid_categorie_af(series: pd.Series, naam: str) -> pd.Series:
    """Derive the dashboard category code using the configured source limits."""
    config = CATEGORIEEN[naam]
    values = pd.to_numeric(series, errors='coerce')
    grenzen = config['grenzen']
    conditions = [values < grenzen[0]]
    for lower, upper in zip(grenzen, grenzen[1:]):
        conditions.append((values >= lower) & (values < upper))
    conditions.append(values >= grenzen[-1])
    return pd.Series(
        np.select(conditions, config['codes'], default=np.nan),
        index=series.index,
    ).where(values.notna())