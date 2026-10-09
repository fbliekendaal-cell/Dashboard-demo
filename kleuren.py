# ── Centrale kleurendefinities ────────────────────────────────────────────────

from categorieen import categorie_labels

HOOFD_KLEUR = '#E87722'  # oranje

# Geslacht: 0 = vrouw, 1 = man
GENDER_COLORS = {
    0:        '#E86FA8',  # roze = vrouw
    1:        '#1F6FBF',  # blauw = man
    'Man':    '#1F6FBF',
    'Vrouw':  '#E86FA8',
}
GENDER_LABELS = {0: 'Vrouw', 1: 'Man'}

# Risico: 0 = laag, 1 = matig, 2 = hoog
RISICO_COLORS = {
    0: '#2ECC71',  # groen = laag
    1: '#E87722',  # oranje = matig
    2: '#E74C3C',  # rood = hoog
}

# Vragenlijst invulpercentage: 0 = groen (>=50%), 1 = oranje (>=20%), 2 = rood (<20%)
VRAGENLIJST_COLORS = {
    0: '#2ECC71',  # groen = hoog (>=50%)
    1: '#E87722',  # oranje = matig (>=20%)
    2: '#E74C3C',  # rood = laag (<20%)
}

HEARTRISK_LABELS = categorie_labels('heartrisk')
STRESS_LABELS    = categorie_labels('stress')

# Leeftijd
LEEFTIJD_LABELS = {0: 'Jong (<40)', 1: 'Middel (40-55)', 2: 'Ouder (>55)'}
LEEFTIJD_COLORS = {
    0: '#F7DC6F',  # geel
    1: '#E67E22',  # oranje
    2: '#C0392B',  # rood
}

# BMI: afwijkingen rood/oranje, normaal gewicht groen.
BMI_LABELS = categorie_labels('bmi')
BMI_COLORS = {
    -2: '#E74C3C',  # rood = ernstig ondergewicht
    -1: '#E87722',  # oranje = ondergewicht
     0: '#2ECC71',  # groen = normaal
     1: '#E87722',  # oranje = overgewicht
     2: '#E74C3C',  # rood = obesitas
}

# Kaart stijl
KAART_STIJL = {
    'tiles':        'CartoDB positron',
    'fill_opacity': 0.75,
    'line_opacity': 0.2,
    'nan_fill':     '#eeeeee',
    'nan_opacity':  0.4,
}
