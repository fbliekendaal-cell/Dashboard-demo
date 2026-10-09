"""
Genereert de HTML-kaarten met consistente stijl en ingebouwde legenda.
Uitvoeren vanuit de demo-map: python kaarten.py
"""
import pandas as pd
import folium
import json
import shapely.ops
from shapely.geometry import shape, mapping
from branca.colormap import linear
from collections import defaultdict
from pathlib import Path

from config import CODE_DIR, HTML_DIR, DB_URL
from data_ingestion import load_table_from_database

BASE = CODE_DIR
HTML_MAP = HTML_DIR
HTML_MAP.mkdir(parents=True, exist_ok=True)
from kleuren import (
    RISICO_COLORS, HEARTRISK_LABELS, BMI_LABELS, BMI_COLORS, STRESS_LABELS,
    KAART_STIJL, GENDER_COLORS, GENDER_LABELS, LEEFTIJD_LABELS, LEEFTIJD_COLORS
)

# ── GeoJSON laden ─────────────────────────────────────────────────────────────
with open(BASE / 'pc4.geojson', 'r') as f:
    geojson_pc4 = json.load(f)

# ── Helper: pc4 aggregeren naar pc3 ──────────────────────────────────────────
def maak_pc3_geojson(extra_properties: dict = None):
    pc3_geometries = defaultdict(list)
    for feature in geojson_pc4['features']:
        pc4_val = str(feature['properties'].get('postcode', ''))
        pc3_val = pc4_val[:3]
        try:
            geom = shape(feature['geometry']).buffer(0)
            if geom.is_valid:
                pc3_geometries[pc3_val].append(geom)
        except Exception:
            pass

    features = []
    for pc3_val, geometries in pc3_geometries.items():
        try:
            merged = shapely.ops.unary_union(geometries)
            props  = {'pc3': pc3_val}
            if extra_properties:
                props.update(extra_properties.get(pc3_val, {}))
            features.append({
                'type': 'Feature',
                'properties': props,
                'geometry': mapping(merged),
            })
        except Exception:
            pass

    # Zorg dat alle features dezelfde keys hebben
    alle_keys = set()
    for feat in features:
        alle_keys.update(feat['properties'].keys())
    for feat in features:
        for key in alle_keys:
            feat['properties'].setdefault(key, None)

    return {'type': 'FeatureCollection', 'features': features}


# ── Helper: basiskaart ────────────────────────────────────────────────────────
def maak_basiskaart(titel: str, subtitel: str = '') -> folium.Map:
    m = folium.Map(location=[52.1, 5.3], zoom_start=7, tiles=KAART_STIJL['tiles'])
    title_html = f'''
    <div style="position:fixed;top:15px;left:60px;z-index:1000;
         background:white;padding:10px 16px;border-radius:6px;
         box-shadow:2px 2px 6px rgba(0,0,0,0.3);font-family:sans-serif;">
        <b style="font-size:15px;">{titel}</b><br>
        <span style="font-size:12px;color:#666;">{subtitel}</span>
    </div>'''
    m.get_root().html.add_child(folium.Element(title_html))
    return m


# ── Helper: legenda ingebouwd in kaart (zelfde stijl als dichtheidskaart) ─────
def voeg_legenda_toe(m: folium.Map, labels: dict, kleuren: dict, titel: str = 'Legenda'):
    items_html = ''.join(
        f'<div style="display:flex;align-items:center;margin-bottom:5px;">'
        f'<div style="width:18px;height:18px;background:{kleuren.get(k,"#ccc")};'
        f'border-radius:3px;margin-right:8px;flex-shrink:0;border:1px solid #aaa;"></div>'
        f'<span style="font-size:12px;">{v}</span></div>'
        for k, v in labels.items()
    )
    legenda_html = f'''
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:10px 14px;border-radius:6px;
         box-shadow:2px 2px 6px rgba(0,0,0,0.3);font-family:sans-serif;min-width:160px;">
        <b style="font-size:13px;display:block;margin-bottom:8px;">{titel}</b>
        {items_html}
        <div style="display:flex;align-items:center;margin-top:5px;">
        <div style="width:18px;height:18px;background:{KAART_STIJL['nan_fill']};
        border-radius:3px;margin-right:8px;flex-shrink:0;border:1px solid #aaa;"></div>
        <span style="font-size:12px;">Geen data</span></div>
    </div>'''
    m.get_root().html.add_child(folium.Element(legenda_html))
    return m


# ── Helper: opacity berekenen op basis van aantal gebruikers ─────────────────
def bereken_opacity(n: int, min_n: int = 1, max_n: int = 50) -> float:
    """
    Schaalt opacity lineair tussen 0.15 (1 gebruiker) en 0.85 (max_n gebruikers).
    Gebieden met veel gebruikers zijn donkerder, gebieden met weinig zijn doorzichtiger.
    """
    if n is None or n == 0:
        return 0.0
    n_clamped = max(min_n, min(n, max_n))
    return 0.15 + (n_clamped - min_n) / (max_n - min_n) * 0.70


def modus_int(series: pd.Series):
    """Geef de meest voorkomende niet-lege categorie terug."""
    s = series.dropna()
    if s.empty:
        return pd.NA
    mode = s.mode()
    return mode.iloc[0] if not mode.empty else pd.NA


def samenvatting_categorie_per_pc3(df_in: pd.DataFrame, waarde_col: str, label_map: dict) -> tuple[pd.DataFrame, dict]:
    """
    Bereken per pc3:
    - meest voorkomende categorie
    - aantal deelnemers met een geldige waarde
    - dominantie van die categorie (% van geldige waarden)
    """
    records = []
    extra = {}

    for pc3, group in df_in.groupby('pc3'):
        s = pd.to_numeric(group[waarde_col], errors='coerce').dropna().round().astype(int)
        if s.empty:
            continue
        counts = s.value_counts()
        top_code = int(counts.index[0])
        top_count = int(counts.iloc[0])
        n_geldig = int(counts.sum())
        dominantie_pct = round(top_count / n_geldig * 100, 1) if n_geldig else 0.0
        label = label_map.get(top_code, str(top_code))

        records.append({
            'pc3': pc3,
            'categorie_num': top_code,
            'aantal_geldig': n_geldig,
            'dominantie_pct': dominantie_pct,
        })
        extra[pc3] = {
            'categorie_num': top_code,
            'categorie_label': label,
            'aantal_gebruikers': n_geldig,
            'dominantie_pct': dominantie_pct,
            'dominantie_label': f'{dominantie_pct:.1f}%',
        }

    return pd.DataFrame(records), extra


def maak_continue_pc3_kaart(
    df_in: pd.DataFrame,
    waarde_col: str,
    titel: str,
    subtitel: str,
    bestandsnaam: str,
    legenda_titel: str,
    tooltip_label: str,
    kleurenschaal='YlOrRd',
):
    """
    Maak een pc3-kaart met mediaan per gebied en opacity op basis van aantal deelnemers.
    """
    df_cont = df_in[['pc3', waarde_col]].copy()
    df_cont[waarde_col] = pd.to_numeric(df_cont[waarde_col], errors='coerce')
    df_cont = df_cont.dropna(subset=[waarde_col])
    if df_cont.empty:
        return

    agg = (
        df_cont.groupby('pc3')[waarde_col]
        .agg(mediaan='median', aantal_geldig='count')
        .reset_index()
    )
    vmin = float(agg['mediaan'].min())
    vmax = float(agg['mediaan'].max())
    if vmin == vmax:
        vmax = vmin + 1.0

    palette = getattr(linear, f'{kleurenschaal}_09', linear.YlOrRd_09).scale(vmin, vmax)
    palette.caption = legenda_titel

    extra = {
        row['pc3']: {
            'mediaan_waarde': round(float(row['mediaan']), 2),
            'mediaan_label': f"{float(row['mediaan']):.2f}",
            'aantal_gebruikers': int(row['aantal_geldig']),
        }
        for _, row in agg.iterrows()
    }
    geojson_cont = maak_pc3_geojson(extra_properties=extra)

    m = maak_basiskaart(titel, subtitel)

    def stijl_cont(feature):
        val = feature['properties'].get('mediaan_waarde')
        n = feature['properties'].get('aantal_gebruikers') or 0
        return {
            'fillColor': palette(val) if val is not None else KAART_STIJL['nan_fill'],
            'fillOpacity': bereken_opacity(n),
            'color': '#555',
            'weight': 0.3,
        }

    folium.GeoJson(
        geojson_cont,
        style_function=stijl_cont,
        tooltip=folium.GeoJsonTooltip(
            fields=['pc3', 'mediaan_label', 'aantal_gebruikers'],
            aliases=['Postcode (pc3):', f'{tooltip_label}:', 'Deelnemers:'],
        ),
        highlight_function=lambda x: {'weight': 2, 'color': '#333'},
    ).add_to(m)
    palette.add_to(m)
    m.save(str(HTML_MAP / bestandsnaam))


# ── Data laden ────────────────────────────────────────────────────────────────
df = load_table_from_database('users_met_scores', DB_URL)
if df.empty:
    raise RuntimeError("De tabel users_met_scores bevat geen data voor de kaartgeneratie.")
# Houd per gebruiker één rij aan, consistent met het dashboard.
if 'user_id' in df.columns:
    df = df.drop_duplicates(subset='user_id', keep='first').reset_index(drop=True)
if 'postal_code' not in df.columns:
    raise KeyError("users_met_scores mist de kolom postal_code voor kaartgeneratie.")
df['pc4'] = df['postal_code'].dropna().astype(str).str.extract(r'(\d{4})')[0]
df['pc3'] = df['pc4'].str[:3]
nl_mask   = df['pc4'].notna() & (df['pc4'].str.len() == 4)
df_nl     = df[nl_mask].copy()
if 'rec_med_age_cat' not in df_nl.columns and 'rec_age_current' in df_nl.columns:
    age = pd.to_numeric(df_nl['rec_age_current'], errors='coerce')
    df_nl['rec_med_age_cat'] = pd.Series(
        pd.NA,
        index=df_nl.index,
        dtype='Int64',
    )
    df_nl.loc[age < 40, 'rec_med_age_cat'] = 0
    df_nl.loc[age.between(40, 55, inclusive='both'), 'rec_med_age_cat'] = 1
    df_nl.loc[age > 55, 'rec_med_age_cat'] = 2


# ══════════════════════════════════════════════════════════════════════════════
# 1. GEBRUIKERSDICHTHEID (pc3)
# ══════════════════════════════════════════════════════════════════════════════
print("Kaart 1: gebruikersdichtheid...")
dichtheid      = df_nl.groupby('pc3').size().reset_index(name='aantal_gebruikers')
dichtheid_dict = dichtheid.set_index('pc3')['aantal_gebruikers'].to_dict()

geojson_d = maak_pc3_geojson(
    extra_properties={k: {'aantal_gebruikers': v} for k, v in dichtheid_dict.items()}
)

m = maak_basiskaart(
    'Gebruikersdichtheid per postcode (pc3)',
    'Aantal PMO deelnemers per gebied · grijs = geen gebruikers'
)
folium.Choropleth(
    geo_data=geojson_d,
    data=dichtheid,
    columns=['pc3', 'aantal_gebruikers'],
    key_on='feature.properties.pc3',
    fill_color='YlOrRd',
    fill_opacity=KAART_STIJL['fill_opacity'],
    line_opacity=KAART_STIJL['line_opacity'],
    nan_fill_color=KAART_STIJL['nan_fill'],
    nan_fill_opacity=KAART_STIJL['nan_opacity'],
    legend_name='Aantal gebruikers',
).add_to(m)
folium.GeoJson(
    geojson_d,
    style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
    tooltip=folium.GeoJsonTooltip(
        fields=['pc3', 'aantal_gebruikers'],
        aliases=['Postcode (pc3):', 'Gebruikers:'],
    ),
).add_to(m)
m.save(str(HTML_MAP / 'gebruikersdichtheid_kaart_pc3.html'))
print("  Opgeslagen.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. HEARTRISK (pc3)
# ══════════════════════════════════════════════════════════════════════════════
print("Kaart 2: heartrisk...")
df_nl['heartrisk_num'] = pd.to_numeric(df_nl['rec_heartrisk_cat'], errors='coerce').round().astype('Int64')
hr_per_pc3, hr_extra_raw = samenvatting_categorie_per_pc3(df_nl, 'heartrisk_num', HEARTRISK_LABELS)
hr_extra = {
    pc3: {
        'heartrisk_num': props['categorie_num'],
        'heartrisk_label': props['categorie_label'],
        'aantal_gebruikers': props['aantal_gebruikers'],
        'dominantie_pct': props['dominantie_pct'],
        'dominantie_label': props['dominantie_label'],
    }
    for pc3, props in hr_extra_raw.items()
}
geojson_hr = maak_pc3_geojson(extra_properties=hr_extra)

m = maak_basiskaart(
    'Cardiovasculair risico per postcode (pc3)',
    'Meest voorkomende risicocategorie per gebied'
)

def stijl_hr(feature):
    val = feature['properties'].get('heartrisk_num')
    n   = feature['properties'].get('aantal_gebruikers') or 0
    return {
        'fillColor':   RISICO_COLORS.get(val, KAART_STIJL['nan_fill']),
        'fillOpacity': bereken_opacity(n),
        'color': '#555', 'weight': 0.3,
    }

folium.GeoJson(
    geojson_hr,
    style_function=stijl_hr,
    tooltip=folium.GeoJsonTooltip(
        fields=['pc3', 'heartrisk_label', 'dominantie_label', 'aantal_gebruikers'],
        aliases=['Postcode (pc3):', 'Risicocategorie:', 'Dominantie:', 'Deelnemers:'],
    ),
    highlight_function=lambda x: {'weight': 2, 'color': '#333'},
).add_to(m)
voeg_legenda_toe(m, HEARTRISK_LABELS, RISICO_COLORS, 'Cardiovasculair risico')
m.save(str(HTML_MAP / 'heartrisk_categorie_kaart.html'))
print("  Opgeslagen.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BMI (pc3)
# ══════════════════════════════════════════════════════════════════════════════
print("Kaart 3: BMI...")
df_nl['bmi_num'] = pd.to_numeric(df_nl['rec_med_bmi_cat'], errors='coerce').round().astype('Int64')
bmi_per_pc3, bmi_extra_raw = samenvatting_categorie_per_pc3(df_nl, 'bmi_num', BMI_LABELS)
bmi_extra = {
    pc3: {
        'bmi_num': props['categorie_num'],
        'bmi_label': props['categorie_label'],
        'aantal_gebruikers': props['aantal_gebruikers'],
        'dominantie_pct': props['dominantie_pct'],
        'dominantie_label': props['dominantie_label'],
    }
    for pc3, props in bmi_extra_raw.items()
}
geojson_bmi = maak_pc3_geojson(extra_properties=bmi_extra)

m = maak_basiskaart(
    'BMI categorie per postcode (pc3)',
    'Meest voorkomende BMI categorie per gebied'
)

def stijl_bmi(feature):
    val = feature['properties'].get('bmi_num')
    n   = feature['properties'].get('aantal_gebruikers') or 0
    return {
        'fillColor':   BMI_COLORS.get(val, KAART_STIJL['nan_fill']),
        'fillOpacity': bereken_opacity(n),
        'color': '#555', 'weight': 0.3,
    }

folium.GeoJson(
    geojson_bmi,
    style_function=stijl_bmi,
    tooltip=folium.GeoJsonTooltip(
        fields=['pc3', 'bmi_label', 'dominantie_label', 'aantal_gebruikers'],
        aliases=['Postcode (pc3):', 'BMI categorie:', 'Dominantie:', 'Deelnemers:'],
    ),
    highlight_function=lambda x: {'weight': 2, 'color': '#333'},
).add_to(m)
# Alleen categorieën tonen die ook voorkomen in de data
bmi_aanwezig = {k: v for k, v in BMI_LABELS.items() if k in bmi_per_pc3['categorie_num'].dropna().astype(int).values}
voeg_legenda_toe(m, bmi_aanwezig, BMI_COLORS, 'BMI categorie')
m.save(str(HTML_MAP / 'bmi_categorie_kaart.html'))
print("  Opgeslagen.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. STRESS (pc3)
# ══════════════════════════════════════════════════════════════════════════════
print("Kaart 4: stress...")
df_nl['stress_num'] = pd.to_numeric(df_nl['rec_ls_stress_cat'], errors='coerce').round().astype('Int64')
stress_per_pc3, stress_extra_raw = samenvatting_categorie_per_pc3(df_nl, 'stress_num', STRESS_LABELS)
stress_extra = {
    pc3: {
        'stress_num': props['categorie_num'],
        'stress_label': props['categorie_label'],
        'aantal_gebruikers': props['aantal_gebruikers'],
        'dominantie_pct': props['dominantie_pct'],
        'dominantie_label': props['dominantie_label'],
    }
    for pc3, props in stress_extra_raw.items()
}
geojson_stress = maak_pc3_geojson(extra_properties=stress_extra)

m = maak_basiskaart(
    'Stress categorie per postcode (pc3)',
    'Meest voorkomende stresscategorie per gebied'
)

def stijl_stress(feature):
    val = feature['properties'].get('stress_num')
    n   = feature['properties'].get('aantal_gebruikers') or 0
    return {
        'fillColor':   RISICO_COLORS.get(val, KAART_STIJL['nan_fill']),
        'fillOpacity': bereken_opacity(n),
        'color': '#555', 'weight': 0.3,
    }

folium.GeoJson(
    geojson_stress,
    style_function=stijl_stress,
    tooltip=folium.GeoJsonTooltip(
        fields=['pc3', 'stress_label', 'dominantie_label', 'aantal_gebruikers'],
        aliases=['Postcode (pc3):', 'Stresscategorie:', 'Dominantie:', 'Deelnemers:'],
    ),
    highlight_function=lambda x: {'weight': 2, 'color': '#333'},
).add_to(m)
voeg_legenda_toe(m, STRESS_LABELS, RISICO_COLORS, 'Stress categorie')
m.save(str(HTML_MAP / 'stress_categorie_kaart.html'))
print("  Opgeslagen.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LEEFTIJDSCATEGORIE (pc3)
# ══════════════════════════════════════════════════════════════════════════════
print("Kaart 5: leeftijdscategorie...")

df_nl['leeftijd_num'] = pd.to_numeric(df_nl['rec_med_age_cat'], errors='coerce').round().astype('Int64')
leeftijd_per_pc3, leeftijd_extra_raw = samenvatting_categorie_per_pc3(df_nl, 'leeftijd_num', LEEFTIJD_LABELS)
leeftijd_extra = {
    pc3: {
        'leeftijd_num': props['categorie_num'],
        'leeftijd_label': props['categorie_label'],
        'aantal_gebruikers': props['aantal_gebruikers'],
        'dominantie_pct': props['dominantie_pct'],
        'dominantie_label': props['dominantie_label'],
    }
    for pc3, props in leeftijd_extra_raw.items()
}
geojson_leeftijd = maak_pc3_geojson(extra_properties=leeftijd_extra)

m = maak_basiskaart(
    'Leeftijdscategorie per postcode (pc3)',
    'Meest voorkomende leeftijdscategorie per gebied'
)

def stijl_leeftijd(feature):
    val = feature['properties'].get('leeftijd_num')
    n   = feature['properties'].get('aantal_gebruikers') or 0
    return {
        'fillColor':   LEEFTIJD_COLORS.get(val, KAART_STIJL['nan_fill']),
        'fillOpacity': bereken_opacity(n),
        'color': '#555', 'weight': 0.3,
    }

folium.GeoJson(
    geojson_leeftijd,
    style_function=stijl_leeftijd,
    tooltip=folium.GeoJsonTooltip(
        fields=['pc3', 'leeftijd_label', 'dominantie_label', 'aantal_gebruikers'],
        aliases=['Postcode (pc3):', 'Leeftijdscategorie:', 'Dominantie:', 'Deelnemers:'],
    ),
    highlight_function=lambda x: {'weight': 2, 'color': '#333'},
).add_to(m)
voeg_legenda_toe(m, LEEFTIJD_LABELS, LEEFTIJD_COLORS, 'Leeftijdscategorie')
m.save(str(HTML_MAP / 'leeftijd_categorie_kaart.html'))
print("  Opgeslagen.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SLAAP (pc3)
# ══════════════════════════════════════════════════════════════════════════════
print("Kaart 6: slaap...")

SLAAP_LABELS = {0: 'Goed', 1: 'Matig', 2: 'Slecht'}

df_nl['slaap_num'] = pd.to_numeric(df_nl['rec_ls_sleep_cat'], errors='coerce').round().astype('Int64')
slaap_per_pc3, slaap_extra_raw = samenvatting_categorie_per_pc3(df_nl, 'slaap_num', SLAAP_LABELS)
slaap_extra = {
    pc3: {
        'slaap_num': props['categorie_num'],
        'slaap_label': props['categorie_label'],
        'aantal_gebruikers': props['aantal_gebruikers'],
        'dominantie_pct': props['dominantie_pct'],
        'dominantie_label': props['dominantie_label'],
    }
    for pc3, props in slaap_extra_raw.items()
}
geojson_slaap = maak_pc3_geojson(extra_properties=slaap_extra)

m = maak_basiskaart(
    'Slaapkwaliteit per postcode (pc3)',
    'Meest voorkomende slaapkwaliteitscategorie per gebied'
)

def stijl_slaap(feature):
    val = feature['properties'].get('slaap_num')
    n   = feature['properties'].get('aantal_gebruikers') or 0
    return {
        'fillColor':   RISICO_COLORS.get(val, KAART_STIJL['nan_fill']),
        'fillOpacity': bereken_opacity(n),
        'color': '#555', 'weight': 0.3,
    }

folium.GeoJson(
    geojson_slaap,
    style_function=stijl_slaap,
    tooltip=folium.GeoJsonTooltip(
        fields=['pc3', 'slaap_label', 'dominantie_label', 'aantal_gebruikers'],
        aliases=['Postcode (pc3):', 'Slaapkwaliteit:', 'Dominantie:', 'Deelnemers:'],
    ),
    highlight_function=lambda x: {'weight': 2, 'color': '#333'},
).add_to(m)
voeg_legenda_toe(m, SLAAP_LABELS, RISICO_COLORS, 'Slaapkwaliteit')
m.save(str(HTML_MAP / 'slaap_categorie_kaart.html'))
print("  Opgeslagen.")

# ══════════════════════════════════════════════════════════════════════════════
# 7. BEWEGING (pc3)
# ══════════════════════════════════════════════════════════════════════════════
#print("Kaart 7: beweging...")
#
#BEWEGING_LABELS = {0: 'Weinig (<5.000)', 1: 'Matig (5.000-10.000)', 2: 'Veel (>10.000)'}
#BEWEGING_COLORS = {0: '#E74C3C', 1: '#E87722', 2: '#2ECC71'}
#
#df_nl['beweging_num'] = pd.to_numeric(df_nl['rec_ls_exercise_steps_cat'], errors='coerce').round().astype('Int64')
#beweging_per_pc3 = df_nl.groupby('pc3')['beweging_num'].agg(modus_int).reset_index()
#beweging_per_pc3.columns = ['pc3', 'beweging_num']
#beweging_gebruikers = df_nl.groupby('pc3').size().to_dict()
#
#beweging_extra = {
#    row['pc3']: {
#        'beweging_num':      int(row['beweging_num']),
#        'beweging_label':    BEWEGING_LABELS.get(int(row['beweging_num']), ''),
#        'aantal_gebruikers': beweging_gebruikers.get(row['pc3'], 0),
#    }
#    for _, row in beweging_per_pc3.iterrows() if pd.notna(row['beweging_num'])
#}
#geojson_beweging = maak_pc3_geojson(extra_properties=beweging_extra)
#
#m = maak_basiskaart(
#    'Stappen per dag per postcode (pc3)',
#    'Meest voorkomende stappencategorie per gebied'
#)
#
#def stijl_beweging(feature):
#    val = feature['properties'].get('beweging_num')
#    n   = feature['properties'].get('aantal_gebruikers') or 0
#    return {
#        'fillColor':   BEWEGING_COLORS.get(val, KAART_STIJL['nan_fill']),
#        'fillOpacity': bereken_opacity(n),
#        'color': '#555', 'weight': 0.3,
#    }
#
#folium.GeoJson(
#    geojson_beweging,
#    style_function=stijl_beweging,
#    tooltip=folium.GeoJsonTooltip(
#        fields=['pc3', 'beweging_label', 'aantal_gebruikers'],
#        aliases=['Postcode (pc3):', 'Stappencategorie:', 'Deelnemers:'],
#    ),
#    highlight_function=lambda x: {'weight': 2, 'color': '#333'},
#).add_to(m)
#voeg_legenda_toe(m, BEWEGING_LABELS, BEWEGING_COLORS, 'Stappen per dag')
#m.save(str(HTML_MAP / 'beweging_categorie_kaart.html'))
#print("  Opgeslagen.")
#
#
## ══════════════════════════════════════════════════════════════════════════════
## 8. WERKVERMOGEN (pc3)
## ══════════════════════════════════════════════════════════════════════════════
#print("Kaart 8: werkvermogen...")
#
#WERKVERMOGEN_LABELS = {0: 'Goed', 1: 'Matig', 2: 'Slecht'}
#
#df_nl['werkvermogen_num'] = pd.to_numeric(df_nl['rec_asr_work_ability_category'], errors='coerce').round().astype('Int64')
#werkvermogen_per_pc3 = df_nl.groupby('pc3')['werkvermogen_num'].agg(modus_int).reset_index()
#werkvermogen_per_pc3.columns = ['pc3', 'werkvermogen_num']
#werkvermogen_gebruikers = df_nl.groupby('pc3').size().to_dict()
#
#werkvermogen_extra = {
#    row['pc3']: {
#        'werkvermogen_num':   int(row['werkvermogen_num']),
#        'werkvermogen_label': WERKVERMOGEN_LABELS.get(int(row['werkvermogen_num']), ''),
#        'aantal_gebruikers':  werkvermogen_gebruikers.get(row['pc3'], 0),
#    }
#    for _, row in werkvermogen_per_pc3.iterrows() if pd.notna(row['werkvermogen_num'])
#}
#geojson_werkvermogen = maak_pc3_geojson(extra_properties=werkvermogen_extra)
#
#m = maak_basiskaart(
#    'Werkvermogen per postcode (pc3)',
#    'Meest voorkomende werkvermogenscategorie per gebied'
#)
#
#def stijl_werkvermogen(feature):
#    val = feature['properties'].get('werkvermogen_num')
#    n   = feature['properties'].get('aantal_gebruikers') or 0
#    return {
#        'fillColor':   WERKVERMOGEN_COLORS.get(val, KAART_STIJL['nan_fill']),
#        'fillOpacity': bereken_opacity(n),
#        'color': '#555', 'weight': 0.3,
#    }
#
#WERKVERMOGEN_COLORS = {0: '#2ECC71', 1: '#E87722', 2: '#E74C3C'}
#
#folium.GeoJson(
#    geojson_werkvermogen,
#    style_function=stijl_werkvermogen,
#    tooltip=folium.GeoJsonTooltip(
#        fields=['pc3', 'werkvermogen_label', 'aantal_gebruikers'],
#        aliases=['Postcode (pc3):', 'Werkvermogen:', 'Deelnemers:'],
#    ),
#    highlight_function=lambda x: {'weight': 2, 'color': '#333'},
#).add_to(m)
#voeg_legenda_toe(m, WERKVERMOGEN_LABELS, WERKVERMOGEN_COLORS, 'Werkvermogen')
#m.save(str(HTML_MAP / 'werkvermogen_categorie_kaart.html'))
#print("  Opgeslagen.")
#
#print("\nKlaar. Alle kaarten opgeslagen in:", HTML_MAP)
#
