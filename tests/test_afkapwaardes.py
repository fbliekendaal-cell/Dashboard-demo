"""Tests voor de afkapwaarde-categorisatie van snapshot scores per opdrachtgever."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visualisaties


# ── Configuratie ───────────────────────────────────────────────────────────────

def test_alle_afkapwaardes_hebben_geldige_richting_en_volgorde():
    for slug, cfg in visualisaties.SCORE_AFKAPWAARDES.items():
        assert cfg['richting'] in visualisaties.AFKAPWAARDE_BANDEN, slug
        assert cfg['grens_goed'] < cfg['grens_hoog'], slug


def test_psqi_grenzen_sluiten_aan_op_rec_ls_sleep_cat_afleiding():
    """PSQI-band moet aansluiten op data_ingestion: 7-14 matig, >=15 hoog."""
    psqi = visualisaties.SCORE_AFKAPWAARDES['rec_ls_sleep_psqi_sum']
    assert psqi == {'richting': 'lager_is_beter', 'grens_goed': 7.0, 'grens_hoog': 15.0}


# ── _afkap_categorie ──────────────────────────────────────────────────────────

def test_categorie_hoger_is_beter():
    cfg = {'richting': 'hoger_is_beter', 'grens_goed': 6.0, 'grens_hoog': 7.5}
    assert visualisaties._afkap_categorie(5.99, cfg) == 'Laag'
    assert visualisaties._afkap_categorie(6.0, cfg) == 'Goed'
    assert visualisaties._afkap_categorie(7.49, cfg) == 'Goed'
    assert visualisaties._afkap_categorie(7.5, cfg) == 'Hoog'
    assert visualisaties._afkap_categorie(np.nan, cfg) == 'Onbekend'


def test_categorie_lager_is_beter():
    cfg = {'richting': 'lager_is_beter', 'grens_goed': 25.0, 'grens_hoog': 30.0}
    assert visualisaties._afkap_categorie(22.0, cfg) == 'Laag'
    assert visualisaties._afkap_categorie(25.0, cfg) == 'Matig'
    assert visualisaties._afkap_categorie(30.0, cfg) == 'Hoog'


def test_hoger_is_beter_goed_is_groen():
    banden = visualisaties.AFKAPWAARDE_BANDEN['hoger_is_beter']
    assert banden['midden']['label'] == 'Goed'
    assert banden['midden']['kleur'] == '#2ECC71'


# ── _controleer_afkapwaardes ─────────────────────────────────────────────────

def test_controleer_geen_waarschuwing_bij_passend_bereik():
    scores = pd.Series([3.0, 6.5, 8.0])
    assert visualisaties._controleer_afkapwaardes(scores, 'rec_ls_lifestyle_score') is None


def test_controleer_waarschuwing_bij_schaal_mismatch():
    """Data onder de banden moet een mismatch-waarschuwing geven."""
    scores = pd.Series([1.0, 4.2, 9.8])
    tekst = visualisaties._controleer_afkapwaardes(scores, 'rec_ls_stress_sum')
    assert tekst is not None and 'grens_' in tekst


def test_controleer_onbekende_indicator_geeft_none():
    scores = pd.Series([1.0, 2.0])
    assert visualisaties._controleer_afkapwaardes(scores, 'onbekende_kolom') is None


# ── Integratie: maak_scores_per_opdrachtgever ────────────────────────────────

@pytest.fixture
def gesimuleerde_data(monkeypatch):
    n = 60
    doel_per_store = {1: 4.0, 2: 6.8, 3: 8.6, 4: 9.0}  # A laag, B goed, C/D hoog
    df_scores = pd.DataFrame({
        'participant_id': range(1, n + 1),
        'rec_ls_lifestyle_score': [doel_per_store[i % 4 + 1] for i in range(n)],
        'rec_med_bmi': np.full(n, 27.0),
        'onbekende_indicator': np.linspace(1.0, 5.0, n),
    })
    store_namen = ['Bedrijf A', 'Bedrijf B', 'Bedrijf C', 'Bedrijf D']
    df_links = pd.DataFrame({
        'participant_id': range(1, n + 1),
        'store_id': [i % 4 + 1 for i in range(n)],
        'store_name': [store_namen[i % 4] for i in range(n)],
    })
    monkeypatch.setattr(
        visualisaties, 'load_my_clic_participants_expanded', lambda db_url=None: df_scores.copy()
    )
    monkeypatch.setattr(
        visualisaties, '_laad_opdrachtgever_links', lambda db_url=None: df_links.copy()
    )
    return df_scores


def test_maak_scores_per_opdrachtgever_absoluut_met_verdeling(gesimuleerde_data):
    resultaat = visualisaties.maak_scores_per_opdrachtgever(
        Path('.'), 'postgresql://test',
        indicator='rec_ls_lifestyle_score', indicator_label='Leefstijlscore', min_n=1,
    )
    assert len(resultaat) == 7
    fig, totaal_gem, n_stores, cat_counts, info, fig_verdeling, verdeling_tabel = resultaat

    assert n_stores == 4
    assert info['methode'] == 'absoluut'
    assert set(cat_counts) == {'Laag', 'Goed', 'Hoog'}
    assert cat_counts == {'Laag': 1, 'Goed': 1, 'Hoog': 2}

    # Verdeling: percentages per opdrachtgever sommeren naar 100
    assert fig_verdeling is not None
    assert not verdeling_tabel.empty
    sommen = verdeling_tabel[['Laag', 'Goed', 'Hoog']].sum(axis=1)
    assert (sommen == 100.0).all()

    # Geen schaal-mismatch bij leefstijlscores rond de drempels
    assert info.get('waarschuwing') is None


def test_maak_scores_per_opdrachtgever_bmi_gebruikt_bmi_categorieen(gesimuleerde_data):
    resultaat = visualisaties.maak_scores_per_opdrachtgever(
        Path('.'), 'postgresql://test',
        indicator='rec_med_bmi', indicator_label='BMI', min_n=1,
    )
    fig, _, _, cat_counts, _, fig_verdeling, _ = resultaat

    assert set(cat_counts) == set(visualisaties.BMI_LABELS.values())
    assert cat_counts['Overgewicht (25-30)'] == 4
    assert 'Laag' not in cat_counts
    assert 'Overgewicht (25-30)' in {trace.name for trace in fig.data}
    assert fig_verdeling is not None


def test_maak_scores_per_opdrachtgever_fallback_relatief_zonder_verdeling(gesimuleerde_data):
    resultaat = visualisaties.maak_scores_per_opdrachtgever(
        Path('.'), 'postgresql://test',
        indicator='onbekende_indicator', indicator_label='Onbekend', min_n=1,
    )
    _, _, _, cat_counts, info, fig_verdeling, verdeling_tabel = resultaat
    assert info['methode'] == 'relatief'
    assert set(cat_counts) == {'Goed', 'Matig', 'Slecht'}
    assert fig_verdeling is None
    assert verdeling_tabel.empty


# ── Correcte schalen per instrument ──────────────────────────────────────────

def test_schalen_kloppen_met_instrumenten():
    """Vergrendel de gecorrigeerde afkapwaardes op instrumenteigen schalen."""
    verwacht = {
        'rec_wellbeing_score':      {'richting': 'hoger_is_beter', 'grens_goed': 55.0, 'grens_hoog': 75.0},   # WHO-5 stijl
        'rec_resilience_score':     {'richting': 'hoger_is_beter', 'grens_goed': 26.0, 'grens_hoog': 33.0},   # CD-RISC-10
        'rec_self_efficacy_score':  {'richting': 'hoger_is_beter', 'grens_goed': 25.0, 'grens_hoog': 32.0},   # GSE 10-40
        'rec_asr_wai_score':        {'richting': 'hoger_is_beter', 'grens_goed': 28.0, 'grens_hoog': 44.0},   # WAI-klassen
        'rec_ls_stress_sum':        {'richting': 'lager_is_beter', 'grens_goed': 10.0, 'grens_hoog': 16.0},
    }
    for slug, cfg in verwacht.items():
        assert visualisaties.SCORE_AFKAPWAARDES[slug] == cfg, slug


def _stub_filters(monkeypatch, intern_participant=None):
    """Monkeypatch loaders zodat de snapshotfunctie zonder DB werkt."""
    import account_filtering

    def _scores(db_url=None):
        return pd.DataFrame({
            'participant_id': [1, 2, 3, 999],
            'rec_ls_score_fruit': [2.0, 3.0, 4.0, 1500.0],   # 1500 = rauwe intake-vervuiler
            'rec_ls_lifestyle_score': [7.0, 7.2, 6.8, 9.9],
        })

    def _links(db_url=None):
        return pd.DataFrame({
            'participant_id': [1, 2, 3, 999],
            'store_id': [1, 1, 1, 1],
            'store_name': ['Bedrijf X'] * 4,
        })

    def _bridge(db_url=None):
        br = pd.DataFrame(columns=['participant_id', 'user_id'])
        if intern_participant is not None:
            br.loc[len(br)] = {'participant_id': intern_participant, 'user_id': 900}
        return br

    monkeypatch.setattr(visualisaties, 'load_my_clic_participants_expanded', _scores)
    monkeypatch.setattr(visualisaties, '_laad_opdrachtgever_links', _links)
    monkeypatch.setattr(visualisaties, '_laad_participant_user_bridge', _bridge)
    # Intern domein voor het testaccount, zodat de bridge-route hem vindt.
    monkeypatch.setattr(
        account_filtering, 'laad_interne_account_ids',
        lambda db_url=None: ({900}, {'smarthealth'}),
    )


def test_snapshot_behoudt_dieetwaarden_uit_de_bron(monkeypatch):
    _stub_filters(monkeypatch)
    resultaat = visualisaties.maak_scores_per_opdrachtgever(
        Path('.'), 'postgresql://test',
        indicator='rec_ls_score_fruit', indicator_label='Fruit score', min_n=1,
    )
    _, totaal_gem, n_stores, cat_counts, info, _, _ = resultaat
    assert n_stores == 1
    # De snapshot toont de bronwaarden: gemiddelde van [2, 3, 4, 1500].
    assert abs(totaal_gem - 377.25) < 1e-6, f'totaalgemiddelde={totaal_gem}'
    assert info['methode'] == 'absoluut'
    # Eén store die in de middenband van hoger_is_beter (3.0-4.0) valt
    assert sum(cat_counts.values()) == 1


def test_snapshot_filtert_interne_testaccounts(monkeypatch):
    _stub_filters(monkeypatch, intern_participant=999)
    resultaat = visualisaties.maak_scores_per_opdrachtgever(
        Path('.'), 'postgresql://test',
        indicator='rec_ls_score_fruit', indicator_label='Fruit score', min_n=1,
    )
    _, totaal_gem, n_stores, _, _, _, _ = resultaat
    assert n_stores == 1
    # Zonder het interne testaccount blijft het gemiddelde op [2,3,4]
    assert abs(totaal_gem - 3.0) < 1e-6, f'totaalgemiddelde={totaal_gem}'
