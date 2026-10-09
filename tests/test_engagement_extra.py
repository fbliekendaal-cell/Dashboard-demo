"""Tests voor de uitgebreide engagement-aggregatie (herhaalkopers,
challenge-voltooiing en recente app-activiteit)."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visualisaties


@pytest.fixture
def gestubde_engagement(monkeypatch):
    """Monkeypatch alle bronnen zodat de engagementpijplijn zonder DB draait."""
    # Deelnemers: store 1 -> p1,p2,p3 ; store 2 -> p4,p5
    def _scores(db_url=None):
        return pd.DataFrame({
            'participant_id': [1, 2, 3, 4, 5],
            'rec_ls_lifestyle_score': [7.0] * 5,
        })

    def _links(db_url=None):
        return pd.DataFrame({
            'participant_id': [1, 2, 3, 4, 5],
            'store_id': [1, 1, 1, 2, 2],
            'store_name': ['Bedrijf A'] * 3 + ['Bedrijf B'] * 2,
            'user_id': [101, 102, 103, 201, 202],
        })

    def _bridge(db_url=None):
        return pd.DataFrame(columns=['participant_id', 'user_id'])

    def _aankopen(base_pad=None, db_url=None):
        # p1: 2 betaalde orders (herhaalkoper), p2: 1 betaald, p3: alleen geannuleerd
        return pd.DataFrame([
            {'participant_id': 1, 'user_id': 101, 'status': 'paid'},
            {'participant_id': 1, 'user_id': 101, 'status': 'paid'},
            {'participant_id': 2, 'user_id': 102, 'status': 'paid'},
            {'participant_id': 3, 'user_id': 103, 'status': 'canceled'},
        ])

    def _challenges(db_url=None):
        return pd.DataFrame([
            {'store_id': 1, 'user_ref': 101, 'status': 'completed', 'created_at': pd.Timestamp('2025-01-01')},
            {'store_id': 1, 'user_ref': 101, 'status': 'completed', 'created_at': pd.Timestamp('2025-01-10')},
            {'store_id': 1, 'user_ref': 102, 'status': 'completed', 'created_at': pd.Timestamp('2025-01-05')},
            {'store_id': 1, 'user_ref': 103, 'status': 'expired', 'created_at': pd.Timestamp('2025-02-01')},
            {'store_id': 2, 'user_ref': 201, 'status': 'expired', 'created_at': pd.Timestamp('2025-02-01')},
            {'store_id': 2, 'user_ref': 202, 'status': 'expired', 'created_at': pd.Timestamp('2025-02-10')},
        ])

    def _logins(db_url=None):
        # Venster van 90 dagen gerekend vanaf max(created_at) = 2025-06-30
        return pd.DataFrame([
            {'user_id': 101, 'created_at': pd.Timestamp('2025-06-28')},
            {'user_id': 103, 'created_at': pd.Timestamp('2025-06-29')},
            {'user_id': 102, 'created_at': pd.Timestamp('2024-01-01')},  # buiten venster
        ])

    monkeypatch.setattr(visualisaties, 'load_my_clic_participants_expanded', _scores)
    monkeypatch.setattr(visualisaties, '_laad_opdrachtgever_links', _links)
    monkeypatch.setattr(visualisaties, '_laad_participant_user_bridge', _bridge)
    monkeypatch.setattr(visualisaties, 'get_user_purchases', _aankopen)
    monkeypatch.setattr(visualisaties, '_laad_user_challenges_per_store', _challenges)
    monkeypatch.setattr(visualisaties, '_laad_app_logins', _logins)
    monkeypatch.setattr(visualisaties, 'load_completions', lambda db_url=None: pd.DataFrame())


# ---- TESTS ----

def test_engagement_summary_includes_top_and_benchmark(gestubde_engagement):
    agg, benchmark, _ = visualisaties.bereken_engagement_per_opdrachtgever(Path('.'), 'x')
    summary = visualisaties.build_engagement_summary_stats(agg, benchmark, lang='nl')

    assert summary['beste_opdrachtgever'] == 'Bedrijf A'
    assert summary['benchmark'] == pytest.approx(benchmark)
    assert summary['top_delta'] >= 0
    assert 'insight' in summary and summary['insight']


def test_agg_berekent_herhaalkopers_challenge_en_app(gestubde_engagement):
    agg, benchmark, _ = visualisaties.bereken_engagement_per_opdrachtgever(Path('.'), 'x')

    assert {'pct_herhaalkopers_van_kopers', 'pct_challenges_afgerond',
            'pct_app_actief_90d'} <= set(agg.columns)

    rA = agg[agg['store_name'] == 'Bedrijf A'].iloc[0]
    rB = agg[agg['store_name'] == 'Bedrijf B'].iloc[0]

    # Bedrijf A: geldige kopers {1,2}, herhaalkoper p1 -> 1/2 van de kopers
    assert abs(rA['pct_herhaalkopers_van_kopers'] - 50.0) < 0.2
    # Bedrijf B: geen enkele koper -> onbekend (NaN)
    assert pd.isna(rB['pct_herhaalkopers_van_kopers'])

    # Challenge-voltooiingsgraad
    assert abs(rA['pct_challenges_afgerond'] - 75.0) < 0.2        # 3/4 challenge-runs uitgekomen
    assert rB['pct_challenges_afgerond'] == 0.0                    # 0/2
    assert int(rB['n_challenges']) == 2                            # expired telt als deelname
    assert int(rB['n_challenges_afgerond']) == 0
    assert int(rA['n_challenges_afgerond']) == 3
    assert int(rA['n_challenges']) == 3                            # unieke deelnemers
    assert int(rA['n_challenge_runs']) == 4                       # challenge-runs

    # App actief laatste 90 dagen (p1 en p3 recent; p2-logs buiten venster)
    assert abs(rA['pct_app_actief_90d'] - 66.667) < 0.2
    assert rB['pct_app_actief_90d'] == 0.0


def test_app_activiteit_gebruikt_expliciete_referentiedatum(gestubde_engagement):
    agg, _, _ = visualisaties.bereken_engagement_per_opdrachtgever(
        Path('.'), 'x', app_activity_reference_date=pd.Timestamp('2025-04-01')
    )

    assert agg['pct_app_actief_90d'].eq(0.0).all()


def test_uitsplitsing_toont_nieuwe_maten(gestubde_engagement):
    agg, _, _ = visualisaties.bereken_engagement_per_opdrachtgever(Path('.'), 'x')
    fig = visualisaties.maak_engagement_breakdown_plot(agg, 'Bedrijf A', lang='nl')
    bar_trace = fig.data[0]
    balk_labels = set(getattr(bar_trace, 'y', []))
    assert 'Herhaalkoper (% van kopers)' in balk_labels
    assert 'Challenge afgerond (% van uitgekomen)' in balk_labels
    assert 'App actief (laatste 90 dagen)' in balk_labels


def test_werkt_ook_zonder_events_of_purchases(monkeypatch):
    """Ontbrekende bronnen mogen geen crash geven; kolommen vallen netjes weg."""
    def _scores(db_url=None):
        return pd.DataFrame({'participant_id': [1, 2], 'rec_ls_lifestyle_score': [6.0, 6.0]})

    def _links(db_url=None):
        return pd.DataFrame({
            'participant_id': [1, 2], 'store_id': [9, 9],
            'store_name': ['Bedrijf C'] * 2, 'user_id': [301, 302],
        })

    def _leeg(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(visualisaties, 'load_my_clic_participants_expanded', _scores)
    monkeypatch.setattr(visualisaties, '_laad_opdrachtgever_links', _links)
    monkeypatch.setattr(visualisaties, '_laad_participant_user_bridge', _leeg)
    monkeypatch.setattr(visualisaties, 'get_user_purchases', _leeg)
    monkeypatch.setattr(visualisaties, '_laad_user_challenges_per_store', _leeg)
    monkeypatch.setattr(visualisaties, '_laad_app_logins', _leeg)
    monkeypatch.setattr(visualisaties, '_laad_interaction_views', _leeg)
    monkeypatch.setattr(visualisaties, 'load_completions', _leeg)

    agg, _, _ = visualisaties.bereken_engagement_per_opdrachtgever(Path('.'), 'x')
    assert len(agg) == 1
    assert agg.iloc[0]['pct_app_actief_90d'] == 0.0  # events leeg -> iedereen inactief


def test_leefstijlscore_risicomigratie_uses_lifestyle_slug():
    assert visualisaties._dashboard_score_to_history_slug('rec_ls_lifestyle_score') == 'lifestyle'


def test_herhaalde_vragenlijst_traject_uses_numeric_x_axis():
    traject = pd.DataFrame({
        'invul_nr': [1, 2, 3],
        'invul_label': ['1e invulling', '2e invulling', '3e invulling'],
        'gem_score': [32.0, 24.0, 20.0],
        'n_deelnemers': [12, 8, 6],
    })
    fig = visualisaties._maak_herhaalde_vragenlijst_traject_fig(traject, 'DASS', 'Score', 'nl')
    assert list(fig.data[0].x) == [1, 2, 3]
    assert fig.layout.xaxis.tickmode == 'array'
    assert fig.layout.xaxis.tickvals == (1, 2, 3)


def test_risicomigratie_excludes_dass_variabelen():
    assert 'dass_stress' in visualisaties.RISICO_MIGRATIE_EXCLUDED_SLUGS
    assert 'dass_anxiety' in visualisaties.RISICO_MIGRATIE_EXCLUDED_SLUGS
    assert 'dass_depression' in visualisaties.RISICO_MIGRATIE_EXCLUDED_SLUGS


def test_risicomigratie_kleuren_matchen_laag_is_beter_variabelen():
    assert not visualisaties._is_higher_better('sleep')
    assert not visualisaties._is_higher_better('heartrisk')
    assert not visualisaties._is_higher_better('workload')
    assert not visualisaties._is_higher_better('dass_stress')
    assert visualisaties._is_higher_better('wellbeing')
    assert visualisaties._is_higher_better('exercise')


def test_bmi_risico_migratie_markeert_normaal_gewicht_groen():
    matrix = pd.DataFrame({
        'Van': [-2, -1, 0, 1, 2, 0],
        'Na': [0, 0, 0, 2, 2, 1],
        'Aantal': [1, 2, 3, 2, 1, 1],
    })
    label_map = {
        -2: 'Ernstig ondergewicht (<16)',
        -1: 'Ondergewicht (16-18.5)',
        0: 'Normaal gewicht (18.5-25)',
        1: 'Overgewicht (25-30)',
        2: 'Obesitas (>=30)',
    }

    fig = visualisaties.maak_risico_migratie_sankey(matrix, 'bmi', label_map, lang='nl')
    node_colors = list(fig.data[0].node.color)
    assert node_colors[2] == '#2ECC71'
    assert node_colors[7] == '#2ECC71'


def test_bmi_risicomigratie_onderste_scores_blijven_binnen_grafiekgebied():
    matrix = pd.DataFrame({
        'Van': [-2, -1, 0, 1, 2],
        'Na': [-2, -1, 0, 1, 2],
        'Aantal': [2, 3, 4, 3, 2],
    })
    label_map = {
        -2: 'Ernstig ondergewicht (<16)',
        -1: 'Ondergewicht (16-18.5)',
        0: 'Normaal gewicht (18.5-25)',
        1: 'Overgewicht (25-30)',
        2: 'Obesitas (>=30)',
    }

    fig = visualisaties.maak_risico_migratie_sankey(matrix, 'bmi', label_map, lang='nl')

    node_y = list(fig.data[0].node.y)
    assert max(node_y) == pytest.approx(0.6)
    assert list(fig.data[0].node.label)[4] == 'Obesitas (>=30) (Eerste) — 2'
    assert list(fig.data[0].node.label)[9] == 'Obesitas (>=30) (Laatste) — 2'


def test_bmi_risico_migratie_binned_haalt_echte_bmi_codes_op():
    df = pd.DataFrame({
        'participant_id': [1, 1, 2, 2, 3, 3],
        'slug': ['bmi'] * 6,
        'completion_created_at': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2025-01-01'),
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2025-01-01'),
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2025-01-01'),
        ],
        'score_value': [17.0, 24.0, 32.0, 27.0, 19.0, 18.0],
    })

    matrix, label_map = visualisaties.bereken_risico_migratie_value_binned(df, 'bmi')

    assert {int(v) for v in matrix['Van']}.issubset({-2, -1, 0, 1, 2})
    assert set(label_map.keys()).issubset({-2, -1, 0, 1, 2})
    assert label_map[0] == 'Normaal gewicht (18.5-25)'


@pytest.mark.parametrize(
    ('slug', 'scores', 'expected_labels'),
    [
        (
            'lifestyle',
            [5.0, 8.0, 7.0, 9.0, 9.0, 6.0],
            ['Matig (<=6)', 'Voldoende (>6-7.5)', 'Uitstekend (>7.5)'],
        ),
        (
            'heartrisk',
            [5.0, 25.0, 15.0, 25.0, 25.0, 5.0],
            ['Laag (<=10%)', 'Matig (>10-20%)', 'Hoog (>20%)'],
        ),
    ],
)
def test_risico_migratie_binned_gebruikt_domeinschalen(slug, scores, expected_labels):
    df = pd.DataFrame({
        'participant_id': [1, 1, 2, 2, 3, 3],
        'slug': [slug] * 6,
        'completion_created_at': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2025-01-01'),
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2025-01-01'),
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2025-01-01'),
        ],
        'score_value': scores,
    })

    matrix, label_map = visualisaties.bereken_risico_migratie_value_binned(df, slug)

    assert set(matrix['Van']).union(matrix['Na']) == {0, 1, 2}
    assert list(label_map.values()) == expected_labels
    assert matrix['Aantal'].sum() == 3


def test_bmi_migratielijnen_kleuren_naar_midden():
    matrix = pd.DataFrame({
        'Van': [2, 1, 0, -1, 0],
        'Na': [1, 0, -1, 0, 1],
        'Aantal': [1, 1, 1, 1, 1],
    })
    label_map = {
        -2: 'Ernstig ondergewicht (<16)',
        -1: 'Ondergewicht (16-18.5)',
        0: 'Normaal gewicht (18.5-25)',
        1: 'Overgewicht (25-30)',
        2: 'Obesitas (>=30)',
    }

    fig = visualisaties.maak_risico_migratie_sankey(matrix, 'bmi', label_map, lang='nl')
    link_colors = list(fig.data[0].link.color)

    assert link_colors[0] == 'rgba(46, 204, 113, 0.4)'
    assert link_colors[1] == 'rgba(46, 204, 113, 0.4)'
    assert link_colors[2] == 'rgba(231, 76, 60, 0.4)'
    assert link_colors[3] == 'rgba(46, 204, 113, 0.4)'
    assert link_colors[4] == 'rgba(231, 76, 60, 0.4)'
