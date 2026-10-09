import pandas as pd

import visualisaties


def test_dedicated_analytics_data_contains_filterable_year(monkeypatch):
    analytics = pd.DataFrame({
        'date': ['2024-06-01T10:00:00', '2025-06-01T10:00:00'],
        'user_id': [1, 2],
    })

    monkeypatch.setattr(
        visualisaties,
        'load_table_from_database',
        lambda table_name, db_url: analytics.copy(),
    )
    visualisaties._laad_analytics_data.clear()

    result = visualisaties._laad_analytics_data('sqlite://')

    assert result['year'].tolist() == [2024, 2025]
    assert result['jaar'].tolist() == [2024, 2025]
    assert result['created_at'].notna().all()


def test_kpi_and_daily_trend_calculate_views_per_user():
    analytics = pd.DataFrame({
        'created_at': pd.to_datetime([
            '2025-01-01 09:00:00',
            '2025-01-01 10:00:00',
            '2025-01-02 09:00:00',
        ]),
        'user_id': [1, 1, 2],
        'views': [1, 1, 1],
        'is_new_user': [True, False, True],
        'is_returning_user': [False, True, False],
    })

    kpis = visualisaties.bereken_kpi_metrics(analytics)
    trends = visualisaties.bereken_dagelijkse_trends(analytics)

    assert kpis['total_users'] == 2
    assert kpis['views'] == 3
    assert kpis['views_per_user'] == 1.5
    assert trends['views_per_user'].tolist() == [2.0, 1.0]