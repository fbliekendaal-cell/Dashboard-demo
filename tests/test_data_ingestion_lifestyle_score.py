from types import SimpleNamespace

import pandas as pd
import pytest

import data_ingestion


@pytest.mark.parametrize("stored_score", [None, 7.2])
def test_lifestyle_score_is_never_derived_from_mixed_scale_inputs(monkeypatch, stored_score):
    source = pd.DataFrame({
        "participant_id": [1],
        "user_id": [1],
        "rec_ls_score_exercise": [570.0],
        "rec_ls_score_fruit": [2.0],
        "rec_ls_score_natrium": [2400.0],
    })
    if stored_score is not None:
        source["rec_ls_lifestyle_score"] = [stored_score]

    monkeypatch.setattr(data_ingestion, "create_engine", lambda _: object())
    monkeypatch.setattr(
        data_ingestion,
        "inspect",
        lambda _: SimpleNamespace(get_table_names=lambda: ["users_met_scores"]),
    )
    monkeypatch.setattr(
        data_ingestion,
        "load_table_from_database",
        lambda *_: source.copy(),
    )
    monkeypatch.setattr(
        data_ingestion,
        "_merge_latest_scores_into",
        lambda df, _: df,
    )

    result = data_ingestion.load_my_clic_participants_expanded("sqlite://")

    assert "rec_ls_lifestyle_score" in result.columns
    if stored_score is None:
        assert pd.isna(result.loc[0, "rec_ls_lifestyle_score"])
    else:
        assert result.loc[0, "rec_ls_lifestyle_score"] == stored_score
