import data


def test_dataframe_has_expected_columns():
    assert list(data.df.columns) == ["cancer_indication", "gene", "median_value"]


def test_dataframe_has_81_rows():
    assert len(data.df) == 81


def test_cancers_list_returns_10_unique_sorted():
    cancers = data.cancers()
    assert len(cancers) == 10
    assert cancers == sorted(cancers)
    assert "lung" in cancers
    assert "breast" in cancers
    assert "esophageal" not in cancers
