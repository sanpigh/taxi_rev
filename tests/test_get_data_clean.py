from taxi_rev.data import get_data, clean_data


def test_get_data_clean():
    df = get_data(100)
    len_df = len(df)
    assert len_df == 100
    df = clean_data(df)
    assert len(df) <= len_df
    