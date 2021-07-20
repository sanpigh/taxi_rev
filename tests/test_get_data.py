from taxi_rev.data import get_data


def test_get_data():
    df = get_data(1)
    assert len(df) == 1