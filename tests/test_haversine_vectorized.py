from taxi_rev.encoders import haversine_vectorized
from taxi_rev.data import get_data, clean_data






def test_haversine_vectorized():
    df = get_data(10)
    df = clean_data(df)
    len_df = len(df)
    assert (haversine_vectorized(df) >= 0.0).sum() == len_df
    # assert haversine_vectorized(df) is float
    
    
