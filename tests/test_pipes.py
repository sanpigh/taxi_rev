import numpy as np
import pandas as pd
from taxi_rev.data import get_data, clean_data
from taxi_rev.encoders import DistanceTransformer, TimeFeaturesEncoder, set_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder




def test_dist_pipe():
    df = clean_data(get_data(100))    
    dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
    ])
    distance = dist_pipe.fit_transform(df)
    assert (np.absolute(distance) < 10.0).sum() == len(distance),\
        'Check outliers: a distance point is outside 10 sigmas from mean taxi distance'
    

def test_time_pipe():
    df = clean_data(get_data(100))
    time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    times = time_pipe.fit_transform(df).toarray()
    logic1 = times == 1. 
    logic0 = times == 0.
    print(np.shape(times))
    shape = np.shape(times)
    assert (logic1+logic0).sum() == shape[0]*shape[1], 'Onehotencoder output != 0 or 1'

def test_set_pipeline():
    df = clean_data(get_data(100))
    y = df.pop('fare_amount')
    set_pipe_line = set_pipeline()
    set_pipe_line.fit(df, y)
    assert set_pipe_line.score(df, y) <= 1.0
    


    
if __name__ == '__main__':
    print(test_set_pipeline())
    # dist_pipe = Pipeline([
    #     ('dist_trans', DistanceTransformer()),
    #     ('stdscaler', StandardScaler())
    # ])
    # distance = dist_pipe.fit_transform(df)
    # print(len(distance))
    # print(type(distance))
    # time_pipe = test_time_pipe()
    # times = time_pipe.fit_transform(df).toarray()
    # print(times)
    
    
    