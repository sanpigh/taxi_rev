
from taxi_rev.parameters import data_path

import pandas as pd



# implement get_data() function
def get_data(nrows=1000):
    '''returns a DataFrame with nrows from s3 bucket'''
    # aws_path = "s3://wagon-public-datasets/taxi-fare-train.csv"
    df = pd.read_csv(data_path, nrows=nrows)
    return df

if __name__ == '__main__':
    df = get_data()
    print(df.head())
    print(len(df))