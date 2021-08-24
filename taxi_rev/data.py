from taxi_rev.parameters import DATA_LOCATION

import pandas as pd


# implement get_data() function
def get_data(nrows=1000):
    """returns a DataFrame with nrows from s3 bucket"""
    df = pd.read_csv(DATA_LOCATION, nrows=nrows)
    return df


def clean_data(df, test=False):
    """returns a DataFrame without outliers and missing values"""
    df = df.dropna(how="any")
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == "__main__":
    df = get_data(100)
    print(df.head())
    print(len(df))

    df = clean_data(df)
    print(len(df))
