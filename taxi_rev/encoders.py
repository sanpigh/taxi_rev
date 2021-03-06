import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# df es el dataframe sobre el que voy a trabajar. El resto de las variables son los nombres de las columnas sobre las que voy a operar
def haversine_vectorized(
    df,
    start_lat="pickup_latitude",
    start_lon="pickup_longitude",
    end_lat="dropoff_latitude",
    end_lon="dropoff_longitude",
):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized version of the haversine distance for pandas df.
    Computes the distance in kms.
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(
        df[start_lon].astype(float)
    )
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(
        df[end_lon].astype(float)
    )
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


# create a TimeFeaturesEncoder
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
    Extracts the day of week (dow), the hour, the month and the year from a time column.
    Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    # Esta clase va a necesitar la variable time_column para ser instanciada

    def __init__(self, time_column, time_zone_name="America/New_York"):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[["dow", "hour", "month", "year"]]


# create a DistanceTransformer
# Si entend?? bien, BaseEstimator se usa para heredar la clase get_params que nos dice todos los par??metros necesarios para instanciar
# la clase. TransformerMixin se usa para heredar el m??todo fit_transform
# Esta clase no necesita parametros de entrada ya que tiene valores default


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Computes the haversine distance between two GPS points.
    Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(
        self,
        start_lat="pickup_latitude",
        start_lon="pickup_longitude",
        end_lat="dropoff_latitude",
        end_lon="dropoff_longitude",
    ):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(
            X_,
            start_lat=self.start_lat,
            start_lon=self.start_lon,
            end_lat=self.end_lat,
            end_lon=self.end_lon,
        )
        return X_[["distance"]]


def set_preproc_pipe():
    """returns a pipelined model"""
    dist_pipe = Pipeline(
        [("dist_trans", DistanceTransformer()), ("stdscaler", StandardScaler())]
    )
    time_pipe = Pipeline(
        [
            ("time_enc", TimeFeaturesEncoder("pickup_datetime")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preproc_pipe = ColumnTransformer(
        [
            (
                "distance",
                dist_pipe,
                [
                    "pickup_latitude",
                    "pickup_longitude",
                    "dropoff_latitude",
                    "dropoff_longitude",
                ],
            ),
            ("time", time_pipe, ["pickup_datetime"]),
        ],
        remainder="drop",
    )
    return preproc_pipe


class GetPipeline():
    
    def set_preproc_pipe(self):
        self.preproc_pipe = set_preproc_pipe()
        
    def get_estimator(self):
        if self.estimator_name == 'Lasso':
            estimator = Lasso()
        elif self.estimator_name == 'Ridge':
            estimator = Ridge()
        elif self.estimator_name == 'Linear':
            estimator = LinearRegression()
        elif self.estimator_name == 'GBM':
            estimator = GradientBoostingRegressor()
        elif self.estimator_name == 'DecisionTree':
            estimator = DecisionTreeRegressor()
        elif self.estimator_name == 'RandomForest':
            estimator = RandomForestRegressor()
            # self.estimator_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
            #     'max_features': ['auto', 'sqrt']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        elif self.estimator_name == 'Xgboost':
            estimator = XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=10, learning_rate=0.05,
                                 gamma=3)
        estimator.set_params(**self.estimator_params)
        self.estimator = estimator

    def set_pipeline(self):
        self.set_preproc_pipe()
        self.get_estimator()
        self.pipeline = Pipeline(
            [('preproc', self.preproc_pipe), ('estimator', self.estimator)]
        )
