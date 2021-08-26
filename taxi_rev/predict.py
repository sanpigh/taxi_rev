

from pickle import FALSE
import pandas as pd
from taxi_rev.parameters import TEST_LOCATION, PREDICT_LOCATION
from taxi_rev.inout import download_model

if __name__ == '__main__':
    X_test = pd.read_csv(TEST_LOCATION)
    trainer_pipeline = download_model(FALSE)
    y_test = trainer_pipeline.predict(X_test)
    df = pd.DataFrame({
        'key' : X_test['key'],
        'fare_amount' : y_test
    })
    df.to_csv(PREDICT_LOCATION, index=False)