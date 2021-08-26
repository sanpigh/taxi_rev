

from pickle import FALSE
import pandas as pd
from taxi_rev.parameters import LOCAL_TEST_LOCATION, LOCAL_PREDICT_LOCATION, get_data_location
from taxi_rev.inout import download_model

if __name__ == '__main__':
    print(get_data_location())
    quit()
    X_test = pd.read_csv(LOCAL_TEST_LOCATION)
    trainer_pipeline = download_model(FALSE)
    y_test = trainer_pipeline.predict(X_test)
    df = pd.DataFrame({
        'key' : X_test['key'],
        'fare_amount' : y_test
    })
    df.to_csv(LOCAL_PREDICT_LOCATION, index=False)