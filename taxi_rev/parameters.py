

ESTIMATOR_NAME = 'Lasso'

MODEL_VERSION = 'v0'

LOCAL_DATA_LOCATION = 'raw_data/train_1k.csv'



STORAGE_LOCATION = f'models/{ESTIMATOR_NAME}/model.joblib'


# ----------------------------------
#      Google Cloud
# ----------------------------------
# CREATE BUCKET
# project id - replace with your GCP project id
PROJECT_ID='le-wagon-bootcamp-313312'
# bucket name - replace with your GCP bucket name
BUCKET_NAME='wagon-data-633-pighin_rev'
BUCKET_DATA_LOCATION = 'data/train_1k.csv'



# for local training
# DATA_LOCATION = LOCAL_DATA_LOCATION

# for GCS training
DATA_LOCATION = f"gs://{BUCKET_NAME}/{BUCKET_DATA_LOCATION}"


