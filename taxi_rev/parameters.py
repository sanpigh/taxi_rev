import os


MODEL_VERSION = 'v1'


# ----------------------------------
#      Google Cloud
# ----------------------------------
# CREATE BUCKET
# project id - replace with your GCP project id



ESTIMATOR_NAME = os.environ['ESTIMATOR_NAME']
LOCAL          = os.environ['LOCAL']

if LOCAL == 'true':
    DATA_LOCATION    = os.environ['LOCAL_DATA_LOCATION']
    TEST_LOCATION    = os.environ['LOCAL_TEST_LOCATION']
    PREDICT_LOCATION = os.environ['LOCAL_PREDICT_LOCATION']
    tmp = os.environ['LOCAL_MODEL_LOCATION']
    LOCAL_MODEL_LOCATION   = f'{tmp}/{ESTIMATOR_NAME}/model.joblib'
else:
    DATA_LOCATION    = 'dfgsdfgsdfg'
    TEST_LOCATION    = 'ddddd'
    PREDICT_LOCATION = 'sssss'
    
# for GCS training
STORAGE_LOCATION = f'models/{ESTIMATOR_NAME}/model.joblib'
PROJECT_ID='le-wagon-bootcamp-313312'
BUCKET_NAME='wagon-data-633-pighin_rev'
BUCKET_DATA_LOCATION = 'data/train_1k.csv'
#DATA_LOCATION = f"gs://{BUCKET_NAME}/{BUCKET_DATA_LOCATION}"
# bucket name - replace with your GCP bucket name
