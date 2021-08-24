
from taxi_rev.parameters import DATA_PATH, STORAGE_LOCATION, BUCKET_NAME
from google.cloud import storage
import joblib


''' CONTAINS INPUT OUTPUT FUNCTIONS'''

def save_estimator_to_gcp():
    ''' Save model into models'''
    print(BUCKET_NAME, STORAGE_LOCATION)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model.joblib')


def save_estimator(model):
    """method that saves the model into a .joblib file and uploads it on 
    Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(model, 'model.joblib')
    print('saved model.joblib locally in ./')

    # Implement here
    save_estimator_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")
