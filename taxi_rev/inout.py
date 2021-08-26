import os
from taxi_rev.parameters import STORAGE_LOCATION, BUCKET_NAME, MODEL_LOCATION
from google.cloud import storage
import joblib


''' CONTAINS INPUT OUTPUT FUNCTIONS'''

def save_estimator_to_gcp():
    ''' Save model into models'''
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model.joblib')


def save_estimator(model):
    """method that saves the model into a .joblib file and uploads it on 
    Google Storage /models folder
    use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(model, 'model.joblib')
    print('saved model.joblib locally in ./')

    # Implement here
    save_estimator_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


def download_model(rm=True):



#    client = storage.Client()
#    bucket = client.get_bucket(BUCKET_NAME)
#    blob = bucket.get_blob(STORAGE_LOCATION)
#    blob.download_to_filename('model.joblib')
#    print("=> pipeline downloaded from storage")
    
    
#    model = joblib.load('model.joblib')
    model = joblib.load(MODEL_LOCATION)
#    print("=> model loaded")
#    if rm:
#        os.remove('model.joblib')
    return model