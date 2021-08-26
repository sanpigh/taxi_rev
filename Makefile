
export ESTIMATOR_NAME=Lasso

# ----------------------------------
#      Run Local
# ----------------------------------
LOCAL_DATA_LOCATION=raw_data/train_1k.csv
LOCAL_MODEL_LOCATION=models
LOCAL_TEST_LOCATION=raw_data/test.csv
LOCAL_PREDICT_LOCATION=raw_data/prediction.csv

# ----------------------------------
#      Google Cloud
# ----------------------------------
# CREATE BUCKET
# project id - replace with your GCP project id
PROJECT_ID=le-wagon-bootcamp-313312
# bucket name - replace with your GCP bucket name
BUCKET_NAME=wagon-data-633-pighin_rev
# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1
BUCKET_TRAIN_DATA_PATH=data/train_1k.csv




# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* taxi_rev/*.py

black:
	@black scripts/* taxi_rev/*.py

test:
	@coverage run -m pytest tests/test*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr taxi_rev-*.dist-info
	@rm -fr taxi_rev.egg-info

install:
	@pip install -U -e .

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''


run_local:
	@python -m taxi_rev.train


run_predict_local: export DATA_LOCATION=${LOCAL_DATA_LOCATION} 
run_predict_local: export TEST_LOCATION=${LOCAL_TEST_LOCATION}
run_predict_local: export PREDICT_LOCATION=${LOCAL_PREDICT_LOCATION}
run_predict_local: export MODEL_LOCATION=${LOCAL_MODEL_LOCATION}
run_predict_local:
	python -m taxi_rev.predict

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)











set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}



# SUBMIT PROCESS TO AI

PACKAGE_NAME=taxi_rev
# Name of the package

FILENAME=train
# Name of the module containing the main

BUCKET_TRAINING_FOLDER=trainings
# see job-dir. It contains the a zip file with the package.

PYTHON_VERSION=3.7
RUNTIME_VERSION=2.5
# These versions must be compatible.

JOB_NAME=taxi_rev_$(shell date +'%Y%m%d_%H%M%S')
# Name of the job

# -- job-dir
# Cloud Storage path in which to store training outputs and other data needed for training.
# This path will be passed to your TensorFlow program as the --job-dir command-line arg. 

# --package-path
#   Path to a Python package to build. This should point to a local directory containing the 
# Python source for the job. It will be built using setuptools (which must be installed) using 
#its parent directory as context. If the parent directory contains a setup.py file, the build 
#will use that; otherwise, it will use a simple built-in one. 

--module-name=train
#    Name of the module to run. 

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
  --job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
  --package-path ${PACKAGE_NAME} \
  --module-name ${PACKAGE_NAME}.${FILENAME} \
  --python-version=${PYTHON_VERSION} \
  --runtime-version=${RUNTIME_VERSION} \
  --region ${REGION} \
  --stream-logs