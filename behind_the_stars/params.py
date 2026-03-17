import os
from dotenv import load_dotenv

##################  VARIABLES  ##################
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")

DATA_FILENAME = os.environ.get("DATA_FILENAME")
LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")


RANDOM_STATE = 42

# #MLflow and Prefect
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

GAR_IMAGE = os.environ.get("GAR_IMAGE")
