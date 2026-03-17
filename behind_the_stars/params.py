import os
from dotenv import load_dotenv

##################  VARIABLES  ##################
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")

DATA_FILENAME = os.environ.get("DATA_FILENAME")
LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")

RANDOM_STATE = 42

GAR_IMAGE = os.environ.get("GAR_IMAGE")
