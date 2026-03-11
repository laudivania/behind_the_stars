import os
from dotenv import load_dotenv

#GCP
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

DATA_FILENAME = os.environ.get("DATA_FILENAME")
LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")

RANDOM_STATE = 42
