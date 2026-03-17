import os
import pickle
import pandas as pd
from google.cloud import storage
from behind_the_stars.params import *
from tensorflow import keras
from behind_the_stars.params import *
import mlflow
from mlflow.tracking import MlflowClient


###------------------# Load functions #-----------------------###
def load_dataset():
    """
    It charges the dataset from the bucket in GCS
    """
    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Downloads demo_names.parquet
        blob = bucket.blob("demo_names.parquet")
        path_to_file = "/tmp/demo_names.parquet"
        blob.download_to_filename(path_to_file)

        return pd.read_parquet(path_to_file)
    return None


def load_model():
    """
    It charges model and vectorizer from GCS
    """
    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Temporary routes
        model_path = "/tmp/xgb_model_final.pkl"
        vec_path = "/tmp/tfidf_vectorizer_final.pkl"

        # Downloads from bucket
        bucket.blob("xgb_model_final.pkl").download_to_filename(model_path)
        bucket.blob("tfidf_vectorizer_final.pkl").download_to_filename(vec_path)

        # Loads with Pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vec_path, 'rb') as f:
            vectorizer = pickle.load(f)

        print("pkl files succesfully downloaded")
        return model, vectorizer

    return None, None

# def load_model_topics():
#     """
#     It charges the topics model information
#     """
#     if MODEL_TARGET == "gcs":
#         client = storage.Client()
#         bucket = client.bucket(BUCKET_NAME)

#         # Temproary routes
#         model_path = "/tmp/xgb_model.pkl"
#         vec_path = "/tmp/tfidf_vectorizer.pkl"

#         # Downloads from bucket
#         bucket.blob("xgb_model.pkl").download_to_filename(model_path)
#         bucket.blob("tfidf_vectorizer.pkl").download_to_filename(vec_path)

#         # Loads with Pickle
#         with open(model_path, 'rb') as f:
#             model = pickle.load(f)
#         with open(vec_path, 'rb') as f:
#             vectorizer = pickle.load(f)

#         print("✅ Archivos pkl cargados exitosamente.")
#         return model, vectorizer

#     return None, None
