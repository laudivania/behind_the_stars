import os
import pickle
import pandas as pd
from google.cloud import storage
from behind_the_stars.params import *
from tensorflow import keras
from behind_the_stars.params import *
# import mlflow
# from mlflow.tracking import MlflowClient
# import mlflow
# from mlflow.tracking import MlflowClient
import zipfile
# import joblib
from sentence_transformers import SentenceTransformer


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
    print(MODEL_TARGET)
    # if MODEL_TARGET == "gcs":
    print('juste apres if model target')
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    print('juste apres instantiation model')
    # Temporary routes
    model_path = "/tmp/xgb_model_final.pkl"
    vec_path = "/tmp/tfidf_vectorizer_final.pkl"
    print('juste apres vec path')
    # Downloads from bucket
    bucket.blob("xgb_model_final.pkl").download_to_filename(model_path)
    bucket.blob("tfidf_vectorizer_final.pkl").download_to_filename(vec_path)
    print('no break here after bucket blob')
    # Loads with Pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(model)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(vectorizer)
    print("pkl files succesfully downloaded")
    assert model is not None
    assert vectorizer is not None
    return model, vectorizer

    # return None, None

def load_embed():
    """
    Télécharge et charge le modèle SentenceTransformer depuis GCS
    """
    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        zip_path = "/tmp/bert_model.zip"
        extract_path = "/tmp/bert_model_folder"


        blob = bucket.blob("bert_model.zip")
        blob.download_to_filename(zip_path)


        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print("BERT model downloaded and extracted")

        model = SentenceTransformer(extract_path)
        return model

    # Fallback
    return SentenceTransformer('all-MiniLM-L6-v2')

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
def load_small_dataset():
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
        small_df = pd.read_parquet(path_to_file)
        small_df = small_df.head(50)
        dic_fin = {"business_id":["s6Nb9L-4r9MkLPy07ajIeg",  "EKSmnS-fup3HNFLR9J17mQ", "cAgwUJ5oMhrm_WVbB0Q1Fg", "ZiEd4l1qEnLJFcF-K_3NgQ", "MnfAzLt3qp0CkMHuQF7cvg"],"text": ['a','b','c','d','e'], 'is_open':[1,1,1,1,1],'name':["Midtown II Restaurant","Quartermaster Store","Peppy Grill","Appetite's Delight","New Town Restaurant"]}
        test_df =pd.DataFrame(dic_fin)
        final = pd.concat([small_df, test_df], ignore_index=True)
        return final
    return None
