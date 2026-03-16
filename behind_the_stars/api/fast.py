import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from behind_the_stars.ml_logic.preprocessor import preprocessing
# from behind_the_stars.ml_logic.registry import load_model --> En attente du travail de Kenny-D

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {'greeting': 'Hello'}


# @app.get("/predict")
# def predict(
#         restaurant : str
#     ):      # 1
#     """
#     given a number of reviews (up to one hundred), predict the likelyhood of closure
#     """
#     df = get_data()
#     X_pred = df[df['restaurant']==restaurant]['text'].tolist()
#     X_vect = vectorizer(X_pred)
#     y_pred = app.state.model.predict(X_vect)
#     proba = {'closure likelyhood' : float(y_pred)}
#     return proba
