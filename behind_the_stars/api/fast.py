import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.registry import load_dataset, load_model
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


@app.get("/predict")
def predict(
        restaurant : str
    ):      # 1
    """
    given a number of reviews (up to one hundred), predict the likelyhood of closure
    """
    df = load_dataset()
    model, vectorizer = load_model()
    try :
        X_pred = df[df['name']==restaurant]['text'].tolist()
    except :
        print('no restaurant found')
    X_vect = vectorizer(X_pred)
    y_pred = model.predict(X_vect)
    if y_pred==0:
        y_proba = model.predict_proba(X_vect)[0]
        proba = {'closure likelyhood' : float(y_proba)}
    else:
        y_proba = model.predict_proba(X_vect)[1]
        proba = {'open likelyhood' : float(y_proba)}
    return proba
