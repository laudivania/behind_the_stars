import pandas as pd
from fastapi import FastAPI, File, UploadFile
import csv
import codecs
from fastapi.middleware.cors import CORSMiddleware
from behind_the_stars.ml_logic.registry import load_dataset, load_model
from io import BytesIO
# import joblib

# from behind_the_stars.models.bertopics_model import master_topics
# import joblib

app = FastAPI()
# df = load_dataset()
model, vectorizer = load_model()
# model_topics = joblib.load("model_saved/bertopic.pkl")

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
    return {'Welcome to the best project of batch #2195! :)'}


@app.get("/predict")
def predict(
        restaurant : str
    ):      # 1
    """
    given a number of reviews (up to one hundred), predict the likelyhood of closure
    """
    if restaurant in df['name'].values:
        X_pred = df[df['name']==restaurant]['text'].tolist()

        X_vect = vectorizer.transform(X_pred)

        y_pred = model.predict(X_vect)

        if y_pred[0]==0:
            y_proba = model.predict_proba(X_vect)[0][0]
            proba = {'closure likelyhood' : float(y_proba)}

        else:
            y_proba = model.predict_proba(X_vect)[0][1]
            proba = {'open likelyhood' : float(y_proba)}

    else:
        proba = {'open likelyhood' : 'could not answer, no restaurant found'}
    return proba

@app.post("/api_csv/")
async def predict_from_csv(data:  UploadFile = File(...)):
    content = await data.read()
    df = pd.read_csv(BytesIO(content))
    X_pred = df['text']
    X_vect = vectorizer.transform(X_pred)[0,]
    y_pred = model.predict(X_vect)
    if y_pred[0]==0:
        y_proba = model.predict_proba(X_vect)[0][0]
        proba = {'closure likelyhood' : float(y_proba)}

    else:
        y_proba = model.predict_proba(X_vect)[0][1]
        proba = {'open likelyhood' : float(y_proba)}
    return proba
