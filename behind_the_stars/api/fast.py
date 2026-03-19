import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
import csv
import codecs
from fastapi.middleware.cors import CORSMiddleware
from behind_the_stars.ml_logic.registry import load_small_dataset, load_model, load_embed
from behind_the_stars.ml_logic.embedding import get_recommendations_for_new_resto
from behind_the_stars.ml_logic.preprocessor import megatron_final
from behind_the_stars.models.bertopics_model import bertopic_model, bertopic_features, add_restaurant_topic_features
from io import BytesIO
import pickle as pkl
import re
# import joblib
from behind_the_stars.params import *
# from behind_the_stars.models.bertopics_model import master_topics
# import joblib

app = FastAPI()
small_df = load_small_dataset()
model, vectorizer = load_model()
embed_model = load_embed()
with open('/raw_data/pickle_embeddings.pkl', 'rb') as f:
    small_embed = pkl.load(f)

# fonction regex
def extract_reviews(text):
    # Extraire tout ce qui est entre "..." ou '...'
    reviews = re.findall(r'"(.*?)"|\'(.*?)\'', text, re.DOTALL)

    # re.findall retourne des tuples → on nettoie
    reviews = [r[0] if r[0] else r[1] for r in reviews]

    return reviews
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
    if restaurant in small_df['name'].values:
        X_pred = small_df[small_df['name']==restaurant]['text'].tolist()

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
    temp = pd.read_csv(BytesIO(content), engine="python")
    df = megatron_final(df)
    X_pred = df['text']
    X_vect = vectorizer.transform(X_pred)[0,]
    y_pred = model.predict(X_vect)
    if y_pred[0]==0:
        y_proba = model.predict_proba(X_vect)[0][0]
        proba = {'closure likelyhood' : float(y_proba)}

    else:
        y_proba = model.predict_proba(X_vect)[0][1]
        proba = {'open likelyhood' : float(y_proba)}
    X_embed = np.mean(embed_model.encode(X_pred), axis=0)
    recommandation = get_recommendations_for_new_resto(X_embed, small_embed, small_df)
    print('etape keny done')
    clean = temp
    # # temp = megatron_final(temp, use_lemmatizer=False)
    # for k in temp['text']:
    #     temp['text']=k
    #     clean = pd.concat([temp,clean])
    clean["text"] = clean["text"].apply(extract_reviews)

    # Explosion en lignes
    clean = clean.explode("text").reset_index(drop=True)
    print(clean)
    topics, probs, topic_model = bertopic_model(clean['text'],nr_topics=4, min_topic_size=2)
    dataframe = bertopic_features(clean,clean['text'],topic_model,topics)
    last_etape = add_restaurant_topic_features(dataframe,text_col='text', restaurant_col='business_id', topic_col='topic_main', date_col='date')
    last_etape = last_etape.drop(["Unnamed: 0", "is_open"], axis=1)
    print(last_etape)
    # topic_dico = dict(last_etape)
    topic_dico = last_etape.to_dict(orient="records")
    print(topic_dico)
    return {'proba':proba , 'recommandation': recommandation, 'topic_dico':topic_dico}

# @app.post("/api_topics/")
# async def predict_from_csv(data:  UploadFile = File(...)):
#     content = await data.read()
#     topic_df = pd.read_csv(BytesIO(content))
#     X_topic_pred = topic_df['text']
#     X_topic_model = topic_model
# topic_df = megatron_final(topic_df, use_lemmatizer=False)
