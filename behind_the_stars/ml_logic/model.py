from preprocessor import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def nlp_model_fitting(X,y):

    '''Fonction permettant la création et le fitting de notre modèle NLP.
        X: notre colonne de review
        y: ?? is_open potentiellemnt
        Retourne le model et le vectorizer (sous forme de model , vectorizer)'''

    model = MultinomialNB()
    vectorizer = TfidfVectorizer()

    X_processed = X.map(lambda x: preprocessing(x))

    X_vectorized = vectorizer.fit_transform(X_processed['text']).toarray()

    model.fit(X_vectorized,y)

    return model,vectorizer

def nlp_predict(X,model,vectorizer):

    '''Fonction permettant la prédiction de is_open en fonction de reviews.
        X: reviews
        model: Model NLP fitté.
        vectorizer: Vectorizer fitté
        Retourne y_pred (ouvert ou pas) et y_pred_proba (probabilité de la predict)'''

    X_processed = X.map(lambda x: preprocessing(x))

    X_vectorized = vectorizer.transform(X_processed['text']).toarray()
    y_pred = model.predict(X_vectorized)
    y_pred_proba = model.predict_proba(X_vectorized)

    return y_pred,y_pred_proba

if __name__ == '__main__':
    path = '../raw_data/yelp_10k_sample_strat.parquet'
    data = pd.read_parquet(path)

    X = data[['text']]
    print(f'X shape : {X.shape}')
    y = data['is_open']
    print(f'y shape : {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y)
    print(f'X train shape : {X_train.shape}')
    print(f'X test shape : {X_test.shape}')
    print(f'y train shape : {y_train.shape}')
    print(f'y test shape : {y_test.shape}')

    model, vectorizer = nlp_model_fitting(X_train,y_train)
    y_pred , y_proba = nlp_predict(X_test,model,vectorizer)

    print(X_test)
    print(y_pred)
