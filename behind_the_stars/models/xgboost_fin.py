import pandas as pd
import numpy as np
from behind_the_stars.ml_logic.registry import load_model
from behind_the_stars.ml_logic.preprocessor import clean_single_text, megatron_final

def predict_batch(df: pd.DataFrame):
    """
    It loads model and vectorizer and make predictions on a given Dataframe.
    It returns:
    - A label list
    - A probability list of the predicted label.
    """
    model, vectorizer = load_model()

    if model is None or vectorizer is None:
        print("Error loading model and vectorizer.")
        return [], []

    print(f"- Preprocessing {len(df)} reviews -")
    df['clean_text'] = df['text'].apply(clean_single_text)

    X_vec = vectorizer.transform(df['clean_text'])

    y_pred = model.predict(X_vec)
    y_proba_all = model.predict_proba(X_vec)

    results_list = ["Open" if p == 1 else "Closed" for p in y_pred]

    probabilities_list = [round(float(y_proba_all[i][y_pred[i]]), 4) for i in range(len(y_pred))]

    return results_list, probabilities_list

if __name__ == "__main__":
    test_df = pd.DataFrame({
        'name': ['Pasta Eater'],
        'text': ['Really amazing staff that truly goes above and beyond! From the moment we walked in, the service was warm, attentive, and thoughtful. The food was incredible, the drinks were perfectly crafted, and the overall atmosphere made it such a great spot for a night out. Highly recommend for dinner and drinks. Can’t wait to come back! ']
    })

    res, prob = predict_batch(test_df)
    print(f"Result: {res[0]} | Confidence: {prob[0]*100}%")
