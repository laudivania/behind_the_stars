import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
# from pandarallel import pandarallel
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')  # For nltk>=3.9.0
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import TextVectorization

from nltk.stem import WordNetLemmatizer
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import CountVectorizer

import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud

#---------------------------- End of libraries -------------------------


# Initialisation une seule fois
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocessing(sentence, minuscule=True, ponctuation=True, remove_stopwords=True):
    # gérer les valeurs manquantes
    if not isinstance(sentence, str):
        return ""
    # enlever espaces
    sentence = sentence.strip()
    # minuscules
    if minuscule:
        sentence = sentence.lower()
    # enlever chiffres
    sentence = ''.join(char for char in sentence if not char.isdigit())
    # enlever ponctuation
    if ponctuation:
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')
    # tokenization
    tokens = word_tokenize(sentence)
    # enlever stopwords
    if remove_stopwords:
        tokens = [w for w in tokens if w not in stop_words]
    # lemmatization verbes
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
    # lemmatization noms
    tokens = [lemmatizer.lemmatize(word, pos="n") for word in tokens]
    # reconstruire la phrase
    sentence = " ".join(tokens)
    return sentence


lemmatizer = WordNetLemmatizer()
DetectorFactory.seed = 42
def detect_language(text):
    """
    Detects language for a given text
    """
    try:
        if pd.isna(text) or str(text).strip() == "":
            return "Unknown language"
        return detect(text)
    except:
        return "Error"

def basic_cleaning(sentence:str) -> str:
    """Removes whitespaces, converts to lowercase, strips digits
    and remove all puntuation"""

    sentence = sentence.strip().lower()
    sentence = "".join(char for char in sentence if not char.isdigit())

    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")
    return sentence

def lemmatize_verbs(sentence: str) -> str:
    """Tokenizes the sentence and reduces verbs to their base forms"""

    tokens = word_tokenize(sentence)
    verb_lemmas = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
    return " ".join(verb_lemmas)

def lemmatize_nouns(sentence: str) -> str:
    """Tokenizes the sentence and reduces verbs to their base forms"""

    tokens = word_tokenize(sentence)
    noun_lemmas = [lemmatizer.lemmatize(word, pos="n") for word in tokens]
    return " ".join(noun_lemmas)

def full_preprocessing(sentence: str) -> str:
    """Conbines basic_cleaning, lemmatize_verb and lemmatize_nouns into a
    single pipeline"""

    sentence = basic_cleaning(sentence)
    sentence = lemmatize_verbs(sentence)
    sentence = lemmatize_nouns(sentence)
    return sentence

# ------- Function Master Preprocessing--------
# Simple functions

def strip_ascii(text: str):
    """ It removes emojis and characters not included
        in standard ASCII"""
    return text.encode("ascii", "ignore").decode("ascii")

def reduce_lengthening(text: str):
    """ It transforms "loooove" in "loove", reducing length
    "goooood" in "good" """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def fine_cleaning(text: str):
    """Advanced cleaning with regex and character filtering"""
    # It begins with URLs and mentions
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = reduce_lengthening(text)
    text = strip_ascii(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_junk_review(text, threshold_vowels=0.15, max_rep=3):
    """ It detects series of repeated words, keyboard patterns and filtering
        by number of vowels (English usually has 30-40% of vowels)"""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return True

    repetition_pattern = re.compile(r'(\w{2,})\1{' + str(max_rep) + ',}', re.IGNORECASE)
    if repetition_pattern.search(text):
        return True

    keyboard_patterns = ['asdf', 'ghjk', 'qwerty', 'zxcv', '12345']
    if any(seq in text.lower() for seq in keyboard_patterns):
        return True

    letters_only = re.sub(r'[^a-zA-Z]', '', text)
    if len(letters_only) > 8:
        vowels = len(re.findall(r'[aeiouAEIOU]', letters_only))
        if (vowels / len(letters_only)) < threshold_vowels:
            return True
    return False

def basic_cleaning(sentence: str) -> str:
    """ It deletes spaces, sets words in lowercase and remove digits."""
    sentence = sentence.strip().lower()
    sentence = "".join(char for char in sentence if not char.isdigit())
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")
    return sentence

def lemmatize_all(sentence: str) -> str:
    """It lemmatizes verbs and nouns."""
    tokens = word_tokenize(sentence)
    # Lematizamos verbos y luego sustantivos sobre el resultado
    lemmas = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
    lemmas = [lemmatizer.lemmatize(word, pos="n") for word in lemmas]
    return " ".join(lemmas)

# The orchestrator

def master_preprocessor(text: str):
    """
    It combines all the functions in an optimum order. It returns
    cleaned_text and is_junk as a boolean.
    """
    # Cleaning (Regex/URLs/ASCII)
    cleaned = fine_cleaning(text)

    # Quality control before lemmatizing (Performance issues)
    junk_flag = is_junk_review(cleaned)

    # 3. Basic cleaning for NLP
    nlp_ready = basic_cleaning(cleaned)
    nlp_ready = lemmatize_all(nlp_ready)

    return nlp_ready, junk_flag

# Use clean_single_text and megatron_final when applied to a dataframe for demo.

def clean_single_text(text, use_regex=True, remove_stopwords=True, use_lemmatizer=True):
    """Applied to a text, it allows cleaning before applying a model."""
    if isinstance(text, (list, np.ndarray)):
        text = " ".join([str(item) for item in text if item is not None])

    if not isinstance(text, str) or text.strip() == "":
        return ""

    if use_regex:
        text = fine_cleaning(text)

    text = basic_cleaning(text)

    if remove_stopwords or use_lemmatizer:
        tokens = word_tokenize(text)

        if remove_stopwords:
            tokens = [w for w in tokens if w not in stop_words]

        if use_lemmatizer:
            text = " ".join([lemmatizer.lemmatize(w) for w in tokens])
        else:
            text = " ".join(tokens)

    return text

def megatron_final(df, column_name='text', use_regex=True, remove_stopwords=True, use_lemmatizer=True):
    """ After text cleaning , it allows applying clean_single_text to a
        dataset, choosing the column_name"""

    print(f"Megatron processsing {len(df)} rows...")

    df[column_name] = df[column_name].apply(
        clean_single_text,
        use_regex=use_regex,
        remove_stopwords=remove_stopwords,
        use_lemmatizer=use_lemmatizer
    )
    print("Dataset ready for modelling.")
    return df

#-------------- End Master Preprocessing before Vectorizing -----------


#For Machine Learning, Tfidvectorizer
def vectorizing_tfid(preprocessed_data,ngram_range=(2,2),min_df=0.01,
                     max_df=0.8, stop_words="english"):
    """Vectorizing the data with TfidVectorizer after the preprocessing
    in order to used it on the models. This method give us the weights of each
    n-gram """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                             min_df=min_df,
                             max_df=max_df,
                             stop_words=stop_words)
    vectorized_data= vectorizer.fit_transform(preprocessed_data)
    vectorized_data = pd.DataFrame(vectorized_data.toarray(),
                    columns = vectorizer.get_feature_names_out())
    return vectorized_data, vectorizer


#For Machine Learning, CountVectorizer
def vectorizing_countv(preprocessed_data,ngram_range = (2,2),min_df=0.01,
                       max_df=0.8,stop_words="english"):
    """Vectorizing the data with Count Vector after the preprocessing
    in order to used it on the models. This methow will split it by word
    """
    vectorizer= CountVectorizer(ngram_range=ngram_range,
                        min_df=min_df, max_df=max_df,
                        stop_words=stop_words)
    vectorized_data= pd.DataFrame(vectorizer.fit_transform(preprocessed_data).toarray(),
                       columns = vectorizer.get_feature_names_out())
    return vectorized_data, vectorizer


#For Machine learning: Sum the results, to have the weights per n-gram
def sum_and_sort (vectorized_data):
    return vectorized_data.sum(axis=0).sort_values(ascending=False)


#For Deep Learning
def get_vectorizer(X_preproc, vocab_size=3000, output_sequence_length=None):
    """

    To vectorize, aparently ideal for Keras (Deep Learning specific)

        X_preproc: The X  already cleaned strings (from our previous preprocessing).
            # change later the preproc 'sentence' to something else TBC
        vocab_size: 3000 to level, it seems we dont have more than that (might be optimized later)
        output_sequence_length: padding length, I have used a np.percentile(review_lengths, 75) in my initial tests
        (might be optimized later).

    """

    # Initialize
    vectorizer = TextVectorization(
        max_tokens=vocab_size,
        standardize=None,
        output_mode='int',
        output_sequence_length=output_sequence_length
    )

    # Adapt to the dataset (it works like .fit in Keras)
    vectorizer.adapt(X_preproc)

    # Transforms the text into a np matrix
    X_vect = vectorizer(X_preproc).numpy()

    return X_vect

#-------------------------------- End of Vectorizers---------------------------

#Create a word cloud
def word_cloud (text:str, max_words, background_color="white"):
    wordcloud = WordCloud(background_color=background_color,
                          max_words=max_words).generate(text)
    display = plt.imshow(wordcloud, interpolation='bilinear')
    display= plt.axis("off")
    display= plt.show()
    return display
