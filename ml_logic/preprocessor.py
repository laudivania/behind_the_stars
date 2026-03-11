import pandas as pd
import string
from nltk import word_tokenize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import TextVectorization

def preprocessing(sentence, minuscule = "minuscule", ponctuation = "ponctuation"):
    sentence = sentence.strip()
    if minuscule == "minuscule":
        sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
    if ponctuation == "ponctuation":
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')
    token_sized_sentence = word_tokenize(sentence)
    liste = [WordNetLemmatizer().lemmatize(word, pos = "v") # v --> verbs
    for word in token_sized_sentence]
    sentence =  " ".join(liste)
    token_sized_sentence = word_tokenize(sentence)
    liste = [WordNetLemmatizer().lemmatize(word, pos = "n") # n --> nouns
    for word in token_sized_sentence]
    sentence =  " ".join(liste)
    return sentence

def preprocessing_stop_words(sentence, minuscule = "minuscule", ponctuation = "ponctuation"):
    sentence = sentence.strip()
    if minuscule == "minuscule":
        sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
    if ponctuation == "ponctuation":
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')
    stop_words = set(stopwords.words('english'))
    token_sized_sentence = word_tokenize(sentence)
    liste = [w for w in token_sized_sentence if w not in stop_words]
    sentence = " ".join(liste)
    token_sized_sentence = word_tokenize(sentence)
    liste = [WordNetLemmatizer().lemmatize(word, pos = "v") # v --> verbs
    for word in token_sized_sentence]
    sentence =  " ".join(liste)
    token_sized_sentence = word_tokenize(sentence)
    liste = [WordNetLemmatizer().lemmatize(word, pos = "n") # n --> nouns
    for word in token_sized_sentence]
    sentence =  " ".join(liste)
    return sentence


from nltk.stem import WordNetLemmatizer
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer

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


#For Machine Learning
def vectorizing(preprocessed_data):
    """Vectorizing the data with TfidVectorizer after the preprocessing
    in order to used it on the models"""
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(preprocessed_data)
    vectorized_data = pd.DataFrame(vectorized_data.toarray(),
                    columns = vectorizer.get_feature_names_out())
    return vectorized_data


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
<<<<<<< HEAD
    vectorizer = Textvectorizer(
=======
    vectorizer = TextVectorization(

>>>>>>> main
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
