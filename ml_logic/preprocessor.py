import pandas as pd
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


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


def preprocessing(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
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
