import pandas as pd
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.layers import TextVectorization


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
