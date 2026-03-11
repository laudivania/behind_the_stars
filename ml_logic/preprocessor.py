import pandas as pd
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import TextVectorization
#fonction customizable de prepoc
def preprocessing(sentence, minuscule = True, ponctuation = True, stopwords = True):
    sentence = sentence.strip()
    if minuscule:
        sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
    if ponctuation:
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')
    if stopwords:
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


#For Deep Learning, TextVectorization
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
    vectorizer = Textvectorizer(
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
