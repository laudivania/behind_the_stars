from sklearn.decomposition import NMF
from pandarallel import pandarallel
import joblib

pandarallel.initialize(progress_bar=True)


def nmf_model(vectorized_data,num_topics, random_state=None):
    """Non-negative matrix factorization, will give us the weigh of the sentence
    on the document and the weight of the word on the sentence"""
    model = NMF (n_components=num_topics,random_state=random_state)
    nmf_matrix = model.fit_transform(vectorized_data)
    joblib.dump(model,'model_saved/nmf.pkl')
    return nmf_matrix, model


def print_topics_with_weights(model, vectorizer, top_n):
    """ To print the words impacting each topic the most and
    the associated weights"""
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):

        print(f"\nTopic {topic_idx + 1}")
        top_words = topic.argsort()[:-top_n - 1:-1]
        for i in top_words:
            print(words[i], round(topic[i], 3))
