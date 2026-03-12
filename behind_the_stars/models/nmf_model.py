from sklearn.decomposition import NMF
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


def nmf_model(vectorized_data,num_topics random_state=None):
    """Non-negative matrix factorization, will give us the weigh of the sentence
    on the document and the weight of the word on the sentence"""
    nmf_model = NMF (n_components=num_topics)
    nmf_matrix = nmf_model.fit_transform(vectorized_data)
    return nmf_matrix


def print_topics_with_weights(model, vectorizer, top_n):
    """ To print the words impacting each topic the most and
    the associated weights"""
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):

        print(f"\nTopic {topic_idx + 1}")
        top_words = topic.argsort()[:-top_n - 1:-1]
        for i in top_words:
            print(words[i], round(topic[i], 3))
