import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


def lda_topics(n_components, vectorized_data, max_iter, random_state=42):
    """"Latent Dirichlet Allocation to find main topics on reviews"""
    model = LatentDirichletAllocation(n_components=n_components,
                                      max_iter = max_iter,
                                      random_state=random_state)
    return model.fit_transform(vectorized_data)


def document_mixture(model, vectorized_data):
    """ Creation of the documents mixture function,
    that will give us the weight of each sentence in the document"""
    doc_mixture = model.transform(vectorized_data)
    return doc_mixture


def topic_mixture(model,vectorizer):
    """ Creation of the topics mixture function,
    that will give us the weight of each word in the sentence"""
    topic_word_mixture = pd.DataFrame(model.components_,
                        columns = vectorizer.get_features_name_out())
    return topic_word_mixture


def print_topics(model, vectorizer):
    """Print the words associated to the potential topics """
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names_out()[i], topic[i])
                        for i in topic.argsort()[:-10 - 1:-1]])
