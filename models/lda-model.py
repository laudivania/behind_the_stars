import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


def lda_topics(n_components, vectorized_data, max_iter):
    """"Latent Dirichlet Allocation to find main topics on reviews"""
    model = LatentDirichletAllocation(n_components=n_components, max_iter = max_iter)
    return model.fit(vectorized_data)


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
