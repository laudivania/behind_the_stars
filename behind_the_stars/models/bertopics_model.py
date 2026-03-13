
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd


# BerTopics model
def bertopics_model(text,
                    embedding_model="all-MiniLM-L6-v2",
                    nr_topics=None,
                    min_topic_size=20,
                    seed_topic_list=None,
                    random_state=42):
    """Embed the model and have topics as output"""
    embedding_model= SentenceTransformer(embedding_model)

    topic_model= BERTopic(
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,    # to avoid micro topics
        seed_topic_list=seed_topic_list)  # the topics we want the model to converge around

    topics, probs = topic_model.fit_transform(text)

    return topics, probs, topic_model


#Print words per topics
def print_bertopic_topics(topic_model, n_words=10):
    """
    Display the main words for each topic.
    """
    topic_info = topic_model.get_topic_info()

    # Exclude the -1 topic (outliers)
    valid_topics = topic_info[topic_info.Topic != -1]["Topic"].tolist()

    for topic_num in valid_topics:
        words_scores = topic_model.get_topic(topic_num)
        # To have unique words
        words = [w[0] for w in words_scores[:n_words]]
        print(f"Topic {topic_num}: {', '.join(words)}\n")


#Convert each topic into feature
def bertopic_features(df, text, topic_model, topics,
                      custom_topic_labels=None,
                      top_words=2):
    """ Convert each topic into a feature, and choose
    the topics we want to model to converge around
    """

    # To have the probability of each topic on the review
    topic_distr, _ = topic_model.approximate_distribution(text)
    #Converting it into a dataframe
    probs_df = pd.DataFrame(topic_distr)

    # Create the labels of each topic
    topic_labels = {}

    for topic in topic_model.get_topic_info()["Topic"]:
        #Exclude outliers
        if topic == -1:
            continue

        # If a list of personalised topics is provided
        if custom_topic_labels and topic in custom_topic_labels:
            topic_labels[topic] = custom_topic_labels[topic]

        else:
            words = topic_model.get_topic(topic)
            label = "_".join([w for w, _ in words[:top_words]])
            topic_labels[topic] = label

    # Rename the columns
    probs_df.columns = [
        topic_labels.get(i, f"topic_{i}")
        for i in range(probs_df.shape[1])]

    # Create the main topic column
    topic_main_names = [
        topic_labels.get(t, "outlier")
        for t in topics]

    df_features = df.copy()
    df_features["topic_main"] = topic_main_names

    # Contactenation of the features
    df_features = pd.concat(
        [df_features.reset_index(drop=True),
         probs_df.reset_index(drop=True)],
        axis=1
    )

    return df_features
