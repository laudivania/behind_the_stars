
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


#Predefining a list of seed topics
seed_topics_list = [["food","taste","delicious","flavor","dish","meal"],
["service","staff","waiter","friendly","rude","attentive"],
["slow","wait","long","minutes","quick","fast"],
["price","expensive","cheap","value","overpriced"],
["wrong","order","missing","incorrect"],
["clean","dirty","bathroom","hygiene"],
["atmosphere","music","loud","quiet","ambiance"],
["parking","location","downtown"],
["manager","management","organized","chaotic"],
["portion","small","large","size"]]

# BerTopics model
def bertopic_model(text,
                    embedding_model="all-MiniLM-L6-v2",
                    nr_topics=None,
                    min_topic_size=20,
                    seed_topic_list=seed_topics_list,
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


#Defining a list of topics to be explored
topic_labels = {
    0: "food_quality",
    1: "service",
    2: "waiting_time",
    3: "price_value",
    4: "order_accuracy",
    5: "cleanliness",
    6: "atmosphere",
    7: "location_access",
    8: "management",
    9: "portion_size"}

#Convert each topic into feature
def bertopic_features(df, text, topic_model, topics,
                      custom_topic_labels=topic_labels,
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
        axis=1)

    return df_features


def add_restaurant_topic_features(df, text_col="text_cleaned",
                                  restaurant_col="business_id", topic_col="topic_main", date_col ="date"):
    """
    Build a restaurant-level dataset combining:
    - original business dataset (target, metadata)
    - aggregated sentiment features
    - topic sentiment
    - topic volume
    - temporal features (sentiment trend and review growth)

    Returns:
        dataframe with one row per restaurant
    """

    #A copy of the original df
    df = df.copy()

    #Formating date
    df[date_col] = pd.to_datetime(df[date_col])

    # Keep the original columns
    restaurant_base = df.drop_duplicates(subset=restaurant_col)

    #Instantiation of the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Sentiment per review (applying analyzer)
    df["sentiment"] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

    # Select the negative reviews
    df["negative_review"] = df["sentiment"] < -0.05

    # Aggregate restaurants, calculate the average sentiment and negative reviews and count the number of topics per restaurant
    agg_global = df.groupby(restaurant_col).agg(
        avg_sentiment=("sentiment", "mean"),
        negative_review_ratio=("negative_review", "mean"),
        review_count=(text_col, "count")).reset_index()

    # Merge global and aggregated df
    df = df.merge(agg_global, on=restaurant_col, how="left")

    # Compute average sentiment per topic, in order to know how it affect restaurants
    topic_sentiment = (
        df.groupby([restaurant_col, topic_col])["sentiment"]
        .mean()
        .unstack()
        .add_prefix("sentiment_")
        .reset_index())

    # volume (proportion) per topic
    topic_count = (
        df.groupby([restaurant_col, topic_col]).size()
        .unstack()
        .fillna(0)
        .reset_index())

    # Normalization based on the total of reviews per restaurant
    total_reviews = df.groupby(restaurant_col)[text_col].count().reset_index(name="total_reviews")
    topic_volume = topic_count.merge(total_reviews, on=restaurant_col)

    topic_cols = [c for c in topic_count.columns if c != restaurant_col]
    for c in topic_cols:
        topic_volume[c] = topic_volume[c] / topic_volume["total_reviews"]
    topic_volume = topic_volume.drop(columns=["total_reviews"]).add_prefix("topic_volume_")
    topic_volume = topic_volume.rename(columns={f"topic_volume_{restaurant_col}": restaurant_col})

    # Merge sentiment and volume
    topic_features = topic_sentiment.merge(topic_volume, on=restaurant_col, how="outer")

    # Reorganise columns: sentiment_X / topic_volume_X side by side
    cols_ordered = [restaurant_col]
    topics = [c.replace("sentiment_", "") for c in topic_sentiment.columns if c.startswith("sentiment_")]
    for t in topics:
        cols_ordered.append(f"sentiment_{t}")
        cols_ordered.append(f"topic_volume_{t}")
    topic_features = topic_features[cols_ordered]

    # Merge all restaurant features
    restaurant_features = agg_global.merge(topic_features, on=restaurant_col, how="left")

    # Merge with original business dataset (target, metadata)
    df_business = restaurant_base.merge(restaurant_features, on=restaurant_col, how="left")

    # Fill missing topics
    df_business = df_business.fillna(0)

    return df_business
