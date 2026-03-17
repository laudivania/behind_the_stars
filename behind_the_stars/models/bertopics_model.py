
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
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
                    embedding_model="paraphrase-mpnet-base-v2",
                    nr_topics=10,
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

    joblib.dump(topic_model,'behind_the_stars/model_saved/bertopic.pkl')

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

#--------------# Master function for topics #--------------#
def master_topics(
    df,
    model,
    text_col="text_cleaned",
    restaurant_col="business_id",
    date_col="date"
):
    """
    Master function:
    - Predict topics
    - Create topic features (review-level)
    - Aggregate to restaurant-level features

    Input:
        df with columns:
        - text_cleaned
        - business_id
        - date

    Output:
        restaurant-level dataframe
    """

    df = df.copy()

    # Predict topics
    topics, _ = model.transform(df[text_col])

    # Review-level features
    df = bertopic_features(
        df,
        text=df[text_col],
        topic_model=model,
        topics=topics
    )

    # Restaurant-level aggregation
    df_restaurant = add_restaurant_topic_features(
        df,
        text_col=text_col,
        restaurant_col=restaurant_col,
        topic_col="topic_main",
        date_col=date_col
    )

    return df_restaurant


"""A SUPPRIMER APRES LES TESTS DE SAUVEGARDE"""
# if __name__ == '__main':
#     text = 'The food here is very authentic and probably the best you can find in the immediate area of Lindenwold/Clementon. They have tamales, pupusa, empanadas, and much more. The only down side about this place is that the owners/workers do not speak a lick of english so it can be hard to order and/or to figure out what you are really getting. The menu is partly in english. Regardless, I keep going back. Just gotta brush up on your spanish before you go.',
#     'Take out preferred. Not much atmosphere but very good salvadorian food. Papuseria with fresh choices and very good chicken dishes.\nVery reasonable prices and large portions',
#     'Dirty, trash all over ,rude clientele ,do not speak english a scar in Clementon.',
#     "Very good San Salvadorian place ! Authentic and yes the owner and workers speak broken English however it doesn't really matter since 95% of there patrons don't speak English either. However if you don't mind being cultured and  want great authentic food cooked by people from San Salvadore this is the place to go. It's not going to win a beauty contest but I would say the atmosphere is very authentic too. It's cheap and portions are large and I love the the rice milk beverage that taste great and refreshing on a hot day and please try the the pupusa! The squash one are really good. It's a hidden gem for people who know food!  Besides they can please the picky eaters like My kids who loved the food at this place ; )",
#     'I have been to El Salvador and this is good, authentic Salvadoran food. I tried the pupusas de calabaza (squash) and the empanadas de platano con frijoles (plantain empanadas with refried bean stuffing), both were good. the decor is an almost overwhelming bombardment of blue and white (colors of the flag of El Salvador), the environment is clean and pleasant. Friendly people. I can see I will be pit stopping here on the way to or from the Lindenwold PATCO station.',
#     'Dont be fooled by the 3 star rating.  This place is great.  Couple of problems hold it back from a higher rating.  \n\nThe food is awesome but be prepared to wait.  It is super slow.  Also, no ac so it got pretty warm.  Limited parking, just hit the side street.\n\nOn to the good stuff...i had never had el salvadorian food before and it was a great surprise.  Do not miss the squash pupusaria, heads and shoulders above the rest.  Empaadas with plantans and beans was pretty great too.\n\nGreat for a low key inexpensive dinner.  If you are on a budget i think you could easily have a memorable dinner for 2 under $25.',
#     'Knowing a thing or two about El Salvadorian food, I can say these papusas are authentic. Looking forward to a return trip soon.\n\nThe only thing holding this back from a 5 star is the wait and disorganization but that is also part of the charm.',
#     'What a great find in Clementon, NJ. Authentic pupusas like they are made in El Salvador! I am definitely coming back when I am in the neighborhood!',
#     "I have eaten a lot of pupusas. these were not the best and these were not the worst. The place itself was pretty cool the staff was very friendly. The curtido didn't cut it for me. Just average. If I was close by and craving El Salvadorian food I'd go back.",
#     '5 stars for food, 3 stars for service. Food is amazing, really, wow. But when I last went I ordered a dish with Spanish sausage, but got chicken! The chicken was delicious, so I didnt bother to complain, but such slip ups shouldnt be made. Also, the food takes forever!!! But it is absolutely delicious otherwise. Would highly reccomend.',
#     "Building very clean, service slow but that's customary with most ethnic food joints. Prices better then Tejas Grill and La Esperanza but menu is small and not impressive. Slight Communication barrier if you don't speak spanish. They offer typical Mexican dishes that are not authentic to El Salvador, only reason I go are for the homemade delicious pupaseras. Squash and cheese is the best, horchata is good also. The locals all get the soups on the weekend. Chicken tamales were a good deal at 2 for $3 but nothing special at all. Sopes here are much different then Mexican, good flavor but come in a small crispy almost ice cream cone like shell. Not the tortilla I'm used to, wouldn't order that again. Only go for those fresh pupaseras.",
#     "If I could give 0 stars, I would. Ordered some pupusas, the menu said that an order contained 3 and cost $1.75- charged me $1.75 per for a total of $5.25. I had ordered lunch too, so I only ate 1. When I told the waitress I wasn't paying $5.25 for it, especially when I only ate 1- she proceeded to call the cops. Menu is very misleading. An order actually isn't 3, it's 1 or you have to order at least 3. I don't know, none of any of this was in writing and the waitress didn't tell us, needless to say - I won't be going back there.",
#     "Didn't even get to try the papusas. Went in and ordered two orders. I left after waiting 45 minutes and no food. They started serving food to people who came after me. Horrible service. Won't be back.",
#     "This place makes the best, most authentic pupusas in the area. Service can be slow but it's part of the charm. Be prepared to speak Spanish and enjoy delicious food!"

#     seed_topics_list = [["food","taste","delicious","flavor","dish","meal"],
#     ["service","staff","waiter","friendly","rude","attentive"],
#     ["slow","wait","long","minutes","quick","fast"],
#     ["price","expensive","cheap","value","overpriced"],
#     ["wrong","order","missing","incorrect"],
#     ["clean","dirty","bathroom","hygiene"],
#     ["atmosphere","music","loud","quiet","ambiance"],
#     ["parking","location","downtown"],
#     ["manager","management","organized","chaotic"],
#     ["portion","small","large","size"]]

#     bertopic_model(text,seed_topic_list=seed_topics_list)
