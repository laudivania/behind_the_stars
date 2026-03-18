from sentence_transformers import SentenceTransformer
import numpy as np
# from preprocessor import fine_cleaning
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from transformers import AutoTokenizer, TFAutoModel
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import os
import joblib
import tqdm

_BERT_MODEL = None

def get_bert_model():
    global _BERT_MODEL
    if _BERT_MODEL is None:
        _BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _BERT_MODEL

def process_embed_bert(list_reviews):
    ''' Fonction renvoyant la moyenne de l'embedding d'une liste de reviews (par resto)
    Moyenne des embeddings plutôt que CLS car ce transformer est fait pour mieux comprendre le contexte
    dans les phrases.
    list_reviews: Une liste de reviews CLEAN (par resto)

    Retourne un array de la moyenne des embeddings (taille 384)'''

    embedder = get_bert_model()
    review_embed = embedder.encode(list_reviews, batch_size=32,show_progress_bar=True)
    mean_embedding = np.mean(review_embed,axis=0)

    return mean_embedding

def embedding_bert(reviews_list,output_path):
    '''Fonction gérant l'embedding, ainsi que l'enregistrement de ses derniers.
        reviews_list = Liste de reviews du resto
        output_path = Chemin d'enregistrement (à adapter pour enregistrer où on veut)'''
    model = get_bert_model()
    if os.path.exists(output_path):
        results = joblib.load(output_path)
        print(f"Checkpoint trouvé : {len(results)} restos déjà traités")
    else:
        results = {}
        print("Pas de checkpoint, start from scratch")

    data_to_process = reviews_list.to_dict('records')
    for i, row in enumerate(tqdm(data_to_process)):
        bid = row['business_id']

        if bid in results:
            continue

        reviews = row['text']
        clean_reviews = [fine_cleaning(r) for r in reviews if isinstance(r, str) and len(r) > 2]

        if not clean_reviews:
            results[bid] = np.zeros(384)
            continue

        embeddings = model.encode(clean_reviews, batch_size=64, show_progress_bar=False)

        results[bid] = np.mean(embeddings, axis=0)

        if (i + 1) % 200 == 0:
            joblib.dump(results, output_path)

    joblib.dump(results, output_path)
    print(f"\n Terminé : {len(results)} restos sont sauvegardés.")
    return results

def process_embed_word2vec(list_reviews_seq, model):
    ''' Fonction renvoyant la moyenne de l'embedding d'une liste de reviews (par resto)
    Model à initier avant pour qu'il puisse'''

    review_vector = []

    for review in list_reviews_seq:
        valid_words = [model.wv[word] for word in review if word in model.wv]
        if valid_words:
            review_vector.append(np.mean(valid_words,axis=0))
    if not review_vector:
        return np.zeros(300)

    return np.mean(review_vector,axis=0)


def get_recommendations_for_new_resto(new_embedding, small_embeddings_dict, df_meta, top_n=5):
    """
    new_embedding : le vecteur (384,) généré en live pour le nouveau resto
    small_embeddings_dict : ton dataset restreint d'embeddings {bid: vector}
    df_meta : le dataframe contenant les noms/infos des restos du dataset restreint
    Retourne une liste de dictionnaires de top_n (int) restos similaires, sous le format {name,city,state,business_id,similarity}
    """

    target_vector = new_embedding.reshape(1, -1)

    all_bids = list(small_embeddings_dict.keys())
    all_vectors = np.array(list(small_embeddings_dict.values()))

    scores = cosine_similarity(target_vector, all_vectors)[0]

    best_indices = scores.argsort()[-top_n:][::-1]

    recommendations = []

    for idx in best_indices:
        bid = all_bids[idx]
        similarity = scores[idx]

        match_info = df_meta[df_meta['business_id'] == bid]

        if not match_info.empty:
            name = match_info['name'].values[0]
            city = match_info['city'].values[0]
            state = match_info['state'].values[0]
            resto_id = match_info['business_id'].values[0]
            print(f"[{similarity:.3f}] {name} ({city} - {state}) - {resto_id}")

            recommendations.append({
                "name": name,
                "city": city,
                "state": state,
                "business_id": resto_id,
                "similarity": similarity
            })

    return recommendations

if __name__=='__main__':

    test = ['The food here is very authentic and probably the best you can find in the immediate area of Lindenwold/Clementon. They have tamales, pupusa, empanadas, and much more. The only down side about this place is that the owners/workers do not speak a lick of english so it can be hard to order and/or to figure out what you are really getting. The menu is partly in english. Regardless, I keep going back. Just gotta brush up on your spanish before you go.',
 'Take out preferred. Not much atmosphere but very good salvadorian food. Papuseria with fresh choices and very good chicken dishes.\nVery reasonable prices and large portions',
 'Dirty, trash all over ,rude clientele ,do not speak english a scar in Clementon.',
 "Very good San Salvadorian place ! Authentic and yes the owner and workers speak broken English however it doesn't really matter since 95% of there patrons don't speak English either. However if you don't mind being cultured and  want great authentic food cooked by people from San Salvadore this is the place to go. It's not going to win a beauty contest but I would say the atmosphere is very authentic too. It's cheap and portions are large and I love the the rice milk beverage that taste great and refreshing on a hot day and please try the the pupusa! The squash one are really good. It's a hidden gem for people who know food!  Besides they can please the picky eaters like My kids who loved the food at this place ; )",
 'I have been to El Salvador and this is good, authentic Salvadoran food. I tried the pupusas de calabaza (squash) and the empanadas de platano con frijoles (plantain empanadas with refried bean stuffing), both were good. the decor is an almost overwhelming bombardment of blue and white (colors of the flag of El Salvador), the environment is clean and pleasant. Friendly people. I can see I will be pit stopping here on the way to or from the Lindenwold PATCO station.',
 'Dont be fooled by the 3 star rating.  This place is great.  Couple of problems hold it back from a higher rating.  \n\nThe food is awesome but be prepared to wait.  It is super slow.  Also, no ac so it got pretty warm.  Limited parking, just hit the side street.\n\nOn to the good stuff...i had never had el salvadorian food before and it was a great surprise.  Do not miss the squash pupusaria, heads and shoulders above the rest.  Empaadas with plantans and beans was pretty great too.\n\nGreat for a low key inexpensive dinner.  If you are on a budget i think you could easily have a memorable dinner for 2 under $25.',
 'Knowing a thing or two about El Salvadorian food, I can say these papusas are authentic. Looking forward to a return trip soon.\n\nThe only thing holding this back from a 5 star is the wait and disorganization but that is also part of the charm.',
 'What a great find in Clementon, NJ. Authentic pupusas like they are made in El Salvador! I am definitely coming back when I am in the neighborhood!',
 "I have eaten a lot of pupusas. these were not the best and these were not the worst. The place itself was pretty cool the staff was very friendly. The curtido didn't cut it for me. Just average. If I was close by and craving El Salvadorian food I'd go back.",
 '5 stars for food, 3 stars for service. Food is amazing, really, wow. But when I last went I ordered a dish with Spanish sausage, but got chicken! The chicken was delicious, so I didnt bother to complain, but such slip ups shouldnt be made. Also, the food takes forever!!! But it is absolutely delicious otherwise. Would highly reccomend.',
 "Building very clean, service slow but that's customary with most ethnic food joints. Prices better then Tejas Grill and La Esperanza but menu is small and not impressive. Slight Communication barrier if you don't speak spanish. They offer typical Mexican dishes that are not authentic to El Salvador, only reason I go are for the homemade delicious pupaseras. Squash and cheese is the best, horchata is good also. The locals all get the soups on the weekend. Chicken tamales were a good deal at 2 for $3 but nothing special at all. Sopes here are much different then Mexican, good flavor but come in a small crispy almost ice cream cone like shell. Not the tortilla I'm used to, wouldn't order that again. Only go for those fresh pupaseras.",
 "If I could give 0 stars, I would. Ordered some pupusas, the menu said that an order contained 3 and cost $1.75- charged me $1.75 per for a total of $5.25. I had ordered lunch too, so I only ate 1. When I told the waitress I wasn't paying $5.25 for it, especially when I only ate 1- she proceeded to call the cops. Menu is very misleading. An order actually isn't 3, it's 1 or you have to order at least 3. I don't know, none of any of this was in writing and the waitress didn't tell us, needless to say - I won't be going back there.",
 "Didn't even get to try the papusas. Went in and ordered two orders. I left after waiting 45 minutes and no food. They started serving food to people who came after me. Horrible service. Won't be back.",
 "This place makes the best, most authentic pupusas in the area. Service can be slow but it's part of the charm. Be prepared to speak Spanish and enjoy delicious food!"]

    test_clean = [fine_cleaning(x) for x in test]
    word_seq = [text_to_word_sequence(_) for _ in test_clean]

    embed_bert = process_embed_bert(test_clean)
    print(f"Shape BERT: {embed_bert.shape}")

    model = Word2Vec(sentences=word_seq,vector_size=300,window=5,min_count=2,workers=4,sg=1)

    embed2vec = process_embed_word2vec(word_seq,model)
    print(f"Shape Word2Vec: {embed2vec.shape}")


def embedding_by_batch(df,tokenizer='tiny_bert', model='tiny_bert'):
    batch_size = 512  # à ajuster selon ta RAM/GPU
    embeddings_batches = []
    if tokenizer == 'tiny_bert':
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", dtype="auto", padding = "right")
        model = TFAutoModel.from_pretrained("prajjwal1/bert-tiny", from_pt = True)

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_texts = df['text'].iloc[start:end].apply(lambda x: ' '.join(x)).tolist()

        # Tokenize
        encodings = tokenizer(batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='tf')

        # Embedding BERT
        outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
        # Utiliser le [CLS] token ou la moyenne des tokens
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)

        embeddings_batches.append(cls_embeddings)

    # Combiner tous les batches
    embeddings = tf.concat(embeddings_batches, axis=0)
    print(embeddings.shape)  # (34562, 128) si BERT-tiny
    return embeddings
