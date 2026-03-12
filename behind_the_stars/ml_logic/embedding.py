from sentence_transformers import SentenceTransformer
import numpy as np
from preprocessor import fine_cleaning

def process_embed(list_reviews):
    ''' Fonction renvoyant la moyenne de l'embedding d'une liste de reviews (par resto)
    Moyenne des embeddings plutôt que CLS car ce transformer est fait pour mieux comprendre le contexte
    dans les phrases.
    list_reviews: Une liste de reviews (par resto)

    Retourne un array de la moyenne des embeddings (taille 384)'''

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    review_embed = embedder.encode(list_reviews, batch_size=32)
    mean_embedding = np.mean(review_embed,axis=0)

    return mean_embedding

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
    embed = process_embed(test_clean)
    print(embed, embed.shape)
