import numpy as np
import tensorflow as tf
import pymongo
import tweepy
import json
import requests
from credentials import *
from nltk import TweetTokenizer
from keras.models import load_model
from train import clean_tweet, pad_tweet, vectorize_tweet
from gensim.models import Word2Vec


# Vectorizes a tweet
def prepare_tweet(tweet):
    tokens = tokenizer.tokenize(tweet)
    tokens = clean_tweet(tokens)
    tokens = pad_tweet(tokens)
    vect = []
    for token in tokens:
        if token in w2v.wv.vocab:
            word_index = w2v.wv.vocab[token].index
            vect.append(word_index)
        else:
            vect.append(w2v.wv.vocab['0'].index)  # 0 is padding idx
    
    return vect

def predict(tweet):
    print(tweet)
    input_vector = prepare_tweet(tweet)
    print(input_vector)

    # Needed to serve saved model with Flask
    res = model.predict(np.array([input_vector]))
    p = res[0][0]
    
    p_scaled = (p * 2) - 1
    print(p_scaled)

    return p_scaled


# Load w2v model, tokenizer, Keras neural network model
model = load_model('model.h5')
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
w2v = Word2Vec.load('w2v.model')

# Credentials
connection_string = get_connection_string()
consumer_key = get_consumer_key()
consumer_secret = get_consumer_secret()
access_token = get_access_token()
access_token_secret = get_access_token_secret()

# Connect to MongoDB client using credentials and navigate to collection
connection_string = get_connection_string()
client = pymongo.MongoClient(connection_string)
my_db = client['test']
my_col = my_db['candidates']

# Connect to Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

candidates = ['joe biden', 'cory booker', 'pete buttigieg', 'tulsi gabbard',
    'kamala harris', 'amy klobuchar', 'bernie sanders', 'tom steyer', 
    'elizabeth warren', 'andrew yang']

for candidate in candidates:
    qry = "{} -filter:retweets".format(candidate)
    #max_id = None
    max_id_cur = my_db['query_ids'].find({"name" : candidate})
    try:
        max_id = max_id_cur.next()['last_id']
    except:
        max_id = None
    print("Starting at tweet id:", max_id, "for candidate:", candidate)
    
    curr_first_tweet_id = None
    curr_last_tweet_id = None
    # get n*100 tweets for the candidate, maximum 180 queries per 15 min=18k tweets
    for i in range(1):
        tweets_about_candidate = api.search(qry, count=100, max_id=max_id)
        for i, tweet in enumerate(tweets_about_candidate):
            #print(i, tweet.text)
            if not curr_first_tweet_id:
                curr_first_tweet_id = tweet.id
            max_id = tweet.id
        curr_last_tweet_id = max_id

        for tweet in tweets_about_candidate:
            prediction = predict(tweet.text)

            row = {
                "name": candidate,
                "time": tweet.created_at,
                "sentiment": prediction
            }
            my_col.insert_one(row)
    

    # Store last tweet ID for each candidate so we know where to start off next query
    id_col = my_db['query_ids']
    id_row = {
        "name": candidate,
        "first_id": curr_first_tweet_id,
        "last_id": curr_last_tweet_id
    }
    id_col.update_one({"name": candidate}, {"$set": {"last_id": curr_last_tweet_id} }, upsert=True)





