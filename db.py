import pymongo
import tweepy
import json
import requests
from credentials import *


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

# Idea: Store max_id values for each candidate

candidates = ['joe biden', 'cory booker', 'pete buttigieg', 'tulsi gabbard',
    'kamala harris', 'amy klobuchar', 'bernie sanders', 'tom steyer', 
    'elizabeth warren', 'andrew yang']

for candidate in candidates:
    qry = "{} -filter:retweets".format(candidate)
    max_id = None

    # get n*100 tweets for the candidate, maximum 180 queries per 15 min=18k tweets
    for i in range(1):
        tweets_about_candidate = api.search(qry, count=5, max_id=max_id)
        for i, tweet in enumerate(tweets_about_candidate):
            print(i, tweet.text)
            max_id = tweet.id

        for tweet in tweets_about_candidate:
            url = 'http://localhost:5000/predict'
            headers = {'content-type': 'application/json'}
            my_obj = {"tweet": tweet.text}
            res = requests.post(url, json=my_obj, headers=headers)
            
            decoded = json.loads(res.text)
            print(decoded['prediction'])
            print(candidate)
            print(tweet.created_at)
            row = {
                "name": candidate,
                "time": tweet.created_at,
                "sentiment": decoded['prediction']
            }
            my_col.insert_one(row)

# Get all last 7 days:
# 1. Query until 18k is reached (n*100)
# 2. Wait 15 minutes
# 3. Go back to step 1 until no more tweets are retrieved from qry
# 4. Repeat every week