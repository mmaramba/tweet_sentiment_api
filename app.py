import numpy as np
import random
import tensorflow as tf
import json
import pymongo
from datetime import datetime
from credentials import *
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from train import clean_tweet, pad_tweet, vectorize_tweet
from nltk import TweetTokenizer
from gensim.models import Word2Vec


app = Flask(__name__)
CORS(app)
model = None
w2v = None
tokenizer = None
graph = None
client = None


# Loads NLTK tweet tokenizer, Keras model, and word2vec model
def load_models():
    global model
    global tokenizer
    global w2v
    global graph
    graph = tf.get_default_graph()
    model = load_model('model.h5')
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    w2v = Word2Vec.load('w2v.model')

def db_connect():
    global client
    connection_string = get_connection_string()
    client = pymongo.MongoClient(connection_string)
    print(client)


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


def get_candidate_from_db(candidate, start_time, end_time):
  # Convert [first]-[last] to [first] [last] to search in DB
  candidate_name = ' '.join(candidate.split('-'))
  format_time = lambda x : datetime.fromtimestamp(int(x) // 1000)
  start, end = format_time(start_time), format_time(end_time)
  candidate_data = client['test']['candidates'].find({
    'name': candidate_name,
    'time': {
      '$gte': start,
      '$lt': end
    }
  })

  # create 12 data points
  start, end = int(start_time), int(end_time)
  step = (end - start) / 12
  bins = [[] for i in range(12)]
  for data in candidate_data:
    index = int(((datetime.timestamp(data['time']) * 1000) - start) / step)
    bins[index].append(data)
  average_sentiment = lambda data : (sum([entry['sentiment'] for entry in data]) / len(data)
    if len(data) > 0 else 0)
  return {
    'success': True,
    'id': candidate,
    'data': [{
      'time': int(start + step * i),
      'sentiment': average_sentiment(bins[i])
    } for i in range(len(bins))]
  }


# Get sentiment analysis data for a candidate
@app.route('/candidate/<candidate>', methods=['GET'])
def getCandidate(candidate):
  start_time = request.args.get('start_time')
  end_time = request.args.get('end_time')
  result = get_candidate_from_db(candidate, start_time, end_time)
  return jsonify(result)


# Get sentiment analysis data for all candidates
@app.route('/all', methods=['GET'])
def getCandidates():
  candidates = [
    'joe-biden',
    'cory-booker',
    'pete-buttigieg',
    'tulsi-gabbard',
    'kamala-harris',
    'amy-klobuchar',
    'bernie-sanders',
    'tom-steyer',
    'elizabeth-warren',
    'andrew-yang'
  ]
  start_time = request.args.get('start_time')
  end_time = request.args.get('end_time')
  result = [get_candidate_from_db(candidate, start_time, end_time) for candidate in candidates]
  return jsonify(result)


# Predicts sentiment of tweet
@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}
    if request.method == 'POST':
        content = request.json
        print(type(content))
        if 'tweet' in content:
            print(content['tweet'])
            in_vec = content['tweet']

            input_vector = prepare_tweet(in_vec)
            print(input_vector)

            # Needed to serve saved model with Flask
            with graph.as_default():
                res = model.predict(np.array([input_vector]))
                p = res[0][0]
                print(p)
                # For testing purposes
                if p < 0.1:
                  p = -1
                elif p > .9:
                  p = 1
                else:
                  p = 0
            
            data['success'] = True
            data['prediction'] = p

    return jsonify(data)

if __name__ == '__main__':
    print(('* Loading Keras model and Flask starting server...'
        'please wait until server has fully started'))
    load_models()
    db_connect()
    app.run()
