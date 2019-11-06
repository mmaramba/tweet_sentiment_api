import numpy as np
import random
import tensorflow as tf
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


# Get sentiment analysis data for candidate
@app.route("/<candidate>", methods=["GET"])
def getCandidate(candidate):
  data = {
    'success': True,
    'id': candidate,
    # TODO get real data
    'data': [(-1 + random.random() * 2) for _ in range(10)]
  }
  return jsonify(data)


# Predicts sentiment of tweet
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if request.method == "POST":
        content = request.json
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
            
            data["success"] = True
            data["prediction"] = int(round(p))

    return jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_models()
    app.run()
