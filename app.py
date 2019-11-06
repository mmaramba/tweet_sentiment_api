# import the necessary packages
import numpy as np
import io
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.models import load_model
from train import clean_tweet, pad_tweet, vectorize_tweet
from nltk import TweetTokenizer
from gensim.models import Word2Vec

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None
w2v = None
tokenizer = None
graph = None

def load_models():
    global model
    global tokenizer
    global w2v
    global graph
    graph = tf.get_default_graph()
    model = load_model('model.h5')
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    w2v = Word2Vec.load('w2v.model')

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
    
    #vect_np = np.array(vect)
    return vect

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        content = request.json
        if 'tweet' in content:
            print(content['tweet'])
            in_vec = content['tweet']

            input_vector = prepare_tweet(in_vec)
            print(input_vector)
            print(len(input_vector))



            with graph.as_default():
                res = model.predict(np.array([input_vector]))
                print(len(res))
                print("PREDICTION:", res)
                p = res[0][0]
            
            # indicate that the request was a success
            data["success"] = True
            data["prediction"] = int(round(p))

    # return the data dictionary as a JSON response
    return jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_models()
    app.run()
