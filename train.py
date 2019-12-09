import pandas as pd
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, LSTM
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def main():
    df = ingest_data()
    df = fix_label(df)
    print(df['label'].head(10))
    df = df.reindex(np.random.permutation(df.index))
    print(df['label'].head(10))
    #df = df[:250000]
    df = tokenize_samples(df)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(df['tokens']),
        np.array(df['label']),
        test_size=0.2
    )
    
    print(x_train[:10])
    print(x_train.shape)
    print(y_train.shape)

    # Create and train word2vec model
    w2v = Word2Vec(size=200, min_count=10)
    w2v.build_vocab(x_train)
    w2v.train(x_train, total_examples=w2v.corpus_count, epochs=w2v.iter)
    print(w2v.most_similar('good'))

    pad_idx = w2v.wv.vocab['0'].index
    print(x_train[0])
    print(y_train[0])
    x_train = vectorize_tweet(w2v, x_train, pad_idx)
    x_test = vectorize_tweet(w2v, x_test, pad_idx)

    # Create embedding layer from w2v model to be passed into Keras as layer
    embedding_layer = w2v.wv.get_keras_embedding(train_embeddings=False)
    w2v.save('w2v.model')
    
    # Create, train, save, and test Keras model
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, verbose=1)
    model.save('model.h5')

    score, acc = model.evaluate(x_test, y_test, verbose=2)

    y_pred1 = model.predict(x_test)
    y_pred = np.argmax(y_pred1, axis=1)

    # Print f1, precision, and recall scores
    print(precision_score(y_test, y_pred , average="micro"))
    print(recall_score(y_test, y_pred , average="micro"))
    print(f1_score(y_test, y_pred , average="micro"))

    print("score: ", score)
    print("acc: ", acc)


# Fixes labels: 0 = Negative, 1 = Positive
def fix_label(df):
    df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    return df

# Stores data in Pandas DataFrame
def ingest_data():
    col_names = ['label', "ids", "date", "flag", "user", "text"]
    encoding = "ISO-8859-1"
    df = pd.read_csv('twitter_sa_train.csv', encoding=encoding, names=col_names)
    df = df[['label', 'text']]
    return df


# Tokenizes text in DataFrame
def tokenize_samples(df):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokenize = lambda tweet : tokenizer.tokenize(tweet)
    df['tokens'] = df['text'].map(tokenize)
    df['tokens'] = df['tokens'].map(clean_tweet)
    df['tokens'] = df['tokens'].map(pad_tweet)
    return df

# Removes '#', links from tweet
def clean_tweet(tokens):
    tokens = [t for t in tokens if not t.startswith('#') and not t.startswith('http')]
    return tokens


# Pads tweet to 32 tokens
def pad_tweet(tokens):
    if len(tokens) >= 32:
        return tokens[:32]
    while len(tokens) < 32:
        tokens.append('0')
    return tokens


# Vectorizes tweet tokens in dataset
def vectorize_tweet(model, x_train, pad_idx):
    source_word_indices = []
    for i in range(len(x_train)):
        source_word_indices.append([])
        for j in range(len(x_train[i])):
            word = x_train[i][j]
            if word in model.wv.vocab:
                word_index = model.wv.vocab[word].index
                source_word_indices[i].append(word_index)
            else:
                # Replace with padding index if not in vocab
                source_word_indices[i].append(pad_idx)
    source = np.array(source_word_indices)
    return source



if __name__ == "__main__":
    main()


