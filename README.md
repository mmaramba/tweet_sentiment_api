# tweet_sentiment_api
Tweet Sentiment Analysis API made with Keras, Flask, word2vec via gensim

Start API on http://localhost:5000
```
python3 app.py
```

Post to API at http://localhost:5000/predict:
```
{
	"tweet": "This is a sad tweet. I hate my life."
}
```

Returns:
```
{
    "prediction": 0,
    "success": true
}
```

Link to dataset:
https://www.kaggle.com/kazanova/sentiment140/
